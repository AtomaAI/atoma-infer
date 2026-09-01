//! The dispatcher: owns dispatch truth for every live batch.

use serde::{Deserialize, Serialize};
use tracing::debug;
use validator::Validate;

use crate::attention::{BreakPoints, CaptureContract, GraphMode};
use crate::dispatch::{
    decide, BucketLadder, EagerFallbackCounters, EagerReason, GraphKey, LiveBatch, PaddingLookup,
};
use crate::types::RequestCount;

/// How the captured set was recorded: whole forward passes, or segments around eager regions.
///
/// Derived from the contract's break points and never configured — a pass with one standing over
/// it cannot be captured whole, and one with none has nothing to split around — so it stays
/// inside the dispatcher. What reaches a caller is the [`DispatchDecision`] it produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CaptureKind {
    /// Each bucket's graph records the whole forward pass.
    Full,
    /// Each bucket's captured pass is split into segments around eager operations.
    Segmented,
}

impl CaptureKind {
    /// How a pass with these break points standing over it is recorded.
    fn for_break_points(break_points: &BreakPoints) -> Self {
        if break_points.is_empty() {
            Self::Full
        } else {
            Self::Segmented
        }
    }
}

/// What an operator configures the dispatcher with, fixed for the process lifetime.
///
/// The support level and the capture kind are not here: backends declare the first and break
/// points settle the second, both through
/// [`CaptureContract`](crate::attention::CaptureContract) at startup. An operator who could write
/// either down could claim capture properties the backends never offered.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
pub struct DispatchConfig {
    /// The buckets the engine captured.
    pub bucket_ladder: BucketLadder,
    /// The largest request count any captured graph serves.
    pub captured_max_requests: RequestCount,
}

/// What the executor does with one live batch. Executors act on this without re-deriving it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchDecision {
    /// Replay the full captured graph the key selects.
    FullReplay(GraphKey),
    /// Replay the captured segments the key selects, with eager regions between them.
    SegmentedReplay(GraphKey),
    /// Run the whole step eagerly.
    Eager(EagerReason),
}

/// Owns dispatch truth: which captured graph serves a live batch, or why none does.
///
/// Built once at Allocation and never modified afterwards — no method changes the bucket ladder,
/// the captured set, the graph mode or the break points, so no code path recaptures at runtime.
/// Dispatch priority is full-graph replay, then segmented replay, then eager: a batch with a key
/// replays the whole pass when the captured set records whole passes, replays segments when it is
/// split, and every batch without one runs eagerly.
#[derive(Debug)]
pub struct Dispatcher {
    lookup: PaddingLookup,
    captured_max_requests: RequestCount,
    graph_mode: GraphMode,
    break_points: BreakPoints,
    fallbacks: EagerFallbackCounters,
}

impl Dispatcher {
    /// Builds the dispatcher from what an operator configured and what the backends and the model
    /// settled, deriving the dense padding lookup from the configured bucket ladder.
    #[must_use]
    pub fn new(config: &DispatchConfig, contract: &CaptureContract) -> Self {
        Self {
            lookup: PaddingLookup::new(&config.bucket_ladder),
            captured_max_requests: config.captured_max_requests,
            graph_mode: contract.graph_mode(),
            break_points: contract.break_points().clone(),
            fallbacks: EagerFallbackCounters::default(),
        }
    }

    /// Decides how `batch` runs, counting and logging the fallback when no graph serves it.
    pub fn dispatch(&mut self, batch: LiveBatch) -> DispatchDecision {
        match decide(
            batch,
            self.graph_mode,
            self.captured_max_requests,
            &self.lookup,
        ) {
            Ok(key) => match CaptureKind::for_break_points(&self.break_points) {
                CaptureKind::Full => DispatchDecision::FullReplay(key),
                CaptureKind::Segmented => DispatchDecision::SegmentedReplay(key),
            },
            Err(reason) => {
                self.fallbacks.count(&reason);
                debug!(%reason, "eager fallback");
                DispatchDecision::Eager(reason)
            }
        }
    }

    /// Eager fallbacks so far, by eager reason.
    #[must_use]
    pub fn fallbacks(&self) -> EagerFallbackCounters {
        self.fallbacks
    }

    /// Every site the pass leaves the graph: the union of what the backends declared they cannot
    /// capture and what the model declared it runs eagerly.
    #[must_use]
    pub fn break_points(&self) -> &BreakPoints {
        &self.break_points
    }
}

#[cfg(test)]
mod tests {
    use super::{CaptureKind, DispatchConfig, DispatchDecision, Dispatcher, EagerFallbackCounters};
    use crate::attention::{BackendDeclaration, CaptureContract, ModelDeclaration, SupportLevel};
    use crate::dispatch::test_support::batch;
    use crate::dispatch::{BucketLadder, EagerReason, Platform};
    use crate::test_support::{captured_log, contract, requests, site, tokens};

    fn dispatcher_under(contract: &CaptureContract) -> Dispatcher {
        Dispatcher::new(
            &DispatchConfig {
                bucket_ladder: BucketLadder::default_for(Platform::Hopper),
                captured_max_requests: requests(512),
            },
            contract,
        )
    }

    fn dispatcher(support_level: SupportLevel) -> Dispatcher {
        dispatcher_under(&contract(support_level, &[]))
    }

    #[test]
    fn a_pass_with_nothing_broken_replays_fully_with_its_key() {
        let mut dispatcher = dispatcher(SupportLevel::Always);
        let DispatchDecision::FullReplay(key) = dispatcher.dispatch(batch(5, 5, true)) else {
            panic!("a pass with nothing broken must replay fully");
        };
        assert_eq!(key.padded_token_count(), tokens(8));
        assert_eq!(key.request_count(), requests(5));
        assert!(key.uniform_decode());
        assert_eq!(dispatcher.fallbacks(), EagerFallbackCounters::default());
    }

    #[test]
    fn a_broken_pass_replays_segments_with_its_key() {
        let mut dispatcher = dispatcher_under(&contract(SupportLevel::Always, &[site(3, 2)]));
        let DispatchDecision::SegmentedReplay(key) = dispatcher.dispatch(batch(5, 5, true)) else {
            panic!("a broken pass must replay segments");
        };
        assert_eq!(key.padded_token_count(), tokens(8));
        assert_eq!(key.request_count(), requests(5));
    }

    #[test]
    fn the_capture_kind_follows_the_break_points_the_contract_settled() {
        assert_eq!(
            CaptureKind::for_break_points(contract(SupportLevel::Always, &[]).break_points()),
            CaptureKind::Full
        );
        assert_eq!(
            CaptureKind::for_break_points(
                contract(SupportLevel::Always, &[site(0, 1)]).break_points()
            ),
            CaptureKind::Segmented
        );
    }

    #[test]
    fn the_dispatcher_returns_the_union_of_both_declarers() {
        let backend =
            BackendDeclaration::new("union-backend", SupportLevel::Always).rank_coupled(site(4, 0));
        let model = ModelDeclaration::new("union-model").eager_at(site(1, 6));
        let dispatcher = dispatcher_under(&CaptureContract::resolve(&[backend], &model));

        assert_eq!(dispatcher.break_points().sites(), [site(1, 6), site(4, 0)]);
    }

    #[test]
    fn keyless_batch_dispatches_eagerly_with_its_reason() {
        let mut dispatcher = dispatcher(SupportLevel::Always);
        assert_eq!(
            dispatcher.dispatch(batch(600, 600, true)),
            DispatchDecision::Eager(EagerReason::TokensAboveBucketLadderMaximum {
                token_count: tokens(600),
                bucket_ladder_maximum: Some(tokens(512)),
            })
        );
    }

    #[test]
    fn eager_fallback_logs_once_with_its_reason() {
        let log = captured_log();
        // The global capture sees every test in the crate, so the batch sizes here are ones no
        // other test dispatches: 509 keys (it pads to 512), 8191 falls back.
        let mut dispatcher = dispatcher(SupportLevel::Always);
        dispatcher.dispatch(batch(509, 509, true));
        dispatcher.dispatch(batch(8191, 8191, true));
        let output = log.contents();
        let fallback_lines: Vec<&str> = output
            .lines()
            .filter(|line| line.contains("token count 8191"))
            .collect();
        assert_eq!(fallback_lines.len(), 1, "one fallback, one log line");
        assert!(fallback_lines[0].contains("eager fallback"));
        assert!(fallback_lines[0].contains(
            "token count 8191 exceeds every captured bucket; the bucket-ladder maximum is 512"
        ));
        assert!(
            !output.contains("token count 509"),
            "keyed batches log nothing"
        );
    }

    #[test]
    fn dispatch_config_round_trips_through_config_json() {
        let config: DispatchConfig = serde_json::from_str(
            r#"{
                "bucket_ladder": [1, 2, 4, 8],
                "captured_max_requests": 16
            }"#,
        )
        .unwrap();
        assert_eq!(config.bucket_ladder.buckets(), [1, 2, 4, 8]);
        assert_eq!(config.captured_max_requests, requests(16));
        let json = serde_json::to_string(&config).unwrap();
        assert_eq!(
            serde_json::from_str::<DispatchConfig>(&json).unwrap(),
            config
        );
    }

    #[test]
    fn a_configured_support_level_is_refused() {
        // The level is the backends' to declare, so a configuration naming one is a mistake worth
        // refusing to start over rather than silently ignoring.
        let error = serde_json::from_str::<DispatchConfig>(
            r#"{
                "bucket_ladder": [1],
                "captured_max_requests": 1,
                "support_level": "always"
            }"#,
        )
        .unwrap_err();
        assert!(error.to_string().contains("support_level"), "got: {error}");
    }

    #[test]
    fn zero_captured_max_requests_is_rejected_in_config() {
        let error = serde_json::from_str::<DispatchConfig>(
            r#"{
                "bucket_ladder": [1],
                "captured_max_requests": 0
            }"#,
        )
        .unwrap_err();
        assert!(error.to_string().contains("nonzero"));
    }

    #[test]
    fn each_eager_reason_counts_separately() {
        let mut full_support = dispatcher(SupportLevel::Always);
        assert_eq!(full_support.fallbacks(), EagerFallbackCounters::default());

        full_support.dispatch(batch(600, 600, true));
        full_support.dispatch(batch(700, 700, true));
        full_support.dispatch(batch(8, 513, true));
        full_support.dispatch(batch(16, 4, false));
        assert_eq!(
            full_support.fallbacks(),
            EagerFallbackCounters {
                tokens_above_bucket_ladder_maximum: 2,
                requests_above_captured_maximum: 1,
                support_level_insufficient: 0,
                not_uniform_decode: 1,
            }
        );

        let mut single_token_only = dispatcher(SupportLevel::UniformSingleTokenDecode);
        single_token_only.dispatch(batch(16, 4, true));
        assert_eq!(single_token_only.fallbacks().support_level_insufficient, 1);
    }
}
