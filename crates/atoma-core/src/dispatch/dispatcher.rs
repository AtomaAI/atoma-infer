//! The dispatcher: owns dispatch truth for every live batch.

use std::num::NonZeroUsize;

use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::dispatch::{
    admit, BucketLadder, EagerFallbackCounters, GraphKey, LiveBatch, PaddingLookup,
    RejectionReason, SupportLevel,
};

/// How the captured set was recorded: whole forward passes, or segments around eager regions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureKind {
    /// Each bucket's graph records the whole forward pass.
    Full,
    /// Each bucket's captured pass is split into segments around eager operations.
    Segmented,
}

/// Everything the dispatcher is built from, fixed for the process lifetime.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DispatchConfig {
    /// The buckets the engine captured.
    pub bucket_ladder: BucketLadder,
    /// The largest request count any captured graph serves.
    pub captured_max_requests: NonZeroUsize,
    /// The minimum support level across the active backends, settled at Allocation.
    pub support_level: SupportLevel,
    /// How the captured set was recorded.
    pub capture_kind: CaptureKind,
}

/// What the executor does with one live batch. Executors act on this without re-deriving it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchDecision {
    /// Replay the full captured graph the key selects.
    FullReplay(GraphKey),
    /// Replay the captured segments the key selects, with eager regions between them.
    SegmentedReplay(GraphKey),
    /// Run the whole step eagerly.
    Eager(RejectionReason),
}

/// Owns dispatch truth: which captured graph serves a live batch, or why none does.
///
/// Built once at Allocation and never modified afterwards — no method changes the bucket ladder,
/// the captured set or the support level, so no code path recaptures at runtime.
/// Dispatch priority is full-graph replay, then segmented replay, then eager: an admitted batch
/// replays the whole pass when the captured set records whole passes, replays segments when it is
/// split, and every rejected batch runs eagerly.
#[derive(Debug)]
pub struct Dispatcher {
    lookup: PaddingLookup,
    captured_max_requests: NonZeroUsize,
    support_level: SupportLevel,
    capture_kind: CaptureKind,
    fallbacks: EagerFallbackCounters,
}

impl Dispatcher {
    /// Builds the dispatcher, deriving the dense padding lookup from the configured bucket
    /// ladder.
    #[must_use]
    pub fn new(config: &DispatchConfig) -> Self {
        Self {
            lookup: PaddingLookup::new(&config.bucket_ladder),
            captured_max_requests: config.captured_max_requests,
            support_level: config.support_level,
            capture_kind: config.capture_kind,
            fallbacks: EagerFallbackCounters::default(),
        }
    }

    /// Decides how `batch` runs, counting and logging the fallback when no graph serves it.
    pub fn dispatch(&mut self, batch: LiveBatch) -> DispatchDecision {
        match admit(
            batch,
            self.support_level,
            self.captured_max_requests,
            &self.lookup,
        ) {
            Ok(key) => match self.capture_kind {
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

    /// Eager fallbacks so far, by rejection reason.
    #[must_use]
    pub fn fallbacks(&self) -> EagerFallbackCounters {
        self.fallbacks
    }
}

#[cfg(test)]
mod tests {
    use super::{CaptureKind, DispatchConfig, DispatchDecision, Dispatcher, EagerFallbackCounters};
    use crate::dispatch::test_support::{batch, nonzero};
    use crate::dispatch::{BucketLadder, Platform, RejectionReason, SupportLevel};

    fn dispatcher(support_level: SupportLevel, capture_kind: CaptureKind) -> Dispatcher {
        Dispatcher::new(&DispatchConfig {
            bucket_ladder: BucketLadder::default_for(Platform::Hopper),
            captured_max_requests: nonzero(512),
            support_level,
            capture_kind,
        })
    }

    #[test]
    fn full_captured_set_replays_fully_with_the_admitted_key() {
        let mut dispatcher = dispatcher(SupportLevel::Always, CaptureKind::Full);
        let DispatchDecision::FullReplay(key) = dispatcher.dispatch(batch(5, 5, true)) else {
            panic!("a full captured set must replay fully");
        };
        assert_eq!(key.padded_token_count(), nonzero(8));
        assert_eq!(key.request_count(), nonzero(5));
        assert!(key.uniform_decode());
        assert_eq!(dispatcher.fallbacks(), EagerFallbackCounters::default());
    }

    #[test]
    fn segmented_captured_set_replays_segments_with_the_admitted_key() {
        let mut dispatcher = dispatcher(SupportLevel::Always, CaptureKind::Segmented);
        let DispatchDecision::SegmentedReplay(key) = dispatcher.dispatch(batch(5, 5, true)) else {
            panic!("a segmented captured set must replay segments");
        };
        assert_eq!(key.padded_token_count(), nonzero(8));
        assert_eq!(key.request_count(), nonzero(5));
    }

    #[test]
    fn rejected_batch_dispatches_eagerly_with_its_reason() {
        let mut dispatcher = dispatcher(SupportLevel::Always, CaptureKind::Full);
        assert_eq!(
            dispatcher.dispatch(batch(600, 600, true)),
            DispatchDecision::Eager(RejectionReason::TokensAboveBucketLadderMaximum {
                token_count: nonzero(600),
                bucket_ladder_maximum: Some(nonzero(512)),
            })
        );
    }

    #[test]
    fn dispatch_config_round_trips_through_config_json() {
        let config: DispatchConfig = serde_json::from_str(
            r#"{
                "bucket_ladder": [1, 2, 4, 8],
                "captured_max_requests": 16,
                "support_level": "uniform_single_token_decode",
                "capture_kind": "full"
            }"#,
        )
        .unwrap();
        assert_eq!(config.bucket_ladder.buckets(), [1, 2, 4, 8]);
        assert_eq!(config.captured_max_requests, nonzero(16));
        assert_eq!(config.support_level, SupportLevel::UniformSingleTokenDecode);
        assert_eq!(config.capture_kind, CaptureKind::Full);
        let json = serde_json::to_string(&config).unwrap();
        assert_eq!(
            serde_json::from_str::<DispatchConfig>(&json).unwrap(),
            config
        );
    }

    #[test]
    fn zero_captured_max_requests_is_rejected_in_config() {
        let error = serde_json::from_str::<DispatchConfig>(
            r#"{
                "bucket_ladder": [1],
                "captured_max_requests": 0,
                "support_level": "always",
                "capture_kind": "full"
            }"#,
        )
        .unwrap_err();
        assert!(error.to_string().contains("nonzero"));
    }

    #[test]
    fn each_rejection_reason_counts_separately() {
        let mut full_support = dispatcher(SupportLevel::Always, CaptureKind::Full);
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

        let mut single_token_only =
            dispatcher(SupportLevel::UniformSingleTokenDecode, CaptureKind::Full);
        single_token_only.dispatch(batch(16, 4, true));
        assert_eq!(single_token_only.fallbacks().support_level_insufficient, 1);
    }
}
