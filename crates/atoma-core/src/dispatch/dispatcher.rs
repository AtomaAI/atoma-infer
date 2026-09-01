//! The dispatcher: owns dispatch truth for every live batch.

use serde::{Deserialize, Serialize};
use tracing::debug;
use validator::Validate;

use crate::dispatch::{
    decide, BucketLadder, EagerFallbackCounters, EagerReason, GraphKey, LiveBatch, PaddingLookup,
    SupportLevel,
};
use crate::types::RequestCount;

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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
pub struct DispatchConfig {
    /// The buckets the engine captured.
    pub bucket_ladder: BucketLadder,
    /// The largest request count any captured graph serves.
    pub captured_max_requests: RequestCount,
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
    Eager(EagerReason),
}

/// Owns dispatch truth: which captured graph serves a live batch, or why none does.
///
/// Built once at Allocation and never modified afterwards — no method changes the bucket ladder,
/// the captured set or the support level, so no code path recaptures at runtime.
/// Dispatch priority is full-graph replay, then segmented replay, then eager: a batch with a key
/// replays the whole pass when the captured set records whole passes, replays segments when it is
/// split, and every batch without one runs eagerly.
#[derive(Debug)]
pub struct Dispatcher {
    lookup: PaddingLookup,
    captured_max_requests: RequestCount,
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
        match decide(
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

    /// Eager fallbacks so far, by eager reason.
    #[must_use]
    pub fn fallbacks(&self) -> EagerFallbackCounters {
        self.fallbacks
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::sync::{Arc, Mutex, OnceLock};

    use tracing::subscriber::set_global_default;
    use tracing::Level;

    use super::{CaptureKind, DispatchConfig, DispatchDecision, Dispatcher, EagerFallbackCounters};
    use crate::dispatch::test_support::batch;
    use crate::dispatch::{BucketLadder, EagerReason, Platform, SupportLevel};
    use crate::test_support::{requests, tokens};

    /// Log output collected behind the process-global subscriber.
    ///
    /// A thread-scoped subscriber is unreliable here: with a single registered dispatcher,
    /// tracing computes a callsite's cached interest on whichever thread first hits it, and a
    /// sibling test's thread without a subscriber caches a no-op interest for everyone. The
    /// global subscriber sees every test's events, so assertions must match on batch numbers
    /// unique to their own test.
    #[derive(Clone, Default)]
    struct CapturedLog(Arc<Mutex<Vec<u8>>>);

    impl CapturedLog {
        fn contents(&self) -> String {
            let bytes = self.0.lock().expect("log capture lock").clone();
            String::from_utf8(bytes).expect("log output is utf-8")
        }
    }

    impl Write for CapturedLog {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0
                .lock()
                .expect("log capture lock")
                .extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn dispatcher(support_level: SupportLevel, capture_kind: CaptureKind) -> Dispatcher {
        Dispatcher::new(&DispatchConfig {
            bucket_ladder: BucketLadder::default_for(Platform::Hopper),
            captured_max_requests: requests(512),
            support_level,
            capture_kind,
        })
    }

    #[test]
    fn full_captured_set_replays_fully_with_its_key() {
        let mut dispatcher = dispatcher(SupportLevel::Always, CaptureKind::Full);
        let DispatchDecision::FullReplay(key) = dispatcher.dispatch(batch(5, 5, true)) else {
            panic!("a full captured set must replay fully");
        };
        assert_eq!(key.padded_token_count(), tokens(8));
        assert_eq!(key.request_count(), requests(5));
        assert!(key.uniform_decode());
        assert_eq!(dispatcher.fallbacks(), EagerFallbackCounters::default());
    }

    #[test]
    fn segmented_captured_set_replays_segments_with_its_key() {
        let mut dispatcher = dispatcher(SupportLevel::Always, CaptureKind::Segmented);
        let DispatchDecision::SegmentedReplay(key) = dispatcher.dispatch(batch(5, 5, true)) else {
            panic!("a segmented captured set must replay segments");
        };
        assert_eq!(key.padded_token_count(), tokens(8));
        assert_eq!(key.request_count(), requests(5));
    }

    #[test]
    fn keyless_batch_dispatches_eagerly_with_its_reason() {
        let mut dispatcher = dispatcher(SupportLevel::Always, CaptureKind::Full);
        assert_eq!(
            dispatcher.dispatch(batch(600, 600, true)),
            DispatchDecision::Eager(EagerReason::TokensAboveBucketLadderMaximum {
                token_count: tokens(600),
                bucket_ladder_maximum: Some(tokens(512)),
            })
        );
    }

    fn captured_log() -> &'static CapturedLog {
        static LOG: OnceLock<CapturedLog> = OnceLock::new();
        LOG.get_or_init(|| {
            let log = CapturedLog::default();
            let writer = log.clone();
            let subscriber = tracing_subscriber::fmt()
                .with_max_level(Level::DEBUG)
                .with_writer(move || writer.clone())
                .finish();
            set_global_default(subscriber).expect("no other test installs a subscriber");
            log
        })
    }

    #[test]
    fn eager_fallback_logs_once_with_its_reason() {
        let log = captured_log();
        // The global capture sees every test in the crate, so the batch sizes here are ones no
        // other test dispatches: 509 keys (it pads to 512), 8191 falls back.
        let mut dispatcher = dispatcher(SupportLevel::Always, CaptureKind::Full);
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
                "captured_max_requests": 16,
                "support_level": "uniform_single_token_decode",
                "capture_kind": "full"
            }"#,
        )
        .unwrap();
        assert_eq!(config.bucket_ladder.buckets(), [1, 2, 4, 8]);
        assert_eq!(config.captured_max_requests, requests(16));
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
    fn each_eager_reason_counts_separately() {
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
