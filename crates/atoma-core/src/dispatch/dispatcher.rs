//! The dispatcher: owns dispatch truth for every live batch.

use std::num::NonZeroUsize;

use tracing::debug;

use crate::dispatch::{
    admit, BucketLadder, EagerFallbackCounters, GraphKey, LiveBatch, PaddingLookup,
    RejectionReason, SupportLevel,
};

/// How the captured set was recorded: whole forward passes, or segments around eager regions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureKind {
    /// Each bucket's graph records the whole forward pass.
    Full,
    /// Each bucket's captured pass is split into segments around eager operations.
    Segmented,
}

/// Everything the dispatcher is built from, fixed for the process lifetime.
#[derive(Debug, Clone, PartialEq, Eq)]
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

    /// Admits `batch` without counting: exactly one key, or exactly one rejection reason.
    ///
    /// # Errors
    ///
    /// Returns the [`RejectionReason`] naming the first failed admission check, carrying the
    /// numbers that caused it.
    pub fn admit(&self, batch: LiveBatch) -> Result<GraphKey, RejectionReason> {
        admit(
            batch,
            self.support_level,
            self.captured_max_requests,
            &self.lookup,
        )
    }

    /// Decides how `batch` runs, counting and logging the fallback when no graph serves it.
    pub fn dispatch(&mut self, batch: LiveBatch) -> DispatchDecision {
        match self.admit(batch) {
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
    use crate::dispatch::test_support::{batch, count};
    use crate::dispatch::{BucketLadder, Platform, RejectionReason, SupportLevel};

    fn dispatcher(support_level: SupportLevel, capture_kind: CaptureKind) -> Dispatcher {
        Dispatcher::new(&DispatchConfig {
            bucket_ladder: BucketLadder::default_for(Platform::Hopper),
            captured_max_requests: count(512),
            support_level,
            capture_kind,
        })
    }

    #[test]
    fn full_captured_set_replays_fully_with_the_admitted_key() {
        let mut dispatcher = dispatcher(SupportLevel::Always, CaptureKind::Full);
        let expected = dispatcher.admit(batch(5, 5, true)).unwrap();
        assert_eq!(
            dispatcher.dispatch(batch(5, 5, true)),
            DispatchDecision::FullReplay(expected)
        );
        assert_eq!(dispatcher.fallbacks(), EagerFallbackCounters::default());
    }

    #[test]
    fn segmented_captured_set_replays_segments_with_the_admitted_key() {
        let mut dispatcher = dispatcher(SupportLevel::Always, CaptureKind::Segmented);
        let expected = dispatcher.admit(batch(5, 5, true)).unwrap();
        assert_eq!(
            dispatcher.dispatch(batch(5, 5, true)),
            DispatchDecision::SegmentedReplay(expected)
        );
    }

    #[test]
    fn rejected_batch_dispatches_eagerly_with_its_reason() {
        let mut dispatcher = dispatcher(SupportLevel::Always, CaptureKind::Full);
        assert_eq!(
            dispatcher.dispatch(batch(600, 600, true)),
            DispatchDecision::Eager(RejectionReason::TokensAboveBucketLadderMaximum {
                token_count: count(600),
                bucket_ladder_maximum: Some(count(512)),
            })
        );
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
