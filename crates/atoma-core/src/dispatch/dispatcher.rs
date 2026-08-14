//! The dispatcher: owns dispatch truth for every live batch.

use tracing::debug;

use crate::dispatch::{
    admit, BatchShape, BucketLadder, GraphKey, PaddingLookup, RejectionReason, SupportLevel,
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
    pub ladder: BucketLadder,
    /// The largest request count any captured graph serves.
    pub captured_max_requests: usize,
    /// The minimum support level across the active backends, settled at startup.
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

/// Eager fallbacks so far, by rejection reason.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct EagerFallbackCounters {
    /// Batches whose token count exceeded the ladder maximum.
    pub tokens_above_ladder_maximum: u64,
    /// Batches with more requests than any captured graph serves.
    pub requests_above_captured_maximum: u64,
    /// Batches the backends' declared support level could not serve.
    pub support_level_insufficient: u64,
    /// Batches that were not uniform decode.
    pub not_uniform_decode: u64,
}

impl EagerFallbackCounters {
    fn count(&mut self, reason: &RejectionReason) {
        match reason {
            RejectionReason::TokensAboveLadderMaximum {
                token_count: _,
                ladder_maximum: _,
            } => {
                self.tokens_above_ladder_maximum += 1;
            }
            RejectionReason::RequestsAboveCapturedMaximum {
                request_count: _,
                captured_maximum: _,
            } => {
                self.requests_above_captured_maximum += 1;
            }
            RejectionReason::SupportLevelInsufficient {
                support_level: _,
                required: _,
                token_count: _,
                request_count: _,
            } => {
                self.support_level_insufficient += 1;
            }
            RejectionReason::NotUniformDecode {
                token_count: _,
                request_count: _,
            } => {
                self.not_uniform_decode += 1;
            }
        }
    }
}

/// Owns dispatch truth: which captured graph serves a live batch, or why none does.
///
/// Built once at startup and never modified afterwards — no method changes the ladder, the
/// captured set or the support level, so no code path recaptures at runtime. Dispatch priority is
/// full-graph replay, then segmented replay, then eager: an admitted batch replays the whole pass
/// when the captured set records whole passes, replays segments when it is split, and every
/// rejected batch runs eagerly.
#[derive(Debug)]
pub struct Dispatcher {
    lookup: PaddingLookup,
    captured_max_requests: usize,
    support_level: SupportLevel,
    capture_kind: CaptureKind,
    fallbacks: EagerFallbackCounters,
}

impl Dispatcher {
    /// Builds the dispatcher, deriving the dense padding lookup from the configured ladder.
    #[must_use]
    pub fn new(config: &DispatchConfig) -> Self {
        Self {
            lookup: PaddingLookup::new(&config.ladder),
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
    pub fn admit(&self, batch: BatchShape) -> Result<GraphKey, RejectionReason> {
        admit(
            batch,
            self.support_level,
            self.captured_max_requests,
            &self.lookup,
        )
    }

    /// Decides how `batch` runs, counting and logging the fallback when no graph serves it.
    pub fn dispatch(&mut self, batch: BatchShape) -> DispatchDecision {
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
    use std::num::NonZeroUsize;

    use super::{CaptureKind, DispatchConfig, DispatchDecision, Dispatcher, EagerFallbackCounters};
    use crate::dispatch::{BatchShape, BucketLadder, Platform, RejectionReason, SupportLevel};

    fn count(value: usize) -> NonZeroUsize {
        NonZeroUsize::new(value).expect("test counts are nonzero")
    }

    fn batch(token_count: usize, request_count: usize, uniform_decode: bool) -> BatchShape {
        BatchShape {
            token_count: count(token_count),
            request_count: count(request_count),
            uniform_decode,
        }
    }

    fn dispatcher(support_level: SupportLevel, capture_kind: CaptureKind) -> Dispatcher {
        Dispatcher::new(&DispatchConfig {
            ladder: BucketLadder::default_for(Platform::Hopper),
            captured_max_requests: 512,
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
            DispatchDecision::Eager(RejectionReason::TokensAboveLadderMaximum {
                token_count: count(600),
                ladder_maximum: 512,
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
                tokens_above_ladder_maximum: 2,
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
