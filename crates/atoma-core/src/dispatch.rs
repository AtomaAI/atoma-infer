//! Which captured graph serves a live batch, or why none does.
//!
//! A padded batch produces a graph key by exactly one pure function; any token count maps through
//! a dense lookup to the next captured bucket; a batch no graph serves produces a named rejection
//! reason carrying the numbers that caused it. The dispatcher owns dispatch truth — executors act
//! on its result without re-deriving it — with priority full-graph replay, then segmented replay,
//! then eager.

mod admission;
mod bucket_ladder;
mod dispatcher;
mod key;
mod lookup;

pub(crate) use admission::admit;
pub use admission::{LiveBatch, RejectionReason, SupportLevel};
pub use bucket_ladder::{BucketLadder, BucketLadderError, Platform};
pub use dispatcher::{
    CaptureKind, DispatchConfig, DispatchDecision, Dispatcher, EagerFallbackCounters,
};
pub use key::GraphKey;
pub use lookup::PaddingLookup;

#[cfg(test)]
pub(crate) mod test_support {
    use std::num::NonZeroUsize;

    use crate::dispatch::LiveBatch;

    pub(crate) fn count(value: usize) -> NonZeroUsize {
        NonZeroUsize::new(value).expect("test counts are nonzero")
    }

    pub(crate) fn batch(
        token_count: usize,
        request_count: usize,
        uniform_decode: bool,
    ) -> LiveBatch {
        LiveBatch {
            token_count: count(token_count),
            request_count: count(request_count),
            uniform_decode,
        }
    }
}
