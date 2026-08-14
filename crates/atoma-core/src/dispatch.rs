//! Which captured graph serves a live batch, or why none does.
//!
//! A padded batch produces a graph key by exactly one pure function; any token count maps through
//! a dense lookup to the next captured bucket; a batch no graph serves produces a named rejection
//! reason carrying the numbers that caused it.

mod admission;
mod key;
mod ladder;
mod lookup;

pub use admission::{admit, BatchShape, RejectionReason, SupportLevel};
pub use key::GraphKey;
pub use ladder::{BucketLadder, LadderError, Platform};
pub use lookup::PaddingLookup;
