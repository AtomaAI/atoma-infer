//! Which captured graph serves a live batch, or why none does.

mod ladder;
mod lookup;

pub use ladder::{BucketLadder, LadderError, Platform};
pub use lookup::PaddingLookup;
