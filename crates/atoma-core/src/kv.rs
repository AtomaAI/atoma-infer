//! The KV substrate the engine owns.
//!
//! Identity here is content, never residence: a block-sized token run is identified by its chain
//! hash, which commits to the whole prefix behind it, and which slot currently holds a hash's
//! bytes is always a separate lookup. Everything is host-side data structures and arithmetic —
//! no device code, no threads, no channels.

mod chain_hash;

pub use chain_hash::{ExtraKeys, HashAlgorithm};
