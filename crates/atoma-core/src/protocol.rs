//! The protocol vocabulary the engine is written against.
//!
//! Every identity and count that crosses a module or wire boundary is a newtype here, so a block
//! id cannot be passed where a request slot is meant and a raw integer cannot impersonate either.
//! Each type is `Copy` and serde-derived with nothing reference-counted, boxed-dyn, or
//! wall-clock reachable from serialized position, so the compiler enforces the serializable
//! boundary.

use std::fmt;
use std::num::NonZeroUsize;

use serde::{Deserialize, Serialize};

/// One request across its whole lifetime, minted at admission and never reused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RequestId(u64);

impl RequestId {
    #[must_use]
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// A live request's position in the scheduler's slab, valid only while the request runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RequestSlot(u32);

impl RequestSlot {
    #[must_use]
    pub const fn new(slot: u32) -> Self {
        Self(slot)
    }

    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// One engine step, monotonically increasing over the process lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct StepId(u64);

impl StepId {
    #[must_use]
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// One physical block in the pool — a slot address, never an identity.
///
/// Identity is [`BlockHash`]; a block id only says which slot holds some hash's bytes right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlockId(u32);

impl BlockId {
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }

    /// The pool-slab index this id addresses.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// The tier-agnostic identity of one block-sized token run: its full chain-hash digest.
///
/// The digest commits to the whole prefix behind the run, so equal hashes mean equal cacheable
/// content regardless of where the bytes reside.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlockHash([u8; 32]);

impl BlockHash {
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Debug for BlockHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BlockHash(")?;
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        write!(f, ")")
    }
}

/// One layer group in the model's cache layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LayerGroupId(u16);

impl LayerGroupId {
    #[must_use]
    pub const fn new(id: u16) -> Self {
        Self(id)
    }

    #[must_use]
    pub const fn get(self) -> u16 {
        self.0
    }
}

/// A count of tokens, nonzero by construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TokenCount(NonZeroUsize);

impl TokenCount {
    #[must_use]
    pub const fn new(count: usize) -> Option<Self> {
        match NonZeroUsize::new(count) {
            Some(count) => Some(Self(count)),
            None => None,
        }
    }

    #[must_use]
    pub const fn get(self) -> usize {
        self.0.get()
    }
}

impl From<NonZeroUsize> for TokenCount {
    fn from(count: NonZeroUsize) -> Self {
        Self(count)
    }
}

/// A count of requests, nonzero by construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RequestCount(NonZeroUsize);

impl RequestCount {
    #[must_use]
    pub const fn new(count: usize) -> Option<Self> {
        match NonZeroUsize::new(count) {
            Some(count) => Some(Self(count)),
            None => None,
        }
    }

    #[must_use]
    pub const fn get(self) -> usize {
        self.0.get()
    }
}

impl From<NonZeroUsize> for RequestCount {
    fn from(count: NonZeroUsize) -> Self {
        Self(count)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BlockHash, BlockId, LayerGroupId, RequestCount, RequestId, RequestSlot, StepId, TokenCount,
    };

    #[test]
    fn ids_round_trip_through_serde() {
        let request = RequestId::new(7);
        let slot = RequestSlot::new(3);
        let step = StepId::new(11);
        let block = BlockId::new(42);
        let group = LayerGroupId::new(1);

        let json = serde_json::to_string(&(request, slot, step, block, group)).unwrap();
        let back: (RequestId, RequestSlot, StepId, BlockId, LayerGroupId) =
            serde_json::from_str(&json).unwrap();

        assert_eq!(back, (request, slot, step, block, group));
    }

    #[test]
    fn counts_serialize_as_the_bare_integer() {
        let tokens = TokenCount::new(8).unwrap();
        let requests = RequestCount::new(2).unwrap();

        assert_eq!(serde_json::to_string(&tokens).unwrap(), "8");
        assert_eq!(serde_json::to_string(&requests).unwrap(), "2");
        assert_eq!(serde_json::from_str::<TokenCount>("8").unwrap(), tokens);
        assert_eq!(serde_json::from_str::<RequestCount>("2").unwrap(), requests);
    }

    #[test]
    fn zero_counts_are_unrepresentable() {
        assert!(TokenCount::new(0).is_none());
        assert!(RequestCount::new(0).is_none());
        assert!(serde_json::from_str::<TokenCount>("0").is_err());
        assert!(serde_json::from_str::<RequestCount>("0").is_err());
    }

    #[test]
    fn block_hash_round_trips_and_debugs_as_hex() {
        let mut bytes = [0u8; 32];
        bytes[0] = 0xab;
        bytes[31] = 0x01;
        let hash = BlockHash::from_bytes(bytes);

        let json = serde_json::to_string(&hash).unwrap();
        assert_eq!(serde_json::from_str::<BlockHash>(&json).unwrap(), hash);

        let debug = format!("{hash:?}");
        assert!(debug.starts_with("BlockHash(ab00"));
        assert!(debug.ends_with("01)"));
    }

    #[test]
    fn step_ids_order_by_value() {
        assert!(StepId::new(1) < StepId::new(2));
        assert!(TokenCount::new(8) < TokenCount::new(9));
        assert!(RequestCount::new(1) < RequestCount::new(4));
    }
}
