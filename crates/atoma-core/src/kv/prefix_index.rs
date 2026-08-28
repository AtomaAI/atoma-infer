//! The prefix index: longest-prefix match over chains of block hashes.
//!
//! A radix tree over block-sized runs, stored as an index-based slab. Because a chain hash
//! commits to its whole prefix, each run's node is found by its hash in one map probe, so
//! longest-prefix match is a single traversal of the query's hashes; parent links and child
//! counts keep the tree shape for leaf-only eviction. The index answers in block hashes, never
//! slot ids — which slot holds a hash's bytes is the pool's separate residence lookup.

use std::collections::HashMap;
use std::mem;

use crate::protocol::BlockHash;

/// One slab slot: an occupied node, or a vacancy threading the free list.
#[derive(Debug)]
enum Slot {
    Occupied(Node),
    /// A removed node's slot, linking the next vacancy; `None` ends the free list.
    Vacant {
        next_free: Option<u32>,
    },
}

impl Slot {
    fn occupied(&self) -> Option<&Node> {
        match self {
            Slot::Occupied(node) => Some(node),
            Slot::Vacant { next_free: _ } => None,
        }
    }
}

#[derive(Debug)]
struct Node {
    hash: BlockHash,
    parent: Option<u32>,
    child_count: u32,
    /// Live paths through this node; a pinned node is never evicted or removed.
    pins: u32,
    /// Tick of the last lookup, insert or unpin that walked this node.
    last_touch: u64,
}

/// Radix index over chains of block hashes, with pins and leaf-only LRU eviction.
///
/// Single-owner, like the pool: `&mut self` everywhere, no lock, no reference-counted sharing.
#[derive(Debug, Default)]
pub struct PrefixIndex {
    slots: Vec<Slot>,
    /// Head of the free list threaded through the vacant slots.
    free_head: Option<u32>,
    node_by_hash: HashMap<BlockHash, u32>,
    clock: u64,
}

impl PrefixIndex {
    /// An empty index.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// How many of `hashes`, in order, are present as a stored path — the longest prefix match,
    /// resolved in one traversal. Matched nodes count as touched for eviction recency.
    pub fn lookup(&mut self, hashes: &[BlockHash]) -> usize {
        self.clock += 1;
        let clock = self.clock;
        let mut matched = 0;
        for hash in hashes {
            let Some(&slot) = self.node_by_hash.get(hash) else {
                break;
            };
            self.node(slot).last_touch = clock;
            matched += 1;
        }
        matched
    }

    /// Stores `hashes` as a path, creating missing nodes, and pins every node on it.
    ///
    /// The pin lasts until the same path is [`PrefixIndex::unpin`]ned, so a live sequence's
    /// prefix can never be evicted from under it.
    pub fn insert(&mut self, hashes: &[BlockHash]) {
        self.clock += 1;
        let clock = self.clock;
        let mut parent = None;
        for hash in hashes {
            let slot = match self.node_by_hash.get(hash) {
                Some(&slot) => slot,
                None => self.create_node(*hash, parent),
            };
            let node = self.node(slot);
            node.pins += 1;
            node.last_touch = clock;
            parent = Some(slot);
        }
    }

    /// Releases one pin on every node of `hashes`. Call with the exact path a prior
    /// [`PrefixIndex::insert`] stored; the nodes cannot have been removed while pinned.
    pub fn unpin(&mut self, hashes: &[BlockHash]) {
        self.clock += 1;
        let clock = self.clock;
        for hash in hashes {
            let Some(&slot) = self.node_by_hash.get(hash) else {
                debug_assert!(false, "unpin of a path never inserted: {hash:?}");
                continue;
            };
            let node = self.node(slot);
            debug_assert!(node.pins > 0, "unpin below zero for {hash:?}");
            node.pins = node.pins.saturating_sub(1);
            node.last_touch = clock;
        }
    }

    /// Removes and returns the least-recently-touched unpinned leaf, or `None` when every node
    /// is pinned or interior. The caller evicts the hash's bytes from wherever they reside.
    pub fn evict_lru(&mut self) -> Option<BlockHash> {
        let victim = self
            .slots
            .iter()
            .filter_map(Slot::occupied)
            .filter(|node| node.pins == 0 && node.child_count == 0)
            .min_by_key(|node| node.last_touch)?
            .hash;
        self.remove_leaf(victim);
        Some(victim)
    }

    /// Removes `hash`'s node if it is an unpinned leaf, returning whether it was removed.
    /// Its parent may become a leaf, and so evictable, in turn.
    pub fn remove_leaf(&mut self, hash: BlockHash) -> bool {
        let Some(&slot) = self.node_by_hash.get(&hash) else {
            return false;
        };
        let node = self.node(slot);
        if node.pins > 0 || node.child_count > 0 {
            return false;
        }
        let parent = node.parent;
        self.slots[slot as usize] = Slot::Vacant {
            next_free: self.free_head,
        };
        self.free_head = Some(slot);
        self.node_by_hash.remove(&hash);
        if let Some(parent) = parent {
            self.node(parent).child_count -= 1;
        }
        true
    }

    /// Whether `hash` is stored, pinned or not.
    #[must_use]
    pub fn contains(&self, hash: BlockHash) -> bool {
        self.node_by_hash.contains_key(&hash)
    }

    /// Stored nodes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.node_by_hash.len()
    }

    /// Whether the index stores nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.node_by_hash.is_empty()
    }

    fn create_node(&mut self, hash: BlockHash, parent: Option<u32>) -> u32 {
        if let Some(parent) = parent {
            self.node(parent).child_count += 1;
        }
        let node = Node {
            hash,
            parent,
            child_count: 0,
            pins: 0,
            last_touch: self.clock,
        };
        let slot = if let Some(slot) = self.free_head {
            let vacated = mem::replace(&mut self.slots[slot as usize], Slot::Occupied(node));
            let Slot::Vacant { next_free } = vacated else {
                unreachable!("the free list links only vacant slots")
            };
            self.free_head = next_free;
            slot
        } else {
            self.slots.push(Slot::Occupied(node));
            u32::try_from(self.slots.len() - 1).expect("node count fits u32")
        };
        self.node_by_hash.insert(hash, slot);
        slot
    }

    fn node(&mut self, slot: u32) -> &mut Node {
        let Slot::Occupied(node) = &mut self.slots[slot as usize] else {
            unreachable!("slot indices are handed out only for occupied slots")
        };
        node
    }
}

#[cfg(test)]
mod tests {
    use proptest::collection::vec;
    use proptest::prelude::*;

    use super::PrefixIndex;
    use crate::kv::{ExtraKeys, HashAlgorithm};
    use crate::protocol::{BlockHash, TokenCount};

    /// Chains `tokens` with block size 2 so short test sequences span several runs.
    fn chain(tokens: &[u32]) -> Vec<BlockHash> {
        let block_size = TokenCount::new(2).expect("test block size is nonzero");
        HashAlgorithm::Sha256V1.chain(block_size, tokens, ExtraKeys::none())
    }

    #[test]
    fn longest_prefix_match_against_a_hand_built_tree() {
        // The tree, built run by run (block size 2):
        //   [1,2] ── [3,4] ── [5,6]
        //        └── [7,8]
        let mut index = PrefixIndex::new();
        let long = chain(&[1, 2, 3, 4, 5, 6]);
        let fork = chain(&[1, 2, 7, 8]);
        index.insert(&long);
        index.insert(&fork);
        assert_eq!(index.len(), 4, "three shared-root runs plus the fork");

        assert_eq!(index.lookup(&long), 3, "the whole stored path matches");
        assert_eq!(index.lookup(&chain(&[1, 2, 3, 4])), 2);
        assert_eq!(index.lookup(&fork), 2);
        assert_eq!(
            index.lookup(&chain(&[1, 2, 3, 4, 9, 9])),
            2,
            "a divergent tail stops the match"
        );
        assert_eq!(
            index.lookup(&chain(&[9, 9, 3, 4])),
            0,
            "an unknown root matches nothing"
        );
        assert_eq!(index.lookup(&[]), 0, "the empty query matches nothing");

        index.unpin(&long);
        index.unpin(&fork);
    }

    #[test]
    fn pinned_paths_are_never_evicted() {
        let mut index = PrefixIndex::new();
        let path = chain(&[1, 2, 3, 4]);
        index.insert(&path);

        assert_eq!(index.evict_lru(), None, "every node is pinned");

        index.unpin(&path);
        assert!(index.evict_lru().is_some(), "unpinned, the path evicts");
    }

    #[test]
    fn eviction_is_leaf_only_deepest_first() {
        let mut index = PrefixIndex::new();
        let path = chain(&[1, 2, 3, 4, 5, 6]);
        index.insert(&path);
        index.unpin(&path);

        assert_eq!(
            index.evict_lru(),
            Some(path[2]),
            "only the deepest run is a leaf"
        );
        assert_eq!(index.evict_lru(), Some(path[1]));
        assert_eq!(index.evict_lru(), Some(path[0]));
        assert_eq!(index.evict_lru(), None);
        assert!(index.is_empty());
    }

    #[test]
    fn eviction_takes_the_least_recently_touched_leaf() {
        let mut index = PrefixIndex::new();
        let old = chain(&[1, 2, 3, 4]);
        let new = chain(&[5, 6, 7, 8]);
        index.insert(&old);
        index.insert(&new);
        index.unpin(&old);
        index.unpin(&new);

        // Touch the older path: its leaf becomes the fresher one.
        assert_eq!(index.lookup(&old), 2);
        assert_eq!(
            index.evict_lru(),
            Some(new[1]),
            "the untouched leaf goes first"
        );
        assert_eq!(index.evict_lru(), Some(new[0]));
        assert_eq!(index.evict_lru(), Some(old[1]));
    }

    #[test]
    fn a_shared_prefix_stays_pinned_until_every_path_releases() {
        let mut index = PrefixIndex::new();
        let left = chain(&[1, 2, 3, 4]);
        let right = chain(&[1, 2, 7, 8]);
        index.insert(&left);
        index.insert(&right);

        index.unpin(&left);
        assert_eq!(
            index.evict_lru(),
            Some(left[1]),
            "only left's divergent tail is unpinned"
        );
        assert_eq!(
            index.evict_lru(),
            None,
            "the shared root is still pinned by right"
        );
        assert!(index.contains(left[0]));

        index.unpin(&right);
        assert!(index.evict_lru().is_some());
    }

    #[test]
    fn remove_leaf_refuses_pinned_and_interior_nodes() {
        let mut index = PrefixIndex::new();
        let path = chain(&[1, 2, 3, 4]);
        index.insert(&path);

        assert!(!index.remove_leaf(path[1]), "pinned");
        index.unpin(&path);
        assert!(!index.remove_leaf(path[0]), "interior");
        assert!(index.remove_leaf(path[1]));
        assert!(index.remove_leaf(path[0]), "the parent became a leaf");
        assert!(!index.remove_leaf(path[0]), "already gone");
        assert!(index.is_empty());
    }

    #[test]
    fn reinserting_an_evicted_path_works_over_recycled_slots() {
        let mut index = PrefixIndex::new();
        let path = chain(&[1, 2, 3, 4]);
        index.insert(&path);
        index.unpin(&path);
        while index.evict_lru().is_some() {}

        index.insert(&path);
        assert_eq!(index.lookup(&path), 2);
        assert_eq!(index.len(), 2);
        index.unpin(&path);
    }

    proptest! {
        /// Oracle: the match length is the longest `k` such that the query's first `k` runs are
        /// a prefix of some inserted chain — computed by scanning the stored chains directly.
        #[test]
        fn lookup_agrees_with_a_scan_over_stored_chains(
            stored in vec(vec(0_u32..3, 0..12), 0..6),
            query in vec(0_u32..3, 0..12),
        ) {
            let mut index = PrefixIndex::new();
            let stored_chains: Vec<Vec<_>> = stored.iter().map(|tokens| chain(tokens)).collect();
            for hashes in &stored_chains {
                index.insert(hashes);
            }

            let query_chain = chain(&query);
            let expected = stored_chains
                .iter()
                .map(|stored_chain| {
                    query_chain
                        .iter()
                        .zip(stored_chain)
                        .take_while(|(q, s)| q == s)
                        .count()
                })
                .max()
                .unwrap_or(0);
            prop_assert_eq!(index.lookup(&query_chain), expected);

            for hashes in &stored_chains {
                index.unpin(hashes);
            }
        }
    }
}
