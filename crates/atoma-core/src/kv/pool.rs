//! The block pool: preallocated device-block bookkeeping owned by one thread.
//!
//! Every block is preallocated and addressed by index; the free list is an intrusive
//! doubly-linked list of indices, so no step-path operation allocates, locks, or chases heap
//! pointers. A block leaves the pool only as a [`BlockLease`], and nothing frees or reassigns a
//! block by id: the only paths back to the free list are surrendering the lease or evicting an
//! unleased cached block, so a leased block cannot be evicted by construction.

use std::collections::HashMap;
use std::thread;

use crate::types::{BlockHash, BlockId};

/// The links' null: where a pointer-based list would hold a null pointer, the index-based free
/// list holds `NIL` — in a block's `prev`/`next` for a missing neighbor, and in `free_head`/
/// `free_tail` for an empty list. Never a valid block index: indices stay below the pool's
/// `u32` block count.
const NIL: u32 = u32::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockState {
    /// On the free list, holding nothing.
    Free,
    /// Held by exactly one live [`BlockLease`].
    Leased,
    /// Holding identified bytes with no lease: evictable.
    Cached,
}

#[derive(Debug, Clone, Copy)]
struct BlockMeta {
    state: BlockState,
    /// The chain hash whose bytes this block holds, once assigned and recorded in `by_hash`.
    hash: Option<BlockHash>,
    prev: u32,
    next: u32,
}

/// Holds one pool block for exactly one owner.
///
/// While the lease exists its block is un-evictable and unreachable by any other pool operation;
/// the block returns to the pool only through [`BlockPool::release`], which consumes the lease.
/// Dropping a lease without releasing it leaks the block until process exit, which the drop guard
/// reports in debug builds.
#[derive(Debug)]
#[must_use = "an unreleased lease leaks its block until process exit"]
pub struct BlockLease {
    block: BlockId,
    released: bool,
}

impl BlockLease {
    /// The block this lease holds.
    #[must_use]
    pub fn block(&self) -> BlockId {
        self.block
    }
}

impl Drop for BlockLease {
    fn drop(&mut self) {
        // Skipped mid-unwind so an unrelated test panic is not masked by a double panic.
        if !thread::panicking() {
            debug_assert!(
                self.released,
                "BlockLease for {:?} dropped without release; the block leaks until process exit",
                self.block
            );
        }
    }
}

/// Preallocated, index-based pool of device blocks with an intrusive free list.
///
/// Single-owner: `&mut self` everywhere, no lock, no reference count. Identity stays separate
/// from residence — [`BlockPool::residence`] answers which slot holds a hash's bytes, and every
/// other consumer speaks hashes.
#[derive(Debug)]
pub struct BlockPool {
    blocks: Box<[BlockMeta]>,
    /// Which slot holds each assigned hash's bytes, leased or cached.
    by_hash: HashMap<BlockHash, BlockId>,
    free_head: u32,
    free_tail: u32,
    free_count: u32,
    cached_count: u32,
}

impl BlockPool {
    /// Builds a pool of `block_count` free blocks, allocating everything it will ever hold.
    #[must_use]
    pub fn new(block_count: u32) -> Self {
        let blocks = (0..block_count)
            .map(|index| BlockMeta {
                state: BlockState::Free,
                hash: None,
                prev: index.checked_sub(1).unwrap_or(NIL),
                next: if index + 1 < block_count {
                    index + 1
                } else {
                    NIL
                },
            })
            .collect();
        Self {
            blocks,
            by_hash: HashMap::with_capacity(block_count as usize),
            free_head: if block_count == 0 { NIL } else { 0 },
            free_tail: block_count.checked_sub(1).unwrap_or(NIL),
            free_count: block_count,
            cached_count: 0,
        }
    }

    /// Leases the block at the head of the free list, or `None` when every block is leased or
    /// cached. The caller decides whether to evict and retry.
    pub fn lease(&mut self) -> Option<BlockLease> {
        let head = self.pop_free_head()?;
        self.blocks[head.index()].state = BlockState::Leased;
        Some(BlockLease {
            block: head,
            released: false,
        })
    }

    /// Records that `lease`'s block now holds `hash`'s bytes.
    ///
    /// Returns `false` without recording when another block already holds that hash — the first
    /// copy keeps the claim, and this block stays anonymous, so it frees rather than caches on
    /// release.
    pub fn assign_hash(&mut self, lease: &BlockLease, hash: BlockHash) -> bool {
        if self.by_hash.contains_key(&hash) {
            return false;
        }
        self.by_hash.insert(hash, lease.block);
        self.blocks[lease.block.index()].hash = Some(hash);
        true
    }

    /// Surrenders `lease`. A block holding an assigned hash stays resident as evictable cache;
    /// an anonymous block returns straight to the free list.
    pub fn release(&mut self, mut lease: BlockLease) {
        lease.released = true;
        let index = lease.block.index();
        if self.blocks[index].hash.is_some() {
            self.blocks[index].state = BlockState::Cached;
            self.cached_count += 1;
        } else {
            self.push_free_tail(lease.block);
        }
    }

    /// Evicts the cached block holding `hash`, returning the freed slot.
    ///
    /// Returns `None` when no cached block holds `hash` — including while the holding block is
    /// leased, since a lease keeps its block out of the cached state entirely.
    pub fn evict(&mut self, hash: BlockHash) -> Option<BlockId> {
        let block = *self.by_hash.get(&hash)?;
        let meta = &mut self.blocks[block.index()];
        if meta.state != BlockState::Cached {
            return None;
        }
        meta.hash = None;
        self.by_hash.remove(&hash);
        self.cached_count -= 1;
        self.push_free_tail(block);
        Some(block)
    }

    /// Evicts some cached block regardless of which hash it holds, returning the hash and the
    /// freed slot; `None` when nothing is cached. For callers whose cached bytes have no
    /// reader; a pin-aware caller evicts by hash instead.
    pub fn evict_any(&mut self) -> Option<(BlockHash, BlockId)> {
        let hash = self.blocks.iter().find_map(|meta| {
            if meta.state == BlockState::Cached {
                meta.hash
            } else {
                None
            }
        })?;
        let block = self.evict(hash)?;
        Some((hash, block))
    }

    /// Which slot holds `hash`'s bytes right now, leased or cached. Identity never carries
    /// residence: consumers that only compare prefixes never need this lookup.
    #[must_use]
    pub fn residence(&self, hash: BlockHash) -> Option<BlockId> {
        self.by_hash.get(&hash).copied()
    }

    /// Blocks on the free list.
    #[must_use]
    pub fn free_count(&self) -> usize {
        self.free_count as usize
    }

    /// Blocks a lease could obtain: free now, plus cached blocks an eviction would free.
    #[must_use]
    pub fn available(&self) -> usize {
        (self.free_count + self.cached_count) as usize
    }

    /// Every block this pool was built with.
    #[must_use]
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Takes the block at the front of the free list, or `None` when the list is empty.
    /// Blocks leave from the head and return at the tail, so freed blocks recycle FIFO.
    fn pop_free_head(&mut self) -> Option<BlockId> {
        if self.free_head == NIL {
            return None;
        }
        let head = self.free_head;
        let next = self.blocks[head as usize].next;
        self.free_head = next;
        if next == NIL {
            // The popped head was also the tail, so the list is now empty. A stale tail would
            // let push_free_tail link the next freed block behind this now-leased one.
            self.free_tail = NIL;
        } else {
            // The new head has no predecessor.
            self.blocks[next as usize].prev = NIL;
        }
        self.free_count -= 1;
        Some(BlockId::new(head))
    }

    /// Appends `block` at the back of the free list, resetting its metadata to `Free`.
    fn push_free_tail(&mut self, block: BlockId) {
        let index = block.get();
        self.blocks[index as usize] = BlockMeta {
            state: BlockState::Free,
            hash: None,
            prev: self.free_tail,
            next: NIL,
        };
        if self.free_tail == NIL {
            // Appending to an empty list: this block becomes head and tail at once.
            self.free_head = index;
        } else {
            self.blocks[self.free_tail as usize].next = index;
        }
        self.free_tail = index;
        self.free_count += 1;
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::{BlockLease, BlockPool};
    use crate::kv::{ExtraKeys, HashAlgorithm};
    use crate::types::BlockHash;

    fn hash_of(tokens: &[u32]) -> BlockHash {
        HashAlgorithm::Sha256V1.hash_run(None, tokens, ExtraKeys::none())
    }

    fn release_all(pool: &mut BlockPool, leases: Vec<BlockLease>) {
        for lease in leases {
            pool.release(lease);
        }
    }

    #[test]
    fn every_block_leases_exactly_once_then_the_pool_is_exhausted() {
        let mut pool = BlockPool::new(4);
        assert_eq!(pool.free_count(), 4);
        assert_eq!(pool.block_count(), 4);

        let leases: Vec<_> = (0..4).map(|_| pool.lease().unwrap()).collect();
        let distinct: HashSet<_> = leases.iter().map(|lease| lease.block().get()).collect();
        assert_eq!(distinct.len(), 4, "no block is leased twice");
        assert_eq!(pool.free_count(), 0);
        assert!(pool.lease().is_none(), "an exhausted pool refuses to lease");

        release_all(&mut pool, leases);
    }

    #[test]
    fn free_count_returns_to_baseline_after_release() {
        let mut pool = BlockPool::new(3);
        let baseline = pool.free_count();

        let leases: Vec<_> = (0..3).map(|_| pool.lease().unwrap()).collect();
        assert_eq!(pool.free_count(), 0);
        release_all(&mut pool, leases);
        assert_eq!(pool.free_count(), baseline);

        let again = pool.lease().expect("released blocks lease again");
        pool.release(again);
    }

    #[test]
    fn a_released_block_with_an_assigned_hash_stays_resident_as_cache() {
        let mut pool = BlockPool::new(2);
        let lease = pool.lease().unwrap();
        let block = lease.block();
        let hash = hash_of(&[1, 2, 3, 4]);

        assert!(pool.assign_hash(&lease, hash));
        assert_eq!(
            pool.residence(hash),
            Some(block),
            "residence answers while leased"
        );

        pool.release(lease);
        assert_eq!(
            pool.residence(hash),
            Some(block),
            "residence survives release"
        );
        assert_eq!(pool.free_count(), 1, "a cached block is not free");
        assert_eq!(pool.available(), 2, "but an eviction could free it");
    }

    #[test]
    fn evicting_a_cached_block_frees_it_and_forgets_its_residence() {
        let mut pool = BlockPool::new(1);
        let lease = pool.lease().unwrap();
        let hash = hash_of(&[1, 2, 3, 4]);
        assert!(pool.assign_hash(&lease, hash));
        pool.release(lease);
        assert_eq!(pool.free_count(), 0);

        let freed = pool.evict(hash).expect("a cached block evicts");
        assert_eq!(pool.residence(hash), None);
        assert_eq!(pool.free_count(), 1);

        let release = pool.lease().expect("the evicted slot leases again");
        assert_eq!(release.block(), freed);
        pool.release(release);
    }

    #[test]
    fn a_leased_block_cannot_be_evicted() {
        let mut pool = BlockPool::new(1);
        let lease = pool.lease().unwrap();
        let hash = hash_of(&[1, 2, 3, 4]);
        assert!(pool.assign_hash(&lease, hash));

        assert_eq!(pool.evict(hash), None, "the lease keeps the block resident");
        assert_eq!(pool.residence(hash), Some(lease.block()));

        pool.release(lease);
        assert!(pool.evict(hash).is_some(), "released, the same hash evicts");
    }

    #[test]
    fn evicting_an_unknown_hash_is_refused() {
        let mut pool = BlockPool::new(1);
        assert_eq!(pool.evict(hash_of(&[9, 9, 9, 9])), None);
        assert_eq!(pool.free_count(), 1);
    }

    #[test]
    fn a_duplicate_hash_claim_keeps_the_first_copy() {
        let mut pool = BlockPool::new(2);
        let first = pool.lease().unwrap();
        let second = pool.lease().unwrap();
        let first_block = first.block();
        let hash = hash_of(&[1, 2, 3, 4]);

        assert!(pool.assign_hash(&first, hash));
        assert!(
            !pool.assign_hash(&second, hash),
            "the claim is not overwritten"
        );
        assert_eq!(pool.residence(hash), Some(first_block));

        pool.release(second);
        assert_eq!(pool.free_count(), 1, "the anonymous copy frees outright");
        pool.release(first);
        assert_eq!(pool.free_count(), 1, "the claiming copy stays cached");
        assert_eq!(pool.residence(hash), Some(first_block));
    }

    #[test]
    fn evict_any_reclaims_only_cached_blocks() {
        let mut pool = BlockPool::new(2);
        let identified = pool.lease().unwrap();
        let hash = hash_of(&[1, 2, 3, 4]);
        assert!(pool.assign_hash(&identified, hash));
        let anonymous = pool.lease().unwrap();

        assert_eq!(
            pool.evict_any(),
            None,
            "leased blocks are untouchable, hash or not"
        );

        let identified_block = identified.block();
        pool.release(identified);
        assert_eq!(pool.evict_any(), Some((hash, identified_block)));
        assert_eq!(pool.residence(hash), None);
        assert_eq!(pool.evict_any(), None, "nothing cached remains");

        pool.release(anonymous);
    }

    #[test]
    fn a_zero_capacity_pool_leases_nothing() {
        let mut pool = BlockPool::new(0);
        assert_eq!(pool.block_count(), 0);
        assert_eq!(pool.free_count(), 0);
        assert_eq!(pool.available(), 0);
        assert!(pool.lease().is_none());
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "dropped without release")]
    fn dropping_a_lease_without_release_is_reported() {
        let mut pool = BlockPool::new(1);
        let lease = pool.lease().unwrap();
        drop(lease);
    }
}
