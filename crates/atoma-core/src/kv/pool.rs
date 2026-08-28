//! The block pool: preallocated device-block bookkeeping owned by one thread.
//!
//! Every block is preallocated and addressed by index; the free list and the cached list are
//! intrusive doubly-linked lists of indices, so no step-path operation allocates, locks, or
//! chases heap pointers. A block leaves the pool only as a [`BlockLease`], and nothing frees or
//! reassigns a block by id: the only paths back to the free list are surrendering the lease or
//! evicting an unleased cached block, so a leased block cannot be evicted by construction.

use std::collections::HashMap;
use std::thread;

use crate::types::{BlockHash, BlockId};

/// The links' null: where a pointer-based list would hold a null pointer, the index-based lists
/// hold `NIL` — in a block's `prev`/`next` for a missing neighbor, and in a list's `head`/`tail`
/// when it is empty. Never a valid block index: indices stay below the pool's `u32` block count.
const NIL: u32 = u32::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockState {
    /// On the free list, holding nothing.
    Free,
    /// Held by exactly one live [`BlockLease`].
    Leased,
    /// On the cached list, holding identified bytes with no lease: evictable.
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

/// The ends and length of one list threaded through the blocks' `prev`/`next` links.
///
/// A block sits on at most one list at a time — free or cached, never both, and a leased block
/// on neither — so every list shares the same two links per block. Blocks leave from the head
/// and join at the tail, so each list recycles FIFO.
#[derive(Debug, Clone, Copy)]
struct BlockList {
    head: u32,
    tail: u32,
    len: u32,
}

impl BlockList {
    const EMPTY: Self = Self {
        head: NIL,
        tail: NIL,
        len: 0,
    };

    fn push_tail(&mut self, blocks: &mut [BlockMeta], index: u32) {
        blocks[index as usize].prev = self.tail;
        blocks[index as usize].next = NIL;
        if self.tail == NIL {
            self.head = index;
        } else {
            blocks[self.tail as usize].next = index;
        }
        self.tail = index;
        self.len += 1;
    }

    fn unlink(&mut self, blocks: &mut [BlockMeta], index: u32) {
        let prev = blocks[index as usize].prev;
        let next = blocks[index as usize].next;
        if prev == NIL {
            self.head = next;
        } else {
            blocks[prev as usize].next = next;
        }
        if next == NIL {
            self.tail = prev;
        } else {
            blocks[next as usize].prev = prev;
        }
        self.len -= 1;
    }

    fn pop_head(&mut self, blocks: &mut [BlockMeta]) -> Option<u32> {
        if self.head == NIL {
            return None;
        }
        let head = self.head;
        self.unlink(blocks, head);
        Some(head)
    }
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

/// Preallocated, index-based pool of device blocks with intrusive free and cached lists.
///
/// Single-owner: `&mut self` everywhere, no lock, no reference count. Identity stays separate
/// from residence — [`BlockPool::residence`] answers which slot holds a hash's bytes, and every
/// other consumer speaks hashes.
#[derive(Debug)]
pub struct BlockPool {
    blocks: Box<[BlockMeta]>,
    /// Which slot holds each assigned hash's bytes, leased or cached.
    by_hash: HashMap<BlockHash, BlockId>,
    free: BlockList,
    /// Cached blocks in the order they were released, so the longest-cached evicts first.
    cached: BlockList,
}

impl BlockPool {
    /// Builds a pool of `block_count` free blocks, allocating everything it will ever hold.
    #[must_use]
    pub fn new(block_count: u32) -> Self {
        let mut blocks: Box<[BlockMeta]> = (0..block_count)
            .map(|_| BlockMeta {
                state: BlockState::Free,
                hash: None,
                prev: NIL,
                next: NIL,
            })
            .collect();
        let mut free = BlockList::EMPTY;
        for index in 0..block_count {
            free.push_tail(&mut blocks, index);
        }
        Self {
            blocks,
            by_hash: HashMap::with_capacity(block_count as usize),
            free,
            cached: BlockList::EMPTY,
        }
    }

    /// Leases the block at the head of the free list, or `None` when every block is leased or
    /// cached. The caller decides whether to evict and retry.
    pub fn lease(&mut self) -> Option<BlockLease> {
        let head = self.free.pop_head(&mut self.blocks)?;
        self.blocks[head as usize].state = BlockState::Leased;
        Some(BlockLease {
            block: BlockId::new(head),
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
        let index = lease.block.get();
        if self.blocks[index as usize].hash.is_some() {
            self.blocks[index as usize].state = BlockState::Cached;
            self.cached.push_tail(&mut self.blocks, index);
        } else {
            self.free_block(index);
        }
    }

    /// Evicts the cached block holding `hash`, returning the freed slot.
    ///
    /// Returns `None` when no cached block holds `hash` — including while the holding block is
    /// leased, since a lease keeps its block out of the cached state entirely.
    pub fn evict(&mut self, hash: BlockHash) -> Option<BlockId> {
        let block = *self.by_hash.get(&hash)?;
        let index = block.get();
        if self.blocks[index as usize].state != BlockState::Cached {
            return None;
        }
        self.cached.unlink(&mut self.blocks, index);
        self.by_hash.remove(&hash);
        self.free_block(index);
        Some(block)
    }

    /// Evicts the longest-cached block regardless of which hash it holds, returning the hash and
    /// the freed slot; `None` when nothing is cached. For callers whose cached bytes have no
    /// reader; a pin-aware caller evicts by hash instead.
    pub fn evict_any(&mut self) -> Option<(BlockHash, BlockId)> {
        if self.cached.head == NIL {
            return None;
        }
        let hash = self.blocks[self.cached.head as usize].hash?;
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
        self.free.len as usize
    }

    /// Blocks a lease could obtain: free now, plus cached blocks an eviction would free.
    #[must_use]
    pub fn available(&self) -> usize {
        (self.free.len + self.cached.len) as usize
    }

    /// Every block this pool was built with.
    #[must_use]
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Returns an off-list block to the back of the free list, anonymous and `Free`.
    fn free_block(&mut self, index: u32) {
        self.blocks[index as usize].state = BlockState::Free;
        self.blocks[index as usize].hash = None;
        self.free.push_tail(&mut self.blocks, index);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::{BlockLease, BlockPool};
    use crate::kv::test_support::hash_of;
    use crate::types::BlockHash;

    fn release_all(pool: &mut BlockPool, leases: Vec<BlockLease>) {
        for lease in leases {
            pool.release(lease);
        }
    }

    /// Leases one block, claims `hash` for it and releases it, so it is resident as cache.
    fn cache(pool: &mut BlockPool, hash: BlockHash) {
        let lease = pool
            .lease()
            .expect("the pool has a free block to cache into");
        assert!(pool.assign_hash(&lease, hash));
        pool.release(lease);
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
        let hash = hash_of(&[1, 2, 3, 4]);
        cache(&mut pool, hash);
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
    fn evict_any_takes_the_longest_cached_block_first() {
        let mut pool = BlockPool::new(3);
        let hashes = [hash_of(&[1]), hash_of(&[2]), hash_of(&[3])];
        for hash in hashes {
            cache(&mut pool, hash);
        }
        assert_eq!(pool.available(), 3);

        assert_eq!(pool.evict_any().map(|(hash, _)| hash), Some(hashes[0]));
        assert_eq!(pool.evict_any().map(|(hash, _)| hash), Some(hashes[1]));
        assert_eq!(pool.evict_any().map(|(hash, _)| hash), Some(hashes[2]));
        assert_eq!(pool.free_count(), 3);
    }

    #[test]
    fn evicting_by_hash_from_the_middle_keeps_the_cached_order() {
        let mut pool = BlockPool::new(3);
        let hashes = [hash_of(&[1]), hash_of(&[2]), hash_of(&[3])];
        for hash in hashes {
            cache(&mut pool, hash);
        }

        assert!(pool.evict(hashes[1]).is_some());
        assert_eq!(pool.available(), 3);
        assert_eq!(pool.free_count(), 1);
        assert_eq!(pool.evict_any().map(|(hash, _)| hash), Some(hashes[0]));
        assert_eq!(pool.evict_any().map(|(hash, _)| hash), Some(hashes[2]));
        assert_eq!(pool.evict_any(), None);
        assert_eq!(pool.free_count(), 3);
    }

    #[test]
    fn a_zero_capacity_pool_leases_nothing() {
        let mut pool = BlockPool::new(0);
        assert_eq!(pool.block_count(), 0);
        assert_eq!(pool.free_count(), 0);
        assert_eq!(pool.available(), 0);
        assert!(pool.lease().is_none());
        assert_eq!(pool.evict_any(), None);
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
