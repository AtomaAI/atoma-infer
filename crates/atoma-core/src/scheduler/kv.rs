//! KV bookkeeping for one sequence: prefix hits at admission, block growth from the pool with
//! eviction before exhaustion, caching of filled blocks, and release.
//!
//! Index and pool stay coupled by one invariant: every hash the index holds has a resident block
//! behind it, so a prefix hit turns straight into a block id.
//!
//! A hit block is protected by the sequence's pin in the prefix index, not by a lease; its id is
//! the pool's residence lookup. Only blocks a sequence computes itself are leased. Eviction goes
//! through the index alone — an unpinned leaf's hash, then the pool's block for it — so a block
//! any live sequence reads is never freed.

use crate::kv::{BlockLease, BlockPool, HashAlgorithm, PrefixIndex};
use crate::request::Sequence;
use crate::types::TokenCount;

/// The pool had no block to lease and the index had nothing left to evict. The caller decides
/// whether to preempt or wait.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PoolExhausted;

/// The KV substrate a scheduler owns, passed together wherever a sequence's blocks change.
pub(crate) struct Kv<'a> {
    pub pool: &'a mut BlockPool,
    pub index: &'a mut PrefixIndex,
    pub algorithm: HashAlgorithm,
    pub block_size: TokenCount,
}

impl Kv<'_> {
    /// Claims every cached prefix block of `sequence` at admission: the matched hashes are
    /// pinned, their resident blocks lead the block table, and the computed count starts past
    /// them. Returns the blocks claimed.
    pub(crate) fn claim_prefix(&mut self, sequence: &mut Sequence) -> usize {
        debug_assert!(sequence.block_table.is_empty() && sequence.pinned == 0);
        let hits = self.index.lookup(sequence.hashable_prefix(self.block_size));
        if hits == 0 {
            return 0;
        }
        self.index.insert(&sequence.chain[..hits]);
        for hash in &sequence.chain[..hits] {
            let block = self
                .pool
                .residence(*hash)
                .expect("a hash the index holds has resident bytes");
            sequence.block_table.push(block);
        }
        sequence.hits = hits;
        sequence.pinned = hits;
        sequence.advance(hits * self.block_size.get());
        hits
    }

    /// Grows `sequence`'s block table until it covers `tokens` tokens, evicting unpinned cache
    /// when the pool runs dry.
    ///
    /// All or nothing: when neither the pool nor eviction can supply a block, the blocks leased
    /// by this call go straight back, so a sequence never holds blocks for a step it is not going
    /// to run.
    pub(crate) fn ensure_blocks(
        &mut self,
        sequence: &mut Sequence,
        tokens: usize,
    ) -> Result<(), PoolExhausted> {
        let needed = tokens.div_ceil(self.block_size.get());
        let held = sequence.block_table.len();
        let leases_held = sequence.leases.len();
        while sequence.block_table.len() < needed {
            let Some(lease) = self.lease_or_evict() else {
                for lease in sequence.leases.drain(leases_held..) {
                    self.pool.release(lease);
                }
                sequence.block_table.truncate(held);
                return Err(PoolExhausted);
            };
            sequence.block_table.push(lease.block());
            sequence.leases.push(lease);
        }
        Ok(())
    }

    /// Publishes every block of `sequence` that filled since the last call — hashed into the
    /// pool, pinned into the index — so later requests with the same prefix hit it instead of
    /// recomputing. Called once per scheduled entry as its step result is applied.
    ///
    /// Only a block whose KV is computed is published, never one merely hashed, and `pinned`
    /// records how far publishing reached, so repeated calls resume rather than repeat.
    ///
    /// Publishing exposes a block's bytes to every other sequence, so the caller must not call
    /// this before the step's KV writes are visible.
    pub(crate) fn cache_filled_blocks(&mut self, sequence: &mut Sequence) {
        sequence.extend_chain(self.algorithm, self.block_size);
        let filled = sequence.computed() / self.block_size.get();
        for block in sequence.pinned..filled {
            let hash = sequence.chain[block];
            let parent = block.checked_sub(1).map(|parent| sequence.chain[parent]);
            if block >= sequence.hits {
                // When another sequence published this hash first, the block stays anonymous and
                // returns to the free list on release rather than becoming duplicate cache. The
                // pin below still lands on the node that first claim created.
                self.pool
                    .assign_hash(&sequence.leases[block - sequence.hits], hash);
            }
            self.index.insert_child(parent, hash);
            sequence.pinned = block + 1;
        }
    }

    /// Gives back everything `sequence` holds: its pins, its leases and its block table. Blocks
    /// with an assigned hash stay resident as evictable cache.
    pub(crate) fn release(&mut self, sequence: &mut Sequence) {
        self.index.unpin(&sequence.chain[..sequence.pinned]);
        sequence.pinned = 0;
        for lease in sequence.leases.drain(..) {
            self.pool.release(lease);
        }
        sequence.block_table.clear();
        sequence.hits = 0;
    }

    /// A free block, or the block behind the least recently used unpinned cache leaf.
    fn lease_or_evict(&mut self) -> Option<BlockLease> {
        loop {
            if let Some(lease) = self.pool.lease() {
                return Some(lease);
            }
            let hash = self.index.evict_lru()?;
            let evicted = self.pool.evict(hash);
            debug_assert!(
                evicted.is_some(),
                "an unpinned index leaf's block is cached, never leased"
            );
        }
    }
}
