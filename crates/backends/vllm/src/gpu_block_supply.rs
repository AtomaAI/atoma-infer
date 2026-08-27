//! GPU block supply: the engine's device blocks, leased from the atoma-core pool.
//!
//! Supply and measurement, not reuse. Every GPU block the engine hands out is leased 1:1 from
//! the preallocated [`BlockPool`]; each admitted prompt is chain-hashed and matched against the
//! [`PrefixIndex`], so prefix-cache hit rate is measurable end to end on this engine. No block
//! is aliased across sequences and no prefill compute is skipped — a hit is counted, not
//! consumed. Released blocks stay resident as evictable cache, reclaimed least-recently-used
//! and leaf-first when allocation runs out of free blocks.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use atoma_core::kv::{BlockLease, BlockPool, ExtraKeys, HashAlgorithm, PrefixIndex};
use atoma_core::protocol::{BlockHash, TokenCount};
use metrics::counter;
use tracing::debug;

use crate::block::{BlockDevice, PhysicalTokenBlock, SyncPhysicalTokenBlock};
use crate::block_allocator::BlockAllocatorError;
use crate::types::{ReadLock, WriteLock};

/// Counter of blocks queried against the prefix index at admission.
pub const PREFIX_CACHE_QUERIES_METRIC: &str = "atoma_kv_prefix_cache_queries";
/// Counter of queried blocks that were already indexed.
pub const PREFIX_CACHE_HITS_METRIC: &str = "atoma_kv_prefix_cache_hits";

/// Prefix-cache traffic so far. The hit rate is `hits / queries`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PrefixCacheStats {
    /// Full prompt blocks looked up at admission.
    pub queries: u64,
    /// Looked-up blocks that were already indexed.
    pub hits: u64,
}

/// Supplies the engine's GPU blocks from an [`atoma_core`] pool and measures prefix reuse.
///
/// Drop-in for the uncached `BlockAllocator` on the GPU side: same `allocate`/`free`/count
/// surface, same error type, plus the per-sequence hooks the block manager calls at admission,
/// retirement and swap-out.
#[derive(Debug)]
pub struct GpuBlockSupply {
    pool: BlockPool,
    index: PrefixIndex,
    algorithm: HashAlgorithm,
    /// Tokens per block, as configured.
    block_size: usize,
    /// The lease behind each outstanding block, by block number.
    leases: HashMap<u32, BlockLease>,
    /// Each indexed sequence's inserted path, for unpinning at retirement.
    chains: HashMap<u64, Vec<BlockHash>>,
    stats: PrefixCacheStats,
}

impl GpuBlockSupply {
    /// Builds the supply over a pool of `num_blocks` preallocated blocks.
    #[must_use]
    pub fn new(block_size: usize, num_blocks: usize) -> Self {
        // A u32 block count bounds pools at four billion blocks; at a kilobyte or more per
        // block, real configurations sit orders of magnitude below that.
        let num_blocks = u32::try_from(num_blocks).expect("block count fits u32");
        Self {
            pool: BlockPool::new(num_blocks),
            index: PrefixIndex::new(),
            algorithm: HashAlgorithm::Sha256V1,
            block_size,
            leases: HashMap::with_capacity(num_blocks as usize),
            chains: HashMap::new(),
            stats: PrefixCacheStats::default(),
        }
    }

    /// Leases one block, evicting cached blocks least-recently-used first under pressure.
    ///
    /// # Errors
    ///
    /// Returns [`BlockAllocatorError::OutOfMemory`] when every block is leased and nothing is
    /// evictable.
    pub fn allocate(&mut self) -> Result<SyncPhysicalTokenBlock, BlockAllocatorError> {
        let lease = loop {
            if let Some(lease) = self.pool.lease() {
                break lease;
            }
            let Some(hash) = self.index.evict_lru() else {
                return Err(BlockAllocatorError::OutOfMemory);
            };
            self.pool.evict(hash);
        };
        let block_number = lease.block().get();
        self.leases.insert(block_number, lease);
        let mut block = PhysicalTokenBlock::new(block_number, self.block_size, BlockDevice::Gpu);
        block.increment_ref_count();
        Ok(Arc::new(RwLock::new(block)))
    }

    /// Frees one reference to `block`; the last reference releases the lease, leaving an
    /// identified block resident as evictable cache.
    ///
    /// # Errors
    ///
    /// Returns [`BlockAllocatorError::CannotDoubleFree`] for a block whose reference count is
    /// already zero, and [`BlockAllocatorError::BlockNotFound`] for a block this supply never
    /// handed out.
    pub fn free(&mut self, block: &SyncPhysicalTokenBlock) -> Result<(), BlockAllocatorError> {
        let block_number = {
            let block = block.read_lock()?;
            if block.ref_count() == 0 {
                return Err(BlockAllocatorError::CannotDoubleFree(block.block_number()));
            }
            block.block_number()
        };
        let remaining = {
            let mut block = block.write_lock()?;
            block.decrease_ref_count()?;
            block.ref_count()
        };
        if remaining == 0 {
            let Some(lease) = self.leases.remove(&block_number) else {
                return Err(BlockAllocatorError::BlockNotFound(block_number));
            };
            self.pool.release(lease);
        }
        Ok(())
    }

    /// Blocks obtainable right now: free, plus cached blocks an eviction would reclaim.
    #[must_use]
    pub fn get_num_free_blocks(&self) -> usize {
        self.pool.available()
    }

    /// Every block the pool was built with.
    #[must_use]
    pub fn get_num_total_blocks(&self) -> usize {
        self.pool.block_count()
    }

    /// Measures and registers one admitted sequence: chain-hashes its prompt, counts the
    /// longest-prefix match as hits, pins the path, and claims each full block's hash for the
    /// block that holds its bytes. Indexing is skipped under a sliding window, where block
    /// contents are overwritten in place.
    pub fn index_sequence(&mut self, sequence_id: u64, token_ids: &[u32], block_numbers: &[u32]) {
        let Some(block_size) = TokenCount::new(self.block_size) else {
            return;
        };
        let chain = self
            .algorithm
            .chain(block_size, token_ids, ExtraKeys::none());
        let matched = self.index.lookup(&chain);
        self.stats.queries += chain.len() as u64;
        self.stats.hits += matched as u64;
        counter!(PREFIX_CACHE_QUERIES_METRIC).increment(chain.len() as u64);
        counter!(PREFIX_CACHE_HITS_METRIC).increment(matched as u64);
        debug!(
            sequence_id,
            queried = chain.len(),
            matched,
            "prefix cache lookup"
        );
        self.index.insert(&chain);
        for (hash, block_number) in chain.iter().zip(block_numbers) {
            if let Some(lease) = self.leases.get(block_number) {
                // A duplicate claim is refused and the first copy keeps the hash; this
                // sequence's copy then frees outright on release instead of caching.
                self.pool.assign_hash(lease, *hash);
            }
        }
        if let Some(previous) = self.chains.insert(sequence_id, chain) {
            self.index.unpin(&previous);
        }
    }

    /// Retires a finished sequence: its path unpins and becomes evictable. Unindexed
    /// sequences retire as a no-op.
    pub fn finish_sequence(&mut self, sequence_id: u64) {
        if let Some(chain) = self.chains.remove(&sequence_id) {
            self.index.unpin(&chain);
        }
    }

    /// Discards a sequence whose device copies are gone (swap-out): its path unpins, and every
    /// block of its chain that was cached is evicted and its node removed, deepest first. Nodes
    /// still pinned by another sequence stay, as their bytes are still claimed.
    pub fn discard_sequence(&mut self, sequence_id: u64) {
        let Some(chain) = self.chains.remove(&sequence_id) else {
            return;
        };
        self.index.unpin(&chain);
        for hash in chain.iter().rev() {
            if self.pool.evict(*hash).is_some() {
                self.index.remove_leaf(*hash);
            }
        }
    }

    /// Forgets every sequence and evicts everything cached, for a full engine reset. The
    /// caller has already freed every outstanding block.
    pub fn clear(&mut self) {
        let chains: Vec<_> = self.chains.drain().map(|(_, chain)| chain).collect();
        for chain in &chains {
            self.index.unpin(chain);
        }
        while let Some(hash) = self.index.evict_lru() {
            self.pool.evict(hash);
        }
    }

    /// Prefix-cache traffic so far.
    #[must_use]
    pub fn prefix_cache_stats(&self) -> PrefixCacheStats {
        self.stats
    }
}

impl Drop for GpuBlockSupply {
    /// Returns every outstanding lease before the pool drops: blocks still out at engine
    /// teardown are owned here, not leaked, so the leases' debug leak guard stays quiet.
    fn drop(&mut self) {
        let leases: Vec<BlockLease> = self.leases.drain().map(|(_, lease)| lease).collect();
        for lease in leases {
            self.pool.release(lease);
        }
    }
}
