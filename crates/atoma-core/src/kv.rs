//! The KV substrate the engine owns.
//!
//! Identity here is content, never residence: a block-sized token run is identified by its chain
//! hash, which commits to the whole prefix behind it, and which slot currently holds a hash's
//! bytes is always a separate lookup. Everything is host-side data structures and arithmetic —
//! no device code, no threads, no channels.

mod chain_hash;
mod layer_group;
mod pool;
mod prefix_index;

pub use chain_hash::{ExtraKeys, HashAlgorithm};
pub use layer_group::{BlockLayout, CacheKind, KvCacheSpec, KvSource, LayerGroup, LayerGroupError};
pub use pool::{BlockLease, BlockPool};
pub use prefix_index::PrefixIndex;

#[cfg(test)]
mod tests {
    use crate::kv::{BlockPool, ExtraKeys, HashAlgorithm, PrefixIndex};
    use crate::protocol::{BlockHash, TokenCount};

    /// One admission-to-eviction cycle across the pool and the index together, as the engine
    /// will drive them: hashes looked up and inserted at admission, blocks leased per run,
    /// releases leaving cache behind, and eviction restoring the pool to baseline.
    #[test]
    fn free_count_returns_to_baseline_across_a_pool_and_index_cycle() {
        let block_size = TokenCount::new(4).expect("test block size is nonzero");
        let algorithm = HashAlgorithm::Sha256V1;
        let mut pool = BlockPool::new(8);
        let mut index = PrefixIndex::new();
        let baseline = pool.free_count();

        // Two requests sharing a four-token prefix, admitted one after the other.
        let first = algorithm.chain(block_size, &[1, 2, 3, 4, 5, 6, 7, 8], ExtraKeys::none());
        let second = algorithm.chain(block_size, &[1, 2, 3, 4, 9, 9, 9, 9], ExtraKeys::none());

        assert_eq!(
            index.lookup(&first),
            0,
            "a cold index has no prefix to offer"
        );
        index.insert(&first);
        let first_leases: Vec<_> = first
            .iter()
            .map(|&hash| {
                let lease = pool.lease().expect("the pool covers both requests");
                pool.assign_hash(&lease, hash);
                lease
            })
            .collect();

        assert_eq!(index.lookup(&second), 1, "the shared run is a hit");
        index.insert(&second);
        let second_leases: Vec<_> = second
            .iter()
            .map(|&hash| {
                let lease = pool.lease().expect("the pool covers both requests");
                pool.assign_hash(&lease, hash);
                lease
            })
            .collect();
        assert_eq!(pool.free_count(), baseline - 4);

        // Both requests finish: their paths unpin and their blocks release into cache.
        index.unpin(&first);
        for lease in first_leases {
            pool.release(lease);
        }
        index.unpin(&second);
        for lease in second_leases {
            pool.release(lease);
        }
        assert_eq!(
            pool.free_count(),
            baseline - 3,
            "hashed blocks stay resident as cache"
        );
        assert_eq!(
            pool.available(),
            baseline,
            "one duplicate-claim block freed outright"
        );

        // Pressure evicts leaf-first until nothing identified remains resident.
        let mut evicted: Vec<BlockHash> = Vec::new();
        while let Some(hash) = index.evict_lru() {
            pool.evict(hash);
            evicted.push(hash);
        }
        assert_eq!(evicted.len(), 3);
        assert!(index.is_empty());
        assert_eq!(
            pool.free_count(),
            baseline,
            "eviction restores the baseline"
        );
        for hash in evicted {
            assert_eq!(pool.residence(hash), None);
        }
    }
}
