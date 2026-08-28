//! Padding dummies: the permanent block leases behind graph padding.
//!
//! A live batch is padded up to its bucket with dummy requests, and a dummy's attention must
//! read valid KV, so every dummy owns its own block for the process lifetime. The engine
//! reserves them from the pool once at startup — a held lease already makes a block
//! un-evictable, so permanence needs no second mechanism — and hands the block ids to the
//! executor's Allocation phase. What a configuration's padding costs is answerable before any
//! pool exists.

use thiserror::Error;

use crate::kv::{BlockLease, BlockPool, KvCacheSpec};
use crate::types::{BlockId, RequestCount};

/// The padding dummies' blocks, held as leases for the process lifetime.
///
/// A batch of one real request padded to the maximum batch needs every other slot filled, so
/// the reservation holds one block per dummy: the configured maximum batch minus one.
#[derive(Debug)]
pub struct PaddingReservation {
    leases: Vec<BlockLease>,
}

impl PaddingReservation {
    /// Reserves one block per dummy from `pool`, permanently.
    ///
    /// # Errors
    ///
    /// Returns [`PaddingError::NotEnoughFreeBlocks`] when the pool cannot cover the reservation;
    /// nothing is reserved in that case.
    pub fn reserve(pool: &mut BlockPool, max_batch: RequestCount) -> Result<Self, PaddingError> {
        let dummy_count = max_batch.get() - 1;
        let mut leases = Vec::with_capacity(dummy_count);
        for _ in 0..dummy_count {
            let Some(lease) = pool.lease() else {
                let free = pool.free_count() + leases.len();
                for lease in leases {
                    pool.release(lease);
                }
                return Err(PaddingError::NotEnoughFreeBlocks {
                    needed: dummy_count,
                    free,
                });
            };
            leases.push(lease);
        }
        Ok(Self { leases })
    }

    /// Bytes the reservation costs under `spec` — answerable at configuration time, before any
    /// pool exists.
    #[must_use]
    pub fn cost_bytes(spec: &KvCacheSpec, max_batch: RequestCount) -> usize {
        (max_batch.get() - 1) * spec.bytes_per_block()
    }

    /// Dummies reserved: the configured maximum batch minus one.
    #[must_use]
    pub fn dummy_count(&self) -> usize {
        self.leases.len()
    }

    /// Each dummy's block, in reservation order — what the executor's Allocation phase receives.
    #[must_use]
    pub fn block_ids(&self) -> Vec<BlockId> {
        self.leases.iter().map(BlockLease::block).collect()
    }

    /// Surrenders the reservation. The engine never calls this — the dummies live as long as
    /// the process — but shutdown and tests return the blocks cleanly.
    pub fn release(self, pool: &mut BlockPool) {
        for lease in self.leases {
            pool.release(lease);
        }
    }
}

/// A padding reservation the pool cannot cover.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum PaddingError {
    /// The configured maximum batch needs more dummy blocks than the pool has free.
    #[error("padding needs {needed} free blocks for its dummies but the pool has {free}")]
    NotEnoughFreeBlocks { needed: usize, free: usize },
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::{PaddingError, PaddingReservation};
    use crate::kv::test_support::full_attention_group;
    use crate::kv::{BlockPool, KvCacheSpec};
    use crate::test_support::requests as max_batch;

    #[test]
    fn the_reservation_holds_one_distinct_block_per_dummy() {
        let mut pool = BlockPool::new(8);
        let reservation = PaddingReservation::reserve(&mut pool, max_batch(4)).unwrap();

        assert_eq!(
            reservation.dummy_count(),
            3,
            "maximum batch minus the one real request"
        );
        let ids = reservation.block_ids();
        let distinct: HashSet<_> = ids.iter().map(|id| id.get()).collect();
        assert_eq!(distinct.len(), 3, "every dummy owns its own block");
        assert_eq!(pool.free_count(), 5);

        reservation.release(&mut pool);
        assert_eq!(pool.free_count(), 8, "released only for shutdown and tests");
    }

    #[test]
    fn reserved_blocks_stay_out_of_reach_for_the_pool_lifetime() {
        let mut pool = BlockPool::new(3);
        let reservation = PaddingReservation::reserve(&mut pool, max_batch(3)).unwrap();

        assert_eq!(
            pool.available(),
            1,
            "the dummies' blocks are not obtainable"
        );
        let only = pool.lease().expect("one block is left");
        assert!(pool.lease().is_none());

        pool.release(only);
        reservation.release(&mut pool);
    }

    #[test]
    fn a_maximum_batch_of_one_reserves_nothing() {
        let mut pool = BlockPool::new(2);
        let reservation = PaddingReservation::reserve(&mut pool, max_batch(1)).unwrap();
        assert_eq!(reservation.dummy_count(), 0);
        assert!(reservation.block_ids().is_empty());
        assert_eq!(pool.free_count(), 2);
        reservation.release(&mut pool);
    }

    #[test]
    fn a_pool_too_small_for_the_dummies_is_reported_with_both_numbers() {
        let mut pool = BlockPool::new(2);
        let error = PaddingReservation::reserve(&mut pool, max_batch(4)).unwrap_err();
        assert_eq!(
            error,
            PaddingError::NotEnoughFreeBlocks { needed: 3, free: 2 }
        );
        assert_eq!(pool.free_count(), 2, "a refused reservation takes nothing");
    }

    #[test]
    fn padding_cost_is_reported_at_configuration_time() {
        let spec = KvCacheSpec::new(vec![full_attention_group(0)]).unwrap();
        assert_eq!(
            PaddingReservation::cost_bytes(&spec, max_batch(4)),
            3 * 2 * 1024 * 1024
        );
        assert_eq!(PaddingReservation::cost_bytes(&spec, max_batch(1)), 0);
    }
}
