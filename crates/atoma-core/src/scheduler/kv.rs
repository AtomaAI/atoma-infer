//! KV bookkeeping for one sequence: growing its block table from the pool and giving it back.

use crate::kv::BlockPool;
use crate::request::Sequence;
use crate::types::TokenCount;

/// The pool had no block to lease. The caller decides whether to evict, preempt or wait.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PoolExhausted;

/// Grows `sequence`'s block table until it covers `tokens` tokens, leasing from `pool`.
///
/// Blocks obtained before the pool runs dry stay in the table: the sequence will need them
/// whenever it next runs, and a preemption releases them with the rest.
pub(crate) fn ensure_blocks(
    pool: &mut BlockPool,
    block_size: TokenCount,
    sequence: &mut Sequence,
    tokens: usize,
) -> Result<(), PoolExhausted> {
    let needed = tokens.div_ceil(block_size.get());
    while sequence.block_table.len() < needed {
        let Some(lease) = pool.lease() else {
            return Err(PoolExhausted);
        };
        sequence.block_table.push(lease.block());
        sequence.leases.push(lease);
    }
    Ok(())
}

/// Surrenders every lease the sequence holds and empties its block table.
pub(crate) fn release_blocks(pool: &mut BlockPool, sequence: &mut Sequence) {
    for lease in sequence.leases.drain(..) {
        pool.release(lease);
    }
    sequence.block_table.clear();
}
