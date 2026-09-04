//! The split count the harness bakes per bucket: the FA2 split-KV heuristic `atoma-kernels`
//! shares, over the harness's decode shape, so the split accumulators are sized and the count
//! fixed before capture.
//!
//! The harness calls `run_mha` directly with an explicit `num_splits`, so this count decides the
//! accumulator allocation and the kernel trusts the accumulators to be large enough. Decode always
//! has `seqlen_q = 1`.

use atoma_kernels::splits::{self, SplitShape};

/// Splits that maximize SM occupancy for a decode call of `batch_size` sequences, as the `u32`
/// the launch and the static sizes take.
pub fn num_splits(
    batch_size: usize,
    num_heads: usize,
    head_dim: usize,
    max_seqlen_k: usize,
    sm_count: usize,
) -> u32 {
    let shape = SplitShape {
        batch_size,
        num_heads,
        head_dim,
        max_seqlen_k,
        max_seqlen_q: 1,
    };
    u32::try_from(splits::num_splits(shape, sm_count))
        .expect("the shared heuristic caps its count at MAX_SPLITS, which fits u32")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_harness_asks_the_shared_heuristic_for_one_query_row() {
        for (batch_size, max_seqlen_k) in [(1, 2048), (8, 8192), (64, 2048)] {
            let shape = SplitShape {
                batch_size,
                num_heads: 32,
                head_dim: 128,
                max_seqlen_k,
                max_seqlen_q: 1,
            };
            assert_eq!(
                num_splits(batch_size, 32, 128, max_seqlen_k, 114) as usize,
                splits::num_splits(shape, 114)
            );
        }
    }
}
