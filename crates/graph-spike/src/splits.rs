//! The FA2 split-KV heuristic, mirrored from `atoma-kernels`' private `compute_num_splits` so
//! the spike can size split accumulators and bake `num_splits` before capture.
//!
//! The spike calls `run_mha` directly with an explicit `num_splits`, so this copy — not the
//! kernels-crate original — decides the accumulator allocation; the kernel then trusts the
//! accumulators to be large enough. Decode always has `seqlen_q = 1`, so `num_m_blocks` is 1.

/// Splits that maximize SM occupancy for a decode call, matching the FA2 heuristic: the best
/// efficiency is found first, then the smallest split count within 85% of it wins.
pub fn num_splits(
    batch_size: usize,
    num_heads: usize,
    head_dim: usize,
    max_seqlen_k: usize,
    sm_count: usize,
) -> u32 {
    let block_n = if head_dim <= 64 {
        256
    } else if head_dim <= 128 {
        128
    } else {
        64
    };
    let num_n_blocks = max_seqlen_k.div_ceil(block_n);
    let splits = num_splits_heuristic(batch_size * num_heads, sm_count * 2, num_n_blocks, 128);
    u32::try_from(splits).expect("split count is bounded by 128")
}

/// Verbatim logic of `atoma-kernels`' `num_splits_heuristic` (flash_attention.rs), kept
/// structurally identical so a diff against the original stays trivial.
#[allow(clippy::cast_precision_loss)]
fn num_splits_heuristic(
    batch_nheads_mblocks: usize,
    num_sms: usize,
    num_n_blocks: usize,
    max_splits: usize,
) -> usize {
    if (batch_nheads_mblocks as f32) >= 0.8 * (num_sms as f32) {
        return 1;
    }

    let max_splits = max_splits.min(num_sms).min(num_n_blocks);
    let mut max_efficiency = 0.0;
    let mut efficiency = Vec::with_capacity(max_splits);

    let is_split_eligible = |num_splits: usize| -> bool {
        num_splits == 1
            || num_n_blocks.div_ceil(num_splits) != num_n_blocks.div_ceil(num_splits - 1)
    };

    for num_splits in 1..=max_splits {
        if is_split_eligible(num_splits) {
            let n_waves = (batch_nheads_mblocks * num_splits) as f32 / num_sms as f32;
            let eff = n_waves / n_waves.ceil();
            if eff > max_efficiency {
                max_efficiency = eff;
            }
            efficiency.push(eff);
        } else {
            efficiency.push(0.0);
        }
    }

    for num_splits in 1..=max_splits {
        if !is_split_eligible(num_splits) {
            continue;
        }
        if efficiency[num_splits - 1] >= 0.85 * max_efficiency {
            return num_splits;
        }
    }

    1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_batch_that_fills_the_sms_uses_one_split() {
        // bs=64 * 32 heads = 2048 >= 0.8 * (114 SMs * 2), so the kernel runs unsplit.
        assert_eq!(num_splits(64, 32, 128, 2048, 114), 1);
    }

    #[test]
    fn a_single_sequence_splits_across_sms_on_an_h100() {
        // Hand-walked for bs=1, h=32, d=128, seqlen_k=2048 on 114 SMs (228 doubled):
        // num_n_blocks = 16, eligible split counts change ceil(16/n); efficiency peaks at
        // n=6 (192/228 = 0.842), and 6 is the first count within 85% of the peak.
        assert_eq!(num_splits(1, 32, 128, 2048, 114), 6);
    }

    #[test]
    fn head_dim_over_128_shrinks_the_kv_block() {
        // d=256 uses block_n=64, doubling num_n_blocks vs d=128 at the same seqlen.
        let wide = num_splits(1, 32, 256, 2048, 114);
        let narrow = num_splits(1, 32, 128, 2048, 114);
        assert!(wide >= narrow);
    }
}
