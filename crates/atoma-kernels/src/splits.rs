//! The flash-attention split-KV heuristic: how many partitions of the key length one call runs.
//!
//! The kernel's own choice, made in the vendored C++ from the launch shape and the device's SM
//! count. The Rust side reproduces it so it can size the split accumulators the kernel writes
//! before any launch, and bake the count into a captured graph: a capture-clean call takes the
//! split count and its workspace from the caller, and the caller can only provide them by
//! computing the same answer the kernel would. Pure arithmetic, compiled and tested without a
//! CUDA toolkit.

/// The shape one attention call runs over, as the heuristic sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitShape {
    pub batch_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    /// The longest key length any sequence in the batch attends over.
    pub max_seqlen_k: usize,
    /// The longest query length in the batch; one for a decode step.
    pub max_seqlen_q: usize,
}

/// Rows of the query tile the split kernels cover per block.
const BLOCK_M: usize = 64;

/// The most split partitions one call may run: the largest count the vendored launch template
/// dispatches a combine kernel for. Past it the split kernel still runs, nothing combines its
/// partitions, and the output is never written.
pub const MAX_SPLITS: usize = 128;

/// Key positions one kernel block covers, by head dimension: wider heads take smaller blocks.
pub fn kv_block_n(head_dim: usize) -> usize {
    if head_dim <= 64 {
        256
    } else if head_dim <= 128 {
        128
    } else {
        64
    }
}

/// The split count for `shape` on a device of `sm_count` streaming multiprocessors: at most
/// [`MAX_SPLITS`], and one when the batch already fills the device.
pub fn num_splits(shape: SplitShape, sm_count: usize) -> usize {
    let num_n_blocks = shape.max_seqlen_k.div_ceil(kv_block_n(shape.head_dim));
    let num_m_blocks = shape.max_seqlen_q.div_ceil(BLOCK_M);
    num_splits_heuristic(
        shape.batch_size * shape.num_heads * num_m_blocks,
        sm_count * 2,
        num_n_blocks,
        MAX_SPLITS,
    )
}

/// The number of splits that maximizes occupancy: the best efficiency is found first, then the
/// smallest split count within 85% of it wins, because every extra split costs HBM traffic.
///
/// Structurally identical to the vendored C++ so a diff against it stays trivial.
#[allow(clippy::cast_precision_loss)]
pub fn num_splits_heuristic(
    batch_nheads_mblocks: usize,
    num_sms: usize,
    num_n_blocks: usize,
    max_splits: usize,
) -> usize {
    // If we have enough to almost fill the SMs, then just use 1 split.
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

    /// An H100's SM count.
    const H100_SMS: usize = 114;

    /// The vendored launch template, so the cap and the combine kernels it dispatches are held to
    /// the same number.
    const LAUNCH_TEMPLATE: &str = include_str!("../kernels/flash_fwd_launch_template.h");

    fn decode(batch_size: usize, head_dim: usize, max_seqlen_k: usize) -> SplitShape {
        SplitShape {
            batch_size,
            num_heads: 32,
            head_dim,
            max_seqlen_k,
            max_seqlen_q: 1,
        }
    }

    #[test]
    fn a_batch_that_fills_the_sms_uses_one_split() {
        // bs=64 * 32 heads = 2048 >= 0.8 * (114 SMs * 2), so the kernel runs unsplit.
        assert_eq!(num_splits(decode(64, 128, 2048), H100_SMS), 1);
    }

    #[test]
    fn a_single_sequence_splits_across_sms_on_an_h100() {
        // Hand-walked for bs=1, h=32, d=128, seqlen_k=2048 on 114 SMs (228 doubled):
        // num_n_blocks = 16, eligible split counts change ceil(16/n); efficiency peaks at
        // n=6 (192/228 = 0.842), and 6 is the first count within 85% of the peak.
        assert_eq!(num_splits(decode(1, 128, 2048), H100_SMS), 6);
    }

    #[test]
    fn head_dim_over_128_shrinks_the_kv_block() {
        assert_eq!(kv_block_n(64), 256);
        assert_eq!(kv_block_n(128), 128);
        assert_eq!(kv_block_n(256), 64);
        // d=256 uses block_n=64, doubling num_n_blocks vs d=128 at the same seqlen.
        let wide = num_splits(decode(1, 256, 2048), H100_SMS);
        let narrow = num_splits(decode(1, 128, 2048), H100_SMS);
        assert!(wide >= narrow);
    }

    #[test]
    fn a_short_key_length_cannot_split_past_its_blocks() {
        // 64 key positions is one 128-wide block, so there is nothing to split.
        assert_eq!(num_splits(decode(1, 128, 64), H100_SMS), 1);
    }

    #[test]
    fn the_split_count_never_exceeds_the_kernel_cap() {
        let shape = SplitShape {
            batch_size: 1,
            num_heads: 1,
            head_dim: 128,
            max_seqlen_k: 1 << 20,
            max_seqlen_q: 1,
        };
        assert!(num_splits(shape, 1024) <= MAX_SPLITS);
    }

    #[test]
    fn a_long_query_adds_row_blocks_and_fills_the_device_sooner() {
        // 64 query rows are one row block; 65 are two, doubling the work units.
        let one_block = SplitShape {
            max_seqlen_q: 64,
            ..decode(4, 128, 2048)
        };
        let two_blocks = SplitShape {
            max_seqlen_q: 65,
            ..decode(4, 128, 2048)
        };
        assert!(num_splits(two_blocks, H100_SMS) <= num_splits(one_block, H100_SMS));
    }

    #[test]
    fn the_cap_is_the_largest_count_the_template_dispatches_a_combine_kernel_for() {
        let dispatched = LAUNCH_TEMPLATE
            .split("params.num_splits <= ")
            .skip(1)
            .map(|rest| {
                let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
                digits
                    .parse::<usize>()
                    .expect("a split count follows every comparison in the dispatch chain")
            })
            .max()
            .expect("the template dispatches the combine kernel by split count");
        assert_eq!(dispatched, MAX_SPLITS);
    }
}
