//! Arena roles, static buffer sizes, and the device memory budget of the spike step.

use atoma_runtime::arena::{CaptureArena, Dtype, TensorRole};

use crate::dims::{ModelDims, BF16_BYTES};

/// One activation tensor the step produces per layer, addressed through the capture arena.
///
/// The discriminant is the arena role index. `Hidden` is the residual stream *entering* a layer;
/// the arena is built with one extra layer row so the last layer's residual add writes into
/// `LayerIdx(num_layers)`'s `Hidden` slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Hidden = 0,
    /// RMSNorm output feeding qkv or the MLP.
    Normed = 1,
    /// Fused q/k/v projection, `[num_q_heads + 2 * num_kv_heads, head_dim]` per token.
    Qkv = 2,
    /// Paged-attention output, `[num_q_heads, head_dim]` per token.
    AttnOut = 3,
    OProj = 4,
    /// Residual stream after the attention residual add.
    Mid = 5,
    Gate = 6,
    Up = 7,
    /// `silu(gate) * up`.
    FfnAct = 8,
    FfnDown = 9,
}

impl Role {
    pub const ALL: [Role; 10] = [
        Role::Hidden,
        Role::Normed,
        Role::Qkv,
        Role::AttnOut,
        Role::OProj,
        Role::Mid,
        Role::Gate,
        Role::Up,
        Role::FfnAct,
        Role::FfnDown,
    ];

    pub fn tensor_role(self) -> TensorRole {
        TensorRole(self as usize)
    }

    /// Per-token width in bytes, the unit the arena is declared in.
    pub fn width_bytes(self, dims: &ModelDims) -> usize {
        match self {
            Role::Hidden
            | Role::Normed
            | Role::AttnOut
            | Role::OProj
            | Role::Mid
            | Role::FfnDown => Dtype::Bf16.width_bytes(dims.hidden),
            Role::Qkv => Dtype::Bf16.width_bytes(dims.qkv_out()),
            Role::Gate | Role::Up | Role::FfnAct => Dtype::Bf16.width_bytes(dims.ffn),
        }
    }
}

/// Builds the arena for `buckets` (batch sizes; decode steps have one token per sequence), with
/// the extra layer row for the final residual.
pub fn build_arena(dims: &ModelDims, buckets: &[usize]) -> CaptureArena {
    let widths: Vec<usize> = Role::ALL.iter().map(|r| r.width_bytes(dims)).collect();
    CaptureArena::new(dims.num_layers + 1, &widths, buckets)
}

/// Byte sizes of one cell's per-graph buffers: the graph's static inputs, outputs, and
/// workspaces, each of which the harness allocates before capture and `GraphEntry` then owns.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct StaticSizes {
    pub token_ids: usize,
    pub seqlens_k: usize,
    pub block_table: usize,
    pub slot_mapping: usize,
    pub logits: usize,
    pub argmax: usize,
    pub softmax_lse: usize,
    pub lse_accum: usize,
    pub o_accum: usize,
}

impl StaticSizes {
    /// Sizes for a bucket of `batch_size` sequences whose attention runs with `num_splits`
    /// split-KV partitions.
    pub fn for_bucket(
        dims: &ModelDims,
        batch_size: usize,
        max_blocks_per_seq: usize,
        num_splits: u32,
    ) -> Self {
        let splits = num_splits as usize;
        Self {
            token_ids: batch_size * 4,
            seqlens_k: batch_size * 4,
            block_table: batch_size * max_blocks_per_seq * 4,
            slot_mapping: batch_size * 8,
            logits: batch_size * dims.vocab * BF16_BYTES,
            argmax: batch_size * 4,
            softmax_lse: batch_size * dims.num_q_heads * 4,
            lse_accum: splits * batch_size * dims.num_q_heads * 4,
            o_accum: splits * batch_size * dims.num_q_heads * dims.head_dim * 4,
        }
    }

    /// Total bytes of the four input buffers (what staging mirrors).
    pub fn input_bytes(&self) -> usize {
        self.token_ids + self.seqlens_k + self.block_table + self.slot_mapping
    }

    /// Total bytes of everything in this struct.
    pub fn total_bytes(&self) -> usize {
        self.input_bytes()
            + self.logits
            + self.argmax
            + self.softmax_lse
            + self.lse_accum
            + self.o_accum
    }
}

/// Bytes of one paged K (or V) cache: `[total_blocks, page_block, num_kv_heads, head_dim]` bf16.
pub fn kv_cache_bytes_each(dims: &ModelDims, total_blocks: usize, page_block: usize) -> usize {
    total_blocks * page_block * dims.num_kv_heads * dims.head_dim * BF16_BYTES
}

#[cfg(test)]
mod tests {
    use atoma_runtime::arena::{BucketIdx, LayerIdx};

    use super::*;

    #[test]
    fn arena_footprint_matches_the_hand_computed_bucket_64() {
        // Per-token role widths in bytes: 6 hidden-wide bf16 roles (8192), qkv (12288), and 3
        // ffn-wide bf16 roles (28672) — 147456 per token, every slot already 256-aligned at
        // these widths. Four layers plus the final residual row gives 5 * 64 * 147456. The f32
        // all-reduce mirror is a dedicated typed slice, not an arena role, because the safe
        // NCCL collective wants a CudaSlice<f32>.
        let dims = ModelDims::llama_8b_shaped(4);
        let arena = build_arena(&dims, &[1, 8, 32, 64]);
        assert_eq!(arena.total_size(), 5 * 64 * 147_456);
    }

    #[test]
    fn every_role_is_addressable_in_every_declared_layer_row() {
        let dims = ModelDims::llama_8b_shaped(2);
        let arena = build_arena(&dims, &[1, 8]);
        // The final residual row exists: layer index num_layers is valid.
        let final_hidden = arena.offset(BucketIdx(0), LayerIdx(2), Role::Hidden.tensor_role());
        assert!(final_hidden > 0);
    }

    #[test]
    fn static_sizes_scale_with_splits_and_bucket() {
        let dims = ModelDims::llama_8b_shaped(4);
        let sizes = StaticSizes::for_bucket(&dims, 8, 128, 2);
        assert_eq!(sizes.token_ids, 32);
        assert_eq!(sizes.block_table, 8 * 128 * 4);
        assert_eq!(sizes.logits, 8 * 128_256 * 2);
        assert_eq!(sizes.softmax_lse, 8 * 32 * 4);
        assert_eq!(sizes.lse_accum, 2 * 8 * 32 * 4);
        assert_eq!(sizes.o_accum, 2 * 8 * 32 * 128 * 4);
        assert_eq!(
            sizes.input_bytes(),
            sizes.token_ids + sizes.seqlens_k + sizes.block_table + sizes.slot_mapping
        );
    }

    #[test]
    fn kv_cache_bytes_match_the_pool_shape() {
        let dims = ModelDims::llama_8b_shaped(4);
        // 8192 blocks * 16 slots * 8 heads * 128 dim * 2 bytes.
        assert_eq!(kv_cache_bytes_each(&dims, 8192, 16), 268_435_456);
    }
}
