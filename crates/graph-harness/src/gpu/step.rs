//! The decode step: one function enqueues every op of the Llama-shaped step onto a raw stream,
//! so the eager run and the captured recording are the same code path by construction.

use anyhow::{anyhow, Result};
use atoma_runtime::arena::{BucketIdx, CaptureArena, LayerIdx};
use cudarc::driver::{result, sys};

use crate::dims::ModelDims;
use crate::fa2::{self, AttentionCall, KvWriteCall};
use crate::gpu::blas::StepBlas;
use crate::gpu::kernels::StepKernels;
use crate::layout::{Role, StaticSizes};

/// Device addresses of one layer's weights and its paged KV caches.
#[derive(Debug, Clone, Copy)]
pub struct LayerPtrs {
    pub w_qkv: u64,
    pub w_o: u64,
    pub w_gate: u64,
    pub w_up: u64,
    pub w_down: u64,
    pub rms1: u64,
    pub rms2: u64,
    pub k_cache: u64,
    pub v_cache: u64,
}

/// Device addresses of everything the step reads and writes. Raw `u64` addresses are snapshotted
/// once after allocation: with cudarc's event tracking disabled, pointer extraction has no
/// ordering side effects, and device allocations never move.
#[derive(Debug, Clone)]
pub struct StepPtrs {
    pub embedding: u64,
    pub final_norm: u64,
    pub lm_head: u64,
    pub layers: Vec<LayerPtrs>,
    /// Base address of the capture arena buffer; slot addresses add `CaptureArena::offset`.
    pub arena_base: u64,
    pub token_ids: u64,
    pub seqlens_k: u64,
    pub block_table: u64,
    pub slot_mapping: u64,
    pub logits: u64,
    pub argmax: u64,
    pub softmax_lse: u64,
    pub lse_accum: u64,
    pub o_accum: u64,
}

/// Device addresses of the persistent staging mirrors of the four input buffers.
#[derive(Debug, Clone, Copy)]
pub struct StagingPtrs {
    pub token_ids: u64,
    pub seqlens_k: u64,
    pub block_table: u64,
    pub slot_mapping: u64,
}

/// Everything a step launch needs besides addresses.
pub struct StepContext<'a> {
    pub kernels: &'a StepKernels,
    pub blas: &'a StepBlas,
    pub dims: &'a ModelDims,
    pub arena: &'a CaptureArena,
    pub bucket: BucketIdx,
    pub batch_size: usize,
    pub page_block: usize,
    pub max_blocks_per_seq: usize,
    pub num_splits: u32,
    pub rope_theta: f32,
    pub rms_eps: f32,
    pub stream: sys::CUstream,
}

impl StepContext<'_> {
    fn slot(&self, ptrs: &StepPtrs, layer: usize, role: Role) -> u64 {
        ptrs.arena_base
            + self
                .arena
                .offset(self.bucket, LayerIdx(layer), role.tensor_role()) as u64
    }
}

/// The per-layer hook the all-reduce cells install: called with the o-projection slot and its
/// element count after the projection and before the residual add. Keeping the collective behind
/// this seam keeps `run_step` NCCL-free.
pub type LayerHook<'a> = dyn FnMut(u64, usize) -> Result<()> + 'a;

/// Enqueues the D2D copies from staging into the graph's static input buffers — the captured
/// copy-in that replays consume; per-step host updates only ever touch staging.
///
/// # Safety
/// All addresses must be live and sized per `sizes`.
pub unsafe fn copy_inputs(
    stream: sys::CUstream,
    ptrs: &StepPtrs,
    staging: &StagingPtrs,
    sizes: &StaticSizes,
) -> Result<()> {
    for (dst, src, bytes) in [
        (ptrs.token_ids, staging.token_ids, sizes.token_ids),
        (ptrs.seqlens_k, staging.seqlens_k, sizes.seqlens_k),
        (ptrs.block_table, staging.block_table, sizes.block_table),
        (ptrs.slot_mapping, staging.slot_mapping, sizes.slot_mapping),
    ] {
        unsafe { result::memcpy_dtod_async(dst, src, bytes, stream) }
            .map_err(|e| anyhow!("copy-in D2D failed: {:?}", e.0))?;
    }
    Ok(())
}

/// Enqueues one full decode step: embedding gather, every layer's attention and MLP blocks, the
/// final norm, lm head, and device argmax.
///
/// The per-layer enqueue order here is the op order that [`crate::layout::OPS_PER_LAYER`] counts
/// and [`Role::lifetime`] indexes. Anyone reordering ops here must update those declarations.
///
/// # Safety
/// Every address in `ptrs` must be live and match the shapes implied by `ctx`.
pub unsafe fn run_step(
    ctx: &StepContext<'_>,
    ptrs: &StepPtrs,
    mut hook: Option<&mut LayerHook<'_>>,
) -> Result<()> {
    let dims = ctx.dims;
    let bs = ctx.batch_size;

    unsafe {
        ctx.kernels.embedding_gather(
            ctx.stream,
            ptrs.embedding,
            ptrs.token_ids,
            ctx.slot(ptrs, 0, Role::Hidden),
            dims.hidden,
            bs,
        )?;
        for layer in 0..dims.num_layers {
            attention_block(ctx, ptrs, layer, hook.as_deref_mut())?;
            mlp_block(ctx, ptrs, layer)?;
        }

        // Final norm reads the last residual (the extra arena row) and reuses its Normed slot.
        let final_hidden = ctx.slot(ptrs, dims.num_layers, Role::Hidden);
        let final_normed = ctx.slot(ptrs, dims.num_layers, Role::Normed);
        ctx.kernels.rmsnorm_bf16(
            ctx.stream,
            final_hidden,
            ptrs.final_norm,
            final_normed,
            dims.hidden,
            bs,
            ctx.rms_eps,
        )?;
        ctx.blas.gemm_xwt(
            ptrs.lm_head,
            final_normed,
            ptrs.logits,
            dims.vocab,
            bs,
            dims.hidden,
        )?;
        ctx.kernels
            .argmax_bf16(ctx.stream, ptrs.logits, ptrs.argmax, dims.vocab, bs)?;
    }
    Ok(())
}

/// # Safety
/// See [`run_step`].
unsafe fn attention_block(
    ctx: &StepContext<'_>,
    ptrs: &StepPtrs,
    layer: usize,
    hook: Option<&mut LayerHook<'_>>,
) -> Result<()> {
    let dims = ctx.dims;
    let bs = ctx.batch_size;
    let weights = &ptrs.layers[layer];
    let hidden = ctx.slot(ptrs, layer, Role::Hidden);
    let normed = ctx.slot(ptrs, layer, Role::Normed);
    let qkv = ctx.slot(ptrs, layer, Role::Qkv);
    let attn_out = ctx.slot(ptrs, layer, Role::AttnOut);
    let o_proj = ctx.slot(ptrs, layer, Role::OProj);
    let mid = ctx.slot(ptrs, layer, Role::Mid);

    unsafe {
        ctx.kernels.rmsnorm_bf16(
            ctx.stream,
            hidden,
            weights.rms1,
            normed,
            dims.hidden,
            bs,
            ctx.rms_eps,
        )?;
        ctx.blas
            .gemm_xwt(weights.w_qkv, normed, qkv, dims.qkv_out(), bs, dims.hidden)?;
        ctx.kernels.rope_qk(
            ctx.stream,
            qkv,
            ptrs.seqlens_k,
            bs,
            dims.num_q_heads,
            dims.num_kv_heads,
            dims.head_dim,
            ctx.rope_theta,
        )?;
        fa2::write_kv(
            &KvWriteCall {
                qkv,
                k_cache: weights.k_cache,
                v_cache: weights.v_cache,
                slot_mapping: ptrs.slot_mapping,
                batch_size: bs,
                page_block: ctx.page_block,
                stream: ctx.stream,
            },
            dims,
        )?;
        fa2::paged_decode_attention(
            &AttentionCall {
                q: qkv,
                k_cache: weights.k_cache,
                v_cache: weights.v_cache,
                out: attn_out,
                softmax_lse: ptrs.softmax_lse,
                lse_accum: ptrs.lse_accum,
                o_accum: ptrs.o_accum,
                seqlens_k: ptrs.seqlens_k,
                block_table: ptrs.block_table,
                batch_size: bs,
                max_blocks_per_seq: ctx.max_blocks_per_seq,
                page_block: ctx.page_block,
                num_splits: ctx.num_splits,
                softmax_scale: dims.softmax_scale(),
                stream: ctx.stream,
            },
            dims,
        )?;
        ctx.blas
            .gemm_xwt(weights.w_o, attn_out, o_proj, dims.hidden, bs, dims.hidden)?;
        if let Some(hook) = hook {
            hook(o_proj, bs * dims.hidden)?;
        }
        ctx.kernels
            .add_bf16(ctx.stream, hidden, o_proj, mid, bs * dims.hidden)?;
    }
    Ok(())
}

/// # Safety
/// See [`run_step`].
unsafe fn mlp_block(ctx: &StepContext<'_>, ptrs: &StepPtrs, layer: usize) -> Result<()> {
    let dims = ctx.dims;
    let bs = ctx.batch_size;
    let weights = &ptrs.layers[layer];
    let mid = ctx.slot(ptrs, layer, Role::Mid);
    let normed = ctx.slot(ptrs, layer, Role::Normed);
    let gate = ctx.slot(ptrs, layer, Role::Gate);
    let up = ctx.slot(ptrs, layer, Role::Up);
    let ffn_act = ctx.slot(ptrs, layer, Role::FfnAct);
    let ffn_down = ctx.slot(ptrs, layer, Role::FfnDown);
    let next_hidden = ctx.slot(ptrs, layer + 1, Role::Hidden);

    unsafe {
        ctx.kernels.rmsnorm_bf16(
            ctx.stream,
            mid,
            weights.rms2,
            normed,
            dims.hidden,
            bs,
            ctx.rms_eps,
        )?;
        ctx.blas
            .gemm_xwt(weights.w_gate, normed, gate, dims.ffn, bs, dims.hidden)?;
        ctx.blas
            .gemm_xwt(weights.w_up, normed, up, dims.ffn, bs, dims.hidden)?;
        ctx.kernels
            .silu_mul(ctx.stream, gate, up, ffn_act, bs * dims.ffn)?;
        ctx.blas
            .gemm_xwt(weights.w_down, ffn_act, ffn_down, dims.hidden, bs, dims.ffn)?;
        ctx.kernels
            .add_bf16(ctx.stream, mid, ffn_down, next_hidden, bs * dims.hidden)?;
    }
    Ok(())
}
