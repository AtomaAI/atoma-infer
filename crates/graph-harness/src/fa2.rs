//! Paged decode attention and the KV-cache write over the FlashAttention-2 FFI, candle-free.
//!
//! The extern declarations are copied verbatim from `atoma-kernels/src/ffi.rs` (which keeps them
//! `pub(crate)`); the symbols resolve because the `cuda` feature links the vendored FA2 static
//! library through `atoma-kernels`' build script. The parameter recipe mirrors
//! `FlashAttentionKvCache::cuda_fwd_t` for the decode case: `seqlen_q = 1`, bf16, block table
//! present, per-sequence lengths in non-cumulative `cu_seqlens_k`, `force_split_kernel = true`.
//!
//! This is the one module a GPU-free machine cannot type-check against its link target; without
//! the `cuda` feature the same API compiles to loud runtime errors so the rest of the harness
//! still builds and unit-tests.

use anyhow::Result;

use crate::dims::ModelDims;

/// One paged decode attention call: every device address plus the shapes `run_mha` bakes.
///
/// Addresses are element-0 device pointers; strides are in elements, mirroring the wrapper.
#[derive(Debug, Clone, Copy)]
pub struct AttentionCall {
    /// Query: the q segment of the fused qkv buffer, `[batch, 1, num_q_heads, head_dim]` rows
    /// strided by the qkv row width.
    pub q: u64,
    /// Paged K cache, `[num_blocks, page_block, num_kv_heads, head_dim]`.
    pub k_cache: u64,
    /// Paged V cache, same shape as K.
    pub v_cache: u64,
    /// Output, `[batch, 1, num_q_heads, head_dim]` contiguous.
    pub out: u64,
    /// f32 `[batch, num_q_heads, 1]`.
    pub softmax_lse: u64,
    /// f32 `[num_splits, batch, num_q_heads, 1]`; unused when `num_splits == 1`.
    pub lse_accum: u64,
    /// f32 `[num_splits, batch, num_q_heads, 1, head_dim]`; unused when `num_splits == 1`.
    pub o_accum: u64,
    /// i32 `[batch]`, current key length per sequence (non-cumulative).
    pub seqlens_k: u64,
    /// i32 `[batch, max_blocks_per_seq]`.
    pub block_table: u64,
    pub batch_size: usize,
    pub max_blocks_per_seq: usize,
    pub page_block: usize,
    pub num_splits: u32,
    pub softmax_scale: f32,
    pub stream: cudarc::driver::sys::CUstream,
}

/// One KV-cache write: scatters the k and v segments of the fused qkv buffer into the paged
/// caches at `slot_mapping`.
#[derive(Debug, Clone, Copy)]
pub struct KvWriteCall {
    /// The fused qkv buffer; k sits at element offset `num_q_heads * head_dim`, v one kv width
    /// later, rows strided by the qkv row width.
    pub qkv: u64,
    pub k_cache: u64,
    pub v_cache: u64,
    /// i64 `[batch]`.
    pub slot_mapping: u64,
    pub batch_size: usize,
    pub page_block: usize,
    pub stream: cudarc::driver::sys::CUstream,
}

#[cfg(feature = "cuda")]
mod real {
    use core::ffi::{c_int, c_void};
    use std::f32::consts::LOG2_E;
    use std::ffi::CStr;
    use std::ptr;

    use anyhow::{bail, Result};
    // The extern block below resolves against the FA2 static library that atoma-kernels' build
    // script links. Nothing else in this crate names atoma-kernels, and an unreferenced dependency
    // is dropped from the link graph together with its native-library directives, so this import is
    // what makes the symbols resolve.
    use atoma_kernels as _;

    use super::{AttentionCall, KvWriteCall};
    use crate::dims::{ModelDims, BF16_BYTES};

    extern "C" {
        // Copied verbatim from atoma-kernels/src/ffi.rs; the C side is the vendored FA2 csrc.
        fn run_mha(
            q_ptr: *const c_void,
            k_ptr: *const c_void,
            v_ptr: *const c_void,
            o_ptr: *const c_void,
            softmax_lse_ptr: *const c_void,
            alibi_slopes_ptr: *const c_void,

            cu_seqlens_q_ptr: *const i32,
            cu_seqlens_k_ptr: *const i32,

            is_seqlens_k_cumulative: bool,

            q_batch_stride: u32,
            k_batch_stride: u32,
            v_batch_stride: u32,
            o_batch_stride: u32,
            alibi_slopes_batch_stride: u32,

            q_row_stride: u32,
            k_row_stride: u32,
            v_row_stride: u32,
            o_row_stride: u32,

            q_head_stride: u32,
            k_head_stride: u32,
            v_head_stride: u32,
            o_head_stride: u32,

            num_splits: u32,

            b: u32,
            h: u32,
            h_k: u32,
            d: u32,
            d_rounded: u32,
            softmax_scale: f32,
            scale_softmax_log2: f32,

            block_table: *const c_int,
            block_table_batch_stride: u32,
            page_block_size: c_int,

            seqused_k: *const c_int,
            seqlen_q: u32,
            seqlen_k: u32,
            total_q: u32,
            seqlen_q_rounded: u32,
            seqlen_k_rounded: u32,

            is_bf16: c_int,
            is_causal: c_int,

            window_size_left: c_int,
            window_size_right: c_int,
            softcap: f32,
            unpadded_lse: bool,
            force_split_kernel: bool,

            softmax_lseaccum_ptr: *const c_void,
            oaccum_ptr: *const c_void,

            stream: *mut c_void,
        );

        fn flash_last_error() -> c_int;

        fn reshape_and_cache_flash_cache(
            source: *const c_void,
            cache: *mut c_void,
            slot_mapping: *const i64,
            block_stride: i64,
            num_tokens: i64,
            num_heads: i64,
            head_size: i64,
            block_size: i64,
            source_stride: i64,
            dtype: u32,
            stream: *mut c_void,
        ) -> c_int;

        fn flash_cuda_error_string(code: c_int) -> *const core::ffi::c_char;
    }

    fn check_status(kernel: &'static str, status: c_int) -> Result<()> {
        if status == 0 {
            return Ok(());
        }
        // SAFETY: cudaGetErrorString returns a static string for every input.
        let message = unsafe { CStr::from_ptr(flash_cuda_error_string(status)) }
            .to_string_lossy()
            .into_owned();
        bail!("{kernel} launch failed with cudaError {status}: {message}");
    }

    /// Enqueues the paged decode attention kernel on `call.stream`.
    ///
    /// # Safety
    /// Every address in `call` must be live and match its documented shape for `dims`.
    pub unsafe fn paged_decode_attention(call: &AttentionCall, dims: &ModelDims) -> Result<()> {
        let seqlen_k = call.max_blocks_per_seq * call.page_block;
        let seqlen_k_rounded = seqlen_k.div_ceil(128) * 128;
        let head_dim = dims.head_dim as u32;
        let qkv_row = dims.qkv_out() as u32;
        let o_row = (dims.num_q_heads * dims.head_dim) as u32;
        let cache_row = (dims.num_kv_heads * dims.head_dim) as u32;
        let split_accums = call.num_splits > 1;

        unsafe {
            run_mha(
                call.q as *const c_void,
                call.k_cache as *const c_void,
                call.v_cache as *const c_void,
                call.out as *const c_void,
                call.softmax_lse as *const c_void,
                /* alibi_slopes_ptr */ ptr::null(),
                /* cu_seqlens_q_ptr */ ptr::null(),
                call.seqlens_k as *const i32,
                /* is_seqlens_k_cumulative */ false,
                /* q_batch_stride */ qkv_row,
                /* k_batch_stride */ (call.page_block as u32) * cache_row,
                /* v_batch_stride */ (call.page_block as u32) * cache_row,
                /* o_batch_stride */ o_row,
                /* alibi_slopes_batch_stride */ 0,
                /* q_row_stride */ qkv_row,
                /* k_row_stride */ cache_row,
                /* v_row_stride */ cache_row,
                /* o_row_stride */ o_row,
                /* q_head_stride */ head_dim,
                /* k_head_stride */ head_dim,
                /* v_head_stride */ head_dim,
                /* o_head_stride */ head_dim,
                call.num_splits,
                call.batch_size as u32,
                dims.num_q_heads as u32,
                dims.num_kv_heads as u32,
                /* d */ head_dim,
                /* d_rounded */ head_dim,
                call.softmax_scale,
                call.softmax_scale * LOG2_E,
                call.block_table as *const c_int,
                call.max_blocks_per_seq as u32,
                call.page_block as c_int,
                /* seqused_k */ ptr::null(),
                /* seqlen_q */ 1,
                seqlen_k as u32,
                /* total_q */ call.batch_size as u32,
                /* seqlen_q_rounded */ 128,
                seqlen_k_rounded as u32,
                /* is_bf16 */ 1,
                /* is_causal */ 0,
                /* window_size_left */ -1,
                /* window_size_right */ -1,
                /* softcap */ 0.0,
                /* unpadded_lse */ false,
                /* force_split_kernel */ true,
                if split_accums {
                    call.lse_accum as *const c_void
                } else {
                    ptr::null()
                },
                if split_accums {
                    call.o_accum as *const c_void
                } else {
                    ptr::null()
                },
                call.stream.cast::<c_void>(),
            );
        }
        check_status("run_mha", unsafe { flash_last_error() })
    }

    /// Enqueues the two paged-cache scatter writes (K then V) on `call.stream`.
    ///
    /// # Safety
    /// Every address in `call` must be live and match its documented shape for `dims`.
    pub unsafe fn write_kv(call: &KvWriteCall, dims: &ModelDims) -> Result<()> {
        let qkv_row = dims.qkv_out();
        let cache_row = dims.num_kv_heads * dims.head_dim;
        let k_source = call.qkv + (dims.num_q_heads * dims.head_dim * BF16_BYTES) as u64;
        let v_source = k_source + (cache_row * BF16_BYTES) as u64;

        for (name, source, cache) in [
            ("reshape_and_cache_flash_cache[k]", k_source, call.k_cache),
            ("reshape_and_cache_flash_cache[v]", v_source, call.v_cache),
        ] {
            let status = unsafe {
                reshape_and_cache_flash_cache(
                    source as *const c_void,
                    cache as *mut c_void,
                    call.slot_mapping as *const i64,
                    /* block_stride */ (call.page_block * cache_row) as i64,
                    call.batch_size as i64,
                    dims.num_kv_heads as i64,
                    dims.head_dim as i64,
                    call.page_block as i64,
                    /* source_stride */ qkv_row as i64,
                    /* dtype: bf16 */ 1,
                    call.stream.cast::<c_void>(),
                )
            };
            check_status(name, status)?;
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
pub use real::{paged_decode_attention, write_kv};

/// Loud stub: the harness binary built without the `cuda` feature cannot run attention.
///
/// # Safety
/// Never dereferences anything; the signature matches the real seam.
#[cfg(not(feature = "cuda"))]
pub unsafe fn paged_decode_attention(_call: &AttentionCall, _dims: &ModelDims) -> Result<()> {
    anyhow::bail!(
        "paged decode attention needs the FA2 kernels: rebuild with --features cuda on a machine \
         with nvcc"
    )
}

/// Loud stub: the harness binary built without the `cuda` feature cannot write the KV cache.
///
/// # Safety
/// Never dereferences anything; the signature matches the real seam.
#[cfg(not(feature = "cuda"))]
pub unsafe fn write_kv(_call: &KvWriteCall, _dims: &ModelDims) -> Result<()> {
    anyhow::bail!(
        "the paged KV write needs the FA2 kernels: rebuild with --features cuda on a machine \
         with nvcc"
    )
}
