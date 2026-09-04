//! Paged decode attention and the paged KV write over the flash-attention FFI, without candle.
//!
//! The candle wrapper derives every stride and length from tensors it is handed; a step that owns
//! its device addresses passes them directly. This module is where such a step reaches the FFI:
//! it takes element addresses and element strides, fixes the decode recipe — one query row per
//! sequence, per-sequence key lengths, a block table, the split kernel forced so the split count
//! and its accumulators are the caller's — and checks the launch status. The recipe is the one
//! the candle wrapper follows for the same case, so the two paths run the same kernel with the
//! same arguments.
//!
//! Without the `cuda` feature the same API compiles to
//! [`KernelError::NotCompiled`](crate::error::KernelError::NotCompiled), so a crate built on it
//! type-checks and unit-tests on a machine with no toolkit.

use core::ffi::c_void;

use crate::error::KernelError;
use crate::splits::MAX_SPLITS;

/// The element type the attention operands and the cache hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    F16,
    Bf16,
}

impl Precision {
    /// The `dtype` code the paged-cache write takes.
    pub fn cache_code(self) -> u32 {
        match self {
            Precision::F16 => 0,
            Precision::Bf16 => 1,
        }
    }

    /// The `is_bf16` flag the attention launch takes.
    pub fn is_bf16(self) -> i32 {
        match self {
            Precision::F16 => 0,
            Precision::Bf16 => 1,
        }
    }
}

/// Element strides of a `[batch, row, head, dim]` operand whose last dimension is contiguous.
///
/// For a paged cache, `[blocks, slot, head, dim]`, the batch stride is the block stride and the
/// row stride the slot stride.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OperandStrides {
    pub batch: usize,
    pub row: usize,
    pub head: usize,
}

/// The query-row count the decode kernel pads its single row to.
pub const SEQLEN_Q_ROUNDED: usize = 128;
/// The head dimension is rounded up to this for the split accumulators.
const HEAD_DIM_ALIGN: usize = 32;
/// The key length is rounded up to this.
const SEQLEN_K_ALIGN: usize = 128;

/// The shape of one paged decode attention call: one query row per sequence over a block table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeShape {
    pub batch_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// Slots per cache block.
    pub page_block: usize,
    /// Columns of the block table: the blocks the longest sequence may occupy.
    pub max_blocks_per_seq: usize,
}

impl DecodeShape {
    /// The key length the kernel is told: every block-table column, filled or not. Each
    /// sequence's real length arrives separately in `seqlens_k`.
    pub fn seqlen_k(&self) -> usize {
        self.max_blocks_per_seq * self.page_block
    }

    /// [`DecodeShape::seqlen_k`] rounded up to the kernel's key tile.
    pub fn seqlen_k_rounded(&self) -> usize {
        self.seqlen_k().next_multiple_of(SEQLEN_K_ALIGN)
    }

    /// The head dimension rounded up to the split accumulators' width.
    pub fn head_dim_rounded(&self) -> usize {
        self.head_dim.next_multiple_of(HEAD_DIM_ALIGN)
    }

    /// f32 values the log-sum-exp output holds: one per sequence and head.
    pub fn softmax_lse_len(&self) -> usize {
        self.batch_size * self.num_heads
    }

    /// f32 values the split log-sum-exp accumulator holds for `num_splits` partitions.
    pub fn lse_accum_len(&self, num_splits: u32) -> usize {
        num_splits as usize * self.softmax_lse_len()
    }

    /// f32 values the split output accumulator holds for `num_splits` partitions.
    pub fn o_accum_len(&self, num_splits: u32) -> usize {
        self.lse_accum_len(num_splits) * self.head_dim_rounded()
    }
}

/// One paged decode attention call: every device address plus the shape the launch bakes.
///
/// Addresses are element-0 device pointers of the documented operand; the accumulators are read
/// only when `num_splits` is above one.
#[derive(Debug, Clone, Copy)]
pub struct DecodeAttentionCall {
    /// Query, `[batch, 1, num_heads, head_dim]`.
    pub q: u64,
    pub q_strides: OperandStrides,
    /// Paged K cache, `[blocks, page_block, num_kv_heads, head_dim]`.
    pub k_cache: u64,
    /// Paged V cache, the same shape and strides as K.
    pub v_cache: u64,
    pub cache_strides: OperandStrides,
    /// Output, `[batch, 1, num_heads, head_dim]`.
    pub out: u64,
    pub out_strides: OperandStrides,
    /// f32 `[batch, num_heads, 1]`.
    pub softmax_lse: u64,
    /// f32 `[num_splits, batch, num_heads, 1]`.
    pub lse_accum: u64,
    /// f32 `[num_splits, batch, num_heads, 1, head_dim_rounded]`.
    pub o_accum: u64,
    /// i32 `[batch]`: each sequence's key length after this step's token, non-cumulative.
    pub seqlens_k: u64,
    /// i32 `[batch, max_blocks_per_seq]`, row-major.
    pub block_table: u64,
    pub shape: DecodeShape,
    pub num_splits: u32,
    pub softmax_scale: f32,
    pub precision: Precision,
    pub stream: *mut c_void,
}

/// One paged KV write: scatters each token's key and value rows into the caches at the token's
/// slot.
#[derive(Debug, Clone, Copy)]
pub struct KvWriteCall {
    /// The key rows, `[num_tokens, num_kv_heads * head_dim]`, rows `source_stride` elements
    /// apart.
    pub k_source: u64,
    /// The value rows, laid out like the keys.
    pub v_source: u64,
    /// Elements between consecutive token rows of either source.
    pub source_stride: usize,
    /// Paged K cache, `[blocks, page_block, num_kv_heads, head_dim]`, contiguous.
    pub k_cache: u64,
    /// Paged V cache, the same shape as K.
    pub v_cache: u64,
    /// i64 `[num_tokens]`: each token's slot, block index times `page_block` plus the offset.
    pub slot_mapping: u64,
    pub num_tokens: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub page_block: usize,
    pub precision: Precision,
    pub stream: *mut c_void,
}

impl KvWriteCall {
    /// Elements one cache block holds.
    pub fn block_stride(&self) -> usize {
        self.page_block * self.num_kv_heads * self.head_dim
    }
}

/// The split count as the launch takes it, held to [`MAX_SPLITS`]. The vendored template
/// dispatches no combine kernel past that, so a larger count would run the split kernel and leave
/// the output unwritten; it is refused here, before any address reaches the device.
fn split_count(call: &DecodeAttentionCall) -> Result<u32, KernelError> {
    let num_splits = call.num_splits;
    if num_splits as usize > MAX_SPLITS {
        return Err(KernelError::SplitCount { num_splits });
    }
    Ok(num_splits)
}

#[cfg(feature = "cuda")]
mod compiled {
    use core::ffi::{c_int, c_void};
    use std::f32::consts::LOG2_E;
    use std::ptr;

    use super::{split_count, DecodeAttentionCall, KvWriteCall, SEQLEN_Q_ROUNDED};
    use crate::attention_window;
    use crate::error::{arg32, arg_i64, arg_int, KernelError};
    use crate::ffi;

    /// Enqueues the paged decode attention kernel on `call.stream`.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError`] when the split count is past what the combine kernel is
    /// dispatched for, an argument does not fit the kernel's parameter, or the launch fails.
    ///
    /// # Safety
    /// Every address in `call` must be live on the stream's device and match its documented
    /// shape; the accumulators must hold what [`super::DecodeShape`] sizes for `num_splits`.
    pub unsafe fn decode_attention(call: &DecodeAttentionCall) -> Result<(), KernelError> {
        let shape = call.shape;
        let seqlen_k = shape.seqlen_k();
        let num_splits = split_count(call)?;
        let split = num_splits > 1;
        let accumulator = |address: u64| -> *const c_void {
            if split {
                address as *const c_void
            } else {
                ptr::null()
            }
        };
        let window = attention_window::decode(arg_int("seqlen_k", seqlen_k)?);
        // SAFETY: the caller's contract; the FFI records its status for the check below.
        unsafe {
            ffi::run_mha(
                call.q as *const c_void,
                call.k_cache as *const c_void,
                call.v_cache as *const c_void,
                call.out as *const c_void,
                call.softmax_lse as *const c_void,
                /* alibi_slopes_ptr */ ptr::null(),
                /* cu_seqlens_q_ptr */ ptr::null(),
                call.seqlens_k as *const i32,
                /* is_seqlens_k_cumulative */ false,
                arg32("q_batch_stride", call.q_strides.batch)?,
                arg32("k_batch_stride", call.cache_strides.batch)?,
                arg32("v_batch_stride", call.cache_strides.batch)?,
                arg32("o_batch_stride", call.out_strides.batch)?,
                /* alibi_slopes_batch_stride */ 0,
                arg32("q_row_stride", call.q_strides.row)?,
                arg32("k_row_stride", call.cache_strides.row)?,
                arg32("v_row_stride", call.cache_strides.row)?,
                arg32("o_row_stride", call.out_strides.row)?,
                arg32("q_head_stride", call.q_strides.head)?,
                arg32("k_head_stride", call.cache_strides.head)?,
                arg32("v_head_stride", call.cache_strides.head)?,
                arg32("o_head_stride", call.out_strides.head)?,
                num_splits,
                arg32("b", shape.batch_size)?,
                arg32("h", shape.num_heads)?,
                arg32("h_k", shape.num_kv_heads)?,
                arg32("d", shape.head_dim)?,
                arg32("d_rounded", shape.head_dim_rounded())?,
                call.softmax_scale,
                call.softmax_scale * LOG2_E,
                call.block_table as *const c_int,
                arg32("block_table_batch_stride", shape.max_blocks_per_seq)?,
                arg_int("page_block_size", shape.page_block)?,
                /* seqused_k */ ptr::null(),
                /* seqlen_q */ 1,
                arg32("seqlen_k", seqlen_k)?,
                arg32("total_q", shape.batch_size)?,
                arg32("seqlen_q_rounded", SEQLEN_Q_ROUNDED)?,
                arg32("seqlen_k_rounded", shape.seqlen_k_rounded())?,
                call.precision.is_bf16(),
                i32::from(window.is_causal),
                window.window_size_left,
                window.window_size_right,
                /* softcap */ 0.0,
                /* unpadded_lse */ false,
                /* force_split_kernel */ true,
                accumulator(call.lse_accum),
                accumulator(call.o_accum),
                call.stream,
            );
        }
        // SAFETY: reads the status the launch above recorded.
        ffi::check_launch("run_mha", unsafe { ffi::flash_last_error() })
    }

    /// Enqueues the paged cache writes, keys then values, on `call.stream`.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError`] when an argument does not fit the kernel's parameter or the
    /// launch fails.
    ///
    /// # Safety
    /// Every address in `call` must be live on the stream's device and match its documented
    /// shape; every slot in the mapping must lie inside the caches.
    pub unsafe fn write_kv(call: &KvWriteCall) -> Result<(), KernelError> {
        let block_stride = arg_i64("block_stride", call.block_stride())?;
        let num_tokens = arg_i64("num_tokens", call.num_tokens)?;
        let num_heads = arg_i64("num_heads", call.num_kv_heads)?;
        let head_size = arg_i64("head_size", call.head_dim)?;
        let block_size = arg_i64("block_size", call.page_block)?;
        let source_stride = arg_i64("source_stride", call.source_stride)?;
        for (source, cache) in [(call.k_source, call.k_cache), (call.v_source, call.v_cache)] {
            // SAFETY: the caller's contract; the FFI returns the launch status.
            let status = unsafe {
                ffi::reshape_and_cache_flash_cache(
                    source as *const c_void,
                    cache as *mut c_void,
                    call.slot_mapping as *const i64,
                    block_stride,
                    num_tokens,
                    num_heads,
                    head_size,
                    block_size,
                    source_stride,
                    call.precision.cache_code(),
                    call.stream,
                )
            };
            ffi::check_launch("reshape_and_cache_flash_cache", status)?;
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
pub use compiled::{decode_attention, write_kv};

/// Named refusal: this build carries no kernels. The split count is still held to the cap, so a
/// call no build could combine is refused the same way on every build.
///
/// # Errors
///
/// Returns [`KernelError::SplitCount`] for a count past [`MAX_SPLITS`], and
/// [`KernelError::NotCompiled`] otherwise.
///
/// # Safety
/// Dereferences nothing; the signature matches the real launch.
#[cfg(not(feature = "cuda"))]
pub unsafe fn decode_attention(call: &DecodeAttentionCall) -> Result<(), KernelError> {
    split_count(call)?;
    Err(KernelError::NotCompiled { kernel: "run_mha" })
}

/// Named refusal: this build carries no kernels.
///
/// # Errors
///
/// Always returns [`KernelError::NotCompiled`].
///
/// # Safety
/// Dereferences nothing; the signature matches the real launch.
#[cfg(not(feature = "cuda"))]
pub unsafe fn write_kv(_call: &KvWriteCall) -> Result<(), KernelError> {
    Err(KernelError::NotCompiled {
        kernel: "reshape_and_cache_flash_cache",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn llama_8b(max_blocks_per_seq: usize) -> DecodeShape {
        DecodeShape {
            batch_size: 8,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            page_block: 16,
            max_blocks_per_seq,
        }
    }

    #[test]
    fn the_key_length_is_every_block_table_column() {
        let shape = llama_8b(256);
        assert_eq!(shape.seqlen_k(), 4096);
        assert_eq!(shape.seqlen_k_rounded(), 4096);
        let ragged = llama_8b(3);
        assert_eq!(ragged.seqlen_k(), 48);
        assert_eq!(ragged.seqlen_k_rounded(), 128);
    }

    #[test]
    fn the_head_dimension_rounds_up_to_the_accumulator_width() {
        assert_eq!(llama_8b(1).head_dim_rounded(), 128);
        let odd = DecodeShape {
            head_dim: 100,
            ..llama_8b(1)
        };
        assert_eq!(odd.head_dim_rounded(), 128);
        let small = DecodeShape {
            head_dim: 96,
            ..llama_8b(1)
        };
        assert_eq!(small.head_dim_rounded(), 96);
    }

    #[test]
    fn accumulators_scale_with_the_split_count() {
        let shape = llama_8b(256);
        assert_eq!(shape.softmax_lse_len(), 8 * 32);
        assert_eq!(shape.lse_accum_len(1), 8 * 32);
        assert_eq!(shape.lse_accum_len(6), 6 * 8 * 32);
        assert_eq!(shape.o_accum_len(6), 6 * 8 * 32 * 128);
    }

    #[test]
    fn a_cache_block_holds_every_slot_of_every_kv_head() {
        let call = KvWriteCall {
            k_source: 0,
            v_source: 0,
            source_stride: 6144,
            k_cache: 0,
            v_cache: 0,
            slot_mapping: 0,
            num_tokens: 8,
            num_kv_heads: 8,
            head_dim: 128,
            page_block: 16,
            precision: Precision::Bf16,
            stream: core::ptr::null_mut(),
        };
        assert_eq!(call.block_stride(), 16 * 8 * 128);
    }

    #[test]
    fn precision_maps_to_the_kernels_codes() {
        assert_eq!(Precision::F16.cache_code(), 0);
        assert_eq!(Precision::Bf16.cache_code(), 1);
        assert_eq!(Precision::F16.is_bf16(), 0);
        assert_eq!(Precision::Bf16.is_bf16(), 1);
    }

    /// A call of `llama_8b(1)` at `num_splits`, every address null: the checks under test read
    /// the count alone.
    fn decode_call(num_splits: u32) -> DecodeAttentionCall {
        let shape = llama_8b(1);
        DecodeAttentionCall {
            q: 0,
            q_strides: OperandStrides {
                batch: 6144,
                row: 6144,
                head: 128,
            },
            k_cache: 0,
            v_cache: 0,
            cache_strides: OperandStrides {
                batch: 16 * 1024,
                row: 1024,
                head: 128,
            },
            out: 0,
            out_strides: OperandStrides {
                batch: 4096,
                row: 4096,
                head: 128,
            },
            softmax_lse: 0,
            lse_accum: 0,
            o_accum: 0,
            seqlens_k: 0,
            block_table: 0,
            shape,
            num_splits,
            softmax_scale: 1.0,
            precision: Precision::Bf16,
            stream: core::ptr::null_mut(),
        }
    }

    #[test]
    fn the_split_count_is_held_to_the_combine_kernels_cap() {
        let cap = u32::try_from(MAX_SPLITS).unwrap();
        assert_eq!(split_count(&decode_call(1)), Ok(1));
        assert_eq!(split_count(&decode_call(cap)), Ok(cap));

        let refused = split_count(&decode_call(cap + 1)).unwrap_err();

        assert_eq!(
            refused,
            KernelError::SplitCount {
                num_splits: cap + 1
            }
        );
        assert!(refused.to_string().contains("129"), "{refused}");
        assert!(refused.to_string().contains("128"), "{refused}");
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn a_count_past_the_cap_is_refused_on_a_build_without_kernels_too() {
        // SAFETY: the stub dereferences nothing.
        let refused = unsafe { decode_attention(&decode_call(129)) }.unwrap_err();
        assert_eq!(refused, KernelError::SplitCount { num_splits: 129 });
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn a_build_without_kernels_refuses_by_name() {
        // SAFETY: the stub dereferences nothing.
        let refused = unsafe { decode_attention(&decode_call(1)) }.unwrap_err();
        assert_eq!(refused, KernelError::NotCompiled { kernel: "run_mha" });
        assert!(refused.to_string().contains("--features cuda"), "{refused}");
    }
}
