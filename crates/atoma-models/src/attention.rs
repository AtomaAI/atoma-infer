//! The decode step's attention seam: paged decode attention over the flash-attention kernels,
//! with the split count and every workspace fixed per bucket at Allocation.
//!
//! Two things sit behind this seam. A plan is what a bucket needs before anything runs: the split
//! count the kernel would choose for that batch size, and the accumulator bytes that choice
//! implies. A call is one launch, assembled from tensors the caller resolved — the query heads of
//! the fused row, the two cache halves, the block table and the sequence lengths — into the
//! addresses and element strides the kernel takes.
//!
//! It is deliberately the smallest thing both the step and a real attention backend can be written
//! against. The backend contract in the engine core states the same split of work: preparation
//! that may allocate, recording that may not. A backend implementing that contract replaces what
//! is here without the step's op table changing.

use core::ffi::c_void;

use atoma_kernels::paged_decode::{
    DecodeAttentionCall, DecodeShape, KvWriteCall, OperandStrides, Precision,
};
use atoma_kernels::splits::{num_splits, SplitShape};
use atoma_runtime::tensor::{Dtype, Tensor};
use thiserror::Error;

use crate::dims::LlamaDims;
use crate::operand::{self, Operand, OperandError, OperandKind};

/// f32 is what the split kernel accumulates in.
const ACCUMULATOR: Dtype = Dtype::F32;

/// The precision the decode step runs. Refused at startup if the model is loaded in another.
pub const PRECISION: Precision = Precision::Bf16;

/// Why an attention plan or call could not be shaped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum AttentionError {
    #[error(
        "the cache is rank {rank} of {dims:?}; a layer's cache is [2, blocks, page_block, \
         kv_width]"
    )]
    CacheShape { rank: usize, dims: [usize; 4] },
    #[error("the cache holds {kv_width} key-value elements per slot, not the model's {expected}")]
    CacheWidth { kv_width: usize, expected: usize },
    #[error(transparent)]
    Operand(#[from] OperandError),
    #[error("a batch of {batch} sequences does not fill the plan's bucket of {bucket}")]
    BatchNotBucket { batch: usize, bucket: usize },
    #[error("the split heuristic chose {splits} partitions, more than the kernel's count holds")]
    SplitCount { splits: usize },
}

/// What one bucket's attention needs before anything runs.
///
/// Computed at Allocation, once per bucket: the accumulators are allocated then, and a captured
/// graph bakes their addresses. The split count is the one the kernel would choose for itself, so
/// stating it changes nothing about what runs — it only lets the caller size the accumulators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionPlan {
    /// Sequences the bucket serves, each contributing one token.
    pub bucket: usize,
    /// Columns of the block table the graph bakes.
    pub max_blocks_per_seq: usize,
    /// Slots per cache block.
    pub page_block: usize,
    pub num_splits: u32,
    shape: DecodeShape,
}

impl AttentionPlan {
    /// The plan for `bucket` decoding sequences over `max_blocks_per_seq` blocks of `page_block`
    /// slots, on a device of `sm_count` multiprocessors.
    ///
    /// # Errors
    ///
    /// Returns [`AttentionError::SplitCount`] when the heuristic's split count does not fit the
    /// kernel's argument.
    pub fn new(
        dims: &LlamaDims,
        bucket: usize,
        page_block: usize,
        max_blocks_per_seq: usize,
        sm_count: usize,
    ) -> Result<Self, AttentionError> {
        let shape = DecodeShape {
            batch_size: bucket,
            num_heads: dims.num_heads,
            num_kv_heads: dims.num_kv_heads,
            head_dim: dims.head_dim,
            page_block,
            max_blocks_per_seq,
        };
        let splits = num_splits(
            SplitShape {
                batch_size: bucket,
                num_heads: dims.num_heads,
                head_dim: dims.head_dim,
                max_seqlen_k: shape.seqlen_k(),
                max_seqlen_q: 1,
            },
            sm_count,
        );
        Ok(Self {
            bucket,
            max_blocks_per_seq,
            page_block,
            num_splits: u32::try_from(splits).map_err(|_| AttentionError::SplitCount { splits })?,
            shape,
        })
    }

    /// The kernel shape this plan launches.
    #[must_use]
    pub fn shape(&self) -> DecodeShape {
        self.shape
    }

    /// Bytes of the log-sum-exp output every call writes.
    #[must_use]
    pub fn softmax_lse_bytes(&self) -> usize {
        ACCUMULATOR.width_bytes(self.shape.softmax_lse_len())
    }

    /// Bytes of the split log-sum-exp accumulator.
    #[must_use]
    pub fn lse_accum_bytes(&self) -> usize {
        ACCUMULATOR.width_bytes(self.shape.lse_accum_len(self.num_splits))
    }

    /// Bytes of the split output accumulator.
    #[must_use]
    pub fn o_accum_bytes(&self) -> usize {
        ACCUMULATOR.width_bytes(self.shape.o_accum_len(self.num_splits))
    }
}

/// The key and value halves of one layer's paged cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheHalves {
    pub k: Tensor,
    pub v: Tensor,
    /// The strides the kernel reads either half with.
    pub strides: OperandStrides,
}

/// Splits a layer's cache into its key and value halves.
///
/// A layer's cache is `[2, blocks, page_block, kv_width]`, keys first: the shape the paged kernels
/// read, with each slot's key-value heads flattened into one dimension since the kernel takes the
/// head stride separately. Both halves are views of the same buffer, so nothing is copied.
///
/// # Errors
///
/// Returns [`AttentionError`] when the cache is not that shape, does not hold this model's
/// key-value width, or is not bf16.
///
/// # Panics
///
/// Panics when the cache's first dimension cannot be selected, which the shape check above has
/// already ruled out.
pub fn cache_halves(cache: &Tensor, dims: &LlamaDims) -> Result<CacheHalves, AttentionError> {
    if cache.rank() != 4 || cache.dim(0) != 2 {
        let mut dims = [0; 4];
        for (slot, &size) in dims.iter_mut().zip(cache.dims()) {
            *slot = size;
        }
        return Err(AttentionError::CacheShape {
            rank: cache.rank(),
            dims,
        });
    }
    operand::dtype(
        Operand::model(OperandKind::Cache),
        cache.layout(),
        Dtype::Bf16,
    )?;
    let kv_width = cache.dim(3);
    if kv_width != dims.kv_width() {
        return Err(AttentionError::CacheWidth {
            kv_width,
            expected: dims.kv_width(),
        });
    }
    let half = |index: usize| {
        cache
            .select(0, index)
            .expect("the cache's first dimension selects its half")
    };
    let k = half(0);
    Ok(CacheHalves {
        strides: OperandStrides {
            batch: k.stride(0),
            row: k.stride(1),
            head: dims.head_dim,
        },
        k,
        v: half(1),
    })
}

/// Where one attention call reads and writes.
#[derive(Debug, Clone, Copy)]
pub struct AttentionTensors<'a> {
    /// The fused qkv activations, `[tokens, qkv_width]`; the query heads lead each row.
    pub qkv: &'a Tensor,
    /// This layer's cache halves.
    pub cache: &'a CacheHalves,
    /// The attention output, `[tokens, q_width]`.
    pub out: &'a Tensor,
    /// i32 `[tokens]`: each sequence's key length after this step's token.
    pub seqlens_k: &'a Tensor,
    /// i32 `[tokens, max_blocks_per_seq]`.
    pub block_table: &'a Tensor,
    /// f32 log-sum-exp output.
    pub softmax_lse: &'a Tensor,
    /// f32 split log-sum-exp accumulator.
    pub lse_accum: &'a Tensor,
    /// f32 split output accumulator.
    pub o_accum: &'a Tensor,
}

/// Assembles one paged decode attention call.
///
/// The query is the head of each fused row, so its address is the row's and its strides are the
/// row's: one sequence per row makes the batch stride and the row stride the same.
///
/// # Errors
///
/// Returns [`AttentionError`] when the batch is not the plan's bucket, an operand has the wrong
/// dtype, or the block table is not one full-width row per sequence.
pub fn decode_call(
    plan: &AttentionPlan,
    tensors: &AttentionTensors<'_>,
    dims: &LlamaDims,
    stream: *mut c_void,
) -> Result<DecodeAttentionCall, AttentionError> {
    check(plan, tensors, dims)?;
    let row = tensors.qkv.stride(0);
    Ok(DecodeAttentionCall {
        q: tensors.qkv.address(),
        q_strides: OperandStrides {
            batch: row,
            row,
            head: dims.head_dim,
        },
        k_cache: tensors.cache.k.address(),
        v_cache: tensors.cache.v.address(),
        cache_strides: tensors.cache.strides,
        out: tensors.out.address(),
        out_strides: OperandStrides {
            batch: tensors.out.stride(0),
            row: tensors.out.stride(0),
            head: dims.head_dim,
        },
        softmax_lse: tensors.softmax_lse.address(),
        lse_accum: tensors.lse_accum.address(),
        o_accum: tensors.o_accum.address(),
        seqlens_k: tensors.seqlens_k.address(),
        block_table: tensors.block_table.address(),
        shape: plan.shape(),
        num_splits: plan.num_splits,
        softmax_scale: dims.softmax_scale(),
        precision: PRECISION,
        stream,
    })
}

/// Assembles the paged cache write for the key and value heads of the fused row.
///
/// The key heads sit after the query heads and the value heads after those, so both sources are
/// column offsets into the same rows, `source_stride` elements apart.
///
/// # Errors
///
/// Returns [`AttentionError`] when the activations are not bf16 or not the fused width.
///
/// # Panics
///
/// Panics when the key or value columns do not lie in the fused row, which the width check
/// above has already ruled out.
pub fn kv_write_call(
    qkv: &Tensor,
    cache: &CacheHalves,
    slot_mapping: &Tensor,
    dims: &LlamaDims,
    stream: *mut c_void,
) -> Result<KvWriteCall, AttentionError> {
    let tokens = operand::rows(
        Operand::model(OperandKind::Activations),
        qkv.layout(),
        Dtype::Bf16,
        dims.qkv_width(),
    )?;
    let k_source = qkv
        .narrow(1, dims.q_width(), dims.kv_width())
        .expect("the key heads follow the query heads in the fused row");
    let v_source = qkv
        .narrow(1, dims.q_width() + dims.kv_width(), dims.kv_width())
        .expect("the value heads follow the key heads in the fused row");
    Ok(KvWriteCall {
        k_source: k_source.address(),
        v_source: v_source.address(),
        source_stride: qkv.stride(0),
        k_cache: cache.k.address(),
        v_cache: cache.v.address(),
        slot_mapping: slot_mapping.address(),
        num_tokens: tokens,
        num_kv_heads: dims.num_kv_heads,
        head_dim: dims.head_dim,
        page_block: cache.k.dim(1),
        precision: PRECISION,
        stream,
    })
}

/// Checks that `tensors` describe a full bucket of this model's shape.
fn check(
    plan: &AttentionPlan,
    tensors: &AttentionTensors<'_>,
    dims: &LlamaDims,
) -> Result<(), AttentionError> {
    let batch = tensors.qkv.dim(0);
    if batch != plan.bucket {
        return Err(AttentionError::BatchNotBucket {
            batch,
            bucket: plan.bucket,
        });
    }
    for (operand, tensor, dtype) in [
        (OperandKind::Activations, tensors.qkv, Dtype::Bf16),
        (OperandKind::AttentionOutput, tensors.out, Dtype::Bf16),
        (OperandKind::KeyLengths, tensors.seqlens_k, Dtype::I32),
        (OperandKind::BlockTable, tensors.block_table, Dtype::I32),
        (OperandKind::LogSumExp, tensors.softmax_lse, Dtype::F32),
        (OperandKind::SplitLogSumExp, tensors.lse_accum, Dtype::F32),
        (OperandKind::SplitOutput, tensors.o_accum, Dtype::F32),
    ] {
        operand::dtype(Operand::model(operand), tensor.layout(), dtype)?;
    }
    for (operand, tensor, expected_columns) in [
        (OperandKind::Activations, tensors.qkv, dims.qkv_width()),
        (OperandKind::AttentionOutput, tensors.out, dims.q_width()),
        (
            OperandKind::BlockTable,
            tensors.block_table,
            plan.max_blocks_per_seq,
        ),
    ] {
        operand::shape(
            Operand::model(operand),
            tensor.layout(),
            &[batch, expected_columns],
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use core::ptr;

    use atoma_runtime::tensor::Layout;

    use super::*;
    use crate::dims::test_support::llama_8b;
    use crate::operand::Shape;

    const PAGE_BLOCK: usize = 16;
    const BLOCKS: usize = 4096;
    const H100_SMS: usize = 114;
    const BUCKET: usize = 8;
    const BLOCK_COLUMNS: usize = 256;

    fn plan(bucket: usize) -> AttentionPlan {
        AttentionPlan::new(&llama_8b(2), bucket, PAGE_BLOCK, BLOCK_COLUMNS, H100_SMS).unwrap()
    }

    /// A view at `address` of a contiguous layout; the arithmetic under test needs no device.
    fn view(address: u64, dims: &[usize], dtype: Dtype) -> Tensor {
        Tensor::for_test(address, Layout::contiguous(dims, dtype).unwrap()).unwrap()
    }

    fn cache(dims: &LlamaDims) -> CacheHalves {
        let cache = view(
            0x10_0000,
            &[2, BLOCKS, PAGE_BLOCK, dims.kv_width()],
            Dtype::Bf16,
        );
        cache_halves(&cache, dims).unwrap()
    }

    #[test]
    fn a_plan_takes_the_split_count_the_kernel_would_choose() {
        assert!(plan(1).num_splits > 1, "one sequence splits across the SMs");
        assert_eq!(plan(64).num_splits, 1, "a full bucket already fills them");
        assert_eq!(plan(BUCKET).shape().seqlen_k(), BLOCK_COLUMNS * PAGE_BLOCK);
    }

    #[test]
    fn the_workspace_is_the_two_split_accumulators_at_that_count() {
        let plan = plan(BUCKET);
        let shape = plan.shape();
        assert_eq!(
            plan.lse_accum_bytes(),
            4 * shape.lse_accum_len(plan.num_splits)
        );
        assert_eq!(plan.o_accum_bytes(), 4 * shape.o_accum_len(plan.num_splits));
        assert_eq!(plan.softmax_lse_bytes(), 4 * BUCKET * 32);
    }

    #[test]
    fn the_cache_halves_are_the_keys_then_the_values() {
        let dims = llama_8b(2);
        let halves = cache(&dims);

        let half_bytes = (BLOCKS * PAGE_BLOCK * dims.kv_width() * 2) as u64;
        assert_eq!(halves.k.address(), 0x10_0000);
        assert_eq!(halves.v.address(), 0x10_0000 + half_bytes);
        assert_eq!(halves.k.dims(), [BLOCKS, PAGE_BLOCK, dims.kv_width()]);
        assert_eq!(
            halves.strides,
            OperandStrides {
                batch: PAGE_BLOCK * dims.kv_width(),
                row: dims.kv_width(),
                head: dims.head_dim,
            }
        );
    }

    #[test]
    fn a_cache_that_is_not_this_model_is_refused_with_both_widths() {
        let dims = llama_8b(2);
        let narrow = view(0x10_0000, &[2, BLOCKS, PAGE_BLOCK, 512], Dtype::Bf16);
        assert_eq!(
            cache_halves(&narrow, &dims).unwrap_err(),
            AttentionError::CacheWidth {
                kv_width: 512,
                expected: 1024
            }
        );

        let flat = view(
            0x10_0000,
            &[BLOCKS, PAGE_BLOCK, dims.kv_width()],
            Dtype::Bf16,
        );
        assert!(matches!(
            cache_halves(&flat, &dims).unwrap_err(),
            AttentionError::CacheShape { rank: 3, .. }
        ));
    }

    /// Every operand of a well-formed call at this bucket, in the order [`AttentionTensors`]
    /// declares them after the cache.
    fn tensors(dims: &LlamaDims) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) {
        (
            view(0x20_0000, &[BUCKET, dims.qkv_width()], Dtype::Bf16),
            view(0x30_0000, &[BUCKET, dims.q_width()], Dtype::Bf16),
            view(0x40_0000, &[BUCKET], Dtype::I32),
            view(0x50_0000, &[BUCKET, BLOCK_COLUMNS], Dtype::I32),
            view(0x60_0000, &[BUCKET, dims.num_heads], Dtype::F32),
            view(0x70_0000, &[BUCKET], Dtype::F32),
            view(0x80_0000, &[BUCKET], Dtype::F32),
        )
    }

    #[test]
    fn a_call_reads_the_query_at_the_fused_rows_address_and_stride() {
        let dims = llama_8b(2);
        let halves = cache(&dims);
        let (qkv, out, seqlens_k, block_table, softmax_lse, lse_accum, o_accum) = tensors(&dims);
        let plan = plan(BUCKET);

        let call = decode_call(
            &plan,
            &AttentionTensors {
                qkv: &qkv,
                cache: &halves,
                out: &out,
                seqlens_k: &seqlens_k,
                block_table: &block_table,
                softmax_lse: &softmax_lse,
                lse_accum: &lse_accum,
                o_accum: &o_accum,
            },
            &dims,
            ptr::null_mut(),
        )
        .unwrap();

        assert_eq!(call.q, qkv.address(), "the query heads lead the fused row");
        assert_eq!(
            call.q_strides,
            OperandStrides {
                batch: dims.qkv_width(),
                row: dims.qkv_width(),
                head: dims.head_dim,
            },
            "one sequence per row makes the batch and row strides the same"
        );
        assert_eq!(call.k_cache, halves.k.address());
        assert_eq!(call.v_cache, halves.v.address());
        assert_eq!(call.cache_strides, halves.strides);
        assert_eq!(call.num_splits, plan.num_splits);
        assert_eq!(call.precision, Precision::Bf16);
        assert!((call.softmax_scale - dims.softmax_scale()).abs() < 1e-9);
    }

    #[test]
    fn a_batch_that_does_not_fill_the_bucket_is_refused() {
        let dims = llama_8b(2);
        let halves = cache(&dims);
        let (_, out, seqlens_k, block_table, softmax_lse, lse_accum, o_accum) = tensors(&dims);
        let short = view(0x20_0000, &[BUCKET - 1, dims.qkv_width()], Dtype::Bf16);

        let refused = decode_call(
            &plan(BUCKET),
            &AttentionTensors {
                qkv: &short,
                cache: &halves,
                out: &out,
                seqlens_k: &seqlens_k,
                block_table: &block_table,
                softmax_lse: &softmax_lse,
                lse_accum: &lse_accum,
                o_accum: &o_accum,
            },
            &dims,
            ptr::null_mut(),
        )
        .unwrap_err();

        assert_eq!(
            refused,
            AttentionError::BatchNotBucket {
                batch: BUCKET - 1,
                bucket: BUCKET
            }
        );
    }

    #[test]
    fn a_block_table_narrower_than_the_graph_baked_is_refused() {
        let dims = llama_8b(2);
        let halves = cache(&dims);
        let (qkv, out, seqlens_k, _, softmax_lse, lse_accum, o_accum) = tensors(&dims);
        let narrow = view(0x50_0000, &[BUCKET, 8], Dtype::I32);

        let refused = decode_call(
            &plan(BUCKET),
            &AttentionTensors {
                qkv: &qkv,
                cache: &halves,
                out: &out,
                seqlens_k: &seqlens_k,
                block_table: &narrow,
                softmax_lse: &softmax_lse,
                lse_accum: &lse_accum,
                o_accum: &o_accum,
            },
            &dims,
            ptr::null_mut(),
        )
        .unwrap_err();

        assert_eq!(
            refused,
            AttentionError::Operand(OperandError::Shape {
                operand: Operand::model(OperandKind::BlockTable),
                shape: Shape::new(&[BUCKET, 8]),
                expected: Shape::new(&[BUCKET, BLOCK_COLUMNS])
            })
        );
    }

    #[test]
    fn the_cache_write_reads_the_key_and_value_segments_of_the_fused_row() {
        let dims = llama_8b(2);
        let halves = cache(&dims);
        let qkv = view(0x20_0000, &[BUCKET, dims.qkv_width()], Dtype::Bf16);
        let slot_mapping = view(0x90_0000, &[BUCKET], Dtype::I64);

        let call = kv_write_call(&qkv, &halves, &slot_mapping, &dims, ptr::null_mut()).unwrap();

        assert_eq!(call.k_source, qkv.address() + (dims.q_width() * 2) as u64);
        assert_eq!(
            call.v_source,
            qkv.address() + ((dims.q_width() + dims.kv_width()) * 2) as u64
        );
        assert_eq!(call.source_stride, dims.qkv_width());
        assert_eq!(call.num_tokens, BUCKET);
        assert_eq!(call.page_block, PAGE_BLOCK);
        assert_eq!(call.block_stride(), PAGE_BLOCK * dims.kv_width());
        assert_eq!(call.slot_mapping, slot_mapping.address());
    }

    #[test]
    fn activations_that_are_not_the_fused_width_cannot_be_written_to_the_cache() {
        let dims = llama_8b(2);
        let halves = cache(&dims);
        let slot_mapping = view(0x90_0000, &[BUCKET], Dtype::I64);
        let narrow = view(0x20_0000, &[BUCKET, dims.q_width()], Dtype::Bf16);

        assert_eq!(
            kv_write_call(&narrow, &halves, &slot_mapping, &dims, ptr::null_mut()).unwrap_err(),
            AttentionError::Operand(OperandError::Shape {
                operand: Operand::model(OperandKind::Activations),
                shape: Shape::new(&[BUCKET, dims.q_width()]),
                expected: Shape::new(&[BUCKET, dims.qkv_width()])
            })
        );
    }
}
