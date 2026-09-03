//! Every address one bucket's decode step touches, resolved once at Allocation.
//!
//! The arena places a role's slot as a pure function of bucket, layer and role, and exposes it
//! only through its offset lookup. This module is the one place that lookup becomes a tensor
//! view: for each bucket, for each arena row, one `[tokens, width]` view per role, and the query,
//! key and value column views of the fused row. Beside them sit the weights, the paged cache
//! halves and the step's fixed buffers — the inputs written before every step, the logits read
//! after it, the attention workspace — each checked against the model's dimensions when the
//! table is built. What the step enqueues is then a walk over these tables: no address
//! arithmetic, no lookup and no check inside a recording.
//!
//! A row is one layer's frame in the arena. There are `layers + 1` rows: the last layer's
//! residual add writes the row after it, which the final norm then reads, and whose `Normed`
//! slot the head projection reads.

use std::fmt;

use atoma_runtime::arena::{BucketIdx, CaptureArena, LayerIdx};
use atoma_runtime::tensor::{Dtype, Tensor, TensorError, MAX_RANK};
use thiserror::Error;

use crate::attention::{cache_halves, AttentionError, AttentionPlan, CacheHalves};
use crate::dims::LlamaDims;
use crate::kernels::RotaryTensors;
use crate::layer::{LayerOffset, LayerWeight, QkvColumns, Role, RoleRef};

/// The activations' element type: every arena slot is read and written as bf16.
const ACTIVATION: Dtype = Dtype::Bf16;

/// A shape, for refusals that show both the one held and the one needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    dims: [usize; MAX_RANK],
    rank: usize,
}

impl Shape {
    /// The shape `dims`; dimensions past the rank a layout holds are dropped.
    #[must_use]
    pub fn new(dims: &[usize]) -> Self {
        let mut shape = Self {
            dims: [0; MAX_RANK],
            rank: dims.len().min(MAX_RANK),
        };
        shape.dims[..shape.rank].copy_from_slice(&dims[..shape.rank]);
        shape
    }

    /// The shape of `tensor`.
    #[must_use]
    pub fn of(tensor: &Tensor) -> Self {
        Self::new(tensor.dims())
    }

    #[must_use]
    pub fn dims(&self) -> &[usize] {
        &self.dims[..self.rank]
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.dims())
    }
}

/// Which tensor a refusal is about: a model-level one, or one of a layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Operand {
    pub what: &'static str,
    pub layer: Option<usize>,
}

impl Operand {
    /// A tensor the model has one of.
    #[must_use]
    pub const fn model(what: &'static str) -> Self {
        Self { what, layer: None }
    }

    /// A tensor each layer has one of.
    #[must_use]
    pub const fn layer(layer: usize, what: &'static str) -> Self {
        Self {
            what,
            layer: Some(layer),
        }
    }
}

impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.layer {
            Some(layer) => write!(f, "layer {layer}'s {}", self.what),
            None => f.write_str(self.what),
        }
    }
}

/// Why a table could not be resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum SlotError {
    #[error("the arena memory is {dtype:?}; the activations are bf16")]
    ArenaDtype { dtype: Dtype },
    #[error("the arena memory holds {held} bytes; the arena addresses {needed}")]
    ArenaTooSmall { held: usize, needed: usize },
    #[error(
        "row {row}'s {role:?} slot holds {slot_bytes} bytes; a bucket of {tokens} tokens views \
         {view_bytes}"
    )]
    SlotTooSmall {
        row: usize,
        role: Role,
        tokens: usize,
        slot_bytes: usize,
        view_bytes: usize,
    },
    #[error("{operand} is {dtype:?}, not {expected:?}")]
    Dtype {
        operand: Operand,
        dtype: Dtype,
        expected: Dtype,
    },
    #[error("{operand} is {shape}, not {expected}")]
    Shape {
        operand: Operand,
        shape: Shape,
        expected: Shape,
    },
    #[error("{operand} has strides {strides:?}; it must be one contiguous buffer")]
    NotContiguous {
        operand: Operand,
        strides: [usize; MAX_RANK],
    },
    #[error("{what} covers {count} layers; the model has {expected}")]
    LayerCount {
        what: &'static str,
        count: usize,
        expected: usize,
    },
    #[error("{operand} holds {len} rows; the bucket reads {needed}")]
    StaticTooShort {
        operand: Operand,
        len: usize,
        needed: usize,
    },
    #[error("the attention plan is for {plan} sequences; the bucket serves {tokens} tokens")]
    PlanBucket { plan: usize, tokens: usize },
    #[error(transparent)]
    Attention(#[from] AttentionError),
    #[error(transparent)]
    Tensor(#[from] TensorError),
}

/// One entry of the bucket ladder: its index into the arena and the tokens it serves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bucket {
    pub index: BucketIdx,
    pub tokens: usize,
}

/// One arena row's activation views for one bucket: a `[tokens, width]` view per role, and the
/// query, key and value columns of the fused row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerSlots {
    roles: [Tensor; Role::ALL.len()],
    q: Tensor,
    k: Tensor,
    v: Tensor,
}

impl LayerSlots {
    /// The views of arena row `row` under `bucket`, each a slice of `memory` at the offset the
    /// arena places it.
    fn resolve(
        memory: &Tensor,
        arena: &CaptureArena,
        bucket: Bucket,
        row: LayerIdx,
        dims: &LlamaDims,
    ) -> Result<Self, SlotError> {
        let mut roles = Vec::with_capacity(Role::ALL.len());
        for role in Role::ALL {
            let width = role.width_elements(dims);
            let elements = bucket.tokens * width;
            let view_bytes = ACTIVATION.width_bytes(elements);
            let slot_bytes = arena.slot_size(bucket.index, role.tensor_role());
            if view_bytes > slot_bytes {
                return Err(SlotError::SlotTooSmall {
                    row: row.0,
                    role,
                    tokens: bucket.tokens,
                    slot_bytes,
                    view_bytes,
                });
            }
            // Slot offsets are aligned to the arena's slot alignment, a multiple of every
            // element size, so the byte offset is a whole number of elements.
            let offset = arena.offset(bucket.index, row, role.tensor_role());
            let start = offset / ACTIVATION.size_in_bytes();
            let slot = memory
                .narrow(0, start, elements)?
                .reshape(&[bucket.tokens, width])?;
            roles.push(slot);
        }
        let roles: [Tensor; Role::ALL.len()] =
            roles.try_into().expect("one view was pushed per role");
        let qkv = &roles[Role::Qkv as usize];
        let (q_width, kv_width) = (dims.q_width(), dims.kv_width());
        Ok(Self {
            q: qkv.narrow(1, 0, q_width)?,
            k: qkv.narrow(1, q_width, kv_width)?,
            v: qkv.narrow(1, q_width + kv_width, kv_width)?,
            roles,
        })
    }

    /// The `[tokens, width]` view of `role`'s slot.
    #[must_use]
    pub fn role(&self, role: Role) -> &Tensor {
        &self.roles[role as usize]
    }

    /// The column view of the fused row a projection writes: `[tokens, width]` with the
    /// fused row's stride.
    #[must_use]
    pub fn columns(&self, columns: QkvColumns) -> &Tensor {
        match columns {
            QkvColumns::Q => &self.q,
            QkvColumns::K => &self.k,
            QkvColumns::V => &self.v,
        }
    }
}

/// Every activation view of one bucket: one [`LayerSlots`] per arena row, `layers + 1` rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActivationSlots {
    rows: Vec<LayerSlots>,
}

impl ActivationSlots {
    /// Resolves every row of `bucket` through `arena`'s offset lookup over `memory`, the bf16
    /// view of the arena's whole buffer.
    ///
    /// # Errors
    ///
    /// Returns [`SlotError`] when the memory is not bf16 or does not cover what the arena
    /// addresses, or a slot is smaller than the bucket's view of it.
    ///
    /// # Panics
    ///
    /// Panics when `arena` was built with fewer than `layers + 1` rows or without this bucket:
    /// the arena's lookup asserts its ranges.
    pub fn resolve(
        memory: &Tensor,
        arena: &CaptureArena,
        bucket: Bucket,
        dims: &LlamaDims,
    ) -> Result<Self, SlotError> {
        if memory.dtype() != ACTIVATION {
            return Err(SlotError::ArenaDtype {
                dtype: memory.dtype(),
            });
        }
        contiguous(Operand::model("the arena memory"), memory)?;
        let held = memory.extent_bytes();
        if held < arena.total_size() {
            return Err(SlotError::ArenaTooSmall {
                held,
                needed: arena.total_size(),
            });
        }
        let flat = memory.reshape(&[memory.element_count()])?;
        let rows = (0..=dims.layers)
            .map(|row| LayerSlots::resolve(&flat, arena, bucket, LayerIdx(row), dims))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { rows })
    }

    /// Rows resolved: the model's layers and one more.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows.len()
    }

    /// The views of arena row `row`.
    ///
    /// # Panics
    ///
    /// Panics when `row` is past the rows resolved.
    #[must_use]
    pub fn row(&self, row: LayerIdx) -> &LayerSlots {
        &self.rows[row.0]
    }

    /// The row `at` names from `layer`'s frame: the layer's own, or the next.
    ///
    /// # Panics
    ///
    /// Panics when the row named is past the rows resolved.
    #[must_use]
    pub fn frame(&self, layer: LayerIdx, at: LayerOffset) -> &LayerSlots {
        let row = match at {
            LayerOffset::Same => layer.0,
            LayerOffset::Next => layer.0 + 1,
        };
        &self.rows[row]
    }

    /// The view `role` names from `layer`'s frame.
    ///
    /// # Panics
    ///
    /// Panics when the row named is past the rows resolved.
    #[must_use]
    pub fn role(&self, layer: LayerIdx, role: RoleRef) -> &Tensor {
        self.frame(layer, role.layer).role(role.role)
    }

    /// The row after the last layer: what the final norm reads and writes.
    ///
    /// # Panics
    ///
    /// Panics when no row was resolved, which [`ActivationSlots::resolve`] never leaves.
    #[must_use]
    pub fn final_row(&self) -> &LayerSlots {
        self.rows.last().expect("at least one row is resolved")
    }
}

/// One layer's weights: `[out_features, in_features]` for each projection, one gain per norm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerWeights {
    pub input_norm: Tensor,
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
    pub o: Tensor,
    pub post_attention_norm: Tensor,
    pub gate: Tensor,
    pub up: Tensor,
    pub down: Tensor,
}

impl LayerWeights {
    #[must_use]
    pub fn get(&self, weight: LayerWeight) -> &Tensor {
        match weight {
            LayerWeight::InputNorm => &self.input_norm,
            LayerWeight::Q => &self.q,
            LayerWeight::K => &self.k,
            LayerWeight::V => &self.v,
            LayerWeight::O => &self.o,
            LayerWeight::PostAttentionNorm => &self.post_attention_norm,
            LayerWeight::Gate => &self.gate,
            LayerWeight::Up => &self.up,
            LayerWeight::Down => &self.down,
        }
    }

    /// Holds every weight of layer `layer` to the shape the step multiplies by.
    fn check(&self, layer: usize, dims: &LlamaDims) -> Result<(), SlotError> {
        let (hidden, ffn) = (dims.hidden, dims.ffn);
        let (q_width, kv_width) = (dims.q_width(), dims.kv_width());
        let expected: [(&'static str, &Tensor, &[usize]); 9] = [
            ("input norm gain", &self.input_norm, &[hidden]),
            ("query projection", &self.q, &[q_width, hidden]),
            ("key projection", &self.k, &[kv_width, hidden]),
            ("value projection", &self.v, &[kv_width, hidden]),
            ("output projection", &self.o, &[hidden, q_width]),
            (
                "post-attention norm gain",
                &self.post_attention_norm,
                &[hidden],
            ),
            ("gate projection", &self.gate, &[ffn, hidden]),
            ("up projection", &self.up, &[ffn, hidden]),
            ("down projection", &self.down, &[hidden, ffn]),
        ];
        for (what, tensor, shape) in expected {
            weight(Operand::layer(layer, what), tensor, shape)?;
        }
        Ok(())
    }
}

/// The model's weights as the step reads them: bf16 device views snapshotted at Allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlamaWeights {
    /// `[vocab, hidden]`.
    pub embedding: Tensor,
    pub layers: Vec<LayerWeights>,
    /// `[hidden]`.
    pub final_norm: Tensor,
    /// `[vocab, hidden]`.
    pub lm_head: Tensor,
}

impl LlamaWeights {
    /// Holds every weight to the shape the step reads it as.
    ///
    /// # Errors
    ///
    /// Returns [`SlotError`] when there is not one layer's weights per layer, or a weight is
    /// not bf16, not contiguous, or not the model's shape.
    pub fn check(&self, dims: &LlamaDims) -> Result<(), SlotError> {
        if self.layers.len() != dims.layers {
            return Err(SlotError::LayerCount {
                what: "the weights",
                count: self.layers.len(),
                expected: dims.layers,
            });
        }
        let (vocab, hidden) = (dims.vocab, dims.hidden);
        weight(
            Operand::model("the embedding table"),
            &self.embedding,
            &[vocab, hidden],
        )?;
        for (layer, weights) in self.layers.iter().enumerate() {
            weights.check(layer, dims)?;
        }
        weight(
            Operand::model("the final norm gain"),
            &self.final_norm,
            &[hidden],
        )?;
        weight(
            Operand::model("the head projection"),
            &self.lm_head,
            &[vocab, hidden],
        )
    }
}

/// Every layer's paged cache, split into its key and value halves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlamaCache {
    layers: Vec<CacheHalves>,
}

impl LlamaCache {
    /// Splits one cache per layer, in layer order.
    ///
    /// # Errors
    ///
    /// Returns [`SlotError`] when there is not one cache per layer, or a cache is not the shape
    /// the attention seam reads.
    pub fn new(caches: &[Tensor], dims: &LlamaDims) -> Result<Self, SlotError> {
        if caches.len() != dims.layers {
            return Err(SlotError::LayerCount {
                what: "the cache",
                count: caches.len(),
                expected: dims.layers,
            });
        }
        let layers = caches
            .iter()
            .map(|cache| cache_halves(cache, dims))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { layers })
    }

    /// The halves of `layer`'s cache.
    ///
    /// # Panics
    ///
    /// Panics when `layer` is past the model's layers.
    #[must_use]
    pub fn layer(&self, layer: LayerIdx) -> &CacheHalves {
        &self.layers[layer.0]
    }
}

/// The step's fixed buffers, sized once at the largest bucket: the inputs the host writes before
/// every step, the logits it reads after, the attention workspace, and the rotary tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepStatics {
    /// u32 `[max_tokens]`.
    pub token_ids: Tensor,
    /// i32 `[max_tokens]`.
    pub positions: Tensor,
    /// i32 `[max_tokens]`: each sequence's key length after this step's token.
    pub seqlens_k: Tensor,
    /// i64 `[max_tokens]`.
    pub slot_mapping: Tensor,
    /// i32 `[max_tokens, max_blocks_per_seq]`.
    pub block_table: Tensor,
    /// f32 `[max_tokens, vocab]`.
    pub logits: Tensor,
    /// f32, at least the largest bucket's log-sum-exp output.
    pub softmax_lse: Tensor,
    /// f32, at least the largest split log-sum-exp accumulator of any bucket.
    pub lse_accum: Tensor,
    /// f32, at least the largest split output accumulator of any bucket.
    pub o_accum: Tensor,
    pub rotary: RotaryTensors,
}

/// One bucket's views of the fixed buffers: the leading rows of each, as many as the bucket has
/// tokens, and the attention workspace at the bucket's split count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BucketStatics {
    pub token_ids: Tensor,
    pub positions: Tensor,
    pub seqlens_k: Tensor,
    pub slot_mapping: Tensor,
    pub block_table: Tensor,
    pub logits: Tensor,
    pub softmax_lse: Tensor,
    pub lse_accum: Tensor,
    pub o_accum: Tensor,
}

impl BucketStatics {
    /// The leading rows of each static that `plan`'s bucket reads and writes.
    ///
    /// # Errors
    ///
    /// Returns [`SlotError`] when a static is not the dtype the kernels read, not contiguous,
    /// not as wide as the plan, or shorter than the bucket.
    pub fn resolve(
        statics: &StepStatics,
        plan: &AttentionPlan,
        dims: &LlamaDims,
    ) -> Result<Self, SlotError> {
        let tokens = plan.bucket;
        let shape = plan.shape();
        let pairs = dims.head_dim / 2;
        let rotary = [dims.rope.max_position, pairs];
        static_shape("the cosine table", &statics.rotary.cos, Dtype::F32, &rotary)?;
        static_shape("the sine table", &statics.rotary.sin, Dtype::F32, &rotary)?;
        Ok(Self {
            token_ids: leading("the token ids", &statics.token_ids, Dtype::U32, tokens)?,
            positions: leading("the positions", &statics.positions, Dtype::I32, tokens)?,
            seqlens_k: leading("the key lengths", &statics.seqlens_k, Dtype::I32, tokens)?,
            slot_mapping: leading(
                "the slot mapping",
                &statics.slot_mapping,
                Dtype::I64,
                tokens,
            )?,
            block_table: leading_rows(
                "the block table",
                &statics.block_table,
                Dtype::I32,
                tokens,
                plan.max_blocks_per_seq,
            )?,
            logits: leading_rows(
                "the logits",
                &statics.logits,
                Dtype::F32,
                tokens,
                dims.vocab,
            )?,
            softmax_lse: leading(
                "the log-sum-exp output",
                &statics.softmax_lse,
                Dtype::F32,
                shape.softmax_lse_len(),
            )?,
            lse_accum: leading(
                "the split log-sum-exp accumulator",
                &statics.lse_accum,
                Dtype::F32,
                shape.lse_accum_len(plan.num_splits),
            )?,
            o_accum: leading(
                "the split output accumulator",
                &statics.o_accum,
                Dtype::F32,
                shape.o_accum_len(plan.num_splits),
            )?,
        })
    }
}

/// Everything one bucket's step addresses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BucketSlots {
    pub bucket: Bucket,
    pub plan: AttentionPlan,
    pub activations: ActivationSlots,
    pub statics: BucketStatics,
}

/// What every bucket's slots are resolved from.
#[derive(Debug, Clone, Copy)]
pub struct SlotSources<'a> {
    /// The bf16 view of the arena's whole buffer.
    pub memory: &'a Tensor,
    pub arena: &'a CaptureArena,
    pub statics: &'a StepStatics,
    pub dims: &'a LlamaDims,
}

impl BucketSlots {
    /// Resolves the activations and statics of `bucket`, whose attention runs `plan`.
    ///
    /// # Errors
    ///
    /// Returns [`SlotError`] when the plan is not the bucket's, or a table cannot be resolved.
    ///
    /// # Panics
    ///
    /// Panics when the arena was not built for this bucket or for `layers + 1` rows.
    pub fn resolve(
        sources: &SlotSources<'_>,
        bucket: Bucket,
        plan: AttentionPlan,
    ) -> Result<Self, SlotError> {
        if plan.bucket != bucket.tokens {
            return Err(SlotError::PlanBucket {
                plan: plan.bucket,
                tokens: bucket.tokens,
            });
        }
        Ok(Self {
            bucket,
            plan,
            activations: ActivationSlots::resolve(
                sources.memory,
                sources.arena,
                bucket,
                sources.dims,
            )?,
            statics: BucketStatics::resolve(sources.statics, &plan, sources.dims)?,
        })
    }
}

/// Holds `tensor` to being one unbroken row-major buffer.
fn contiguous(operand: Operand, tensor: &Tensor) -> Result<(), SlotError> {
    if tensor.is_contiguous() {
        return Ok(());
    }
    let mut strides = [0; MAX_RANK];
    for (slot, &stride) in strides.iter_mut().zip(tensor.strides()) {
        *slot = stride;
    }
    Err(SlotError::NotContiguous { operand, strides })
}

/// Holds `tensor` to `dtype`, contiguity and exactly `shape`.
fn exact(
    operand: Operand,
    tensor: &Tensor,
    dtype: Dtype,
    shape: &[usize],
) -> Result<(), SlotError> {
    if tensor.dtype() != dtype {
        return Err(SlotError::Dtype {
            operand,
            dtype: tensor.dtype(),
            expected: dtype,
        });
    }
    contiguous(operand, tensor)?;
    if tensor.dims() != shape {
        return Err(SlotError::Shape {
            operand,
            shape: Shape::of(tensor),
            expected: Shape::new(shape),
        });
    }
    Ok(())
}

/// A bf16 weight of exactly `shape`.
fn weight(operand: Operand, tensor: &Tensor, shape: &[usize]) -> Result<(), SlotError> {
    exact(operand, tensor, Dtype::Bf16, shape)
}

/// A model-level static of exactly `shape`.
fn static_shape(
    what: &'static str,
    tensor: &Tensor,
    dtype: Dtype,
    shape: &[usize],
) -> Result<(), SlotError> {
    exact(Operand::model(what), tensor, dtype, shape)
}

/// The first `needed` elements of a contiguous vector static of `dtype`.
fn leading(
    what: &'static str,
    tensor: &Tensor,
    dtype: Dtype,
    needed: usize,
) -> Result<Tensor, SlotError> {
    leading_rows(what, tensor, dtype, needed, 0)
}

/// The first `rows` rows of a contiguous static of `dtype`: a vector when `columns` is zero, a
/// `[rows, columns]` matrix otherwise.
fn leading_rows(
    what: &'static str,
    tensor: &Tensor,
    dtype: Dtype,
    rows: usize,
    columns: usize,
) -> Result<Tensor, SlotError> {
    let operand = Operand::model(what);
    if tensor.dtype() != dtype {
        return Err(SlotError::Dtype {
            operand,
            dtype: tensor.dtype(),
            expected: dtype,
        });
    }
    contiguous(operand, tensor)?;
    let held = tensor.dims().first().copied().unwrap_or(0);
    let expected_rank = if columns == 0 { 1 } else { 2 };
    if tensor.rank() != expected_rank || (columns != 0 && tensor.dim(1) != columns) {
        let expected = [held, columns];
        return Err(SlotError::Shape {
            operand,
            shape: Shape::of(tensor),
            expected: Shape::new(&expected[..expected_rank]),
        });
    }
    if held < rows {
        return Err(SlotError::StaticTooShort {
            operand,
            len: held,
            needed: rows,
        });
    }
    Ok(tensor.narrow(0, 0, rows)?)
}

#[cfg(test)]
mod tests {
    use atoma_runtime::arena::{ArenaLayout, TensorRole};
    use atoma_runtime::tensor::Layout;

    use super::*;
    use crate::dims::test_support::llama_8b;
    use crate::layer::LLAMA_LAYER;

    const LAYERS: usize = 2;
    const LADDER: [usize; 2] = [1, 8];
    const PAGE_BLOCK: usize = 16;
    const BLOCKS: usize = 64;
    const BLOCK_COLUMNS: usize = 32;
    const SMS: usize = 114;
    const ARENA_BASE: u64 = 0x1000_0000;

    fn view(address: u64, dims: &[usize], dtype: Dtype) -> Tensor {
        Tensor::for_test(address, Layout::contiguous(dims, dtype).unwrap()).unwrap()
    }

    fn dims() -> LlamaDims {
        llama_8b(LAYERS)
    }

    fn arena() -> CaptureArena {
        CaptureArena::new(
            LAYERS + 1,
            LLAMA_LAYER.role_table(&dims()),
            &LADDER,
            ArenaLayout::Greedy,
        )
        .unwrap()
    }

    /// A bf16 view of the whole arena at [`ARENA_BASE`].
    fn memory(arena: &CaptureArena) -> Tensor {
        view(ARENA_BASE, &[arena.total_size() / 2], Dtype::Bf16)
    }

    fn bucket(index: usize) -> Bucket {
        Bucket {
            index: BucketIdx(index),
            tokens: LADDER[index],
        }
    }

    fn plan(tokens: usize) -> AttentionPlan {
        AttentionPlan::new(&dims(), tokens, PAGE_BLOCK, BLOCK_COLUMNS, SMS)
    }

    #[test]
    fn every_slot_is_the_arena_offset_from_the_memory_base_at_the_roles_width() {
        let dims = dims();
        let arena = arena();
        let slots = ActivationSlots::resolve(&memory(&arena), &arena, bucket(1), &dims).unwrap();

        assert_eq!(slots.rows(), LAYERS + 1);
        for row in 0..=LAYERS {
            for role in Role::ALL {
                let slot = slots.row(LayerIdx(row)).role(role);
                let offset = arena.offset(BucketIdx(1), LayerIdx(row), role.tensor_role());
                assert_eq!(
                    slot.address(),
                    ARENA_BASE + offset as u64,
                    "row {row} {role:?}"
                );
                assert_eq!(slot.dims(), [8, role.width_elements(&dims)]);
                assert_eq!(slot.dtype(), Dtype::Bf16);
                assert!(slot.is_contiguous());
            }
        }
    }

    #[test]
    fn the_fused_row_splits_into_query_key_and_value_columns_a_row_apart() {
        let dims = dims();
        let arena = arena();
        let slots = ActivationSlots::resolve(&memory(&arena), &arena, bucket(1), &dims).unwrap();
        let row = slots.row(LayerIdx(0));
        let qkv = row.role(Role::Qkv);

        let (q, k, v) = (
            row.columns(QkvColumns::Q),
            row.columns(QkvColumns::K),
            row.columns(QkvColumns::V),
        );
        assert_eq!(q.address(), qkv.address());
        assert_eq!(k.address(), qkv.address() + (dims.q_width() * 2) as u64);
        assert_eq!(
            v.address(),
            qkv.address() + ((dims.q_width() + dims.kv_width()) * 2) as u64
        );
        assert_eq!(q.dims(), [8, dims.q_width()]);
        assert_eq!(k.dims(), [8, dims.kv_width()]);
        assert_eq!(v.dims(), [8, dims.kv_width()]);
        for view in [q, k, v] {
            assert_eq!(view.strides(), [dims.qkv_width(), 1]);
        }
    }

    #[test]
    fn the_next_layers_residual_is_the_following_rows_hidden_slot() {
        let arena = arena();
        let slots = ActivationSlots::resolve(&memory(&arena), &arena, bucket(0), &dims()).unwrap();

        for layer in 0..LAYERS {
            assert_eq!(
                slots.role(LayerIdx(layer), RoleRef::next(Role::Hidden)),
                slots.role(LayerIdx(layer + 1), RoleRef::same(Role::Hidden))
            );
        }
        assert_eq!(
            slots.final_row().role(Role::Hidden),
            slots.role(LayerIdx(LAYERS - 1), RoleRef::next(Role::Hidden))
        );
    }

    #[test]
    fn a_smaller_bucket_views_fewer_rows_of_each_slot() {
        let arena = arena();
        let slots = ActivationSlots::resolve(&memory(&arena), &arena, bucket(0), &dims()).unwrap();
        let hidden = slots.row(LayerIdx(1)).role(Role::Hidden);
        assert_eq!(hidden.dims(), [1, 4096]);
        assert_eq!(
            hidden.address(),
            ARENA_BASE + arena.offset(BucketIdx(0), LayerIdx(1), TensorRole(0)) as u64
        );
    }

    #[test]
    fn memory_that_does_not_cover_the_arena_is_refused_with_both_sizes() {
        let arena = arena();
        let short = view(ARENA_BASE, &[arena.total_size() / 2 - 128], Dtype::Bf16);
        assert_eq!(
            ActivationSlots::resolve(&short, &arena, bucket(1), &dims()).unwrap_err(),
            SlotError::ArenaTooSmall {
                held: arena.total_size() - 256,
                needed: arena.total_size()
            }
        );
        let f32_memory = view(ARENA_BASE, &[arena.total_size() / 4], Dtype::F32);
        assert_eq!(
            ActivationSlots::resolve(&f32_memory, &arena, bucket(1), &dims()).unwrap_err(),
            SlotError::ArenaDtype { dtype: Dtype::F32 }
        );
    }

    #[test]
    fn a_bucket_of_more_tokens_than_its_rung_is_refused_at_the_first_slot() {
        let arena = arena();
        let oversized = Bucket {
            index: BucketIdx(0),
            tokens: 2,
        };
        let refused =
            ActivationSlots::resolve(&memory(&arena), &arena, oversized, &dims()).unwrap_err();
        assert_eq!(
            refused,
            SlotError::SlotTooSmall {
                row: 0,
                role: Role::Hidden,
                tokens: 2,
                slot_bytes: 8192,
                view_bytes: 16384
            }
        );
        assert!(refused.to_string().contains("16384"));
    }

    fn layer_weights(base: u64, dims: &LlamaDims) -> LayerWeights {
        let (hidden, ffn) = (dims.hidden, dims.ffn);
        LayerWeights {
            input_norm: view(base, &[hidden], Dtype::Bf16),
            q: view(base + 0x1_0000, &[dims.q_width(), hidden], Dtype::Bf16),
            k: view(base + 0x2_0000, &[dims.kv_width(), hidden], Dtype::Bf16),
            v: view(base + 0x3_0000, &[dims.kv_width(), hidden], Dtype::Bf16),
            o: view(base + 0x4_0000, &[hidden, dims.q_width()], Dtype::Bf16),
            post_attention_norm: view(base + 0x5_0000, &[hidden], Dtype::Bf16),
            gate: view(base + 0x6_0000, &[ffn, hidden], Dtype::Bf16),
            up: view(base + 0x7_0000, &[ffn, hidden], Dtype::Bf16),
            down: view(base + 0x8_0000, &[hidden, ffn], Dtype::Bf16),
        }
    }

    fn weights(dims: &LlamaDims) -> LlamaWeights {
        LlamaWeights {
            embedding: view(0x2000_0000, &[dims.vocab, dims.hidden], Dtype::Bf16),
            layers: (0..dims.layers)
                .map(|layer| layer_weights(0x3000_0000 + layer as u64 * 0x10_0000, dims))
                .collect(),
            final_norm: view(0x4000_0000, &[dims.hidden], Dtype::Bf16),
            lm_head: view(0x5000_0000, &[dims.vocab, dims.hidden], Dtype::Bf16),
        }
    }

    #[test]
    fn the_weights_check_and_answer_by_name() {
        let dims = dims();
        let table = weights(&dims);
        table.check(&dims).unwrap();
        let layer = &table.layers[1];
        assert_eq!(layer.get(LayerWeight::Q), &layer.q);
        assert_eq!(layer.get(LayerWeight::Down), &layer.down);
        assert_eq!(
            layer.get(LayerWeight::PostAttentionNorm),
            &layer.post_attention_norm
        );
    }

    #[test]
    fn a_projection_of_the_wrong_shape_is_refused_naming_its_layer() {
        let dims = dims();
        let mut table = weights(&dims);
        table.layers[1].k = view(0x3010_0000, &[dims.q_width(), dims.hidden], Dtype::Bf16);

        let refused = table.check(&dims).unwrap_err();

        assert_eq!(
            refused,
            SlotError::Shape {
                operand: Operand::layer(1, "key projection"),
                shape: Shape::new(&[4096, 4096]),
                expected: Shape::new(&[1024, 4096])
            }
        );
        assert_eq!(
            refused.to_string(),
            "layer 1's key projection is [4096, 4096], not [1024, 4096]"
        );
    }

    #[test]
    fn weights_in_another_precision_or_for_another_depth_are_refused() {
        let dims = dims();
        let mut table = weights(&dims);
        table.final_norm = view(0x4000_0000, &[dims.hidden], Dtype::F16);
        assert_eq!(
            table.check(&dims).unwrap_err(),
            SlotError::Dtype {
                operand: Operand::model("the final norm gain"),
                dtype: Dtype::F16,
                expected: Dtype::Bf16
            }
        );

        let mut table = weights(&dims);
        table.layers.pop();
        assert_eq!(
            table.check(&dims).unwrap_err(),
            SlotError::LayerCount {
                what: "the weights",
                count: 1,
                expected: 2
            }
        );
    }

    fn caches(dims: &LlamaDims, layers: usize) -> Vec<Tensor> {
        (0..layers)
            .map(|layer| {
                view(
                    0x6000_0000 + layer as u64 * 0x100_0000,
                    &[2, BLOCKS, PAGE_BLOCK, dims.kv_width()],
                    Dtype::Bf16,
                )
            })
            .collect()
    }

    #[test]
    fn the_cache_splits_every_layer_into_its_halves_in_layer_order() {
        let dims = dims();
        let caches = caches(&dims, LAYERS);
        let cache = LlamaCache::new(&caches, &dims).unwrap();
        for (layer, tensor) in caches.iter().enumerate() {
            let halves = cache.layer(LayerIdx(layer));
            assert_eq!(halves.k.address(), tensor.address());
            assert_eq!(
                halves.v.address(),
                tensor.address() + (BLOCKS * PAGE_BLOCK * dims.kv_width() * 2) as u64
            );
        }
        assert_eq!(
            LlamaCache::new(&caches[..1], &dims).unwrap_err(),
            SlotError::LayerCount {
                what: "the cache",
                count: 1,
                expected: 2
            }
        );
    }

    fn statics(dims: &LlamaDims) -> StepStatics {
        let max = *LADDER.iter().max().unwrap();
        let widest = plan(max);
        let shape = widest.shape();
        StepStatics {
            token_ids: view(0x7000_0000, &[max], Dtype::U32),
            positions: view(0x7001_0000, &[max], Dtype::I32),
            seqlens_k: view(0x7002_0000, &[max], Dtype::I32),
            slot_mapping: view(0x7003_0000, &[max], Dtype::I64),
            block_table: view(0x7004_0000, &[max, BLOCK_COLUMNS], Dtype::I32),
            logits: view(0x7100_0000, &[max, dims.vocab], Dtype::F32),
            softmax_lse: view(0x7200_0000, &[shape.softmax_lse_len()], Dtype::F32),
            lse_accum: view(
                0x7300_0000,
                &[shape.lse_accum_len(plan(1).num_splits)],
                Dtype::F32,
            ),
            o_accum: view(
                0x7400_0000,
                &[shape.o_accum_len(plan(1).num_splits)],
                Dtype::F32,
            ),
            rotary: RotaryTensors {
                cos: view(
                    0x7500_0000,
                    &[dims.rope.max_position, dims.head_dim / 2],
                    Dtype::F32,
                ),
                sin: view(
                    0x7600_0000,
                    &[dims.rope.max_position, dims.head_dim / 2],
                    Dtype::F32,
                ),
            },
        }
    }

    #[test]
    fn a_buckets_statics_are_the_leading_rows_of_each_fixed_buffer() {
        let dims = dims();
        let statics = statics(&dims);
        let one = plan(1);

        let bucket = BucketStatics::resolve(&statics, &one, &dims).unwrap();

        assert_eq!(bucket.token_ids.address(), statics.token_ids.address());
        assert_eq!(bucket.token_ids.dims(), [1]);
        assert_eq!(bucket.block_table.dims(), [1, BLOCK_COLUMNS]);
        assert!(bucket.block_table.is_contiguous());
        assert_eq!(bucket.logits.dims(), [1, dims.vocab]);
        assert_eq!(bucket.softmax_lse.dims(), [one.shape().softmax_lse_len()]);
        assert_eq!(
            bucket.lse_accum.dims(),
            [one.shape().lse_accum_len(one.num_splits)]
        );
        assert_eq!(
            bucket.o_accum.dims(),
            [one.shape().o_accum_len(one.num_splits)]
        );

        let full = BucketStatics::resolve(&statics, &plan(8), &dims).unwrap();
        assert_eq!(full.logits.dims(), [8, dims.vocab]);
        assert_eq!(full.slot_mapping.dims(), [8]);
    }

    #[test]
    fn a_static_shorter_than_the_bucket_is_refused_with_the_rows_it_holds() {
        let dims = dims();
        let mut statics = statics(&dims);
        statics.positions = view(0x7001_0000, &[4], Dtype::I32);

        assert_eq!(
            BucketStatics::resolve(&statics, &plan(8), &dims).unwrap_err(),
            SlotError::StaticTooShort {
                operand: Operand::model("the positions"),
                len: 4,
                needed: 8
            }
        );
    }

    #[test]
    fn a_scalar_static_is_refused_by_shape_rather_than_indexed() {
        let dims = dims();
        let mut statics = statics(&dims);
        statics.seqlens_k = view(0x7002_0000, &[], Dtype::I32);

        assert_eq!(
            BucketStatics::resolve(&statics, &plan(8), &dims).unwrap_err(),
            SlotError::Shape {
                operand: Operand::model("the key lengths"),
                shape: Shape::new(&[]),
                expected: Shape::new(&[0])
            }
        );
    }

    #[test]
    fn a_block_table_narrower_than_the_plan_is_refused_with_both_shapes() {
        let dims = dims();
        let mut statics = statics(&dims);
        statics.block_table = view(0x7004_0000, &[8, 16], Dtype::I32);

        assert_eq!(
            BucketStatics::resolve(&statics, &plan(8), &dims).unwrap_err(),
            SlotError::Shape {
                operand: Operand::model("the block table"),
                shape: Shape::new(&[8, 16]),
                expected: Shape::new(&[8, BLOCK_COLUMNS])
            }
        );
    }

    #[test]
    fn a_bucket_resolves_whole_and_refuses_a_plan_for_another_bucket() {
        let dims = dims();
        let arena = arena();
        let memory = memory(&arena);
        let statics = statics(&dims);
        let sources = SlotSources {
            memory: &memory,
            arena: &arena,
            statics: &statics,
            dims: &dims,
        };

        let slots = BucketSlots::resolve(&sources, bucket(1), plan(8)).unwrap();
        assert_eq!(slots.bucket, bucket(1));
        assert_eq!(slots.plan.bucket, 8);
        assert_eq!(slots.activations.rows(), LAYERS + 1);
        assert_eq!(slots.statics.token_ids.dims(), [8]);

        assert_eq!(
            BucketSlots::resolve(&sources, bucket(1), plan(1)).unwrap_err(),
            SlotError::PlanBucket { plan: 1, tokens: 8 }
        );
    }
}
