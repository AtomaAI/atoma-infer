//! The decode step's own kernels over checked tensors: embedding gather, `RMSNorm`, the rotary
//! embedding, the SiLU-gated multiply and the residual add.
//!
//! The launchers in `atoma_kernels` take raw addresses and counts, and trust them. What sits in
//! front of them here reads every count off a tensor view instead, after holding the view to
//! what the kernel assumes: the dtype it reads, the rank it indexes, one unbroken row-major
//! buffer, and the widths this model's dimensions fix. A call that passes is a plain value the
//! launcher takes as it stands; one that does not is refused by the operand's name and both
//! numbers, before any address reaches a kernel.

use core::ffi::c_void;

use atoma_kernels::decode_ops::{AddCall, EmbeddingGatherCall, RmsNormCall, RopeCall, SiluMulCall};
use atoma_runtime::tensor::{Dtype, Tensor};

use crate::dims::LlamaDims;
use crate::operand::{matrix, rows, vector, vector_len, Operand, OperandError, OperandKind};

/// The rotary embedding's tables on the device, f32 `[max_position, head_dim / 2]` each.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RotaryTensors {
    pub cos: Tensor,
    pub sin: Tensor,
}

/// The decode kernels, shaped by one model's dimensions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecodeKernels {
    dims: LlamaDims,
}

impl DecodeKernels {
    #[must_use]
    pub fn new(dims: LlamaDims) -> Self {
        Self { dims }
    }

    /// One embedding row per token: `out[t] = table[token_ids[t]]`.
    ///
    /// # Errors
    ///
    /// Returns [`OperandError`] when the table is not the vocabulary by the hidden size, the
    /// token ids are not a u32 vector, or the output is not one hidden row per token.
    pub fn embedding_gather(
        &self,
        table: &Tensor,
        token_ids: &Tensor,
        out: &Tensor,
        stream: *mut c_void,
    ) -> Result<EmbeddingGatherCall, OperandError> {
        let hidden = self.dims.hidden;
        matrix(
            Operand::model(OperandKind::EmbeddingTable),
            table.layout(),
            Dtype::Bf16,
            self.dims.vocab,
            hidden,
        )?;
        let tokens = vector_len(
            Operand::model(OperandKind::TokenIds),
            token_ids.layout(),
            Dtype::U32,
        )?;
        matrix(
            Operand::model(OperandKind::GatheredRows),
            out.layout(),
            Dtype::Bf16,
            tokens,
            hidden,
        )?;
        Ok(EmbeddingGatherCall {
            table: table.address(),
            token_ids: token_ids.address(),
            out: out.address(),
            hidden,
            n_tokens: tokens,
            stream,
        })
    }

    /// Each token row normalized by its root mean square and scaled by the gain, at the model's
    /// epsilon.
    ///
    /// # Errors
    ///
    /// Returns [`OperandError`] when the rows are not the hidden width, the gain is not one
    /// hidden vector, or the output is not the rows' shape.
    pub fn rmsnorm(
        &self,
        x: &Tensor,
        gain: &Tensor,
        out: &Tensor,
        stream: *mut c_void,
    ) -> Result<RmsNormCall, OperandError> {
        let hidden = self.dims.hidden;
        let tokens = rows(
            Operand::model(OperandKind::InputRows),
            x.layout(),
            Dtype::Bf16,
            hidden,
        )?;
        vector(
            Operand::model(OperandKind::Gain),
            gain.layout(),
            Dtype::Bf16,
            hidden,
        )?;
        matrix(
            Operand::model(OperandKind::NormalizedRows),
            out.layout(),
            Dtype::Bf16,
            tokens,
            hidden,
        )?;
        Ok(RmsNormCall {
            x: x.address(),
            gain: gain.address(),
            out: out.address(),
            hidden,
            n_tokens: tokens,
            eps: self.dims.rms_eps,
            stream,
        })
    }

    /// The rotary embedding over the query and key heads of each fused row, in place, at each
    /// token's position.
    ///
    /// # Errors
    ///
    /// Returns [`OperandError`] when the rows are not the fused width, the positions are not one
    /// i32 per row, or a table does not cover every position with one f32 per rotary pair.
    pub fn rope(
        &self,
        qkv: &Tensor,
        positions: &Tensor,
        tables: &RotaryTensors,
        stream: *mut c_void,
    ) -> Result<RopeCall, OperandError> {
        let dims = &self.dims;
        let tokens = rows(
            Operand::model(OperandKind::FusedRows),
            qkv.layout(),
            Dtype::Bf16,
            dims.qkv_width(),
        )?;
        vector(
            Operand::model(OperandKind::Positions),
            positions.layout(),
            Dtype::I32,
            tokens,
        )?;
        let (max_position, pairs) = (dims.rope.max_position, dims.head_dim / 2);
        matrix(
            Operand::model(OperandKind::CosineTable),
            tables.cos.layout(),
            Dtype::F32,
            max_position,
            pairs,
        )?;
        matrix(
            Operand::model(OperandKind::SineTable),
            tables.sin.layout(),
            Dtype::F32,
            max_position,
            pairs,
        )?;
        Ok(RopeCall {
            qkv: qkv.address(),
            positions: positions.address(),
            cos_table: tables.cos.address(),
            sin_table: tables.sin.address(),
            n_tokens: tokens,
            rot_heads: dims.rotary_heads(),
            head_dim: dims.head_dim,
            row_width: qkv.stride(0),
            stream,
        })
    }

    /// `out = silu(gate) * up` over the feed-forward rows.
    ///
    /// # Errors
    ///
    /// Returns [`OperandError`] when an operand is not one feed-forward row per token, or the
    /// three do not hold the same number of rows.
    pub fn silu_mul(
        &self,
        gate: &Tensor,
        up: &Tensor,
        out: &Tensor,
        stream: *mut c_void,
    ) -> Result<SiluMulCall, OperandError> {
        let ffn = self.dims.ffn;
        let tokens = rows(
            Operand::model(OperandKind::Gate),
            gate.layout(),
            Dtype::Bf16,
            ffn,
        )?;
        matrix(
            Operand::model(OperandKind::UpProjection),
            up.layout(),
            Dtype::Bf16,
            tokens,
            ffn,
        )?;
        matrix(
            Operand::model(OperandKind::Activation),
            out.layout(),
            Dtype::Bf16,
            tokens,
            ffn,
        )?;
        Ok(SiluMulCall {
            gate: gate.address(),
            up: up.address(),
            out: out.address(),
            len: tokens * ffn,
            stream,
        })
    }

    /// `out = residual + delta` over the hidden rows.
    ///
    /// # Errors
    ///
    /// Returns [`OperandError`] when an operand is not one hidden row per token, or the three do
    /// not hold the same number of rows.
    pub fn add(
        &self,
        residual: &Tensor,
        delta: &Tensor,
        out: &Tensor,
        stream: *mut c_void,
    ) -> Result<AddCall, OperandError> {
        let hidden = self.dims.hidden;
        let tokens = rows(
            Operand::model(OperandKind::Residual),
            residual.layout(),
            Dtype::Bf16,
            hidden,
        )?;
        matrix(
            Operand::model(OperandKind::Delta),
            delta.layout(),
            Dtype::Bf16,
            tokens,
            hidden,
        )?;
        matrix(
            Operand::model(OperandKind::Sum),
            out.layout(),
            Dtype::Bf16,
            tokens,
            hidden,
        )?;
        Ok(AddCall {
            lhs: residual.address(),
            rhs: delta.address(),
            out: out.address(),
            len: tokens * hidden,
            stream,
        })
    }
}

#[cfg(test)]
mod tests {
    use core::ptr;

    use atoma_runtime::tensor::Layout;

    use super::*;
    use crate::dims::test_support::llama_8b;
    use crate::operand::Shape;

    const TOKENS: usize = 8;

    /// A view at `address` of a contiguous layout; the checks under test need no device.
    fn view(address: u64, dims: &[usize], dtype: Dtype) -> Tensor {
        Tensor::for_test(address, Layout::contiguous(dims, dtype).unwrap()).unwrap()
    }

    fn kernels() -> DecodeKernels {
        DecodeKernels::new(llama_8b(2))
    }

    fn tables(dims: &LlamaDims) -> RotaryTensors {
        let shape = [dims.rope.max_position, dims.head_dim / 2];
        RotaryTensors {
            cos: view(0xA0_0000, &shape, Dtype::F32),
            sin: view(0xB0_0000, &shape, Dtype::F32),
        }
    }

    #[test]
    fn the_gather_reads_the_vocabulary_table_and_writes_one_row_per_token() {
        let dims = llama_8b(2);
        let table = view(0x10_0000, &[dims.vocab, dims.hidden], Dtype::Bf16);
        let ids = view(0x20_0000, &[TOKENS], Dtype::U32);
        let out = view(0x30_0000, &[TOKENS, dims.hidden], Dtype::Bf16);

        let call = kernels()
            .embedding_gather(&table, &ids, &out, ptr::null_mut())
            .unwrap();

        assert_eq!(call.table, 0x10_0000);
        assert_eq!(call.token_ids, 0x20_0000);
        assert_eq!(call.out, 0x30_0000);
        assert_eq!(call.hidden, dims.hidden);
        assert_eq!(call.n_tokens, TOKENS);
    }

    #[test]
    fn token_ids_that_are_not_u32_are_refused_with_what_the_kernel_reads() {
        let dims = llama_8b(2);
        let table = view(0x10_0000, &[dims.vocab, dims.hidden], Dtype::Bf16);
        let ids = view(0x20_0000, &[TOKENS], Dtype::I64);
        let out = view(0x30_0000, &[TOKENS, dims.hidden], Dtype::Bf16);

        let refused = kernels()
            .embedding_gather(&table, &ids, &out, ptr::null_mut())
            .unwrap_err();

        assert_eq!(
            refused,
            OperandError::Dtype {
                operand: Operand::model(OperandKind::TokenIds),
                dtype: Dtype::I64,
                expected: Dtype::U32
            }
        );
        assert!(refused.to_string().contains("token ids"));
        assert!(refused.to_string().contains("U32"));
    }

    #[test]
    fn a_table_that_is_not_this_models_vocabulary_is_refused_with_both_shapes() {
        let dims = llama_8b(2);
        let table = view(0x10_0000, &[32_000, dims.hidden], Dtype::Bf16);
        let ids = view(0x20_0000, &[TOKENS], Dtype::U32);
        let out = view(0x30_0000, &[TOKENS, dims.hidden], Dtype::Bf16);

        assert_eq!(
            kernels()
                .embedding_gather(&table, &ids, &out, ptr::null_mut())
                .unwrap_err(),
            OperandError::Shape {
                operand: Operand::model(OperandKind::EmbeddingTable),
                shape: Shape::new(&[32_000, dims.hidden]),
                expected: Shape::new(&[dims.vocab, dims.hidden])
            }
        );
    }

    #[test]
    fn the_norm_takes_the_models_epsilon_and_one_gain_per_column() {
        let dims = llama_8b(2);
        let x = view(0x10_0000, &[TOKENS, dims.hidden], Dtype::Bf16);
        let gain = view(0x20_0000, &[dims.hidden], Dtype::Bf16);
        let out = view(0x30_0000, &[TOKENS, dims.hidden], Dtype::Bf16);

        let call = kernels().rmsnorm(&x, &gain, &out, ptr::null_mut()).unwrap();

        assert_eq!(call.x, 0x10_0000);
        assert_eq!(call.gain, 0x20_0000);
        assert_eq!(call.out, 0x30_0000);
        assert_eq!(call.hidden, dims.hidden);
        assert_eq!(call.n_tokens, TOKENS);
        assert!((call.eps - dims.rms_eps).abs() < f32::EPSILON);
    }

    #[test]
    fn a_gain_given_as_a_matrix_is_refused_by_rank() {
        let dims = llama_8b(2);
        let x = view(0x10_0000, &[TOKENS, dims.hidden], Dtype::Bf16);
        let gain = view(0x20_0000, &[1, dims.hidden], Dtype::Bf16);
        let out = view(0x30_0000, &[TOKENS, dims.hidden], Dtype::Bf16);

        assert_eq!(
            kernels()
                .rmsnorm(&x, &gain, &out, ptr::null_mut())
                .unwrap_err(),
            OperandError::Rank {
                operand: Operand::model(OperandKind::Gain),
                rank: 2,
                expected: 1
            }
        );
    }

    #[test]
    fn an_output_narrower_than_the_rows_is_refused_with_the_row_count_it_must_match() {
        let dims = llama_8b(2);
        let x = view(0x10_0000, &[TOKENS, dims.hidden], Dtype::Bf16);
        let gain = view(0x20_0000, &[dims.hidden], Dtype::Bf16);
        let out = view(0x30_0000, &[TOKENS - 1, dims.hidden], Dtype::Bf16);

        assert_eq!(
            kernels()
                .rmsnorm(&x, &gain, &out, ptr::null_mut())
                .unwrap_err(),
            OperandError::Shape {
                operand: Operand::model(OperandKind::NormalizedRows),
                shape: Shape::new(&[TOKENS - 1, dims.hidden]),
                expected: Shape::new(&[TOKENS, dims.hidden])
            }
        );
    }

    #[test]
    fn the_rotation_turns_the_query_and_key_heads_a_fused_row_apart() {
        let dims = llama_8b(2);
        let qkv = view(0x10_0000, &[TOKENS, dims.qkv_width()], Dtype::Bf16);
        let positions = view(0x20_0000, &[TOKENS], Dtype::I32);
        let tables = tables(&dims);

        let call = kernels()
            .rope(&qkv, &positions, &tables, ptr::null_mut())
            .unwrap();

        assert_eq!(call.qkv, 0x10_0000);
        assert_eq!(call.positions, 0x20_0000);
        assert_eq!(call.cos_table, 0xA0_0000);
        assert_eq!(call.sin_table, 0xB0_0000);
        assert_eq!(call.n_tokens, TOKENS);
        assert_eq!(call.rot_heads, 40, "32 query heads and 8 key heads");
        assert_eq!(call.head_dim, 128);
        assert_eq!(call.row_width, dims.qkv_width());
    }

    #[test]
    fn a_column_view_of_the_fused_row_cannot_be_rotated() {
        let dims = llama_8b(2);
        let qkv = view(0x10_0000, &[TOKENS, dims.qkv_width()], Dtype::Bf16);
        let q_only = qkv.narrow(1, 0, dims.q_width()).unwrap();
        let positions = view(0x20_0000, &[TOKENS], Dtype::I32);

        let refused = kernels()
            .rope(&q_only, &positions, &tables(&dims), ptr::null_mut())
            .unwrap_err();

        assert_eq!(
            refused,
            OperandError::NotContiguous {
                operand: Operand::model(OperandKind::FusedRows),
                strides: [dims.qkv_width(), 1, 0, 0]
            }
        );
        assert!(refused.to_string().contains("6144"));
    }

    #[test]
    fn positions_must_be_one_per_row() {
        let dims = llama_8b(2);
        let qkv = view(0x10_0000, &[TOKENS, dims.qkv_width()], Dtype::Bf16);
        let positions = view(0x20_0000, &[TOKENS - 2], Dtype::I32);

        assert_eq!(
            kernels()
                .rope(&qkv, &positions, &tables(&dims), ptr::null_mut())
                .unwrap_err(),
            OperandError::Length {
                operand: Operand::model(OperandKind::Positions),
                len: TOKENS - 2,
                expected: TOKENS
            }
        );
    }

    #[test]
    fn a_table_that_does_not_cover_every_position_is_refused() {
        let dims = llama_8b(2);
        let qkv = view(0x10_0000, &[TOKENS, dims.qkv_width()], Dtype::Bf16);
        let positions = view(0x20_0000, &[TOKENS], Dtype::I32);
        let mut tables = tables(&dims);
        tables.sin = view(0xB0_0000, &[4096, dims.head_dim / 2], Dtype::F32);

        assert_eq!(
            kernels()
                .rope(&qkv, &positions, &tables, ptr::null_mut())
                .unwrap_err(),
            OperandError::Shape {
                operand: Operand::model(OperandKind::SineTable),
                shape: Shape::new(&[4096, 64]),
                expected: Shape::new(&[dims.rope.max_position, 64])
            }
        );
    }

    #[test]
    fn the_gated_multiply_runs_over_every_element_of_the_feed_forward_rows() {
        let dims = llama_8b(2);
        let gate = view(0x10_0000, &[TOKENS, dims.ffn], Dtype::Bf16);
        let up = view(0x20_0000, &[TOKENS, dims.ffn], Dtype::Bf16);
        let out = view(0x30_0000, &[TOKENS, dims.ffn], Dtype::Bf16);

        let call = kernels()
            .silu_mul(&gate, &up, &out, ptr::null_mut())
            .unwrap();

        assert_eq!(call.gate, 0x10_0000);
        assert_eq!(call.up, 0x20_0000);
        assert_eq!(call.out, 0x30_0000);
        assert_eq!(call.len, TOKENS * dims.ffn);
    }

    #[test]
    fn a_hidden_row_cannot_stand_in_for_a_feed_forward_row() {
        let dims = llama_8b(2);
        let gate = view(0x10_0000, &[TOKENS, dims.ffn], Dtype::Bf16);
        let up = view(0x20_0000, &[TOKENS, dims.hidden], Dtype::Bf16);
        let out = view(0x30_0000, &[TOKENS, dims.ffn], Dtype::Bf16);

        assert_eq!(
            kernels()
                .silu_mul(&gate, &up, &out, ptr::null_mut())
                .unwrap_err(),
            OperandError::Shape {
                operand: Operand::model(OperandKind::UpProjection),
                shape: Shape::new(&[TOKENS, dims.hidden]),
                expected: Shape::new(&[TOKENS, dims.ffn])
            }
        );
    }

    #[test]
    fn the_residual_add_runs_over_every_element_of_the_hidden_rows() {
        let dims = llama_8b(2);
        let residual = view(0x10_0000, &[TOKENS, dims.hidden], Dtype::Bf16);
        let delta = view(0x20_0000, &[TOKENS, dims.hidden], Dtype::Bf16);
        let out = view(0x30_0000, &[TOKENS, dims.hidden], Dtype::Bf16);

        let call = kernels()
            .add(&residual, &delta, &out, ptr::null_mut())
            .unwrap();

        assert_eq!(call.lhs, 0x10_0000);
        assert_eq!(call.rhs, 0x20_0000);
        assert_eq!(call.out, 0x30_0000);
        assert_eq!(call.len, TOKENS * dims.hidden);
    }

    #[test]
    fn an_f32_operand_is_refused_where_the_kernel_reads_bf16() {
        let dims = llama_8b(2);
        let residual = view(0x10_0000, &[TOKENS, dims.hidden], Dtype::Bf16);
        let delta = view(0x20_0000, &[TOKENS, dims.hidden], Dtype::F32);
        let out = view(0x30_0000, &[TOKENS, dims.hidden], Dtype::Bf16);

        assert_eq!(
            kernels()
                .add(&residual, &delta, &out, ptr::null_mut())
                .unwrap_err(),
            OperandError::Dtype {
                operand: Operand::model(OperandKind::Delta),
                dtype: Dtype::F32,
                expected: Dtype::Bf16
            }
        );
    }
}
