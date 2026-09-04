//! The step's matrix multiplications: `y = x · wᵀ` over bf16 operands, through a cuBLAS handle
//! bound to the capture stream.
//!
//! A weight is stored `[out_features, in_features]` and a row of activations `[tokens,
//! in_features]`, which is the transposed-A case cuBLAS takes directly, with no copy and no
//! transposition of the weight. The shapes and leading dimensions come from the operands' tensor
//! layouts rather than from numbers spelled out at each call site, so a projection writing into a
//! column range of the fused qkv row is the same call as one writing a whole buffer: only the
//! output's row stride differs.
//!
//! The handle is bound once, in Allocation, through the session's bind seam, and is handed an
//! explicit workspace at the same time. cuBLAS allocates a workspace on first use otherwise, and
//! an allocation inside a captured region invalidates the capture.

use core::ffi::c_void;
use core::fmt;

use atoma_runtime::session::Allocation;
use atoma_runtime::tensor::{Dtype, Layout, Tensor};
use cudarc::cublas::{result as cublas, sys};
use cudarc::driver::{CudaSlice, DevicePtr};
use thiserror::Error;

use crate::operand::{self, Operand, OperandError, OperandKind};

/// Bytes of workspace the handle is given. cuBLAS documents 32 MiB as the size that lets Hopper
/// choose among all its kernels; it is allocated once, before any capture.
pub const WORKSPACE_BYTES: usize = 32 * 1024 * 1024;

/// Why a multiplication could not be shaped or enqueued.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum GemmError {
    #[error(transparent)]
    Operand(#[from] OperandError),
    #[error("the weight's {weight_in} input features are not the activations' {input_in}")]
    InputMismatch { weight_in: usize, input_in: usize },
    #[error("the output is {rows} by {columns}, not {tokens} by {out_features}")]
    OutputShape {
        rows: usize,
        columns: usize,
        tokens: usize,
        out_features: usize,
    },
    #[error("the output is {dtype:?}; it must be bf16 or f32")]
    OutputDtype { dtype: Dtype },
    #[error("{dimension} of {value} does not fit the dimension cuBLAS takes")]
    DimensionOverflow {
        dimension: GemmDimension,
        value: usize,
    },
    #[error("cuBLAS refused the call: {status:?}")]
    Cublas { status: sys::cublasStatus_t },
}

impl From<cublas::CublasError> for GemmError {
    fn from(error: cublas::CublasError) -> Self {
        Self::Cublas { status: error.0 }
    }
}

/// The element type a multiplication writes: bf16 for an activation, f32 for logits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmOutput {
    Bf16,
    F32,
}

impl GemmOutput {
    /// The cuBLAS data type of the output.
    fn data_type(self) -> sys::cudaDataType {
        match self {
            GemmOutput::Bf16 => sys::cudaDataType::CUDA_R_16BF,
            GemmOutput::F32 => sys::cudaDataType::CUDA_R_32F,
        }
    }
}

/// One `y = x · wᵀ` multiplication, as cuBLAS takes it.
///
/// In column-major terms the same call reads `Y(out × tokens) = W(in × out)ᵀ · X(in × tokens)`:
/// a row-major matrix with row stride `s` is the column-major transpose with leading dimension
/// `s`, so no operand is copied or transposed in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmShape {
    /// Rows of the weight: the features the multiplication produces per token.
    pub out_features: usize,
    /// Tokens in the batch.
    pub tokens: usize,
    /// Columns of the weight: the features it consumes per token.
    pub in_features: usize,
    /// Leading dimension of the weight, its row stride in elements.
    pub weight_stride: usize,
    /// Leading dimension of the activations, their row stride in elements.
    pub input_stride: usize,
    /// Leading dimension of the output, its row stride in elements. A projection writing into a
    /// column range of a fused row leaves this at the fused row's width.
    pub output_stride: usize,
    pub output: GemmOutput,
}

impl GemmShape {
    /// The shape of `y = x · wᵀ` for operands of these layouts.
    ///
    /// Layouts rather than tensors: what cuBLAS is told is shape and stride arithmetic, and the
    /// addresses are handed to [`StepBlas::enqueue`] alongside it.
    ///
    /// # Errors
    ///
    /// Returns [`GemmError`] when an operand is not a matrix, its last dimension is not
    /// contiguous, the weight's input features are not the activations', the output does not
    /// hold the result, or a dtype is not one the multiplication takes.
    pub fn x_wt(x: &Layout, w: &Layout, y: &Layout) -> Result<Self, GemmError> {
        for (operand, layout) in [
            (OperandKind::Activations, x),
            (OperandKind::Weight, w),
            (OperandKind::Output, y),
        ] {
            let operand = Operand::model(operand);
            operand::rank(operand, layout, 2)?;
            operand::inner_contiguous(operand, layout)?;
        }
        for (operand, layout) in [(OperandKind::Activations, x), (OperandKind::Weight, w)] {
            operand::dtype(Operand::model(operand), layout, Dtype::Bf16)?;
        }
        let output = match y.dtype() {
            Dtype::Bf16 => GemmOutput::Bf16,
            Dtype::F32 => GemmOutput::F32,
            dtype => return Err(GemmError::OutputDtype { dtype }),
        };
        let (tokens, input_in) = (x.dim(0), x.dim(1));
        let (out_features, weight_in) = (w.dim(0), w.dim(1));
        if weight_in != input_in {
            return Err(GemmError::InputMismatch {
                weight_in,
                input_in,
            });
        }
        if y.dim(0) != tokens || y.dim(1) != out_features {
            return Err(GemmError::OutputShape {
                rows: y.dim(0),
                columns: y.dim(1),
                tokens,
                out_features,
            });
        }
        Ok(Self {
            out_features,
            tokens,
            in_features: input_in,
            weight_stride: w.stride(0),
            input_stride: x.stride(0),
            output_stride: y.stride(0),
            output,
        })
    }
}

/// One of the six dimensions cuBLAS takes as a signed 32-bit integer, each naming the field of a
/// [`GemmShape`] it is read from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmDimension {
    /// `m`: the output features.
    OutFeatures,
    /// `n`: the tokens.
    Tokens,
    /// `k`: the input features.
    InFeatures,
    /// `lda`: the weight's row stride.
    WeightStride,
    /// `ldb`: the activations' row stride.
    InputStride,
    /// `ldc`: the output's row stride.
    OutputStride,
}

impl GemmDimension {
    /// This dimension of `shape`, as the argument cuBLAS takes.
    ///
    /// # Errors
    ///
    /// Returns [`GemmError::DimensionOverflow`] when the value does not fit a signed 32-bit
    /// integer.
    pub fn argument(self, shape: &GemmShape) -> Result<i32, GemmError> {
        let value = match self {
            GemmDimension::OutFeatures => shape.out_features,
            GemmDimension::Tokens => shape.tokens,
            GemmDimension::InFeatures => shape.in_features,
            GemmDimension::WeightStride => shape.weight_stride,
            GemmDimension::InputStride => shape.input_stride,
            GemmDimension::OutputStride => shape.output_stride,
        };
        i32::try_from(value).map_err(|_| GemmError::DimensionOverflow {
            dimension: self,
            value,
        })
    }
}

impl fmt::Display for GemmDimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            GemmDimension::OutFeatures => "the output feature count",
            GemmDimension::Tokens => "the token count",
            GemmDimension::InFeatures => "the input feature count",
            GemmDimension::WeightStride => "the weight's row stride",
            GemmDimension::InputStride => "the activations' row stride",
            GemmDimension::OutputStride => "the output's row stride",
        })
    }
}

/// The cuBLAS handle the step's multiplications run on, bound to the capture stream and holding
/// its own workspace.
pub struct StepBlas {
    handle: sys::cublasHandle_t,
    /// The workspace the handle was given; freed after the handle is destroyed.
    _workspace: CudaSlice<u8>,
}

impl StepBlas {
    /// Creates the handle, binds it to the session's capture stream and hands it `workspace`.
    ///
    /// Taking the Allocation phase is what fixes the order: the binding and the workspace are in
    /// place before anything is captured, so no multiplication reaches for memory of its own
    /// inside a recording.
    ///
    /// # Errors
    ///
    /// Returns [`GemmError::Cublas`] when the handle cannot be created, bound, or given its
    /// workspace.
    pub fn new(allocation: &Allocation, workspace: CudaSlice<u8>) -> Result<Self, GemmError> {
        let bytes = workspace.len();
        // The address is read before the buffer moves into the handle's keeping, and the read
        // guard is dropped with the block; device allocations do not move, so the address stays
        // the workspace's.
        let address = {
            let (address, _reads) = workspace.device_ptr(workspace.stream());
            address
        };
        let handle = cublas::create_handle()?;
        // Owned from here on, so a failure below still frees the handle through the destructor.
        let blas = Self {
            handle,
            _workspace: workspace,
        };
        allocation.stream().bind(|raw| {
            // SAFETY: `handle` is the handle just created and `raw` is the live capture stream;
            // cuBLAS stores the stream and retains nothing else.
            unsafe { cublas::set_stream(handle, raw.cast()) }?;
            Ok::<(), GemmError>(())
        })?;
        // SAFETY: `address` is the buffer this value owns and it holds `bytes` bytes; cuBLAS
        // keeps the pointer for the handle's lifetime, which ends first because the destructor
        // destroys the handle before the field is dropped.
        unsafe { sys::cublasSetWorkspace_v2(handle, address as *mut c_void, bytes) }.result()?;
        Ok(blas)
    }

    /// Enqueues `output = input · weightᵀ` on the bound capture stream, the shape read off the
    /// three views.
    ///
    /// # Errors
    ///
    /// Returns [`GemmError`] when the views do not describe one multiplication, a dimension does
    /// not fit cuBLAS, or the call is refused.
    ///
    /// # Safety
    ///
    /// Every view must be over live device memory.
    pub unsafe fn enqueue(
        &self,
        input: &Tensor,
        weight: &Tensor,
        output: &Tensor,
    ) -> Result<(), GemmError> {
        let shape = GemmShape::x_wt(input.layout(), weight.layout(), output.layout())?;
        let (weight, input, output) = (weight.address(), input.address(), output.address());
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let m = GemmDimension::OutFeatures.argument(&shape)?;
        let n = GemmDimension::Tokens.argument(&shape)?;
        let k = GemmDimension::InFeatures.argument(&shape)?;
        let lda = GemmDimension::WeightStride.argument(&shape)?;
        let ldb = GemmDimension::InputStride.argument(&shape)?;
        let ldc = GemmDimension::OutputStride.argument(&shape)?;
        // SAFETY: the caller's contract on the addresses; `alpha` and `beta` live across the
        // call, and the handle is live for as long as this value is.
        unsafe {
            cublas::gemm_ex(
                self.handle,
                sys::cublasOperation_t::CUBLAS_OP_T,
                sys::cublasOperation_t::CUBLAS_OP_N,
                m,
                n,
                k,
                (&raw const alpha).cast(),
                weight as *const c_void,
                sys::cudaDataType::CUDA_R_16BF,
                lda,
                input as *const c_void,
                sys::cudaDataType::CUDA_R_16BF,
                ldb,
                (&raw const beta).cast(),
                output as *mut c_void,
                shape.output.data_type(),
                ldc,
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
        }?;
        Ok(())
    }
}

impl Drop for StepBlas {
    fn drop(&mut self) {
        // The process is exiting or the session is being torn down; a destroy failure leaves
        // nothing to act on, and the workspace is freed after this by field order.
        // SAFETY: the handle is live until here and nothing uses it afterwards.
        let _ = unsafe { cublas::destroy_handle(self.handle) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A contiguous layout of `dims`.
    fn rows(dims: &[usize], dtype: Dtype) -> Layout {
        Layout::contiguous(dims, dtype).unwrap()
    }

    #[test]
    fn a_projection_reads_its_shape_and_strides_off_the_operands() {
        let x = rows(&[8, 4096], Dtype::Bf16);
        let w = rows(&[14336, 4096], Dtype::Bf16);
        let y = rows(&[8, 14336], Dtype::Bf16);

        let shape = GemmShape::x_wt(&x, &w, &y).unwrap();

        assert_eq!(
            shape,
            GemmShape {
                out_features: 14336,
                tokens: 8,
                in_features: 4096,
                weight_stride: 4096,
                input_stride: 4096,
                output_stride: 14336,
                output: GemmOutput::Bf16,
            }
        );
    }

    #[test]
    fn writing_into_a_column_range_leaves_the_output_stride_at_the_fused_width() {
        // The k projection writes columns 4096..5120 of the 6144-wide fused qkv row.
        let x = rows(&[8, 4096], Dtype::Bf16);
        let w = rows(&[1024, 4096], Dtype::Bf16);
        let qkv = rows(&[8, 6144], Dtype::Bf16);
        let (k_columns, offset) = qkv.narrow(1, 4096, 1024).unwrap();

        let shape = GemmShape::x_wt(&x, &w, &k_columns).unwrap();

        assert_eq!(shape.out_features, 1024);
        assert_eq!(
            shape.output_stride, 6144,
            "the column view's rows are still a fused row apart"
        );
        assert_eq!(
            offset,
            4096 * 2,
            "the column view starts at its first column"
        );
    }

    #[test]
    fn the_head_projection_writes_f32_from_bf16_operands() {
        let x = rows(&[64, 4096], Dtype::Bf16);
        let w = rows(&[128_256, 4096], Dtype::Bf16);
        let logits = rows(&[64, 128_256], Dtype::F32);

        let shape = GemmShape::x_wt(&x, &w, &logits).unwrap();

        assert_eq!(shape.output, GemmOutput::F32);
        assert_eq!(shape.output.data_type(), sys::cudaDataType::CUDA_R_32F);
    }

    #[test]
    fn an_output_dtype_the_multiplication_cannot_write_is_refused() {
        let x = rows(&[8, 4096], Dtype::Bf16);
        let w = rows(&[4096, 4096], Dtype::Bf16);
        let y = rows(&[8, 4096], Dtype::F16);

        assert_eq!(
            GemmShape::x_wt(&x, &w, &y).unwrap_err(),
            GemmError::OutputDtype { dtype: Dtype::F16 }
        );
    }

    #[test]
    fn operands_that_are_not_bf16_are_refused_by_name() {
        let x = rows(&[8, 4096], Dtype::F32);
        let w = rows(&[4096, 4096], Dtype::Bf16);
        let y = rows(&[8, 4096], Dtype::Bf16);

        assert_eq!(
            GemmShape::x_wt(&x, &w, &y).unwrap_err(),
            GemmError::Operand(OperandError::Dtype {
                operand: Operand::model(OperandKind::Activations),
                dtype: Dtype::F32,
                expected: Dtype::Bf16
            })
        );
    }

    #[test]
    fn a_weight_that_does_not_consume_the_activations_is_refused() {
        let x = rows(&[8, 4096], Dtype::Bf16);
        let w = rows(&[4096, 2048], Dtype::Bf16);
        let y = rows(&[8, 4096], Dtype::Bf16);

        assert_eq!(
            GemmShape::x_wt(&x, &w, &y).unwrap_err(),
            GemmError::InputMismatch {
                weight_in: 2048,
                input_in: 4096
            }
        );
    }

    #[test]
    fn an_output_that_does_not_hold_the_result_is_refused_with_both_shapes() {
        let x = rows(&[8, 4096], Dtype::Bf16);
        let w = rows(&[14336, 4096], Dtype::Bf16);
        let y = rows(&[8, 4096], Dtype::Bf16);

        assert_eq!(
            GemmShape::x_wt(&x, &w, &y).unwrap_err(),
            GemmError::OutputShape {
                rows: 8,
                columns: 4096,
                tokens: 8,
                out_features: 14336
            }
        );
    }

    #[test]
    fn a_non_matrix_operand_and_a_gapped_inner_dimension_are_refused() {
        let x = rows(&[8, 4096], Dtype::Bf16);
        let w = rows(&[14336, 4096], Dtype::Bf16);
        let vector = rows(&[4096], Dtype::Bf16);
        assert_eq!(
            GemmShape::x_wt(&vector, &w, &x).unwrap_err(),
            GemmError::Operand(OperandError::Rank {
                operand: Operand::model(OperandKind::Activations),
                rank: 1,
                expected: 2
            })
        );

        let gapped = Layout::strided(&[8, 4096], &[8192, 2], Dtype::Bf16).unwrap();
        assert_eq!(
            GemmShape::x_wt(&gapped, &w, &x).unwrap_err(),
            GemmError::Operand(OperandError::InnerStride {
                operand: Operand::model(OperandKind::Activations),
                stride: 2
            })
        );
    }

    /// A shape whose six dimensions all differ, so a dimension read from the wrong field shows.
    fn distinct() -> GemmShape {
        GemmShape {
            out_features: 1,
            tokens: 2,
            in_features: 3,
            weight_stride: 4,
            input_stride: 5,
            output_stride: 6,
            output: GemmOutput::Bf16,
        }
    }

    #[test]
    fn each_dimension_reads_its_own_field_of_the_shape() {
        let shape = distinct();
        assert_eq!(GemmDimension::OutFeatures.argument(&shape), Ok(1));
        assert_eq!(GemmDimension::Tokens.argument(&shape), Ok(2));
        assert_eq!(GemmDimension::InFeatures.argument(&shape), Ok(3));
        assert_eq!(GemmDimension::WeightStride.argument(&shape), Ok(4));
        assert_eq!(GemmDimension::InputStride.argument(&shape), Ok(5));
        assert_eq!(GemmDimension::OutputStride.argument(&shape), Ok(6));
    }

    #[test]
    fn a_dimension_past_the_cublas_limit_is_refused_by_name() {
        let shape = GemmShape {
            tokens: usize::MAX,
            ..distinct()
        };
        let refused = GemmDimension::Tokens.argument(&shape).unwrap_err();
        assert_eq!(
            refused,
            GemmError::DimensionOverflow {
                dimension: GemmDimension::Tokens,
                value: usize::MAX
            }
        );
        assert!(refused.to_string().contains("token count"));
    }
}
