//! cuBLAS GEMMs over the raw handle, bound to the capture stream in Allocation.
//!
//! The handle is bound once, through the session's Allocation-only bind seam, and every GEMM
//! enqueues on that stream; nothing rebinds it after capture has begun. The first eager warmup
//! run lets cuBLAS allocate its workspace and select algorithms before any capture begins — a
//! lazy workspace allocation inside a captured region is one of the predicted capture killers.

use std::ffi::c_void;

use anyhow::{anyhow, Result};
use atoma_runtime::stream::CaptureStream;
use cudarc::cublas::{result as cublas, sys};

/// Raw cuBLAS handle bound to the capture stream.
pub struct StepBlas {
    handle: sys::cublasHandle_t,
}

/// One `y = x @ w^T` GEMM over row-major bf16 operands, accumulated in f32.
#[derive(Debug, Clone, Copy)]
pub struct GemmXwt {
    /// `[out_features, in_features]`.
    pub w: u64,
    /// `[n_tokens, in_features]`.
    pub x: u64,
    /// `[n_tokens, out_features]`.
    pub y: u64,
    pub out_features: usize,
    pub n_tokens: usize,
    pub in_features: usize,
}

impl StepBlas {
    /// Creates the handle and binds it to the capture stream. Only the session's Allocation
    /// phase can hand the stream over, so the binding cannot happen after capture has begun.
    pub fn new(stream: &CaptureStream) -> Result<Self> {
        let handle = cublas::create_handle().map_err(|e| anyhow!("cublasCreate: {e:?}"))?;
        let blas = Self { handle };
        stream.bind(|raw| {
            // SAFETY: `handle` is the live handle just created, and `raw` is the live capture
            // stream on its device; cuBLAS stores the stream and nothing else retains it.
            unsafe { cublas::set_stream(handle, raw.cast()) }
                .map_err(|e| anyhow!("cublasSetStream: {e:?}"))
        })?;
        Ok(blas)
    }

    /// Enqueues `gemm` on the bound capture stream. In cuBLAS column-major terms that is
    /// `Y(out×n) = W_col(in×out)^T · X_col(in×n)`.
    ///
    /// # Safety
    /// Every address in `gemm` must be a live device address of the documented shape.
    pub unsafe fn gemm_xwt(&self, gemm: &GemmXwt) -> Result<()> {
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let m = i32::try_from(gemm.out_features)?;
        let n = i32::try_from(gemm.n_tokens)?;
        let k = i32::try_from(gemm.in_features)?;
        // SAFETY: the caller's contract on the operands; `alpha` and `beta` outlive the call and
        // the handle is live for as long as `self` is.
        unsafe {
            cublas::gemm_ex(
                self.handle,
                sys::cublasOperation_t::CUBLAS_OP_T,
                sys::cublasOperation_t::CUBLAS_OP_N,
                m,
                n,
                k,
                (&raw const alpha).cast(),
                gemm.w as *const c_void,
                sys::cudaDataType_t::CUDA_R_16BF,
                k,
                gemm.x as *const c_void,
                sys::cudaDataType_t::CUDA_R_16BF,
                k,
                (&raw const beta).cast(),
                gemm.y as *mut c_void,
                sys::cudaDataType_t::CUDA_R_16BF,
                m,
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
        }
        .map_err(|e| {
            anyhow!(
                "cublasGemmEx {}x{}: {e:?}",
                gemm.out_features,
                gemm.in_features
            )
        })
    }
}

impl Drop for StepBlas {
    fn drop(&mut self) {
        // Best-effort: the process is exiting or the harness is tearing down after every graph
        // is gone; a destroy failure has nothing actionable left.
        // SAFETY: the handle is live until here and nothing uses it afterwards.
        let _ = unsafe { cublas::destroy_handle(self.handle) };
    }
}
