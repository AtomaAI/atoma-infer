//! cuBLAS GEMMs over the raw handle, bound to the stream of each call.
//!
//! The handle is created unbound and every GEMM binds it to the stream it was handed before
//! enqueueing, so the raw capture stream reaches cuBLAS only through the descriptor seam. The
//! first eager warmup run lets cuBLAS allocate its workspace and select algorithms before any
//! capture begins — a lazy workspace allocation inside a captured region is one of the predicted
//! capture killers.

use std::ffi::c_void;

use anyhow::{anyhow, Result};
use cudarc::cublas::{result as cublas, sys};

/// Raw cuBLAS handle whose GEMMs enqueue on the stream given per call.
pub struct StepBlas {
    handle: sys::cublasHandle_t,
}

impl StepBlas {
    /// Creates the handle, bound to no stream until the first GEMM.
    pub fn new() -> Result<Self> {
        let handle = cublas::create_handle().map_err(|e| anyhow!("cublasCreate: {e:?}"))?;
        Ok(Self { handle })
    }

    /// `y = x @ w^T` for row-major bf16 operands, enqueued on `stream`: `x` is
    /// `[n_tokens, in_features]`, `w` is `[out_features, in_features]`, `y` is
    /// `[n_tokens, out_features]`, accumulated in f32.
    ///
    /// In cuBLAS column-major terms that is `Y(out×n) = W_col(in×out)^T · X_col(in×n)`.
    ///
    /// # Safety
    /// `stream` must be a live stream on the handle's device, and all pointers must be live
    /// device addresses of the documented shapes.
    #[allow(clippy::too_many_arguments)] // three raw operands and three dims; a struct would only rename them
    pub unsafe fn gemm_xwt(
        &self,
        stream: cudarc::driver::sys::CUstream,
        w: cudarc::driver::sys::CUdeviceptr,
        x: cudarc::driver::sys::CUdeviceptr,
        y: cudarc::driver::sys::CUdeviceptr,
        out_features: usize,
        n_tokens: usize,
        in_features: usize,
    ) -> Result<()> {
        unsafe { cublas::set_stream(self.handle, stream.cast()) }
            .map_err(|e| anyhow!("cublasSetStream: {e:?}"))?;
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let m = i32::try_from(out_features)?;
        let n = i32::try_from(n_tokens)?;
        let k = i32::try_from(in_features)?;
        unsafe {
            cublas::gemm_ex(
                self.handle,
                sys::cublasOperation_t::CUBLAS_OP_T,
                sys::cublasOperation_t::CUBLAS_OP_N,
                m,
                n,
                k,
                (&raw const alpha).cast(),
                w as *const c_void,
                sys::cudaDataType_t::CUDA_R_16BF,
                k,
                x as *const c_void,
                sys::cudaDataType_t::CUDA_R_16BF,
                k,
                (&raw const beta).cast(),
                y as *mut c_void,
                sys::cudaDataType_t::CUDA_R_16BF,
                m,
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
        }
        .map_err(|e| anyhow!("cublasGemmEx {out_features}x{in_features}: {e:?}"))
    }
}

impl Drop for StepBlas {
    fn drop(&mut self) {
        // Best-effort: the process is exiting or the harness is tearing down after every graph
        // is gone; a destroy failure has nothing actionable left.
        let _ = unsafe { cublas::destroy_handle(self.handle) };
    }
}
