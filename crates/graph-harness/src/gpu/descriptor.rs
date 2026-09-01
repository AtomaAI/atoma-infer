//! The harness's descriptor seam: one [`Descriptor`] implementation through which every enqueue
//! onto the capture stream passes, and the only place this backend's C-ABI kernels receive the
//! raw stream handle.

#[cfg(not(feature = "nccl"))]
use anyhow::bail;
use anyhow::{anyhow, Result};
use atoma_runtime::session::Descriptor;
use cudarc::driver::{result, sys, CudaSlice};
#[cfg(feature = "nccl")]
use cudarc::nccl::{Comm, ReduceOp};

use crate::gpu::step::{self, StagingPtrs, StepContext, StepPtrs};
use crate::layout::StaticSizes;
use crate::variation::StepInputs;

/// The pieces the per-layer all-reduce hook needs. Only all-reduce cells construct one, and only
/// the `nccl` build can run them.
pub struct AllReduce<'a> {
    #[cfg(feature = "nccl")]
    pub comm: &'a Comm,
    pub buffer: &'a mut CudaSlice<f32>,
    pub buffer_ptr: u64,
}

/// One enqueue of harness work onto the capture stream.
pub enum StepWork<'a> {
    /// H2D upload of one step's inputs into their persistent staging mirrors — the pre-work
    /// every step, replayed or eager, consumes. Pageable H2D is host-synchronous, so it is never
    /// part of a recording; the graph reads staging only through captured D2D nodes.
    Upload {
        staging: &'a StagingPtrs,
        inputs: &'a StepInputs,
    },
    /// The captured copy-in from staging plus the full decode step — what a recording bakes and
    /// what an eager pass runs.
    Decode {
        ctx: &'a StepContext<'a>,
        ptrs: &'a StepPtrs,
        staging: &'a StagingPtrs,
        sizes: &'a StaticSizes,
        all_reduce: Option<AllReduce<'a>>,
    },
}

impl Descriptor for StepWork<'_> {
    type Error = anyhow::Error;

    unsafe fn enqueue(&mut self, stream: sys::CUstream) -> Result<()> {
        match self {
            Self::Upload { staging, inputs } => unsafe { upload_staging(stream, staging, inputs) },
            Self::Decode {
                ctx,
                ptrs,
                staging,
                sizes,
                all_reduce,
            } => unsafe {
                step::copy_inputs(stream, ptrs, staging, sizes)?;
                decode_step(stream, ctx, ptrs, all_reduce.as_mut())
            },
        }
    }
}

/// Uploads one step's inputs into staging. Pageable H2D is host-synchronous, so the borrowed
/// host slices cannot outlive the copy.
///
/// # Safety
/// The staging addresses must be live and sized for `inputs`.
unsafe fn upload_staging(
    stream: sys::CUstream,
    staging: &StagingPtrs,
    inputs: &StepInputs,
) -> Result<()> {
    unsafe {
        result::memcpy_htod_async(staging.token_ids, &inputs.token_ids, stream)
            .and_then(|()| result::memcpy_htod_async(staging.seqlens_k, &inputs.seqlens_k, stream))
            .and_then(|()| {
                result::memcpy_htod_async(staging.block_table, &inputs.block_table, stream)
            })
            .and_then(|()| {
                result::memcpy_htod_async(staging.slot_mapping, &inputs.slot_mapping, stream)
            })
    }
    .map_err(|e| anyhow!("staging upload: {:?}", e.0))
}

/// Runs the step with the cell's all-reduce hook when it has one.
///
/// # Safety
/// See [`step::run_step`].
unsafe fn decode_step(
    stream: sys::CUstream,
    ctx: &StepContext<'_>,
    ptrs: &StepPtrs,
    all_reduce: Option<&mut AllReduce<'_>>,
) -> Result<()> {
    match all_reduce {
        #[cfg(feature = "nccl")]
        Some(parts) => {
            let kernels = ctx.kernels;
            let buffer_ptr = parts.buffer_ptr;
            let comm = parts.comm;
            let buffer = &mut *parts.buffer;
            let mut hook = |o_proj: u64, elements: usize| -> Result<()> {
                unsafe { kernels.bf16_to_f32(stream, o_proj, buffer_ptr, elements) }?;
                let mut view = buffer.slice_mut(0..elements);
                comm.all_reduce_in_place(&mut view, &ReduceOp::Sum)
                    .map_err(|e| anyhow!("ncclAllReduce: {:?}", e.0))?;
                unsafe { kernels.f32_to_bf16(stream, buffer_ptr, o_proj, elements) }?;
                Ok(())
            };
            unsafe { step::run_step(stream, ctx, ptrs, Some(&mut hook)) }
        }
        #[cfg(not(feature = "nccl"))]
        Some(_) => bail!("all-reduce cell in a build without the nccl feature"),
        None => unsafe { step::run_step(stream, ctx, ptrs, None) },
    }
}
