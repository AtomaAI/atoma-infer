//! The harness's descriptor seam: one [`Descriptor`] implementation through which every enqueue
//! onto the capture stream passes, and the only place this backend's C-ABI kernels receive the
//! raw stream handle.

use anyhow::{anyhow, Result};
#[cfg(feature = "nccl")]
use atoma_runtime::communicator::Communicator;
use atoma_runtime::session::Descriptor;
#[cfg(feature = "nccl")]
use cudarc::driver::CudaSlice;
use cudarc::driver::{result, sys};
#[cfg(feature = "nccl")]
use cudarc::nccl::ReduceOp;

#[cfg(feature = "nccl")]
use crate::gpu::alloc;
use crate::gpu::step::{self, InputPtrs, StepContext, StepPtrs};
use crate::layout::StaticSizes;
use crate::variation::StepInputs;

/// The per-layer all-reduce: the cell's communicator and the f32 mirror it reduces in.
#[cfg(feature = "nccl")]
pub struct AllReduce<'a> {
    pub comm: &'a Communicator,
    pub mirror: &'a mut CudaSlice<f32>,
}

/// One enqueue of harness work onto the capture stream.
pub enum StepDescriptor<'a> {
    /// H2D upload of one step's inputs into their persistent staging mirrors — the pre-work
    /// every step, replayed or eager, consumes. Pageable H2D is host-synchronous, so it is never
    /// part of a recording; the graph reads staging only through captured D2D nodes.
    Upload {
        staging: &'a InputPtrs,
        inputs: &'a StepInputs,
    },
    /// The captured copy-in from staging plus the full decode step — what a recording bakes and
    /// what an eager pass runs.
    Decode {
        ctx: &'a StepContext<'a>,
        ptrs: &'a StepPtrs,
        sizes: &'a StaticSizes,
        #[cfg(feature = "nccl")]
        all_reduce: Option<AllReduce<'a>>,
    },
}

impl Descriptor for StepDescriptor<'_> {
    type Error = anyhow::Error;

    unsafe fn enqueue(&mut self, stream: sys::CUstream) -> Result<()> {
        match self {
            Self::Upload { staging, inputs } => {
                // SAFETY: the session hands a live stream, and staging was fixed in Allocation
                // at the sizes this step's inputs have.
                unsafe { upload_staging(stream, staging, inputs) }
            }
            Self::Decode {
                ctx,
                ptrs,
                sizes,
                #[cfg(feature = "nccl")]
                all_reduce,
            } => {
                // SAFETY: the session hands a live stream, and every address in `ptrs` was
                // fixed in Allocation at the shapes `ctx` and `sizes` describe.
                unsafe { step::copy_inputs(stream, &ptrs.statics, sizes) }?;
                #[cfg(feature = "nccl")]
                if let Some(all_reduce) = all_reduce {
                    // SAFETY: as above, plus a live communicator bound to this stream.
                    return unsafe { reduce_step(stream, ctx, ptrs, all_reduce) };
                }
                // SAFETY: as above.
                unsafe { step::run_step(stream, ctx, ptrs, None) }
            }
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
    staging: &InputPtrs,
    inputs: &StepInputs,
) -> Result<()> {
    // SAFETY: the caller's contract, forwarded to each upload.
    unsafe {
        upload(stream, staging.token_ids, &inputs.token_ids)?;
        upload(stream, staging.seqlens_k, &inputs.seqlens_k)?;
        upload(stream, staging.block_table, &inputs.block_table)?;
        upload(stream, staging.slot_mapping, &inputs.slot_mapping)
    }
}

/// One H2D upload of `src` to `dst`.
///
/// # Safety
/// `dst` must be live and sized for `src`.
unsafe fn upload<T>(stream: sys::CUstream, dst: u64, src: &[T]) -> Result<()> {
    // SAFETY: the caller's contract.
    unsafe { result::memcpy_htod_async(dst, src, stream) }
        .map_err(|e| anyhow!("staging upload: {:?}", e.0))
}

/// Runs the step with the cell's per-layer all-reduce installed: each layer's o-projection is
/// widened into the mirror, summed across the ranks, and narrowed back in place.
///
/// # Safety
/// See [`step::run_step`]; the communicator must be bound to `stream`.
#[cfg(feature = "nccl")]
unsafe fn reduce_step(
    stream: sys::CUstream,
    ctx: &StepContext<'_>,
    ptrs: &StepPtrs,
    all_reduce: &mut AllReduce<'_>,
) -> Result<()> {
    let kernels = ctx.kernels;
    let comm = all_reduce.comm;
    let mirror = &mut *all_reduce.mirror;
    let mirror_ptr = alloc::addr(mirror);
    let mut hook = |o_proj: u64, elements: usize| -> Result<()> {
        // SAFETY: `o_proj` is a live arena slot of `elements` bf16 values, and the mirror was
        // allocated for at least `elements` f32 values.
        unsafe { kernels.bf16_to_f32(stream, o_proj, mirror_ptr, elements) }?;
        let mut view = mirror.slice_mut(0..elements);
        comm.all_reduce_in_place(&mut view, &ReduceOp::Sum)?;
        // SAFETY: as above, in the other direction.
        unsafe { kernels.f32_to_bf16(stream, mirror_ptr, o_proj, elements) }?;
        Ok(())
    };
    // SAFETY: the caller's contract.
    unsafe { step::run_step(stream, ctx, ptrs, Some(&mut hook)) }
}
