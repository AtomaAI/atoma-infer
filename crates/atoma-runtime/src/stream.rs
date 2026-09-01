//! The dedicated capture stream: relaxed-mode begin on a side stream, behind a surface with no
//! synchronize and no allocate.

use std::sync::Arc;

use cudarc::driver::sys;
use cudarc::driver::CudaStream;

use crate::capture::{CaptureOp, CaptureState};
use crate::context::RuntimeContext;
use crate::error::RuntimeError;

/// The stream a capture records on: always a dedicated non-blocking side stream, never the
/// default stream, so unrelated work on other streams cannot invalidate a capture.
///
/// The public surface deliberately exposes **no synchronize and no allocate**. A synchronize or
/// an allocation across a capture boundary invalidates the capture — or bakes a dangling address
/// into the graph — so "no sync or alloc during capture" is a compile error here instead of a
/// replay-time illegal memory access.
pub struct CaptureStream {
    stream: Arc<CudaStream>,
}

impl CaptureStream {
    /// Creates the dedicated side stream on `ctx`'s device. Only the session's Allocation phase
    /// constructs one, so every capture stream lives inside a session.
    pub(crate) fn new(ctx: &RuntimeContext) -> Result<Self, RuntimeError> {
        Ok(Self {
            stream: ctx.cuda().new_stream()?,
        })
    }

    /// Begins recording in relaxed mode.
    ///
    /// Relaxed mode does not fence other threads' driver calls, so a concurrent thread doing
    /// legal work cannot invalidate this capture — the global mode would let it. The capture
    /// lifecycle table rejects a begin while a capture is already active or invalidated. Only
    /// the session's Capture phase records, so no phase — Allocation included — can begin a
    /// capture from outside it.
    pub(crate) fn begin_capture(&self) -> Result<(), RuntimeError> {
        self.state()?.apply(CaptureOp::Begin)?;
        self.stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)?;
        Ok(())
    }

    /// The capture state the driver reports for this stream.
    pub(crate) fn state(&self) -> Result<CaptureState, RuntimeError> {
        Ok(CaptureState::from_status(self.stream.capture_status()?))
    }

    /// Raw handle for launching kernels and FFI work onto the captured stream.
    ///
    /// The handle exists because kernel launches need it; using it to synchronize, allocate, or
    /// destroy the stream reintroduces exactly the failures this type's surface removes. Only the
    /// session hands it out, and only into [`Descriptor::enqueue`](crate::session::Descriptor)
    /// implementations.
    pub(crate) fn cu_stream(&self) -> sys::CUstream {
        self.stream.cu_stream()
    }

    /// The underlying cudarc stream, for this crate's end-capture paths only: the public surface
    /// must never grow a synchronize or an allocate.
    pub(crate) fn cudarc_stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Creates the NCCL communicator whose collectives enqueue on this capture stream, so an
    /// all-reduce issued during capture is recorded into the graph instead of landing on a
    /// foreign stream and invalidating the recording.
    ///
    /// Communicator creation allocates device memory, so it must happen strictly before the
    /// first capture, like every other allocation. After the record that captures its
    /// collective, hand the communicator to
    /// [`Capture::attach_comm`](crate::session::Capture::attach_comm) so teardown ordering
    /// holds: abort blocks until no live graph references the communicator.
    #[cfg(feature = "nccl")]
    pub fn nccl_comm(
        &self,
        rank: usize,
        world_size: usize,
        id: cudarc::nccl::Id,
    ) -> Result<cudarc::nccl::Comm, RuntimeError> {
        Ok(cudarc::nccl::Comm::from_rank(
            self.stream.clone(),
            rank,
            world_size,
            id,
        )?)
    }
}
