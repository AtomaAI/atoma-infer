//! The NCCL communicator a session's collectives run on, behind a surface that reaches no
//! stream.
//!
//! cudarc's `Comm` hands back the owning `Arc<CudaStream>` it was created on, which is exactly
//! the reference the Capture and Replay phases deny: a communicator attached to a graph entry
//! would otherwise be a route from either phase to a synchronize, an allocation or a
//! begin-capture on the capture stream. This wrapper exposes the collectives and nothing that
//! names a stream.

use cudarc::driver::DevicePtrMut;
use cudarc::nccl::{Comm, NcclType, ReduceOp};

use crate::error::RuntimeError;

/// A communicator whose collectives enqueue on the capture stream it was created on. Only
/// [`CaptureStream::nccl_comm`](crate::stream::CaptureStream::nccl_comm) mints one, in the
/// session's Allocation phase.
///
/// Its one operation compiles against the collective alone:
///
/// ```no_run
/// use atoma_runtime::communicator::Communicator;
/// use atoma_runtime::error::RuntimeError;
/// use cudarc::driver::CudaSlice;
/// use cudarc::nccl::ReduceOp;
///
/// fn sum(comm: &Communicator, buffer: &mut CudaSlice<f32>) -> Result<(), RuntimeError> {
///     comm.all_reduce_in_place(buffer, &ReduceOp::Sum)
/// }
/// ```
///
/// Asking it for the stream does not:
///
/// ```compile_fail
/// use atoma_runtime::communicator::Communicator;
///
/// fn reach(comm: &Communicator) {
///     comm.stream();
/// }
/// ```
pub struct Communicator {
    comm: Comm,
}

impl Communicator {
    pub(crate) fn new(comm: Comm) -> Self {
        Self { comm }
    }

    /// Reduces `buffer` in place across the ranks with `op`, enqueued on the capture stream.
    /// Launch only: inside a recording the collective is captured into the graph, and outside
    /// one the session's synchronize waits for it.
    pub fn all_reduce_in_place<T: NcclType, R: DevicePtrMut<T>>(
        &self,
        buffer: &mut R,
        op: &ReduceOp,
    ) -> Result<(), RuntimeError> {
        self.comm.all_reduce_in_place(buffer, op)?;
        Ok(())
    }
}
