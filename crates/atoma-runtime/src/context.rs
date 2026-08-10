//! Device context ownership: construction, the global event-tracking disable, and loud failure
//! when no driver is present.

use std::sync::Arc;

use cudarc::driver::CudaContext;

use crate::error::RuntimeError;

/// The engine's handle to one CUDA device, and the only way this crate opens one.
///
/// Construction disables cudarc's event tracking globally, before anything is allocated: cudarc
/// attaches a tracking event to every buffer at allocation time, and a wait on an event recorded
/// before a capture began invalidates the capture. Disabling per capture would be too late — the
/// events attach at allocation, not at use.
pub struct RuntimeContext {
    ctx: Arc<CudaContext>,
}

impl RuntimeContext {
    /// Opens device `ordinal` and disables event tracking for everything allocated afterward.
    ///
    /// Fails loudly and early when no usable driver or device exists, so a misconfigured
    /// deployment dies at startup rather than mid-serve: with no driver library at all, cudarc's
    /// loader panics; with a driver but no usable device, this returns
    /// [`RuntimeError::NoDriver`] carrying the remediation text.
    pub fn new(ordinal: usize) -> Result<Self, RuntimeError> {
        let ctx = CudaContext::new(ordinal)?;
        // SAFETY: cross-stream synchronization is this crate's responsibility from here on. The
        // capture substrate orders work explicitly — buffers are allocated before capture, the
        // capture stream never frees them (GraphEntry owns every buffer for the graph's
        // lifetime), and replay is serialized on the executor thread — so no CudaSlice relies on
        // cudarc's event-based synchronization.
        unsafe { ctx.disable_event_tracking() };
        Ok(Self { ctx })
    }

    /// The underlying cudarc context, for allocating buffers and creating streams.
    pub fn cuda(&self) -> &Arc<CudaContext> {
        &self.ctx
    }
}
