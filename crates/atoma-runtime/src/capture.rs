//! Capture lifecycle: which operations are legal in which capture state, and the end-capture
//! paths that instantiate or discard a recording.
//!
//! The driver reports a stream's capture status as none, active, or invalidated. The transition
//! rules — begin only when idle, instantiate only when active, and an invalidated capture can
//! only be discarded — are pure logic, kept out of the driver-calling seams so they are testable
//! on a machine with no GPU.
//!
//! The end-capture paths are this crate's own rather than cudarc's: cudarc's `end_capture`
//! always instantiates, and its instantiate-flags parameter is an enum with no zero value, so
//! neither "instantiate with flags 0" nor "discard without instantiating" is expressible through
//! it. Both paths are built over cudarc's public `result` and `sys` layers — no fork, no patch.

use std::sync::Arc;

use cudarc::driver::sys::{CUresult, CUstreamCaptureStatus};
use cudarc::driver::{result, sys, CudaStream};

use crate::error::RuntimeError;
use crate::stream::CaptureStream;

/// Capture state of a stream, as the driver reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureState {
    /// No capture is in progress.
    Idle,
    /// A capture is recording.
    Active,
    /// A capture is recording but a previous operation broke it; it can only be discarded.
    Invalidated,
}

impl CaptureState {
    /// The state the driver's capture-status query maps to.
    pub fn from_status(status: CUstreamCaptureStatus) -> Self {
        match status {
            CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE => Self::Idle,
            CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE => Self::Active,
            CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_INVALIDATED => Self::Invalidated,
        }
    }

    /// The state after `op`, or the named error telling the operator what to do instead.
    pub fn apply(self, op: CaptureOp) -> Result<Self, RuntimeError> {
        match (self, op) {
            (Self::Idle, CaptureOp::Begin) => Ok(Self::Active),
            (Self::Active, CaptureOp::EndInstantiate) => Ok(Self::Idle),
            (Self::Active | Self::Invalidated, CaptureOp::Discard) => Ok(Self::Idle),
            (Self::Active, CaptureOp::Begin) => Err(RuntimeError::BeginWhileActive),
            (Self::Invalidated, CaptureOp::Begin) => Err(RuntimeError::BeginAfterInvalidation),
            (Self::Idle, CaptureOp::EndInstantiate) => Err(RuntimeError::EndWithoutCapture),
            (Self::Invalidated, CaptureOp::EndInstantiate) => {
                Err(RuntimeError::EndAfterInvalidation)
            }
            (Self::Idle, CaptureOp::Discard) => Err(RuntimeError::DiscardWithoutCapture),
        }
    }
}

/// An operation the capture substrate can attempt on a stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureOp {
    /// Start recording.
    Begin,
    /// Stop recording and instantiate the recorded graph.
    EndInstantiate,
    /// Stop recording and destroy the recorded graph without instantiating it.
    Discard,
}

/// A recorded graph and its instantiated executable, replayed with [`CapturedGraph::replay`].
///
/// `!Send` and `!Sync` by construction (raw driver handles): NVIDIA documents graph objects as
/// not internally synchronized, so warmup, capture, and replay all run on the executor thread
/// that owns the stream — a graph cannot be captured on a setup thread and moved.
pub struct CapturedGraph {
    cu_graph: sys::CUgraph,
    cu_graph_exec: sys::CUgraphExec,
    stream: Arc<CudaStream>,
}

impl CapturedGraph {
    /// Replays the graph on the stream it was captured from.
    pub fn replay(&self) -> Result<(), RuntimeError> {
        let ctx = self.stream.context();
        ctx.bind_to_thread()?;
        unsafe { result::graph::launch(self.cu_graph_exec, self.stream.cu_stream()) }?;
        Ok(())
    }

    /// Pre-uploads the executable's device state so the first replay pays no setup cost.
    pub fn upload(&self) -> Result<(), RuntimeError> {
        let ctx = self.stream.context();
        ctx.bind_to_thread()?;
        unsafe { result::graph::upload(self.cu_graph_exec, self.stream.cu_stream()) }?;
        Ok(())
    }

    /// The count of recorded nodes, for asserting a capture's shape without a raw handle.
    pub fn node_count(&self) -> Result<usize, RuntimeError> {
        // SAFETY: this type owns a live graph handle.
        Ok(unsafe { graph_nodes(self.cu_graph) }?.len())
    }

    /// Writes the recorded topology to `path` as Graphviz dot; see [`debug_dot_print`].
    pub fn write_debug_dot(&self, path: &std::path::Path, flags: u32) -> Result<(), RuntimeError> {
        // SAFETY: this type owns a live graph handle.
        unsafe { debug_dot_print(self.cu_graph, path, flags) }
    }

    /// Raw graph handle for the diagnostic wrappers. Do not destroy it; this type owns it.
    pub fn cu_graph(&self) -> sys::CUgraph {
        self.cu_graph
    }

    /// Raw executable handle for the update wrappers. Do not destroy it; this type owns it.
    pub fn cu_graph_exec(&self) -> sys::CUgraphExec {
        self.cu_graph_exec
    }
}

impl Drop for CapturedGraph {
    fn drop(&mut self) {
        let ctx = self.stream.context();
        ctx.record_err(ctx.bind_to_thread());
        // The executable references the graph, so it is destroyed first — the same order
        // cudarc's own graph destructor uses.
        let cu_graph_exec = std::mem::replace(&mut self.cu_graph_exec, std::ptr::null_mut());
        if !cu_graph_exec.is_null() {
            ctx.record_err(unsafe { result::graph::exec_destroy(cu_graph_exec) });
        }
        let cu_graph = std::mem::replace(&mut self.cu_graph, std::ptr::null_mut());
        if !cu_graph.is_null() {
            ctx.record_err(unsafe { result::graph::destroy(cu_graph) });
        }
    }
}

/// Ends the capture on `stream` and instantiates the recording with instantiate flags 0 —
/// deterministic memory from the pre-allocated arena, no auto-free, no device-launch, no memory
/// pool coupling.
///
/// No path leaks a graph or an executable. The lifecycle pre-checks reject an idle or
/// invalidated stream before anything is drained — an invalidated capture stays live for
/// [`end_capture_discard`], as its error directs. Past the pre-checks, an end-capture failure
/// leaves no graph behind (the driver drains the capture even when it reports it invalidated),
/// and an instantiate failure destroys the drained graph before returning.
pub fn end_capture_instantiate(stream: &CaptureStream) -> Result<CapturedGraph, RuntimeError> {
    stream.state()?.apply(CaptureOp::EndInstantiate)?;
    let cudarc_stream = stream.cudarc_stream();
    let ctx = cudarc_stream.context();
    ctx.bind_to_thread()?;

    let cu_graph = unsafe { result::stream::end_capture(cudarc_stream.cu_stream()) }?;
    if cu_graph.is_null() {
        return Err(RuntimeError::EndWithoutCapture);
    }

    let mut cu_graph_exec = std::ptr::null_mut();
    let instantiated =
        unsafe { sys::cuGraphInstantiateWithFlags(&mut cu_graph_exec, cu_graph, 0) }.result();
    if let Err(err) = instantiated {
        ctx.record_err(unsafe { result::graph::destroy(cu_graph) });
        return Err(err.into());
    }

    Ok(CapturedGraph {
        cu_graph,
        cu_graph_exec,
        stream: cudarc_stream.clone(),
    })
}

/// Ends the capture on `stream` and destroys whatever was recorded without instantiating it —
/// the path an invalidated capture must take, which cudarc's always-instantiating `end_capture`
/// cannot express. Discarding costs nothing and leaks nothing.
pub fn end_capture_discard(stream: &CaptureStream) -> Result<(), RuntimeError> {
    stream.state()?.apply(CaptureOp::Discard)?;
    let cudarc_stream = stream.cudarc_stream();
    cudarc_stream.context().bind_to_thread()?;

    match unsafe { result::stream::end_capture(cudarc_stream.cu_stream()) } {
        Ok(cu_graph) if !cu_graph.is_null() => {
            unsafe { result::graph::destroy(cu_graph) }?;
            Ok(())
        }
        Ok(_) => Ok(()),
        // Ending an invalidated capture reports the invalidation but still drains the recording
        // with no graph to destroy; for a discard that is success, not an error.
        Err(err) if err.0 == CUresult::CUDA_ERROR_STREAM_CAPTURE_INVALIDATED => Ok(()),
        Err(err) => Err(err.into()),
    }
}

/// Writes the graph's recorded topology to `path` as Graphviz dot, so a failed or suspect
/// capture is diagnosed by inspecting what was actually recorded rather than by guessing.
///
/// `flags` is a bitmask of [`sys::CUgraphDebugDot_flags`] verbosity bits; `0` prints the basic
/// topology.
///
/// # Safety
/// `cu_graph` must be a live graph handle, e.g. from [`CapturedGraph::cu_graph`].
pub unsafe fn debug_dot_print(
    cu_graph: sys::CUgraph,
    path: &std::path::Path,
    flags: u32,
) -> Result<(), RuntimeError> {
    let path = std::ffi::CString::new(path.to_string_lossy().as_bytes())
        .map_err(|_| RuntimeError::DotPrintPathHasNul)?;
    unsafe { sys::cuGraphDebugDotPrint(cu_graph, path.as_ptr(), flags) }.result()?;
    Ok(())
}

/// The graph's recorded nodes, for asserting a capture's shape (node count, kernel nodes only)
/// on the rig.
///
/// # Safety
/// `cu_graph` must be a live graph handle, e.g. from [`CapturedGraph::cu_graph`].
pub unsafe fn graph_nodes(cu_graph: sys::CUgraph) -> Result<Vec<sys::CUgraphNode>, RuntimeError> {
    let mut num_nodes = 0usize;
    unsafe { sys::cuGraphGetNodes(cu_graph, std::ptr::null_mut(), &mut num_nodes) }.result()?;
    let mut nodes = vec![std::ptr::null_mut(); num_nodes];
    unsafe { sys::cuGraphGetNodes(cu_graph, nodes.as_mut_ptr(), &mut num_nodes) }.result()?;
    nodes.truncate(num_nodes);
    Ok(nodes)
}

/// Applies a re-recorded graph's parameters to an existing executable without re-instantiating.
///
/// Returns the driver's result info in both outcomes the operator must distinguish: an applied
/// update, and an update the driver rejected because the topology changed — the info names the
/// offending node and reason. Other driver failures classify as usual.
///
/// # Safety
/// `cu_graph_exec` and `cu_graph` must be live handles, e.g. from a [`CapturedGraph`].
pub unsafe fn exec_update(
    cu_graph_exec: sys::CUgraphExec,
    cu_graph: sys::CUgraph,
) -> Result<sys::CUgraphExecUpdateResultInfo, RuntimeError> {
    let mut info = std::mem::MaybeUninit::uninit();
    let status = unsafe { sys::cuGraphExecUpdate_v2(cu_graph_exec, cu_graph, info.as_mut_ptr()) };
    match status {
        CUresult::CUDA_SUCCESS | CUresult::CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE => {
            Ok(unsafe { info.assume_init() })
        }
        other => Err(cudarc::driver::DriverError(other).into()),
    }
}

/// Exchanges the calling thread's capture-interaction mode, returning the previous mode.
///
/// Wraps a region of driver calls that must not interact with this thread's capture — the
/// caller restores the returned mode afterwards.
pub fn thread_exchange_capture_mode(
    mode: sys::CUstreamCaptureMode,
) -> Result<sys::CUstreamCaptureMode, RuntimeError> {
    let mut mode = mode;
    unsafe { sys::cuThreadExchangeStreamCaptureMode(&mut mode) }.result()?;
    Ok(mode)
}

/// The launch parameters recorded in a kernel node, for asserting baked pointers on the rig.
///
/// Uses the `_v2` symbol: under the CUDA 12 bindings [`sys::CUDA_KERNEL_NODE_PARAMS`] is the v2
/// struct, and the un-versioned driver symbol keeps the v1 ABI.
///
/// # Safety
/// `node` must be a kernel node in a live graph, e.g. from [`graph_nodes`].
pub unsafe fn kernel_node_params(
    node: sys::CUgraphNode,
) -> Result<sys::CUDA_KERNEL_NODE_PARAMS, RuntimeError> {
    let mut params = std::mem::MaybeUninit::uninit();
    unsafe { sys::cuGraphKernelNodeGetParams_v2(node, params.as_mut_ptr()) }.result()?;
    Ok(unsafe { params.assume_init() })
}

/// Replaces the launch parameters of a kernel node in a graph (before instantiation).
///
/// # Safety
/// `node` must be a kernel node in a live graph, and `params` must describe a launch valid for
/// that node's kernel.
pub unsafe fn set_kernel_node_params(
    node: sys::CUgraphNode,
    params: &sys::CUDA_KERNEL_NODE_PARAMS,
) -> Result<(), RuntimeError> {
    unsafe { sys::cuGraphKernelNodeSetParams_v2(node, params) }.result()?;
    Ok(())
}

/// Replaces the launch parameters of a kernel node in an instantiated executable — the patch
/// path for updating baked pointers without re-instantiating.
///
/// # Safety
/// `cu_graph_exec` must be live, `node` must be a kernel node of the graph it was instantiated
/// from, and `params` must describe a launch valid for that node's kernel.
pub unsafe fn set_exec_kernel_node_params(
    cu_graph_exec: sys::CUgraphExec,
    node: sys::CUgraphNode,
    params: &sys::CUDA_KERNEL_NODE_PARAMS,
) -> Result<(), RuntimeError> {
    unsafe { sys::cuGraphExecKernelNodeSetParams_v2(cu_graph_exec, node, params) }.result()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legal_transitions_produce_the_expected_state() {
        assert_eq!(
            CaptureState::Idle.apply(CaptureOp::Begin).unwrap(),
            CaptureState::Active
        );
        assert_eq!(
            CaptureState::Active
                .apply(CaptureOp::EndInstantiate)
                .unwrap(),
            CaptureState::Idle
        );
        assert_eq!(
            CaptureState::Active.apply(CaptureOp::Discard).unwrap(),
            CaptureState::Idle
        );
        assert_eq!(
            CaptureState::Invalidated.apply(CaptureOp::Discard).unwrap(),
            CaptureState::Idle
        );
    }

    #[test]
    fn illegal_transitions_produce_named_errors() {
        assert!(matches!(
            CaptureState::Active.apply(CaptureOp::Begin),
            Err(RuntimeError::BeginWhileActive)
        ));
        assert!(matches!(
            CaptureState::Invalidated.apply(CaptureOp::Begin),
            Err(RuntimeError::BeginAfterInvalidation)
        ));
        assert!(matches!(
            CaptureState::Idle.apply(CaptureOp::EndInstantiate),
            Err(RuntimeError::EndWithoutCapture)
        ));
        assert!(matches!(
            CaptureState::Invalidated.apply(CaptureOp::EndInstantiate),
            Err(RuntimeError::EndAfterInvalidation)
        ));
        assert!(matches!(
            CaptureState::Idle.apply(CaptureOp::Discard),
            Err(RuntimeError::DiscardWithoutCapture)
        ));
    }

    #[test]
    fn an_invalidated_capture_directs_the_operator_to_the_discard_path() {
        let err = CaptureState::Invalidated
            .apply(CaptureOp::EndInstantiate)
            .unwrap_err();
        assert!(err.to_string().contains("end_capture_discard"));
    }

    #[test]
    fn driver_statuses_map_onto_states() {
        assert_eq!(
            CaptureState::from_status(CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE),
            CaptureState::Idle
        );
        assert_eq!(
            CaptureState::from_status(CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE),
            CaptureState::Active
        );
        assert_eq!(
            CaptureState::from_status(CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_INVALIDATED),
            CaptureState::Invalidated
        );
    }
}
