//! Graph-lifetime ownership: every pointer a graph baked stays alive exactly as long as the
//! graph does.

use cudarc::driver::CudaSlice;

use crate::capture::CapturedGraph;

/// One captured graph together with every device buffer whose address is baked into it, so a
/// dropped buffer cannot leave a dangling pointer in a live graph.
///
/// Field order is load-bearing. Rust drops fields in declaration order, so this declaration IS
/// the teardown order — no hand-written cleanup exists to get wrong in a future edit. Buffers
/// go first, then the graph (whose own drop destroys the executable before the graph), and the
/// NCCL communicator last: `ncclCommAbort` blocks until no captured graph references the
/// communicator, so a communicator declared before the graph would deadlock teardown.
pub struct GraphEntry {
    /// Buffers the replay writes before each launch (token ids, positions, block tables).
    inputs: Vec<CudaSlice<u8>>,
    /// Buffers the replay reads after each launch (logits, sampled tokens).
    outputs: Vec<CudaSlice<u8>>,
    /// Kernel workspaces the caller preallocated; kernels may not allocate their own.
    workspaces: Vec<CudaSlice<u8>>,
    /// Dropped after every buffer above and before the communicator below.
    graph: CapturedGraph,
    /// The communicator captured collectives run on. Last: abort blocks on live graphs.
    #[cfg(feature = "nccl")]
    comm: Option<cudarc::nccl::Comm>,
}

impl GraphEntry {
    /// Takes ownership of the graph and every buffer it baked.
    pub fn new(
        inputs: Vec<CudaSlice<u8>>,
        outputs: Vec<CudaSlice<u8>>,
        workspaces: Vec<CudaSlice<u8>>,
        graph: CapturedGraph,
    ) -> Self {
        Self {
            inputs,
            outputs,
            workspaces,
            graph,
            #[cfg(feature = "nccl")]
            comm: None,
        }
    }

    /// Attaches the communicator the captured collectives run on; it outlives the graph and is
    /// torn down after it.
    #[cfg(feature = "nccl")]
    pub fn with_comm(mut self, comm: cudarc::nccl::Comm) -> Self {
        self.comm = Some(comm);
        self
    }

    /// The captured graph, for replay and the diagnostic wrappers.
    pub fn graph(&self) -> &CapturedGraph {
        &self.graph
    }

    /// Input buffers, written before each replay.
    pub fn inputs_mut(&mut self) -> &mut [CudaSlice<u8>] {
        &mut self.inputs
    }

    /// Output buffers, read after each replay.
    pub fn outputs(&self) -> &[CudaSlice<u8>] {
        &self.outputs
    }

    /// Kernel workspaces, handed to kernels that declare a workspace requirement.
    pub fn workspaces_mut(&mut self) -> &mut [CudaSlice<u8>] {
        &mut self.workspaces
    }
}
