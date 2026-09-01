//! A backend that implements the contract over host memory, so every rule the contract states is
//! driven without a GPU.
//!
//! Its scheduling metadata is one buffer allocated at construction — where a real backend
//! allocates during Allocation — and the addresses it reports are that buffer's own. "Preparation
//! re-plans at fixed addresses" is therefore a property of what the fake does rather than a
//! constant it repeats: a preparation that reallocated would report a different address and the
//! tests would say so.

use thiserror::Error;

use crate::attention::{
    AttentionBackend, BackendDeclaration, BreakSite, Captured, DeviceAddress, PlanInput,
    PreparedPlan, SupportLevel, Workspace, WorkspaceRequirement,
};
use crate::dispatch::GraphKey;

/// Bytes of workspace one token of a recorded step needs. Arbitrary: what matters is that the
/// requirement is stated by the call and checked against what the caller allocated.
const WORKSPACE_BYTES_PER_TOKEN: usize = 4;

/// Tokens one scheduling tile covers.
const TOKENS_PER_TILE: usize = 16;

/// Why a fake preparation or recording failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub(crate) enum FakeError {
    /// More entries than the metadata buffer allocated at construction holds. A real backend
    /// hits this the same way: the metadata is sized once, for the largest bucket.
    #[error("a batch of {entries} entries needs more metadata than the {allocated} allocated")]
    BatchOverMetadata { entries: usize, allocated: usize },
    /// The workspace the caller handed over is smaller than the recorded call needs.
    #[error("the recorded call needs {needed} bytes of workspace but was handed {handed}")]
    WorkspaceTooSmall { needed: usize, handed: usize },
}

/// One launch a recording enqueued.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Launch {
    /// What the launch computes.
    pub(crate) name: &'static str,
    /// Threads the launch covers, fixed by the graph key the plan was prepared for.
    pub(crate) threads: usize,
    /// Where the launch reads its scheduling metadata.
    pub(crate) metadata: DeviceAddress,
}

/// Where the fake backend's recording enqueues: a log of launches.
///
/// Stands in for the capture stream, and like it exposes no synchronize, no allocate and no
/// readback — there is nothing here for a record call to reach for.
#[derive(Debug, Default)]
pub(crate) struct FakeRecorder {
    launches: Vec<Launch>,
}

impl FakeRecorder {
    /// Enqueues one launch.
    fn enqueue(&mut self, launch: Launch) {
        self.launches.push(launch);
    }

    /// What has been enqueued, in order.
    pub(crate) fn launches(&self) -> &[Launch] {
        &self.launches
    }
}

/// What one fake preparation produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FakePlan {
    key: GraphKey,
    metadata_addresses: Vec<DeviceAddress>,
    workspace_bytes: usize,
}

impl PreparedPlan for FakePlan {
    fn metadata_addresses(&self) -> &[DeviceAddress] {
        &self.metadata_addresses
    }
}

impl WorkspaceRequirement for FakePlan {
    fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }
}

/// A backend implementing the whole contract over host memory.
#[derive(Debug)]
pub(crate) struct FakeBackend {
    name: String,
    support_level: SupportLevel,
    cannot_capture: Vec<BreakSite>,
    rank_coupled: Vec<BreakSite>,
    /// Scheduling metadata: one tile count per entry, allocated once and re-planned in place.
    metadata: Vec<u32>,
}

impl FakeBackend {
    /// A backend named `name`, valid at `support_level`, whose metadata holds `entries` entries.
    pub(crate) fn new(name: &str, support_level: SupportLevel, entries: usize) -> Self {
        Self {
            name: name.to_owned(),
            support_level,
            cannot_capture: Vec::new(),
            rank_coupled: Vec::new(),
            metadata: vec![0; entries],
        }
    }

    /// Declares that this backend cannot capture the op at `site`.
    pub(crate) fn cannot_capture(mut self, site: BreakSite) -> Self {
        self.cannot_capture.push(site);
        self
    }

    /// Declares that this backend's work at `site` is rank-coupled.
    pub(crate) fn rank_coupled(mut self, site: BreakSite) -> Self {
        self.rank_coupled.push(site);
        self
    }

    /// The scheduling metadata as it stands, for asserting that preparation re-planned it.
    pub(crate) fn metadata(&self) -> &[u32] {
        &self.metadata
    }
}

impl AttentionBackend for FakeBackend {
    type Buffer = Vec<u8>;
    type Recorder = FakeRecorder;
    type Plan = FakePlan;
    type Error = FakeError;

    fn declaration(&self) -> BackendDeclaration {
        let declaration = BackendDeclaration::new(self.name.clone(), self.support_level);
        let declaration = self
            .cannot_capture
            .iter()
            .fold(declaration, |declaration, &site| {
                declaration.cannot_capture(site)
            });
        self.rank_coupled
            .iter()
            .fold(declaration, |declaration, &site| {
                declaration.rank_coupled(site)
            })
    }

    fn workspace_bytes(&self, key: GraphKey) -> usize {
        key.padded_token_count().get() * WORKSPACE_BYTES_PER_TOKEN
    }

    fn prepare(&mut self, input: PlanInput<'_>) -> Result<Self::Plan, Self::Error> {
        if input.sequence_lens.len() > self.metadata.len() {
            return Err(FakeError::BatchOverMetadata {
                entries: input.sequence_lens.len(),
                allocated: self.metadata.len(),
            });
        }
        // Re-planned in place: the buffer is never grown, so its address outlives every
        // preparation, which is what a captured graph baked.
        self.metadata.fill(0);
        for (tiles, &sequence_len) in self.metadata.iter_mut().zip(input.sequence_lens) {
            *tiles = u32::try_from(sequence_len.div_ceil(TOKENS_PER_TILE))
                .expect("a tile count fits in u32");
        }
        let address = u64::try_from(self.metadata.as_ptr().addr())
            .expect("a host address stands in for a device one");
        Ok(FakePlan {
            key: input.key,
            metadata_addresses: vec![DeviceAddress::new(address)],
            workspace_bytes: self.workspace_bytes(input.key),
        })
    }

    fn record(
        &self,
        plan: &Self::Plan,
        workspace: &mut Workspace<Captured, Self::Buffer>,
        recorder: &mut Self::Recorder,
    ) -> Result<(), Self::Error> {
        if !workspace.covers(plan) {
            return Err(FakeError::WorkspaceTooSmall {
                needed: plan.workspace_bytes(),
                handed: workspace.bytes(),
            });
        }
        // Every byte written comes from the caller's workspace, and the shapes come from the key
        // the plan was prepared for — never from this step's live counts.
        let threads = plan.key.padded_token_count().get();
        let caller_owned = workspace.buffer_mut();
        caller_owned[..threads * WORKSPACE_BYTES_PER_TOKEN].fill(0);
        for name in ["decode_attention", "output_projection"] {
            recorder.enqueue(Launch {
                name,
                threads,
                metadata: plan.metadata_addresses[0],
            });
        }
        Ok(())
    }
}
