//! The two calls an attention backend exposes, and what each may do.

use crate::attention::declaration::BackendDeclaration;
use crate::attention::workspace::{Captured, Workspace, WorkspaceRequirement};
use crate::dispatch::GraphKey;

/// One address in device memory.
///
/// Carried by value through this seam so the engine core can state and check where a backend
/// plans, without linking a driver or owning the allocation behind it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DeviceAddress(u64);

impl DeviceAddress {
    /// The address at `address`.
    #[must_use]
    pub fn new(address: u64) -> Self {
        Self(address)
    }
}

/// What one preparation call plans over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanInput<'a> {
    /// The captured graph this preparation serves.
    pub key: GraphKey,
    /// KV length per entry after the step, in batch order.
    ///
    /// Padding dummies are included — they occupy request slots and own real KV, so a backend
    /// schedules over them like any other entry — which is why this is longer than the key's
    /// live request count whenever the batch was padded. Host-native, so no preparation needs a
    /// device-to-host synchronization to read a length.
    pub sequence_lens: &'a [usize],
}

/// What one preparation call produced, and recording reads.
///
/// The addresses are the contract's load-bearing part: preparation re-plans before every replay,
/// and a replay is only valid if it re-plans into the same bytes the capture baked. A caller can
/// therefore check the invariant it depends on instead of trusting it.
pub trait PreparedPlan {
    /// The device addresses this preparation wrote its scheduling metadata to.
    ///
    /// Fixed during Allocation and identical for every preparation this backend makes, whatever
    /// the batch shape: the contents change, the addresses do not.
    fn metadata_addresses(&self) -> &[DeviceAddress];
}

/// The seam an attention backend implements to be capture-safe.
///
/// Two calls, split by what each is allowed to do:
///
/// - **Preparation** runs on the host before every replay. It may allocate and it may synchronize,
///   because it runs outside every captured region. It re-plans scheduling metadata — the tile,
///   split and ordering decisions that follow from this step's lengths — writing it at the
///   addresses fixed during Allocation.
/// - **Recording** runs once, while a capture is active. It issues static-shape device work and
///   nothing else: no host readback, no synchronization, no dynamic allocation. Each of those
///   either invalidates the capture or bakes an address that will not survive it.
///
/// How much of that the signatures hold, exactly: preparation takes `&mut self` and returns a
/// plan, recording takes `&self` — so it can neither re-plan nor swap a buffer of its own — and
/// the bytes it writes come from the caller's [`Workspace`] rather than from an allocator it
/// reaches for. The rest rests on [`AttentionBackend::Recorder`] and
/// [`AttentionBackend::Buffer`]: a recording can only do what those two types let it, which is
/// why the contract on `Recorder` is that its surface has no synchronize, no allocate and no
/// readback, and why the production buffer is a device allocation a recording cannot resize.
pub trait AttentionBackend {
    /// The device buffer kind this backend's kernels read and write.
    type Buffer;

    /// Where recording enqueues device work.
    ///
    /// Its surface must expose no synchronize, no allocate and no host readback, so a record call
    /// cannot reach one. The capture stream is the production choice; a test double that logs
    /// launches is the other.
    type Recorder;

    /// What preparation produces and recording reads: the scheduling metadata's fixed addresses,
    /// plus whatever else the recorded launches need.
    type Plan: PreparedPlan + WorkspaceRequirement;

    /// Why a preparation or a recording failed.
    type Error;

    /// What this backend declares to the engine at startup.
    fn declaration(&self) -> BackendDeclaration;

    /// Bytes of caller-owned workspace the routine recorded for `key` needs.
    ///
    /// Asked during Allocation, at the largest bucket, because the workspace is allocated once
    /// and its address is baked into every captured graph. Each prepared plan restates what its
    /// own call needs through [`WorkspaceRequirement`], which is what recording checks.
    fn workspace_bytes(&self, key: GraphKey) -> usize;

    /// Plans this step on the host: may allocate, may synchronize, and re-plans the scheduling
    /// metadata at the addresses fixed during Allocation.
    ///
    /// Runs before every replay, not only before a capture: a replay launches the work the
    /// capture recorded, and that work reads whatever this call last wrote.
    ///
    /// # Errors
    ///
    /// Returns [`AttentionBackend::Error`] when the step cannot be planned.
    fn prepare(&mut self, input: PlanInput<'_>) -> Result<Self::Plan, Self::Error>;

    /// Records the prepared step's device work into whatever `recorder` is recording.
    ///
    /// Issues static-shape work only. Everything it writes lives in `workspace`, which the
    /// caller allocated before capture and which must cover `plan`'s
    /// [`WorkspaceRequirement`].
    ///
    /// # Errors
    ///
    /// Returns [`AttentionBackend::Error`] when the work cannot be recorded, including when
    /// `workspace` is smaller than `plan` requires.
    fn record(
        &self,
        plan: &Self::Plan,
        workspace: &mut Workspace<Captured, Self::Buffer>,
        recorder: &mut Self::Recorder,
    ) -> Result<(), Self::Error>;
}
