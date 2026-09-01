//! The seam an attention backend implements to be capture-safe, and how a backend or a model
//! declares where a forward pass must leave the graph.
//!
//! A backend exposes two calls. Preparation runs on the host before every replay: it may allocate
//! and synchronize, and it re-plans scheduling metadata at device addresses fixed once during
//! Allocation. Recording runs once, during Capture: it issues static-shape device work and
//! nothing else. Alongside them a backend declares the live batches its captured routine is valid
//! for — its [`SupportLevel`] — and the sites it cannot capture.
//!
//! Break points come from two independent declarers. A backend declares a capability: an op it
//! cannot capture, or one that is rank-coupled. A model declares a policy: where its eager region
//! lies. [`CaptureContract::resolve`] takes the union and settles the graph mode at the minimum
//! level across the active backends; the dispatcher is built from the result.
//!
//! # Where this lives, and why
//!
//! A backend declares a support level, and admission consumes it. The whole contract is placed
//! here, in the GPU-free engine core, rather than beside the device code that implements it:
//!
//! - Both consumers are already here. Admission reads the support level, and the dispatcher returns
//!   the break-point union. A declaration seam whose consumers all sit in one crate belongs in that
//!   crate, and [`SupportLevel`] is then defined once rather than declared in one crate and
//!   mirrored in another.
//! - The seam carries no device type. Preparation is a host call, recording enqueues through
//!   [`AttentionBackend::Recorder`], and the workspace is [`Workspace<Captured,
//!   Buffer>`](Workspace) — both associated types. Whichever crate implements this for a real
//!   backend names its own capture stream and device buffers there and takes the dependency on this
//!   one; nothing here links a driver, and no backend implements it yet.
//! - A fake backend therefore drives every case in this crate's own tests, on a machine with no
//!   GPU.
//! - [`GraphKey`](crate::dispatch::GraphKey) has exactly one constructor, crate-internal. A backend
//!   outside this crate receives keys through [`PlanInput`] and can never mint one.

mod backend;
mod break_point;
mod declaration;
mod graph_mode;
mod workspace;

pub use backend::{AttentionBackend, DeviceAddress, PlanInput, PreparedPlan};
pub use break_point::{BreakPoint, BreakPoints, BreakSite, Declarer, DeclarerKind};
pub use declaration::{BackendDeclaration, CaptureContract, ModelDeclaration};
pub use graph_mode::{GraphMode, SupportLevel};
pub use workspace::{Captured, Eager, Workspace, WorkspaceRequirement};

#[cfg(test)]
mod fake;
#[cfg(test)]
mod tests;
