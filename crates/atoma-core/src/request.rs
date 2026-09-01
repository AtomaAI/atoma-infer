//! Requests: the client unit the engine schedules, and the channel its output leaves on.
//!
//! A request is one prompt, one set of sampling parameters and one egress sink. It is born with
//! one sequence and may hold more. Its phase — Waiting, Running, Preempted, Finished, or Padding
//! for a dummy — is a value only a legal transition can produce; prefilling and decoding are
//! derived from a sequence's computed count against its prompt length, never phases of their own.
//! What the engine tells a client travels as [`RequestEvent`]s over the request's egress channel,
//! and the receiver end of that channel being dropped is the one and only cancel.

mod egress;
mod event;
mod params;
mod phase;
mod slab;
mod state;

pub(crate) use egress::Egress;
pub use egress::{egress, EgressReceiver, EgressSender};
pub use event::{FinishReason, RequestEvent, Usage};
pub use params::{Priority, SamplingParams, StopCriteria};
pub use phase::{Finished, Preempted, RequestPhase, Running, Waiting};
pub use slab::RequestSlab;
pub use state::{NewRequest, Request, Sequence, PADDING_TOKEN};
