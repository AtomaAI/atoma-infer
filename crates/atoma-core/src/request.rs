//! Requests: the client unit the engine schedules, and the channel its output leaves on.
//!
//! A request is one prompt, one set of sampling parameters and one egress sink. What the engine
//! tells a client about its request travels as [`RequestEvent`]s over the request's egress channel,
//! and the receiver end of that channel being dropped is the one and only cancel.

mod egress;
mod event;

pub use egress::{egress, EgressReceiver, EgressSender};
pub use event::{FinishReason, RequestEvent, Usage};
