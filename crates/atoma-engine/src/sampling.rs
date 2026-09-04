//! What the device sampler computes, stated on the host: the record each request slot holds on
//! the device, the counter-based generator its draws come from, the reference computation the
//! kernel is held to, and the mirror of which request each slot holds, so a seeded request's
//! draws are a pure function of its seed and how many draws it has made, and never of the batch
//! it sits in or the slot it occupies.

pub mod owners;
pub mod philox;
pub mod record;
pub mod reference;
