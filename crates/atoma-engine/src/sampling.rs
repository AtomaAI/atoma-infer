//! What the device sampler computes, stated on the host: the counter-based generator its draws
//! come from, so a seeded request's draws are a pure function of its seed and how many draws it
//! has made, and never of the batch it sits in or the slot it occupies.

pub mod philox;
