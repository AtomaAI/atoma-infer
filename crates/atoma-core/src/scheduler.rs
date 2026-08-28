//! The token-budget scheduler.
//!
//! One scheduling pass spends a per-step token budget: running requests first, then admission
//! from the preempted stack and the waiting queue over a bounded window. It answers in indices
//! and counts — which sequences run, how many tokens each computes, which entries sample — never
//! in copied request state.

mod budget;

pub use budget::TokenBudget;
