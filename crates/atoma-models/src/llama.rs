//! The Llama decode step on the tensor path: where it reads and writes, and how it is enqueued.
//!
//! | Module | Responsibility |
//! |---|---|
//! | [`slots`] | Every address one bucket's step touches, resolved through the arena once |

pub mod slots;
