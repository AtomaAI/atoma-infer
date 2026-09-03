//! The Llama decode step on the tensor path: where it reads and writes, and how it is enqueued.
//!
//! | Module | Responsibility |
//! |---|---|
//! | [`slots`] | Every address one bucket's step touches, resolved through the arena once |
//! | [`step`] | The step enqueued from the op table, through one launcher seam |

pub mod slots;
pub mod step;
