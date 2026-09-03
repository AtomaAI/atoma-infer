//! Model definitions on the runtime tensor path: the Llama decode step over engine-owned device
//! memory, addressed through [`atoma_runtime::tensor::Tensor`] views.
//!
//! Candle is not a dependency. Weights and the KV cache are loaded and allocated elsewhere; this
//! crate receives their device addresses at Allocation and describes the step over them. What a
//! model states is host-visible data: its dimensions, and per layer class the linear op order and
//! the roles each op reads and writes, from which the arena's role table follows.
//!
//! | Module | Responsibility |
//! |---|---|
//! | [`dims`] | The dimensions a Llama decode step is shaped by, checked once |
//! | [`layer`] | A layer class: the op order and the roles each op reads and writes; Llama's one class |
//! | [`rope`] | The rotary embedding's cosine and sine tables, computed on the host once |

pub mod dims;
pub mod layer;
pub mod rope;
