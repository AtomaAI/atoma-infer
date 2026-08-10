//! CUDA graph capture substrate: context, streams, capture and graph lifetime, and the capture
//! arena.
//!
//! This crate owns device execution — the CUDA context, stream topology, graph capture, graph
//! lifetime, and the arena from which every captured step's activations are addressed. It knows
//! nothing about models, attention, or kernels; the layer whose allocation-freedom must be
//! provable stays small enough to prove.
//!
//! The crate links cudarc unconditionally under the workspace's `dynamic-loading` pin, so it
//! compiles, links, and runs `cargo test` on a machine with no CUDA toolkit, driver, or GPU.
//! Only paths that call the driver require one, and they fail loudly at the call.
//!
//! Module map:
//!
//! | Module | Responsibility |
//! |---|---|
//! | [`context`] | Device context: construction, global event-tracking disable, loud no-driver failure |
//! | [`stream`] | The dedicated capture stream, whose surface has no synchronize and no allocate |
//! | [`capture`] | Capture lifecycle, end-capture instantiate/discard, raw-sys graph wrappers |
//! | [`graph_entry`] | Graph-lifetime ownership with load-bearing teardown order |
//! | [`arena`] | Activation addresses as a pure function of (bucket, layer, role); workspace contract |
//! | [`error`] | Driver statuses classified into named errors with remediation text |

pub mod arena;
pub mod capture;
pub mod context;
pub mod error;
pub mod graph_entry;
pub mod stream;
