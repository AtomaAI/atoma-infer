//! Benchmark harness for `atoma-infer`.
//!
//! The harness offers an open-loop workload at a target arrival rate, records TTFT, inter-token
//! latency and end-to-end latency as full distributions, and reports **goodput at a fixed SLO** as
//! the headline number. It drives `atoma-infer` and a pinned vLLM baseline over the same OpenAI
//! surface, on the same hardware, and emits one results artifact containing both engines' configs,
//! the absolute numbers, and the ratios between them.
//!
//! Module map:
//!
//! | Module | Responsibility |
//! |---|---|
//! | [`arrival`] | Poisson arrival process |
//! | [`workload`] | ShareGPT-derived and long-context request streams |
//! | [`client`] | Streaming OpenAI client, one [`record::RequestRecord`] per request |
//! | [`record`] | Latency distributions (hdrhistogram) and run summaries |
//! | [`goodput`] | Goodput at a fixed TTFT/ITL SLO |
//! | [`kv_probe`] | Free-KV-block sampling; a monotonic decline fails the run |
//! | [`vllm`] | Pinned vLLM baseline: launch, readiness, version capture |
//! | [`report`] | Results artifact and the rendered baseline table |
//! | [`runner`] | Drives one run: arrivals in, records out |

pub mod arrival;
pub mod client;
pub mod config;
pub mod error;
pub mod goodput;
pub mod kv_probe;
pub mod record;
pub mod report;
pub mod runner;
pub mod vllm;
pub mod workload;

pub use error::{BenchError, Result};
