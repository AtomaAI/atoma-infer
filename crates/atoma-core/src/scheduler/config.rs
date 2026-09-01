//! What the scheduler is built from, and the configurations it refuses to start under.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::kv::HashAlgorithm;
use crate::scheduler::AdmissionPolicy;
use crate::types::{RequestCount, TokenCount};

/// What the scheduler is built from, fixed for the process lifetime.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SchedulerConfig {
    /// Query tokens one step may compute, summed over entries.
    pub token_budget: TokenCount,
    /// Entries one step may hold: the largest bucket graphs were captured for, which the
    /// dispatch config knows as `captured_max_requests`. It bounds one step.
    pub max_batch: RequestCount,
    /// The longest sequence the model serves.
    pub max_model_len: TokenCount,
    /// Tokens per KV block.
    pub block_size: TokenCount,
    /// Admission candidates one pass examines.
    pub window: RequestCount,
    /// How admission orders the window.
    pub admission: AdmissionPolicy,
    /// Requests the slab holds at once — running, waiting and preempted together — before
    /// ingress is refused. Where `max_batch` bounds one step, this bounds the whole population
    /// a step is drawn from.
    pub max_requests: RequestCount,
    /// Generated tokens a client may leave unread on its egress channel before its request is
    /// retired. A client that keeps up leaves none, so this is what one stalled client costs the
    /// host: at most this many events, plus the one pass that runs before the next sweep sees it.
    pub max_client_backlog: TokenCount,
    /// The model's end-of-sequence token ids; sampling one finishes a request that does not
    /// ignore it.
    pub eos_token_ids: Vec<u32>,
    /// The chain-hash algorithm block identity is minted under.
    pub hash_algorithm: HashAlgorithm,
}

/// A configuration the scheduler refuses to start under.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum SchedulerError {
    /// The pool cannot hold one request of the maximum model length, so such a request could
    /// never finish and would bounce through preemption forever.
    #[error(
        "the pool has {free} free blocks but one request of the maximum model length needs \
         {needed}"
    )]
    PoolTooSmallForMaxModelLength { needed: usize, free: usize },
}
