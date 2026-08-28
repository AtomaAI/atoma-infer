//! The token-budget scheduler.
//!
//! One scheduling pass spends a per-step token budget: running requests first, then admission
//! from the waiting queue over a bounded window. It answers in indices and counts — which
//! sequences run, how many tokens each computes, which entries sample — never in copied request
//! state. The scheduler owns the request slab and the block pool outright; nothing here is
//! shared or locked.

mod budget;
mod kv;
#[cfg(test)]
mod tests;

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::debug;

use crate::kv::BlockPool;
use crate::request::{
    FinishReason, NewRequest, Request, RequestEvent, RequestPhase, RequestSlab, Usage,
};
use crate::scheduler::kv::{ensure_blocks, release_blocks, PoolExhausted};
use crate::types::{RequestCount, RequestId, RequestSlot, SequenceIndex, StepId, TokenCount};

pub use budget::TokenBudget;

/// What the scheduler is built from, fixed for the process lifetime.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SchedulerConfig {
    /// Query tokens one step may compute, summed over entries.
    pub token_budget: TokenCount,
    /// Entries one step may hold: the largest bucket.
    pub max_batch: RequestCount,
    /// The longest sequence the model serves.
    pub max_model_len: TokenCount,
    /// Tokens per KV block.
    pub block_size: TokenCount,
    /// Admission candidates one pass examines.
    pub window: RequestCount,
    /// Requests the slab holds before intake is refused.
    pub max_requests: RequestCount,
}

/// One row of a [`Scheduled`]: a sequence, what it computes this step, and whether it samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Entry {
    pub slot: RequestSlot,
    pub sequence: SequenceIndex,
    /// Tokens the sequence already holds in KV before the step.
    pub context_len: usize,
    /// Tokens the entry computes this step.
    pub query_len: TokenCount,
    /// Whether the step samples a token for this entry: only when the query reaches the
    /// sequence's total, so a non-final prefill chunk never does.
    pub samples: bool,
}

impl Entry {
    /// Tokens the sequence's KV holds after the step.
    #[must_use]
    pub fn sequence_len(&self) -> usize {
        self.context_len + self.query_len.get()
    }
}

/// The output of one scheduling pass: indices and counts, never copied request state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Scheduled {
    pub step: StepId,
    pub entries: Vec<Entry>,
}

impl Scheduled {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Query tokens summed over entries.
    #[must_use]
    pub fn token_count(&self) -> usize {
        self.entries.iter().map(|entry| entry.query_len.get()).sum()
    }

    /// Whether every entry has query length one: the condition full-graph replay requires.
    #[must_use]
    pub fn is_uniform_decode(&self) -> bool {
        !self.entries.is_empty() && self.entries.iter().all(|entry| entry.query_len.get() == 1)
    }

    /// Entries that sample, in order.
    pub fn sampling_entries(&self) -> impl Iterator<Item = &Entry> {
        self.entries.iter().filter(|entry| entry.samples)
    }
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

/// The token-budget scheduler: request slab, block pool and queues, owned by one thread.
#[derive(Debug)]
pub struct Scheduler {
    config: SchedulerConfig,
    requests: RequestSlab,
    pool: BlockPool,
    /// Admission order, which is batch order. The last admitted is the preemption victim.
    running: Vec<RequestSlot>,
    /// Arrival order. Admission examines a bounded window from the front.
    waiting: VecDeque<RequestSlot>,
    budget: TokenBudget,
    step: StepId,
    next_request_id: u64,
}

impl Scheduler {
    /// Builds a scheduler over `pool`, which must hold at least one maximum-length request.
    ///
    /// # Errors
    ///
    /// Returns [`SchedulerError::PoolTooSmallForMaxModelLength`] when it does not.
    pub fn new(config: SchedulerConfig, pool: BlockPool) -> Result<Self, SchedulerError> {
        let needed = config.max_model_len.get().div_ceil(config.block_size.get());
        let free = pool.free_count();
        if free < needed {
            return Err(SchedulerError::PoolTooSmallForMaxModelLength { needed, free });
        }
        let budget = TokenBudget::new(config.token_budget, config.max_batch);
        Ok(Self {
            requests: RequestSlab::with_capacity(config.max_requests.get()),
            config,
            pool,
            running: Vec::with_capacity(budget.max_requests().get()),
            waiting: VecDeque::new(),
            budget,
            step: StepId::new(0),
            next_request_id: 0,
        })
    }

    #[must_use]
    pub fn config(&self) -> &SchedulerConfig {
        &self.config
    }

    #[must_use]
    pub fn pool(&self) -> &BlockPool {
        &self.pool
    }

    /// The step the last pass scheduled.
    #[must_use]
    pub fn step(&self) -> StepId {
        self.step
    }

    #[must_use]
    pub fn request(&self, slot: RequestSlot) -> Option<&Request> {
        self.requests.get(slot)
    }

    /// Live requests, in every phase.
    #[must_use]
    pub fn request_count(&self) -> usize {
        self.requests.len()
    }

    /// Whether the slab can take another request.
    #[must_use]
    pub fn has_room(&self) -> bool {
        self.requests.len() < self.config.max_requests.get()
    }

    #[must_use]
    pub fn running(&self) -> &[RequestSlot] {
        &self.running
    }

    #[must_use]
    pub fn waiting(&self) -> &VecDeque<RequestSlot> {
        &self.waiting
    }

    /// Takes `new` in as a Waiting request, or finishes it on the spot when it can never run.
    ///
    /// # Errors
    ///
    /// Returns the [`FinishReason`] already sent to the client when the prompt is empty or leaves
    /// no room to generate under the maximum model length.
    pub fn intake(&mut self, new: NewRequest) -> Result<RequestSlot, FinishReason> {
        let id = RequestId::new(self.next_request_id);
        self.next_request_id += 1;
        let prompt_tokens = new.prompt.len();
        let max_model_length = self.config.max_model_len.get();
        let rejection = if prompt_tokens == 0 {
            Some(FinishReason::EmptyPrompt)
        } else if prompt_tokens >= max_model_length {
            Some(FinishReason::PromptExceedsMaxModelLength {
                prompt_tokens,
                max_model_length,
            })
        } else {
            None
        };
        if let Some(reason) = rejection {
            new.egress.send(RequestEvent::Finished {
                request: id,
                reason,
                usage: Usage {
                    prompt_tokens,
                    generated_tokens: 0,
                },
            });
            debug!(request = id.get(), ?reason, "request finished at intake");
            return Err(reason);
        }
        let slot = self.requests.insert(Request::new(id, new, self.step));
        self.waiting.push_back(slot);
        Ok(slot)
    }

    /// One scheduling pass: running requests spend the budget first, then admission offers the
    /// remainder to the waiting queue over the configured window.
    pub fn schedule(&mut self) -> Scheduled {
        self.step = StepId::new(self.step.get() + 1);
        self.budget.reset();
        let mut entries = Vec::with_capacity(self.budget.max_requests().get());
        self.schedule_running(&mut entries);
        self.admit(&mut entries);
        Scheduled {
            step: self.step,
            entries,
        }
    }

    /// Records the tokens step `scheduled` computed, appending each sampling entry's token from
    /// `sampled`, in entry order.
    ///
    /// # Panics
    ///
    /// Panics when `sampled` does not hold exactly one token per sampling entry: the executor
    /// broke the step protocol, and the caller validates before applying.
    pub fn apply(&mut self, scheduled: &Scheduled, sampled: &[u32]) {
        let mut sampled = sampled.iter().copied();
        for entry in &scheduled.entries {
            let request = self
                .requests
                .get_mut(entry.slot)
                .expect("a scheduled slot stays live until its result is applied");
            let sequence = &mut request.sequences_mut()[entry.sequence.get() as usize];
            sequence.advance(entry.query_len.get());
            if entry.samples {
                let token = sampled
                    .next()
                    .expect("one sampled token per sampling entry");
                sequence.push_token(token);
            }
        }
        assert!(
            sampled.next().is_none(),
            "more sampled tokens than sampling entries"
        );
    }

    /// Budgets every running request in batch order. A request the budget or the pool cannot
    /// serve this step stays running and simply has no entry.
    fn schedule_running(&mut self, entries: &mut Vec<Entry>) {
        let block_size = self.config.block_size;
        for &slot in &self.running {
            let request = self.requests.get_mut(slot).expect("running slots are live");
            for (index, sequence) in request.sequences_mut().iter_mut().enumerate() {
                let Some(query_len) = self.budget.offer(sequence.remaining()) else {
                    return;
                };
                let context_len = sequence.computed();
                if ensure_blocks(
                    &mut self.pool,
                    block_size,
                    sequence,
                    context_len + query_len.get(),
                ) == Err(PoolExhausted)
                {
                    continue;
                }
                self.budget.spend(query_len);
                entries.push(Entry {
                    slot,
                    sequence: SequenceIndex::new(
                        u16::try_from(index).expect("sequence indices fit u16"),
                    ),
                    context_len,
                    query_len,
                    samples: context_len + query_len.get() == sequence.total(),
                });
            }
        }
    }

    /// Admission: examines up to the window from the front of the waiting queue, first come
    /// first served, and stops at the first request the budget or the pool cannot serve.
    fn admit(&mut self, entries: &mut Vec<Entry>) {
        let block_size = self.config.block_size;
        let step = self.step;
        for _ in 0..self.config.window.get() {
            let Some(&slot) = self.waiting.front() else {
                return;
            };
            let request = self.requests.get_mut(slot).expect("waiting slots are live");
            let RequestPhase::Waiting(waiting) = request.phase() else {
                unreachable!("the waiting queue holds only Waiting requests")
            };
            let sequence = &mut request.sequences_mut()[0];
            let total = sequence.total();
            let Some(query_len) = self.budget.offer(sequence.remaining()) else {
                return;
            };
            if ensure_blocks(&mut self.pool, block_size, sequence, query_len.get())
                == Err(PoolExhausted)
            {
                return;
            }
            request.set_phase(RequestPhase::Running(waiting.admit(step)));
            self.waiting.pop_front();
            self.running.push(slot);
            self.budget.spend(query_len);
            entries.push(Entry {
                slot,
                sequence: SequenceIndex::new(0),
                context_len: 0,
                query_len,
                samples: query_len.get() == total,
            });
        }
    }
}

impl Drop for Scheduler {
    /// Returns every block to the pool so no lease outlives the scheduler unreleased.
    fn drop(&mut self) {
        for (_, request) in self.requests.iter_mut() {
            for sequence in request.sequences_mut() {
                release_blocks(&mut self.pool, sequence);
            }
        }
    }
}
