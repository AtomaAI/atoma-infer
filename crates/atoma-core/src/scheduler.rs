//! The token-budget scheduler.
//!
//! One scheduling pass spends a per-step token budget: running requests first, then admission
//! from the preempted stack and the waiting queue over a bounded window. Admission consults the
//! prefix index, so a request starts past every block already cached; blocks that fill are
//! hashed and indexed at once. A running request the pool cannot grow evicts unpinned cache
//! first and only then preempts the most recently admitted request, which releases its KV and
//! later recomputes from whatever the index still holds; nothing is ever swapped out to be
//! brought back. It answers in indices
//! and counts — which sequences run, how many tokens each computes, which entries sample — never in
//! copied request state. The scheduler owns the request slab and the block pool outright; nothing
//! here is shared or locked.

mod admission;
mod budget;
mod kv;
#[cfg(test)]
mod tests;

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::debug;

use crate::dispatch::LiveBatch;
use crate::kv::{BlockPool, HashAlgorithm, PaddingReservation, PrefixIndex};
use crate::request::{
    FinishReason, Finished, NewRequest, Request, RequestEvent, RequestPhase, RequestSlab, Usage,
};
use crate::scheduler::kv::{Kv, PoolExhausted};
use crate::types::{RequestCount, RequestId, RequestSlot, SequenceIndex, StepId, TokenCount};

pub use admission::AdmissionPolicy;
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
    /// How admission orders the window.
    pub admission: AdmissionPolicy,
    /// Requests the slab holds before intake is refused.
    pub max_requests: RequestCount,
    /// The model's end-of-sequence token ids; sampling one finishes a request that does not
    /// ignore it.
    pub eos_token_ids: Vec<u32>,
    /// The chain-hash algorithm block identity is minted under.
    pub hash_algorithm: HashAlgorithm,
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
    /// Requests this pass displaced from Running, most recent last.
    pub preempted: Vec<RequestSlot>,
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

    /// Live requests in the batch: entries address sequences, and a request's sequences sit
    /// together in batch order.
    #[must_use]
    pub fn request_count(&self) -> usize {
        self.entries
            .iter()
            .fold((0, None), |(count, last), entry| {
                if last == Some(entry.slot) {
                    (count, last)
                } else {
                    (count + 1, Some(entry.slot))
                }
            })
            .0
    }

    /// The shape of this pass before padding, or `None` when nothing was scheduled.
    #[must_use]
    pub fn live_batch(&self) -> Option<LiveBatch> {
        Some(LiveBatch {
            token_count: TokenCount::new(self.token_count())?,
            request_count: RequestCount::new(self.request_count())?,
            uniform_decode: self.is_uniform_decode(),
        })
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
    index: PrefixIndex,
    /// Admission order, which is batch order. The last admitted is the preemption victim.
    running: Vec<RequestSlot>,
    /// Arrival order. Admission examines a bounded window from the front.
    waiting: VecDeque<RequestSlot>,
    /// Displaced requests, most recent last; admission offers the top first.
    preempted: Vec<RequestSlot>,
    /// The padding dummies' slots, in reservation order. Never in any queue.
    padding: Vec<RequestSlot>,
    /// The leases behind the dummies' blocks, held until the scheduler is dropped.
    padding_reservation: Option<PaddingReservation>,
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
            index: PrefixIndex::new(),
            running: Vec::with_capacity(budget.max_requests().get()),
            waiting: VecDeque::new(),
            preempted: Vec::new(),
            padding: Vec::new(),
            padding_reservation: None,
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

    #[must_use]
    pub fn index(&self) -> &PrefixIndex {
        &self.index
    }

    /// The KV substrate and the request slab, borrowed apart so a sequence and the pool can
    /// change together.
    fn kv_and_requests(&mut self) -> (Kv<'_>, &mut RequestSlab) {
        let kv = Kv {
            pool: &mut self.pool,
            index: &mut self.index,
            algorithm: self.config.hash_algorithm,
            block_size: self.config.block_size,
        };
        (kv, &mut self.requests)
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

    /// Live requests, in every phase; padding dummies are not counted.
    #[must_use]
    pub fn request_count(&self) -> usize {
        self.requests.len() - self.padding.len()
    }

    /// Whether the slab can take another request.
    #[must_use]
    pub fn has_room(&self) -> bool {
        self.request_count() < self.config.max_requests.get()
    }

    /// Builds a scheduler over `pool` with the padding dummies of `reservation` in its slab:
    /// one request per reserved block, occupying its slot from now on, never finishing and
    /// never entering admission. The reservation was taken from `pool`, and returns to it when
    /// the scheduler is dropped.
    ///
    /// # Errors
    ///
    /// Returns [`SchedulerError::PoolTooSmallForMaxModelLength`] when what the reservation left
    /// of the pool cannot hold one maximum-length request.
    pub fn with_padding(
        config: SchedulerConfig,
        pool: BlockPool,
        reservation: PaddingReservation,
    ) -> Result<Self, SchedulerError> {
        let mut scheduler = Self::new(config, pool)?;
        for block in reservation.block_ids() {
            let id = RequestId::new(scheduler.next_request_id);
            scheduler.next_request_id += 1;
            let slot = scheduler.requests.insert(Request::padding(id, block));
            scheduler.padding.push(slot);
        }
        scheduler.padding_reservation = Some(reservation);
        Ok(scheduler)
    }

    /// The padding dummies' slots, in reservation order.
    #[must_use]
    pub fn padding(&self) -> &[RequestSlot] {
        &self.padding
    }

    #[must_use]
    pub fn running(&self) -> &[RequestSlot] {
        &self.running
    }

    #[must_use]
    pub fn waiting(&self) -> &VecDeque<RequestSlot> {
        &self.waiting
    }

    /// Displaced requests, most recent last.
    #[must_use]
    pub fn preempted(&self) -> &[RequestSlot] {
        &self.preempted
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
        let mut request = Request::new(id, new, self.step);
        for sequence in request.sequences_mut() {
            sequence.extend_chain(self.config.hash_algorithm, self.config.block_size);
        }
        let slot = self.requests.insert(request);
        self.waiting.push_back(slot);
        Ok(slot)
    }

    /// One scheduling pass: requests whose client hung up retire, running requests spend the
    /// budget first — preempting the most recently admitted when the pool cannot grow them —
    /// then admission offers the remainder to the preempted stack and the waiting queue over the
    /// configured window.
    pub fn schedule(&mut self) -> Scheduled {
        self.step = StepId::new(self.step.get() + 1);
        self.budget.reset();
        self.retire_cancelled_running();
        let mut entries = Vec::with_capacity(self.budget.max_requests().get());
        let preempted = self.schedule_running(&mut entries);
        self.admit(&mut entries);
        Scheduled {
            step: self.step,
            entries,
            preempted,
        }
    }

    /// Records the tokens step `scheduled` computed, appending each sampling entry's token from
    /// `sampled` in entry order, telling every client what it got, and finishing the requests
    /// that reached a stop criterion.
    ///
    /// # Panics
    ///
    /// Panics when `sampled` does not hold exactly one token per sampling entry: the executor
    /// broke the step protocol, and the caller validates before applying.
    pub fn apply(&mut self, scheduled: &Scheduled, sampled: &[u32]) {
        let mut sampled = sampled.iter().copied();
        let mut finished = Vec::new();
        for entry in &scheduled.entries {
            let (mut kv, requests) = self.kv_and_requests();
            let request = requests
                .get_mut(entry.slot)
                .expect("a scheduled slot stays live until its result is applied");
            let sequence = &mut request.sequences_mut()[entry.sequence.get() as usize];
            sequence.advance(entry.query_len.get());
            let token = entry.samples.then(|| {
                let token = sampled
                    .next()
                    .expect("one sampled token per sampling entry");
                sequence.push_token(token);
                token
            });
            kv.cache_filled_blocks(sequence);
            let Some(token) = token else {
                continue;
            };
            request.send(RequestEvent::Token {
                request: request.id(),
                sequence: entry.sequence,
                token,
            });
            if let Some(reason) = self.stop_reason(entry.slot, entry.sequence, token) {
                finished.push((entry.slot, reason));
            }
        }
        assert!(
            sampled.next().is_none(),
            "more sampled tokens than sampling entries"
        );
        for (slot, reason) in finished {
            self.retire(slot, reason);
        }
    }

    /// Why the request at `slot` stops after sampling `token`, if it does.
    fn stop_reason(
        &self,
        slot: RequestSlot,
        sequence: SequenceIndex,
        token: u32,
    ) -> Option<FinishReason> {
        let request = self.requests.get(slot).expect("checked live by the caller");
        let sequence = &request.sequences()[sequence.get() as usize];
        let stop = request.stop();
        if self.config.eos_token_ids.contains(&token) && !stop.ignore_eos {
            Some(FinishReason::EndOfSequence)
        } else if sequence.generated_count() >= stop.max_new_tokens.get() {
            Some(FinishReason::MaxNewTokens)
        } else if sequence.total() >= self.config.max_model_len.get() {
            Some(FinishReason::MaxModelLength)
        } else {
            None
        }
    }

    /// Finishes the request at `slot` for `reason`: its KV returns to the pool, its client hears
    /// the finish, and its slot is freed. Works from any live phase.
    fn retire(&mut self, slot: RequestSlot, reason: FinishReason) {
        let mut request = self.requests.remove(slot);
        let finished: Finished = match request.phase() {
            RequestPhase::Waiting(waiting) => {
                if self.waiting.front() == Some(&slot) {
                    self.waiting.pop_front();
                } else {
                    self.waiting.retain(|waiting| *waiting != slot);
                }
                waiting.finish(reason)
            }
            RequestPhase::Running(running) => {
                self.running.retain(|running| *running != slot);
                running.finish(reason)
            }
            RequestPhase::Preempted(preempted) => {
                self.preempted.retain(|preempted| *preempted != slot);
                preempted.finish(reason)
            }
            RequestPhase::Finished(_) | RequestPhase::Padding => {
                unreachable!("only live requests retire")
            }
        };
        let (mut kv, _) = self.kv_and_requests();
        for sequence in request.sequences_mut() {
            kv.release(sequence);
        }
        request.set_phase(RequestPhase::Finished(finished));
        request.send(RequestEvent::Finished {
            request: request.id(),
            reason: finished.reason(),
            usage: request.usage(),
        });
        debug!(request = request.id().get(), reason = ?finished.reason(), "request finished");
    }

    /// Retires every running request whose client hung up, before any budget is spent on it.
    fn retire_cancelled_running(&mut self) {
        let mut cancelled = Vec::new();
        for &slot in &self.running {
            if self.requests.get(slot).is_some_and(Request::is_cancelled) {
                cancelled.push(slot);
            }
        }
        for slot in cancelled {
            self.retire(slot, FinishReason::Cancelled);
        }
    }

    /// Budgets every running request in batch order, returning the requests preempted on the
    /// way. A request the budget cannot serve this step stays running with no entry; one the
    /// pool cannot grow preempts the most recently admitted request — itself, if it is the
    /// newest — and retries.
    fn schedule_running(&mut self, entries: &mut Vec<Entry>) -> Vec<RequestSlot> {
        let mut preempted = Vec::new();
        let mut position = 0;
        while position < self.running.len() {
            let slot = self.running[position];
            let request = self.requests.get_mut(slot).expect("running slots are live");
            let sequence_count = request.sequences().len();
            let mut displaced_self = false;
            for index in 0..sequence_count {
                let sequence =
                    &mut self.requests.get_mut(slot).expect("live").sequences_mut()[index];
                let Some(query_len) = self.budget.offer(sequence.remaining()) else {
                    return preempted;
                };
                let context_len = sequence.computed();
                let total = sequence.total();
                let sequence_len = context_len + query_len.get();
                while {
                    let (mut kv, requests) = self.kv_and_requests();
                    let sequence =
                        &mut requests.get_mut(slot).expect("live").sequences_mut()[index];
                    kv.ensure_blocks(sequence, sequence_len)
                } == Err(PoolExhausted)
                {
                    let victim = *self.running.last().expect("this request is running");
                    self.preempt(victim);
                    preempted.push(victim);
                    if victim == slot {
                        displaced_self = true;
                        break;
                    }
                }
                if displaced_self {
                    break;
                }
                self.budget.spend(query_len);
                entries.push(Entry {
                    slot,
                    sequence: SequenceIndex::new(
                        u16::try_from(index).expect("sequence indices fit u16"),
                    ),
                    context_len,
                    query_len,
                    samples: sequence_len == total,
                });
            }
            if !displaced_self {
                position += 1;
            }
        }
        preempted
    }

    /// Preemption: the request at `slot`, the most recently admitted, releases its KV and will
    /// recompute from the start when it re-enters Running.
    fn preempt(&mut self, slot: RequestSlot) {
        assert_eq!(
            self.running.pop(),
            Some(slot),
            "the preemption victim is the most recently admitted request"
        );
        let Scheduler {
            config,
            requests,
            pool,
            index,
            running: _,
            waiting: _,
            preempted,
            padding: _,
            padding_reservation: _,
            budget: _,
            step,
            next_request_id: _,
        } = self;
        let mut kv = Kv {
            pool,
            index,
            algorithm: config.hash_algorithm,
            block_size: config.block_size,
        };
        let request = requests.get_mut(slot).expect("running slots are live");
        let RequestPhase::Running(running) = request.phase() else {
            unreachable!("the running queue holds only Running requests")
        };
        request.set_phase(RequestPhase::Preempted(running.preempt(*step)));
        for sequence in request.sequences_mut() {
            kv.release(sequence);
            sequence.reset_for_recompute();
        }
        preempted.push(slot);
        debug!(request = request.id().get(), "request preempted");
    }

    /// Admission: examines up to the window of candidates — the preempted stack top first, last
    /// in first out, then the waiting request the policy picks from the window — claims each
    /// one's cached prefix, and stops at the first request the budget or the pool cannot serve.
    /// Admission never preempts.
    fn admit(&mut self, entries: &mut Vec<Entry>) {
        for _ in 0..self.config.window.get() {
            let Some((slot, from)) = self.next_candidate() else {
                return;
            };
            if self.requests.get(slot).is_some_and(Request::is_cancelled) {
                self.retire(slot, FinishReason::Cancelled);
                continue;
            }
            let Scheduler {
                config,
                requests,
                pool,
                index,
                running,
                waiting,
                preempted,
                padding: _,
                padding_reservation: _,
                budget,
                step,
                next_request_id: _,
            } = self;
            let mut kv = Kv {
                pool,
                index,
                algorithm: config.hash_algorithm,
                block_size: config.block_size,
            };
            let request = requests.get_mut(slot).expect("candidates are live");
            let sequence = &mut request.sequences_mut()[0];
            let total = sequence.total();
            kv.claim_prefix(sequence);
            let context_len = sequence.computed();
            let Some(query_len) = budget.offer(sequence.remaining()) else {
                kv.release(sequence);
                sequence.reset_for_recompute();
                return;
            };
            if kv.ensure_blocks(sequence, context_len + query_len.get()) == Err(PoolExhausted) {
                kv.release(sequence);
                sequence.reset_for_recompute();
                return;
            }
            let admitted = match (from, request.phase()) {
                (Candidate::Preempted, RequestPhase::Preempted(phase)) => {
                    preempted.pop();
                    phase.admit(*step)
                }
                (Candidate::Waiting { position }, RequestPhase::Waiting(phase)) => {
                    waiting.remove(position);
                    phase.admit(*step)
                }
                (Candidate::Preempted | Candidate::Waiting { .. }, _) => {
                    unreachable!("each admission queue holds only its own phase")
                }
            };
            request.set_phase(RequestPhase::Running(admitted));
            running.push(slot);
            budget.spend(query_len);
            entries.push(Entry {
                slot,
                sequence: SequenceIndex::new(0),
                context_len,
                query_len,
                samples: context_len + query_len.get() == total,
            });
        }
    }
}

impl Scheduler {
    /// The next admission candidate: the preempted stack top, or the waiting request the policy
    /// picks from the window. Cancelled waiting requests inside the window retire on the way.
    fn next_candidate(&mut self) -> Option<(RequestSlot, Candidate)> {
        if let Some(&slot) = self.preempted.last() {
            return Some((slot, Candidate::Preempted));
        }
        self.retire_cancelled_in_window();
        let position = match self.config.admission {
            AdmissionPolicy::Fcfs => 0,
            AdmissionPolicy::LongestPrefixMatch => self.longest_prefix_position()?,
        };
        self.waiting
            .get(position)
            .map(|&slot| (slot, Candidate::Waiting { position }))
    }

    /// Retires every cancelled request within the window, so none sits there unexamined.
    fn retire_cancelled_in_window(&mut self) {
        let mut cancelled = Vec::new();
        for &slot in self.waiting.iter().take(self.config.window.get()) {
            if self.requests.get(slot).is_some_and(Request::is_cancelled) {
                cancelled.push(slot);
            }
        }
        for slot in cancelled {
            self.retire(slot, FinishReason::Cancelled);
        }
    }

    /// The position in the waiting queue, within the window, of the request with the most
    /// cached prefix blocks; ties go to the earliest arrival. A selection, never a sort.
    fn longest_prefix_position(&mut self) -> Option<usize> {
        let block_size = self.config.block_size;
        let mut best: Option<(usize, usize)> = None;
        for (position, &slot) in self
            .waiting
            .iter()
            .take(self.config.window.get())
            .enumerate()
        {
            let sequence = &self
                .requests
                .get(slot)
                .expect("waiting slots are live")
                .sequences()[0];
            let candidates = sequence.hashable_prefix_blocks(block_size);
            let hits = self.index.lookup(&sequence.chain[..candidates]);
            if best.is_none_or(|(_, best_hits)| hits > best_hits) {
                best = Some((position, hits));
            }
        }
        best.map(|(position, _)| position)
    }
}

/// Which admission queue a candidate came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Candidate {
    Preempted,
    Waiting { position: usize },
}

impl Drop for Scheduler {
    /// Returns every block to the pool so no lease outlives the scheduler unreleased.
    fn drop(&mut self) {
        let (mut kv, requests) = self.kv_and_requests();
        for (_, request) in requests.iter_mut() {
            for sequence in request.sequences_mut() {
                kv.release(sequence);
            }
        }
        if let Some(reservation) = self.padding_reservation.take() {
            reservation.release(&mut self.pool);
        }
    }
}
