//! The token-budget scheduler.
//!
//! One scheduling pass spends a per-step token budget: running requests first, then admission
//! from the preempted stack and the waiting queue over a bounded window. Admission consults the
//! prefix index, so a request starts past every block already cached; blocks that fill are
//! hashed and indexed at once. A running request the pool cannot grow evicts unpinned cache
//! first and only then preempts the most recently admitted request, which releases its KV and
//! later recomputes from whatever the index still holds; nothing is ever swapped out to be
//! brought back. A pass answers in indices and counts — which sequences run, how many tokens each
//! computes, which entries sample — never in copied request state. The scheduler owns the request
//! slab and the block pool outright; nothing here is shared or locked.

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
    FinishReason, Finished, NewRequest, Request, RequestEvent, RequestPhase, RequestSlab, Sequence,
    Usage,
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
    /// Whether admission may move waiting requests into Running; a drain closes it for good.
    /// Preempted requests are offered either way: they have already run.
    admission_open: bool,
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
            admission_open: true,
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

    /// The scheduler borrowed apart, so a sequence, the KV substrate, the queues and the budget
    /// can change together in one pass.
    fn parts(&mut self) -> Parts<'_> {
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
            admission_open: _,
            budget,
            step,
            next_request_id: _,
        } = self;
        Parts {
            kv: Kv {
                pool,
                index,
                algorithm: config.hash_algorithm,
                block_size: config.block_size,
            },
            requests,
            running,
            waiting,
            preempted,
            budget,
            step: *step,
        }
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

    /// Live requests, in every phase; padding dummies are not counted. Not the batch's request
    /// count, which is a [`Scheduled`]'s own.
    #[must_use]
    pub fn live_request_count(&self) -> usize {
        self.requests.len() - self.padding.len()
    }

    /// Whether the slab can take another request.
    #[must_use]
    pub fn has_room(&self) -> bool {
        self.live_request_count() < self.config.max_requests.get()
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

    /// Stops admission for waiting requests, for good. Running requests finish, and preempted
    /// requests — running requests the pool displaced — still re-enter to finish; nothing that
    /// has never run enters Running again.
    pub fn close_admission(&mut self) {
        self.admission_open = false;
    }

    #[must_use]
    pub fn is_admission_open(&self) -> bool {
        self.admission_open
    }

    /// Retires every live request for `reason` — waiting, running and preempted alike —
    /// returning its KV and telling its client. The padding dummies stay.
    pub fn retire_all(&mut self, reason: FinishReason) {
        let live: Vec<RequestSlot> = self
            .requests
            .iter()
            .filter(|(_, request)| !request.is_padding())
            .map(|(slot, _)| slot)
            .collect();
        for slot in live {
            self.retire(slot, reason);
        }
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
    /// Returns the slot it waits in, or `None` when it was finished on the spot: an empty prompt,
    /// or one leaving no room to generate under the maximum model length. A request finished here
    /// has already been told why and never took a slot, so a caller with nothing else to do with
    /// the slot can ignore the answer.
    pub fn intake(&mut self, new: NewRequest) -> Option<RequestSlot> {
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
            return None;
        }
        let mut request = Request::new(id, new, self.step);
        for sequence in request.sequences_mut() {
            sequence.extend_chain(self.config.hash_algorithm, self.config.block_size);
        }
        let slot = self.requests.insert(request);
        self.waiting.push_back(slot);
        Some(slot)
    }

    /// One scheduling pass: requests whose client hung up retire in every queue, running requests
    /// spend the budget first — preempting the most recently admitted when the pool cannot grow
    /// them — then admission offers the remainder to the preempted stack and the waiting queue
    /// over the configured window.
    pub fn schedule(&mut self) -> Scheduled {
        self.step = StepId::new(self.step.get() + 1);
        self.budget.reset();
        self.retire_cancelled();
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
    /// `sampled` in entry order, telling every client what it got, and retiring every request
    /// that reached a stop criterion — each of them once, however many of its sequences stopped.
    ///
    /// # Panics
    ///
    /// Panics when `sampled` does not hold exactly one token per sampling entry: the executor
    /// broke the step protocol, and the caller validates before applying.
    pub fn apply(&mut self, scheduled: &Scheduled, sampled: &[u32]) {
        let mut sampled = sampled.iter().copied();
        let mut finished = Vec::new();
        for entry in &scheduled.entries {
            let mut parts = self.parts();
            let request = parts
                .requests
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
            parts.kv.cache_filled_blocks(sequence);
            let Some(token) = token else {
                continue;
            };
            request.send(RequestEvent::Token {
                request: request.id(),
                sequence: entry.sequence,
                token,
            });
            if let Some(reason) = self.stop_reason(entry.slot, entry.sequence, token) {
                // A request finishes once, for the first of its sequences to reach a stop
                // criterion: retiring frees the slot, so a second entry for the same request
                // would retire a slot that is no longer there.
                if !finished.iter().any(|(finished, _)| *finished == entry.slot) {
                    finished.push((entry.slot, reason));
                }
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
        let mut parts = self.parts();
        for sequence in request.sequences_mut() {
            parts.kv.release(sequence);
        }
        request.set_phase(RequestPhase::Finished(finished));
        request.send(RequestEvent::Finished {
            request: request.id(),
            reason: finished.reason(),
            usage: request.usage(),
        });
        debug!(request = request.id().get(), reason = ?finished.reason(), "request finished");
    }

    /// Retires every request whose client hung up, before any budget is spent on it: the whole
    /// running batch and preempted stack, and the waiting requests admission would examine. It
    /// runs at the head of every pass, so a drain that admits nothing still lets cancels through.
    fn retire_cancelled(&mut self) {
        let mut cancelled = Vec::new();
        let examined = self
            .running
            .iter()
            .chain(&self.preempted)
            .chain(self.waiting.iter().take(self.config.window.get()));
        for &slot in examined {
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
            match self.budget_running(slot, entries, &mut preempted) {
                // The queue shrank under this position, so the next request is at it already.
                Budgeted::DisplacedItself => {}
                Budgeted::Scheduled => position += 1,
                Budgeted::BudgetSpent => return preempted,
            }
        }
        preempted
    }

    /// Gives every sequence of the running request at `slot` an entry, growing its blocks and
    /// spending the budget for each.
    fn budget_running(
        &mut self,
        slot: RequestSlot,
        entries: &mut Vec<Entry>,
        preempted: &mut Vec<RequestSlot>,
    ) -> Budgeted {
        let sequence_count = self
            .requests
            .get(slot)
            .expect("running slots are live")
            .sequences()
            .len();
        for index in 0..sequence_count {
            let sequence = &self.requests.get(slot).expect("live").sequences()[index];
            let Some(query_len) = self.budget.offer(sequence.remaining()) else {
                return Budgeted::BudgetSpent;
            };
            let context_len = sequence.computed();
            let total = sequence.total();
            let sequence_len = context_len + query_len.get();
            if self.grow(slot, index, sequence_len, preempted) == Grown::DisplacedItself {
                return Budgeted::DisplacedItself;
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
        Budgeted::Scheduled
    }

    /// Grows sequence `index` of `slot` to cover `sequence_len` tokens, preempting the most
    /// recently admitted request as often as it takes and recording each victim in `preempted`.
    /// The victim is this request itself when it is the newest one running.
    fn grow(
        &mut self,
        slot: RequestSlot,
        index: usize,
        sequence_len: usize,
        preempted: &mut Vec<RequestSlot>,
    ) -> Grown {
        loop {
            let mut parts = self.parts();
            let sequence = &mut parts.requests.get_mut(slot).expect("live").sequences_mut()[index];
            if parts.kv.ensure_blocks(sequence, sequence_len) != Err(PoolExhausted) {
                return Grown::Fits;
            }
            let victim = self.preempt_newest();
            preempted.push(victim);
            if victim == slot {
                return Grown::DisplacedItself;
            }
        }
    }

    /// Preemption: the most recently admitted running request releases its KV and computes from
    /// its first token again when it re-enters Running. Returns the request displaced.
    fn preempt_newest(&mut self) -> RequestSlot {
        let slot = self
            .running
            .pop()
            .expect("preemption has a running request to displace");
        let mut parts = self.parts();
        let request = parts
            .requests
            .get_mut(slot)
            .expect("running slots are live");
        let RequestPhase::Running(running) = request.phase() else {
            unreachable!("the running queue holds only Running requests")
        };
        request.set_phase(RequestPhase::Preempted(running.preempt(parts.step)));
        for sequence in request.sequences_mut() {
            parts.kv.release(sequence);
            sequence.forget_computed();
        }
        parts.preempted.push(slot);
        debug!(request = request.id().get(), "request preempted");
        slot
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
            if self.admit_candidate(slot, from, entries) == Admitted::No {
                return;
            }
        }
    }

    /// Admits one candidate: it claims its cached prefix, takes an entry out of the budget and
    /// leaves the queue it came from. A candidate the budget or the pool cannot serve goes back
    /// untouched, and its refusal ends this pass's admission.
    fn admit_candidate(
        &mut self,
        slot: RequestSlot,
        from: Candidate,
        entries: &mut Vec<Entry>,
    ) -> Admitted {
        let mut parts = self.parts();
        let request = parts.requests.get_mut(slot).expect("candidates are live");
        let sequence = &mut request.sequences_mut()[0];
        let total = sequence.total();
        parts.kv.claim_prefix(sequence);
        let context_len = sequence.computed();
        let Some(query_len) = parts.budget.offer(sequence.remaining()) else {
            return give_back(&mut parts.kv, sequence);
        };
        if parts
            .kv
            .ensure_blocks(sequence, context_len + query_len.get())
            == Err(PoolExhausted)
        {
            return give_back(&mut parts.kv, sequence);
        }
        let admitted = match (from, request.phase()) {
            (Candidate::Preempted, RequestPhase::Preempted(phase)) => {
                parts.preempted.pop();
                phase.admit(parts.step)
            }
            (Candidate::Waiting { position }, RequestPhase::Waiting(phase)) => {
                parts.waiting.remove(position);
                phase.admit(parts.step)
            }
            (
                Candidate::Preempted | Candidate::Waiting { .. },
                RequestPhase::Waiting(_)
                | RequestPhase::Running(_)
                | RequestPhase::Preempted(_)
                | RequestPhase::Finished(_)
                | RequestPhase::Padding,
            ) => unreachable!("each admission queue holds only its own phase"),
        };
        request.set_phase(RequestPhase::Running(admitted));
        parts.running.push(slot);
        parts.budget.spend(query_len);
        entries.push(Entry {
            slot,
            sequence: SequenceIndex::new(0),
            context_len,
            query_len,
            samples: context_len + query_len.get() == total,
        });
        Admitted::Yes
    }
}

/// Puts a refused candidate back as it was: what its claim pinned or leased goes back to the
/// pool, and the sequence forgets what the claim counted as computed.
fn give_back(kv: &mut Kv<'_>, sequence: &mut Sequence) -> Admitted {
    kv.release(sequence);
    sequence.forget_computed();
    Admitted::No
}

impl Scheduler {
    /// The next admission candidate: the preempted stack top, or — while admission is open —
    /// the waiting request the policy picks from the window. A closed admission still offers the
    /// preempted stack, since those are running requests on their way to finishing. Cancelled
    /// candidates are gone already: the pass sweeps them before it spends anything.
    fn next_candidate(&mut self) -> Option<(RequestSlot, Candidate)> {
        if let Some(&slot) = self.preempted.last() {
            return Some((slot, Candidate::Preempted));
        }
        if !self.admission_open {
            return None;
        }
        let position = match self.config.admission {
            AdmissionPolicy::Fcfs => 0,
            AdmissionPolicy::LongestPrefixMatch => self.longest_prefix_position()?,
        };
        self.waiting
            .get(position)
            .map(|&slot| (slot, Candidate::Waiting { position }))
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
            let hits = self.index.lookup(sequence.hashable_prefix(block_size));
            if best.is_none_or(|(_, best_hits)| hits > best_hits) {
                best = Some((position, hits));
            }
        }
        best.map(|(position, _)| position)
    }
}

/// The scheduler's state borrowed field by field, so one pass can move a request between
/// queues while its sequences change what they hold.
struct Parts<'a> {
    kv: Kv<'a>,
    requests: &'a mut RequestSlab,
    running: &'a mut Vec<RequestSlot>,
    waiting: &'a mut VecDeque<RequestSlot>,
    preempted: &'a mut Vec<RequestSlot>,
    budget: &'a mut TokenBudget,
    /// The step being scheduled; the phase a transition produces is stamped with it.
    step: StepId,
}

/// Whether a candidate was admitted, or put back for the pool or the budget to allow later.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Admitted {
    Yes,
    No,
}

/// How a running request's turn at the budget ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Budgeted {
    /// Every sequence has an entry this step.
    Scheduled,
    /// The pool could not grow the request and it was the newest running one, so it displaced
    /// itself and is no longer in the running queue.
    DisplacedItself,
    /// The budget has nothing left for another entry; the pass is over.
    BudgetSpent,
}

/// How growing a sequence's blocks ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Grown {
    /// The sequence holds blocks for every token the step computes.
    Fits,
    /// The request needing the room was the newest running one, so it was its own victim.
    DisplacedItself,
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
        let mut parts = self.parts();
        for (_, request) in parts.requests.iter_mut() {
            for sequence in request.sequences_mut() {
                parts.kv.release(sequence);
            }
        }
        if let Some(reservation) = self.padding_reservation.take() {
            reservation.release(&mut self.pool);
        }
    }
}
