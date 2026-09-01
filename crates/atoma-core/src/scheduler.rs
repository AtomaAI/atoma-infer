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
mod config;
mod kv;
mod pass;
mod scheduled;
#[cfg(test)]
mod tests;

use std::collections::VecDeque;

use tracing::debug;

use crate::kv::{BlockPool, PaddingReservation, PrefixIndex};
use crate::request::{
    FinishReason, Finished, NewRequest, Request, RequestEvent, RequestPhase, RequestSlab, Usage,
};
use crate::scheduler::admission::AdmissionWindow;
use crate::scheduler::kv::Kv;
use crate::types::{RequestId, RequestSlot, SequenceIndex, StepId};

pub use admission::AdmissionPolicy;
pub use budget::TokenBudget;
pub use config::{SchedulerConfig, SchedulerError};
pub use scheduled::{Entry, Scheduled};

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
        self.retire_lost_clients();
        let window = AdmissionWindow {
            size: self.config.window,
            policy: self.config.admission,
            open: self.admission_open,
        };
        pass::schedule(self.parts(), window)
    }

    /// Records the tokens step `scheduled` computed, appending each sampling entry's token from
    /// `sampled` in entry order, telling every client what it got, and retiring every request
    /// that reached a stop criterion — each of them once, however many of its sequences stopped.
    ///
    /// Blocks that fill here become cache other requests can hit, so `sampled` must come from a
    /// step whose KV writes are complete, not merely submitted.
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

    /// Retires every request whose client hung up or stopped reading, before any budget is spent
    /// on it: the whole running batch and preempted stack, and the waiting requests admission
    /// would examine. It runs at the head of every pass, so a drain that admits nothing still
    /// lets cancels through.
    ///
    /// A backlogged client is retired rather than throttled, so its blocks return to the pool
    /// instead of being held for a reader that may never come back. It keeps every event already
    /// queued for it, since the finish is appended behind them.
    fn retire_lost_clients(&mut self) {
        let max_backlog = self.config.max_client_backlog.get();
        let mut lost = Vec::new();
        let examined = self
            .running
            .iter()
            .chain(&self.preempted)
            .chain(self.waiting.iter().take(self.config.window.get()));
        for &slot in examined {
            let Some(request) = self.requests.get(slot) else {
                continue;
            };
            let queued = request.backlog();
            if request.is_cancelled() {
                lost.push((slot, FinishReason::Cancelled));
            } else if queued > max_backlog {
                lost.push((
                    slot,
                    FinishReason::ClientBacklogged {
                        queued,
                        max_backlog,
                    },
                ));
            }
        }
        for (slot, reason) in lost {
            self.retire(slot, reason);
        }
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
