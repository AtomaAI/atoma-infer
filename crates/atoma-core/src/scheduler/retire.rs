//! Retiring a request: why one stops, how it leaves whichever queue holds it, and the sweep that
//! clears out the requests whose clients are gone.
//!
//! A retire works from any live phase and always ends the same way: the request's KV goes back to
//! the pool, its client hears the finish, and its slot is freed. Nothing here decides what runs;
//! it decides what stops running.

use tracing::debug;

use crate::request::{FinishReason, Finished, RequestEvent, RequestPhase};
use crate::scheduler::Scheduler;
use crate::types::{RequestSlot, SequenceIndex};

impl Scheduler {
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

    /// Why the request at `slot` stops after sampling `token`, if it does.
    pub(crate) fn stop_reason(
        &self,
        slot: RequestSlot,
        sequence: SequenceIndex,
        token: u32,
    ) -> Option<FinishReason> {
        let request = self
            .requests
            .get(slot)
            .expect("a scheduled slot stays live until its result is applied");
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
    pub(crate) fn retire(&mut self, slot: RequestSlot, reason: FinishReason) {
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
    pub(crate) fn retire_lost_clients(&mut self) {
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
