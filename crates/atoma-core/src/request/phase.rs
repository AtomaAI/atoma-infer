//! Request phase: where a request is in its lifecycle, with illegal transitions unrepresentable.
//!
//! Each phase is a value that only a legal transition can produce: [`Running`] comes only from
//! admitting a [`Waiting`] or [`Preempted`] request, [`Preempted`] only from preempting a running
//! one, and [`Finished`] from any live phase. A transition that does not exist — reviving a
//! finished request, preempting one that never ran — has no method to call, so it does not compile:
//!
//! ```compile_fail
//! use atoma_core::request::Finished;
//! use atoma_core::types::StepId;
//! fn revive(finished: Finished) {
//!     finished.admit(StepId::new(1));
//! }
//! ```
//!
//! ```compile_fail
//! use atoma_core::request::Waiting;
//! use atoma_core::types::StepId;
//! fn skip_running(waiting: Waiting) {
//!     waiting.preempt(StepId::new(1));
//! }
//! ```
//!
//! ```compile_fail
//! use atoma_core::request::{Running, Preempted};
//! use atoma_core::types::StepId;
//! fn forge(admitted_at: StepId) -> Running {
//!     Running { admitted_at }
//! }
//! ```
//!
//! Nor can a phase be installed on a request from outside: minting a [`Waiting`] and moving a
//! request back into it are both the crate's, so a finished request cannot be sent round again.
//!
//! ```compile_fail
//! use atoma_core::request::{Request, RequestPhase, Waiting};
//! use atoma_core::types::StepId;
//! fn requeue(request: &mut Request, at: StepId) {
//!     request.set_phase(RequestPhase::Waiting(Waiting::new(at)));
//! }
//! ```

use crate::request::FinishReason;
use crate::types::StepId;

/// Where a request is in its lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestPhase {
    Waiting(Waiting),
    Running(Running),
    Preempted(Preempted),
    Finished(Finished),
    /// A padding dummy: it occupies its slot for the process lifetime, never finishes and never
    /// enters admission.
    Padding,
}

/// Taken in and not yet admitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Waiting {
    arrived_at: StepId,
}

impl Waiting {
    /// A request taken in during step `arrived_at`. Intake is the only way in, so this is the
    /// crate's to mint.
    #[must_use]
    pub(crate) fn new(arrived_at: StepId) -> Self {
        Self { arrived_at }
    }

    /// The step the request was taken in during.
    #[must_use]
    pub fn arrived_at(self) -> StepId {
        self.arrived_at
    }

    /// Admission: the one transition into Running.
    #[must_use]
    pub fn admit(self, at: StepId) -> Running {
        Running { admitted_at: at }
    }

    #[must_use]
    pub fn finish(self, reason: FinishReason) -> Finished {
        Finished { reason }
    }
}

/// Admitted and holding KV; prefilling or decoding is derived from its sequences' counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Running {
    admitted_at: StepId,
}

impl Running {
    /// The step of the admission that produced this phase; the newest is the preemption victim.
    #[must_use]
    pub fn admitted_at(self) -> StepId {
        self.admitted_at
    }

    /// Preemption: KV released, to run again from whatever the prefix index still holds.
    #[must_use]
    pub fn preempt(self, at: StepId) -> Preempted {
        Preempted { preempted_at: at }
    }

    #[must_use]
    pub fn finish(self, reason: FinishReason) -> Finished {
        Finished { reason }
    }
}

/// Displaced from Running; re-enters through admission ahead of every waiting request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Preempted {
    preempted_at: StepId,
}

impl Preempted {
    /// The step that preempted the request.
    #[must_use]
    pub fn preempted_at(self) -> StepId {
        self.preempted_at
    }

    /// Admission: the one transition into Running.
    #[must_use]
    pub fn admit(self, at: StepId) -> Running {
        Running { admitted_at: at }
    }

    #[must_use]
    pub fn finish(self, reason: FinishReason) -> Finished {
        Finished { reason }
    }
}

/// Over, for `reason`. No transition leaves this phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Finished {
    reason: FinishReason,
}

impl Finished {
    #[must_use]
    pub fn reason(self) -> FinishReason {
        self.reason
    }
}

#[cfg(test)]
mod tests {
    use super::{RequestPhase, Waiting};
    use crate::request::FinishReason;
    use crate::types::StepId;

    #[test]
    fn a_request_runs_only_through_admission_and_returns_through_preemption() {
        let waiting = Waiting::new(StepId::new(1));
        assert_eq!(waiting.arrived_at(), StepId::new(1));

        let running = waiting.admit(StepId::new(2));
        assert_eq!(running.admitted_at(), StepId::new(2));

        let preempted = running.preempt(StepId::new(5));
        assert_eq!(preempted.preempted_at(), StepId::new(5));

        let readmitted = preempted.admit(StepId::new(6));
        assert_eq!(readmitted.admitted_at(), StepId::new(6));
    }

    #[test]
    fn every_live_phase_finishes_with_its_reason() {
        let waiting = Waiting::new(StepId::new(1));
        let running = waiting.admit(StepId::new(2));
        let preempted = running.preempt(StepId::new(3));

        assert_eq!(
            waiting.finish(FinishReason::Cancelled).reason(),
            FinishReason::Cancelled
        );
        assert_eq!(
            running.finish(FinishReason::EndOfSequence).reason(),
            FinishReason::EndOfSequence
        );
        assert_eq!(
            preempted.finish(FinishReason::Shutdown).reason(),
            FinishReason::Shutdown
        );
        assert!(matches!(
            RequestPhase::Finished(running.finish(FinishReason::MaxNewTokens)),
            RequestPhase::Finished(_)
        ));
    }
}
