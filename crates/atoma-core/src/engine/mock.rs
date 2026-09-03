//! A mock executor for tests: the far side of the rings, answering every step command with one
//! chosen token per sampling entry. No device, no thread of its own unless a test gives it one.

use crate::engine::ExecutorRings;
use crate::step::{CommandEntry, StepCommand, StepResult};

pub(crate) struct MockExecutor {
    rings: ExecutorRings,
    sample: Box<dyn FnMut(&CommandEntry) -> u32 + Send>,
    /// Every command served so far, in order.
    pub(crate) served: Vec<StepCommand>,
}

impl MockExecutor {
    /// An executor that samples `token` for every entry.
    pub(crate) fn constant(rings: ExecutorRings, token: u32) -> Self {
        Self::with(rings, move |_| token)
    }

    /// An executor that samples whatever `sample` says for each entry.
    pub(crate) fn with(
        rings: ExecutorRings,
        sample: impl FnMut(&CommandEntry) -> u32 + Send + 'static,
    ) -> Self {
        Self {
            rings,
            sample: Box::new(sample),
            served: Vec::new(),
        }
    }

    /// Serves one command if one is waiting, returning whether it did.
    pub(crate) fn serve_one(&mut self) -> bool {
        let Some(command) = self.rings.pop_command() else {
            return false;
        };
        let sampled = command
            .entries
            .iter()
            .filter(|entry| entry.samples())
            .map(|entry| (self.sample)(entry))
            .collect();
        let result = StepResult {
            step: command.step,
            sampled,
        };
        self.served.push(command);
        self.rings
            .push_result(result)
            .expect("the engine keeps at most one step in flight");
        true
    }

    /// Pushes a result of the mock's own making, for protocol-violation tests.
    pub(crate) fn push_raw(&mut self, result: StepResult) {
        self.rings.push_result(result).expect("room for one result");
    }

    pub(crate) fn engine_gone(&self) -> bool {
        self.rings.engine_gone()
    }

    /// Serves commands as they come until the engine is gone, parking between them. For tests
    /// that give the mock a thread of its own.
    pub(crate) fn run_until_engine_gone(mut self) {
        while !self.engine_gone() {
            if !self.serve_one() {
                self.rings.park();
            }
        }
    }
}
