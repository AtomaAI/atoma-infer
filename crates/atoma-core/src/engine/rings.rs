//! The two single-producer single-consumer rings between the engine thread and the executor
//! thread: step commands one way, step results the other.
//!
//! Rings are not channels: each has exactly one producer and one consumer, no lock and no
//! allocation on push or pop. Their capacity is the pipeline depth — one step in flight today,
//! with room for a second when the executor can overlap.

use rtrb::{Consumer, Producer, PushError, RingBuffer};

use crate::step::{StepCommand, StepResult};

/// Steps either ring holds: pipeline-ready, not overlap.
pub const RING_CAPACITY: usize = 2;

/// The engine thread's ends: it produces commands and consumes results.
#[derive(Debug)]
pub struct EngineRings {
    commands: Producer<StepCommand>,
    results: Consumer<StepResult>,
}

/// The executor thread's ends: it consumes commands and produces results.
#[derive(Debug)]
pub struct ExecutorRings {
    commands: Consumer<StepCommand>,
    results: Producer<StepResult>,
}

/// Opens both rings, handing each thread its ends.
#[must_use]
pub fn rings() -> (EngineRings, ExecutorRings) {
    let (command_producer, command_consumer) = RingBuffer::new(RING_CAPACITY);
    let (result_producer, result_consumer) = RingBuffer::new(RING_CAPACITY);
    (
        EngineRings {
            commands: command_producer,
            results: result_consumer,
        },
        ExecutorRings {
            commands: command_consumer,
            results: result_producer,
        },
    )
}

impl EngineRings {
    /// Whether the command ring has room for another step.
    #[must_use]
    pub fn can_push(&self) -> bool {
        self.commands.slots() > 0
    }

    /// Pushes `command` for the executor, handing it back when the ring is full.
    ///
    /// # Errors
    ///
    /// Returns the command when [`RING_CAPACITY`] steps are already in flight.
    pub fn push_command(&mut self, command: StepCommand) -> Result<(), StepCommand> {
        self.commands
            .push(command)
            .map_err(|PushError::Full(command)| command)
    }

    /// The next step result, if the executor has produced one.
    pub fn pop_result(&mut self) -> Option<StepResult> {
        self.results.pop().ok()
    }

    /// Whether the executor dropped its ends: no result will ever arrive again.
    #[must_use]
    pub fn executor_gone(&self) -> bool {
        self.results.is_abandoned() || self.commands.is_abandoned()
    }
}

impl ExecutorRings {
    /// The next step command, if the engine has issued one.
    pub fn pop_command(&mut self) -> Option<StepCommand> {
        self.commands.pop().ok()
    }

    /// Pushes `result` for the engine, handing it back when the ring is full.
    ///
    /// # Errors
    ///
    /// Returns the result when [`RING_CAPACITY`] results are already waiting.
    pub fn push_result(&mut self, result: StepResult) -> Result<(), StepResult> {
        self.results
            .push(result)
            .map_err(|PushError::Full(result)| result)
    }

    /// Whether the engine dropped its ends: no command will ever arrive again.
    #[must_use]
    pub fn engine_gone(&self) -> bool {
        self.commands.is_abandoned() || self.results.is_abandoned()
    }
}

#[cfg(test)]
mod tests {
    use super::{rings, RING_CAPACITY};
    use crate::step::{StepCommand, StepResult};
    use crate::types::StepId;

    fn command(step: u64) -> StepCommand {
        StepCommand {
            step: StepId::new(step),
            entries: Vec::new(),
        }
    }

    #[test]
    fn commands_and_results_cross_in_order_and_the_rings_bound_what_is_in_flight() {
        let (mut engine, mut executor) = rings();
        assert!(engine.can_push());
        assert_eq!(executor.pop_command(), None, "nothing issued yet");
        assert_eq!(engine.pop_result(), None, "nothing produced yet");

        for step in 1..=RING_CAPACITY as u64 {
            engine.push_command(command(step)).unwrap();
        }
        assert!(!engine.can_push());
        assert_eq!(
            engine.push_command(command(99)),
            Err(command(99)),
            "a full ring hands the command back"
        );

        assert_eq!(executor.pop_command(), Some(command(1)));
        assert!(engine.can_push(), "popping frees a slot");
        for step in 1..=RING_CAPACITY as u64 {
            executor
                .push_result(StepResult {
                    step: StepId::new(step),
                    sampled: vec![7],
                })
                .unwrap();
        }
        assert!(executor
            .push_result(StepResult {
                step: StepId::new(3),
                sampled: Vec::new()
            })
            .is_err());
        assert_eq!(
            engine.pop_result().map(|result| result.step),
            Some(StepId::new(1))
        );
        assert_eq!(
            engine.pop_result().map(|result| result.step),
            Some(StepId::new(2))
        );
        assert_eq!(engine.pop_result(), None);
    }

    #[test]
    fn each_side_sees_the_other_leave() {
        let (engine, executor) = rings();
        assert!(!engine.executor_gone());
        assert!(!executor.engine_gone());
        drop(executor);
        assert!(engine.executor_gone());

        let (engine, executor) = rings();
        drop(engine);
        assert!(executor.engine_gone());
    }
}
