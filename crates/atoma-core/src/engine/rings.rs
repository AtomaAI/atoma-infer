//! The two single-producer single-consumer rings between the engine thread and the executor
//! thread: step commands one way, step results the other.
//!
//! Rings are not channels: each has exactly one producer and one consumer, no lock and no
//! allocation on push or pop. Each side parks between steps and is woken by the other: a pushed
//! command wakes the executor, a pushed result wakes the engine, and either side dropping its ends
//! wakes the other so the loss is seen rather than waited on.

use crossbeam_utils::sync::{Parker, Unparker};
use rtrb::{Consumer, Producer, PushError, RingBuffer};

use crate::step::{StepCommand, StepResult};

/// Steps either ring holds. The engine keeps one step in flight, so two always leaves room for
/// the push that follows a pop.
pub const RING_CAPACITY: usize = 2;

/// The engine thread's ends: it produces commands and consumes results, waking the executor
/// thread after every command.
#[derive(Debug)]
pub struct EngineRings {
    commands: Producer<StepCommand>,
    results: Consumer<StepResult>,
    /// Declared after the ring ends, and so dropped after them: fields drop in declaration order,
    /// and it is those ends dropping that makes the loss visible through
    /// [`ExecutorRings::engine_gone`]. Waking first would let a woken executor look, find both
    /// rings intact, and park again.
    wake: WakeOnDrop,
}

/// The executor thread's ends: it consumes commands and produces results, waking the engine
/// thread after every result, and parks here until the engine wakes it.
#[derive(Debug)]
pub struct ExecutorRings {
    commands: Consumer<StepCommand>,
    results: Producer<StepResult>,
    parker: Parker,
    /// Declared after the ring ends for the same reason as [`EngineRings::wake`]: the loss must
    /// be visible through [`EngineRings::executor_gone`] before the engine is woken to look.
    wake: WakeOnDrop,
}

/// An unparker that wakes the thread on the other side of a ring when it is dropped.
///
/// The wake is a field's drop rather than a `Drop` on the rings struct itself, because that runs
/// before any of the struct's fields drop — it would wake the other thread while the rings it is
/// about to examine are still held. Declare it after the ring ends it goes with.
#[derive(Debug)]
pub struct WakeOnDrop(Unparker);

impl WakeOnDrop {
    #[must_use]
    pub fn new(unparker: Unparker) -> Self {
        Self(unparker)
    }

    /// Wakes the other thread now.
    pub fn wake(&self) {
        self.0.unpark();
    }
}

impl Drop for WakeOnDrop {
    fn drop(&mut self) {
        self.wake();
    }
}

/// Opens both rings, handing each thread its ends; `wake` unparks the engine thread after
/// every result the executor pushes. The executor's own parker lives in its ends.
#[must_use]
pub fn rings(wake: Unparker) -> (EngineRings, ExecutorRings) {
    let (command_producer, command_consumer) = RingBuffer::new(RING_CAPACITY);
    let (result_producer, result_consumer) = RingBuffer::new(RING_CAPACITY);
    let executor_parker = Parker::new();
    let wake_executor = WakeOnDrop::new(executor_parker.unparker().clone());
    (
        EngineRings {
            commands: command_producer,
            results: result_consumer,
            wake: wake_executor,
        },
        ExecutorRings {
            commands: command_consumer,
            results: result_producer,
            parker: executor_parker,
            wake: WakeOnDrop::new(wake),
        },
    )
}

impl EngineRings {
    /// Pushes `command` for the executor and wakes it, handing the command back when the ring
    /// is full.
    ///
    /// # Errors
    ///
    /// Returns the command when [`RING_CAPACITY`] steps are already in flight.
    pub fn push_command(&mut self, command: StepCommand) -> Result<(), StepCommand> {
        self.commands
            .push(command)
            .map_err(|PushError::Full(command)| command)?;
        self.wake.wake();
        Ok(())
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

    /// Pushes `result` for the engine and wakes it, handing the result back when the ring is
    /// full.
    ///
    /// # Errors
    ///
    /// Returns the result when [`RING_CAPACITY`] results are already waiting.
    pub fn push_result(&mut self, result: StepResult) -> Result<(), StepResult> {
        self.results
            .push(result)
            .map_err(|PushError::Full(result)| result)?;
        self.wake.wake();
        Ok(())
    }

    /// Whether the engine dropped its ends: no command will ever arrive again.
    #[must_use]
    pub fn engine_gone(&self) -> bool {
        self.commands.is_abandoned() || self.results.is_abandoned()
    }

    /// Parks until the engine pushes a command or drops its ends. A wake that arrived since the
    /// last park returns at once, so a command pushed between a failed pop and the park is never
    /// waited on.
    pub fn park(&self) {
        self.parker.park();
    }

    /// An unparker for the executor thread, for whatever else must wake it from its park.
    #[must_use]
    pub fn unparker(&self) -> Unparker {
        self.parker.unparker().clone()
    }
}

#[cfg(test)]
mod tests {
    use std::thread;
    use std::time::Duration;

    use crossbeam_utils::sync::Parker;

    use super::{rings, RING_CAPACITY};
    use crate::dispatch::{DispatchDecision, EagerReason};
    use crate::step::{StepCommand, StepResult};
    use crate::test_support::tokens;
    use crate::types::StepId;

    fn command(step: u64) -> StepCommand {
        StepCommand {
            step: StepId::new(step),
            entries: Vec::new(),
            padding_count: 0,
            dispatch: DispatchDecision::Eager(EagerReason::TokensAboveBucketLadderMaximum {
                token_count: tokens(1),
                bucket_ladder_maximum: None,
            }),
        }
    }

    #[test]
    fn commands_and_results_cross_in_order_and_the_rings_bound_what_is_in_flight() {
        let parker = Parker::new();
        let (mut engine, mut executor) = rings(parker.unparker().clone());
        assert_eq!(executor.pop_command(), None, "nothing issued yet");
        assert_eq!(engine.pop_result(), None, "nothing produced yet");

        for step in 1..=RING_CAPACITY as u64 {
            engine.push_command(command(step)).unwrap();
        }
        assert_eq!(
            engine.push_command(command(99)),
            Err(command(99)),
            "a full ring hands the command back"
        );

        assert_eq!(executor.pop_command(), Some(command(1)));
        engine
            .push_command(command(3))
            .expect("popping frees a slot");
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
        let parker = Parker::new();
        let (engine, executor) = rings(parker.unparker().clone());
        assert!(!engine.executor_gone());
        assert!(!executor.engine_gone());
        drop(executor);
        assert!(engine.executor_gone());

        let (engine, executor) = rings(parker.unparker().clone());
        drop(engine);
        assert!(executor.engine_gone());
    }

    #[test]
    fn the_executor_leaving_wakes_a_parked_engine_thread() {
        let parker = Parker::new();
        let (engine, executor) = rings(parker.unparker().clone());
        let engine_thread = thread::spawn(move || {
            parker.park();
            engine.executor_gone()
        });
        thread::sleep(Duration::from_millis(10));
        assert!(!engine_thread.is_finished(), "parked until woken");

        drop(executor);
        assert!(
            engine_thread.join().unwrap(),
            "woken, and the loss is visible"
        );
    }

    #[test]
    fn a_pushed_result_wakes_a_parked_engine_thread() {
        let parker = Parker::new();
        let (_engine, mut executor) = rings(parker.unparker().clone());
        let engine_thread = thread::spawn(move || parker.park());
        thread::sleep(Duration::from_millis(10));
        assert!(!engine_thread.is_finished(), "parked until woken");

        executor
            .push_result(StepResult {
                step: StepId::new(1),
                sampled: Vec::new(),
            })
            .unwrap();
        engine_thread.join().unwrap();
    }

    #[test]
    fn the_engine_leaving_wakes_a_parked_executor_thread() {
        let parker = Parker::new();
        let (engine, executor) = rings(parker.unparker().clone());
        let executor_thread = thread::spawn(move || {
            executor.park();
            executor.engine_gone()
        });
        thread::sleep(Duration::from_millis(10));
        assert!(!executor_thread.is_finished(), "parked until woken");

        drop(engine);
        assert!(
            executor_thread.join().unwrap(),
            "woken, and the loss is visible"
        );
    }

    #[test]
    fn a_pushed_command_wakes_a_parked_executor_thread() {
        let parker = Parker::new();
        let (mut engine, mut executor) = rings(parker.unparker().clone());
        let executor_thread = thread::spawn(move || {
            executor.park();
            executor.pop_command().map(|command| command.step)
        });
        thread::sleep(Duration::from_millis(10));
        assert!(!executor_thread.is_finished(), "parked until woken");

        engine.push_command(command(1)).unwrap();
        assert_eq!(
            executor_thread.join().unwrap(),
            Some(StepId::new(1)),
            "woken, and the command is there"
        );
    }

    /// A command pushed before the executor parks is not waited on: the wake is a token, not a
    /// signal that can be missed.
    #[test]
    fn a_command_pushed_before_the_park_returns_the_park_at_once() {
        let parker = Parker::new();
        let (mut engine, executor) = rings(parker.unparker().clone());
        engine.push_command(command(1)).unwrap();
        executor.park();
        assert!(
            !executor.engine_gone(),
            "returned by the wake, not the loss"
        );
    }

    #[test]
    fn the_executors_unparker_wakes_its_park_from_anywhere() {
        let parker = Parker::new();
        let (_engine, executor) = rings(parker.unparker().clone());
        let wake = executor.unparker();
        let executor_thread = thread::spawn(move || executor.park());
        thread::sleep(Duration::from_millis(10));
        assert!(!executor_thread.is_finished(), "parked until woken");

        wake.unpark();
        executor_thread.join().unwrap();
    }
}
