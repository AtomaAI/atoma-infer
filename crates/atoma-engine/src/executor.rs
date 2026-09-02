//! The executor loop: step commands off the ring, through the forward and the sampler, and step
//! results back, until the engine is gone. One pinned thread per rank runs it.
//!
//! The loop parks with nothing to serve and is woken by the next command or by the engine's ends
//! dropping. Any error ends the loop: the rings drop with it, which the engine sees as the
//! executor being lost, and the cause is logged and returned from the thread.

use std::io::ErrorKind;
use std::panic::resume_unwind;
use std::sync::mpsc::{sync_channel, SendError, SyncSender};
use std::thread::{self, JoinHandle};

use atoma_core::engine::ExecutorRings;
use atoma_core::step::{StepCommand, StepResult};
use atoma_core::types::TokenCount;
use core_affinity::{get_core_ids, set_for_current, CoreId as AffinityCoreId};
use thiserror::Error;
use tracing::{debug, error, info};

use crate::batch::{BatchLayout, LayoutError};
use crate::config::{CoreId, Rank};
use crate::forward::Forward;
use crate::sampler::{SampleError, Sampler};

/// An error whose type the executor does not name: the forward's own, or whatever building an
/// executor failed with.
pub type Cause = Box<dyn std::error::Error + Send + Sync + 'static>;

/// Why an executor stopped.
#[derive(Debug, Error)]
pub enum ExecutorError {
    #[error(transparent)]
    Layout(#[from] LayoutError),
    #[error("the forward failed: {0}")]
    Forward(#[source] Cause),
    #[error(transparent)]
    Sample(#[from] SampleError),
    /// The engine keeps one step in flight, so a full result ring means it broke the protocol.
    #[error("the result ring is full: more than one step is in flight")]
    ResultRingFull,
}

/// The executor of the rank that owns the engine's rings: it serves every step command and
/// answers with the step result.
pub struct Executor<F> {
    rings: ExecutorRings,
    forward: F,
    sampler: Sampler,
    block_size: usize,
}

impl<F: Forward> Executor<F> {
    /// An executor serving `rings` through `forward`, over `block_size`-token KV blocks.
    pub fn new(rings: ExecutorRings, forward: F, block_size: TokenCount) -> Self {
        Self {
            rings,
            forward,
            sampler: Sampler::new(),
            block_size: block_size.get(),
        }
    }

    /// Serves step commands as they come, parking between them, until the engine is gone.
    ///
    /// # Errors
    ///
    /// Returns [`ExecutorError`] for the first step that could not be served; the executor is
    /// over once it does.
    pub fn run(mut self) -> Result<(), ExecutorError> {
        info!("executor running");
        loop {
            if let Some(command) = self.rings.pop_command() {
                self.serve(&command)?;
            } else if self.rings.engine_gone() {
                info!("engine gone; executor returning");
                return Ok(());
            } else {
                self.rings.park();
            }
        }
    }

    fn serve(&mut self, command: &StepCommand) -> Result<(), ExecutorError> {
        let layout = BatchLayout::lay_out(command, self.block_size)?;
        let logits = self
            .forward
            .forward(command, &layout)
            .map_err(|cause| ExecutorError::Forward(Box::new(cause)))?;
        let sampled = self
            .sampler
            .sample(command, layout.sampling_rows(), logits)?;
        debug!(
            step = command.step.get(),
            entries = command.entries.len(),
            sampled = sampled.len(),
            "step served"
        );
        self.rings
            .push_result(StepResult {
                step: command.step,
                sampled,
            })
            .map_err(|_| ExecutorError::ResultRingFull)
    }
}

/// Why an executor thread could not be started.
#[derive(Debug, Error)]
pub enum SpawnError {
    /// The operating system refused the thread.
    #[error("the executor thread for rank {rank} could not be spawned: {kind:?}")]
    Thread { rank: Rank, kind: ErrorKind },
    /// The core is not one this process may run on, or the affinity could not be set.
    #[error(
        "rank {rank} cannot be pinned to core {core}; this process may run on cores {allowed:?}"
    )]
    Pin {
        rank: Rank,
        core: CoreId,
        allowed: Vec<usize>,
    },
    /// Building the executor on its thread failed.
    #[error("rank {rank} could not be built: {source}")]
    Build {
        rank: Rank,
        #[source]
        source: Cause,
    },
}

/// The running executor thread of one rank.
#[derive(Debug)]
pub struct ExecutorThread {
    rank: Rank,
    join: JoinHandle<Result<(), ExecutorError>>,
}

impl ExecutorThread {
    #[must_use]
    pub fn rank(&self) -> Rank {
        self.rank
    }

    /// Waits for the thread to return, handing back why it stopped if it failed.
    ///
    /// # Errors
    ///
    /// Returns the [`ExecutorError`] the loop stopped on.
    ///
    /// # Panics
    ///
    /// Panics when the executor thread panicked, carrying its panic on.
    pub fn join(self) -> Result<(), ExecutorError> {
        self.join.join().expect("the executor thread panicked")
    }

    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.join.is_finished()
    }
}

/// Spawns rank `rank`'s executor thread pinned to `core`, builds its executor there with `build`
/// and runs it until the engine is gone. Returns once the executor is built and running, so
/// whatever stops it from starting is this call's error.
///
/// The thread is named `atoma-executor-{rank}`. The executor is built on the thread because what
/// it holds — the device, the session — belongs to that thread alone.
///
/// # Errors
///
/// Returns [`SpawnError`] when the thread cannot be spawned, cannot be pinned to `core`, or
/// `build` fails.
pub fn spawn<F, B>(rank: Rank, core: CoreId, build: B) -> Result<ExecutorThread, SpawnError>
where
    F: Forward + 'static,
    B: FnOnce() -> Result<Executor<F>, Cause> + Send + 'static,
{
    let (ready, readiness) = sync_channel::<Result<(), SpawnError>>(1);
    let join = thread::Builder::new()
        .name(format!("atoma-executor-{rank}"))
        .spawn(move || {
            let executor = match start(rank, core, build) {
                Ok(executor) => executor,
                Err(error) => {
                    report(&ready, Err(error));
                    return Ok(());
                }
            };
            report(&ready, Ok(()));
            executor
                .run()
                .inspect_err(|cause| error!(%rank, %cause, "executor failed"))
        })
        .map_err(|error| SpawnError::Thread {
            rank,
            kind: error.kind(),
        })?;
    match readiness.recv() {
        Ok(Ok(())) => Ok(ExecutorThread { rank, join }),
        Ok(Err(error)) => Err(error),
        Err(_) => {
            // The thread reports before it returns, so a report that never came is a panic
            // while starting.
            let panic = join
                .join()
                .expect_err("the executor thread returned without reporting");
            resume_unwind(panic)
        }
    }
}

/// Pins the current thread and builds its executor.
fn start<F, B>(rank: Rank, core: CoreId, build: B) -> Result<Executor<F>, SpawnError>
where
    F: Forward,
    B: FnOnce() -> Result<Executor<F>, Cause>,
{
    pin(rank, core)?;
    build().map_err(|source| SpawnError::Build { rank, source })
}

/// Pins the current thread to `core`, refusing a core this process may not run on.
fn pin(rank: Rank, core: CoreId) -> Result<(), SpawnError> {
    let allowed: Vec<usize> = get_core_ids()
        .unwrap_or_default()
        .into_iter()
        .map(|id| id.id)
        .collect();
    let pinned =
        allowed.contains(&core.get()) && set_for_current(AffinityCoreId { id: core.get() });
    if !pinned {
        return Err(SpawnError::Pin {
            rank,
            core,
            allowed,
        });
    }
    info!(%rank, %core, "executor thread pinned");
    Ok(())
}

/// Reports whether starting succeeded; the spawner is waiting, so the send cannot fail.
fn report(ready: &SyncSender<Result<(), SpawnError>>, outcome: Result<(), SpawnError>) {
    match ready.send(outcome) {
        Ok(()) | Err(SendError(_)) => {}
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    use atoma_core::engine::{Control, Engine, EngineHandle, EngineThread, ExecutorRings};
    use atoma_core::request::{FinishReason, RequestEvent};
    use core_affinity::get_core_ids;

    use super::{spawn, Executor, ExecutorError, SpawnError};
    use crate::config::{CoreId, Rank};
    use crate::test_support::{
        contract, engine_config, submit, FakeForward, FakeForwardError, BLOCK_SIZE, WAIT,
    };

    fn engine() -> (EngineHandle, ExecutorRings, EngineThread) {
        Engine::spawn(&engine_config(), &contract()).unwrap()
    }

    fn executor(rings: ExecutorRings, forward: FakeForward) -> Executor<FakeForward> {
        Executor::new(rings, forward, BLOCK_SIZE)
    }

    #[test]
    fn a_request_flows_through_the_executor_to_its_finish() {
        let (handle, rings, engine) = engine();
        let forward = FakeForward::constant(5);
        let served = forward.served();
        let executor = thread::spawn(move || executor(rings, forward).run());
        let client = submit(&handle, 3, 2);

        assert!(matches!(
            client.recv_timeout(WAIT).unwrap(),
            RequestEvent::Token { token: 5, .. }
        ));
        assert!(matches!(
            client.recv_timeout(WAIT).unwrap(),
            RequestEvent::Token { token: 5, .. }
        ));
        assert!(matches!(
            client.recv_timeout(WAIT).unwrap(),
            RequestEvent::Finished {
                reason: FinishReason::MaxNewTokens,
                ..
            }
        ));
        let served = served.lock().unwrap().clone();
        assert_eq!(served.len(), 2, "one prefill, one decode");
        assert_eq!(served[0].entries[0].input_tokens, [1, 2, 3]);
        assert_eq!(
            served[1].entries[0].input_tokens,
            [5],
            "decoding what was sampled"
        );
        assert_eq!(served[1].entries[0].context_len, 3);

        handle.control.try_send(Control::Shutdown).unwrap();
        engine.join();
        executor
            .join()
            .unwrap()
            .expect("the executor returns cleanly once the engine is gone");
    }

    #[test]
    fn a_failing_forward_ends_the_executor_with_the_cause_and_the_engine_fails_the_request() {
        let (handle, rings, engine) = engine();
        let forward = FakeForward::constant(5).failing_at_step(2);
        let executor = thread::spawn(move || executor(rings, forward).run());
        let client = submit(&handle, 3, 16);

        assert!(matches!(
            client.recv_timeout(WAIT).unwrap(),
            RequestEvent::Token { token: 5, .. }
        ));
        assert!(matches!(
            client.recv_timeout(WAIT).unwrap(),
            RequestEvent::Finished {
                reason: FinishReason::ExecutorLost,
                ..
            }
        ));
        let error = executor.join().unwrap().unwrap_err();
        assert!(
            matches!(&error, ExecutorError::Forward(cause) if cause.is::<FakeForwardError>()),
            "the forward's own error is the cause: {error}"
        );
        assert!(error.to_string().contains("step 2"), "{error}");
        engine.join();
    }

    #[test]
    fn spawn_pins_the_thread_names_it_and_runs_the_executor_on_it() {
        let Some(core) = get_core_ids().and_then(|cores| cores.first().copied()) else {
            return;
        };
        let (handle, rings, engine) = engine();
        let (observed, observations) = mpsc::channel();
        let thread = spawn(Rank::new(3), CoreId::new(core.id), move || {
            let name = thread::current().name().map(str::to_owned);
            let pinned_to: Vec<usize> = get_core_ids()
                .unwrap_or_default()
                .into_iter()
                .map(|id| id.id)
                .collect();
            observed.send((name, pinned_to)).unwrap();
            Ok(Executor::new(rings, FakeForward::constant(7), BLOCK_SIZE))
        })
        .unwrap();
        assert_eq!(thread.rank(), Rank::new(3));
        let (name, pinned_to) = observations.recv_timeout(WAIT).unwrap();
        assert_eq!(name.as_deref(), Some("atoma-executor-3"));
        assert_eq!(
            pinned_to,
            [core.id],
            "the thread may run on that core alone"
        );

        let client = submit(&handle, 2, 1);
        assert!(matches!(
            client.recv_timeout(WAIT).unwrap(),
            RequestEvent::Token { token: 7, .. }
        ));
        handle.control.try_send(Control::Shutdown).unwrap();
        engine.join();
        assert!(thread.join().is_ok());
    }

    #[test]
    fn spawn_refuses_a_core_the_process_may_not_run_on() {
        let allowed = get_core_ids().unwrap_or_default();
        let beyond = allowed
            .iter()
            .map(|id| id.id)
            .max()
            .map_or(0, |max| max + 1);
        let (_handle, rings, _engine) = engine();
        let error = spawn(Rank::ZERO, CoreId::new(beyond), move || {
            Ok(Executor::new(rings, FakeForward::constant(7), BLOCK_SIZE))
        })
        .unwrap_err();
        assert!(
            matches!(&error, SpawnError::Pin { rank: Rank::ZERO, core, .. } if core.get() == beyond),
            "{error}"
        );
        assert!(error.to_string().contains("may run on cores"), "{error}");
    }

    #[test]
    fn spawn_returns_a_build_failure_from_the_thread() {
        let Some(core) = get_core_ids().and_then(|cores| cores.first().copied()) else {
            return;
        };
        let error = spawn::<FakeForward, _>(Rank::ZERO, CoreId::new(core.id), || {
            Err("no model at /nowhere".into())
        })
        .unwrap_err();
        assert!(matches!(&error, SpawnError::Build { .. }), "{error}");
        assert!(
            error.to_string().contains("no model at /nowhere"),
            "{error}"
        );
    }

    #[test]
    fn the_executor_returns_once_the_engine_is_gone_with_nothing_to_serve() {
        let (handle, rings, engine) = engine();
        let executor = thread::spawn(move || executor(rings, FakeForward::constant(1)).run());
        thread::sleep(Duration::from_millis(10));
        assert!(!executor.is_finished(), "parked, waiting for a command");
        handle.control.try_send(Control::Shutdown).unwrap();
        engine.join();
        assert!(executor.join().unwrap().is_ok());
    }
}
