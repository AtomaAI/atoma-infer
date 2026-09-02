//! Fanning step commands out from rank zero to the follower ranks.
//!
//! The engine has one ring pair, owned by rank zero. Every other rank is a follower: it is fed
//! each step command over its own single-producer single-consumer ring, runs the forward for it
//! and produces nothing, since rank zero alone reads the logits back and samples. A feed wakes the
//! follower on push and the leader on pop, so the leader can wait on a full feed; both wake the
//! far side on drop, so a rank that dies is seen by the others at once: a follower going ends the
//! leader, and the leader going ends every follower.

use std::sync::Arc;

use atoma_core::engine::WakeOnDrop;
use atoma_core::step::StepCommand;
use atoma_core::types::TokenCount;
use crossbeam_utils::sync::{Parker, Unparker};
use rtrb::{Consumer, Producer, PushError, RingBuffer};
use tracing::{debug, info};

use crate::batch::BatchLayout;
use crate::config::Rank;
use crate::executor::{ExecutorError, ExecutorLoop};
use crate::forward::Forward;

/// Commands a feed holds. The leader serves one step at a time, so two always leaves room.
const FEED_CAPACITY: usize = 2;

/// The leader's end of one follower's feed: it produces commands and wakes the follower after
/// each.
#[derive(Debug)]
pub struct FollowerFeed {
    rank: Rank,
    commands: Producer<Arc<StepCommand>>,
    /// Declared after the ring end, and so dropped after it, so the loss is visible through
    /// [`FollowerRings::leader_gone`] before the follower is woken to look.
    wake: WakeOnDrop,
}

/// A follower's ends: it consumes commands, parks between them, and wakes the leader after each
/// pop and when it goes.
#[derive(Debug)]
pub struct FollowerRings {
    rank: Rank,
    commands: Consumer<Arc<StepCommand>>,
    parker: Parker,
    /// Declared after the ring end for the same reason as [`FollowerFeed::wake`].
    wake: WakeOnDrop,
}

/// Opens the feed to follower `rank`; `wake_leader` unparks the leader when the follower goes.
#[must_use]
pub fn feed(rank: Rank, wake_leader: Unparker) -> (FollowerFeed, FollowerRings) {
    let (producer, consumer) = RingBuffer::new(FEED_CAPACITY);
    let parker = Parker::new();
    let wake_follower = WakeOnDrop::new(parker.unparker().clone());
    (
        FollowerFeed {
            rank,
            commands: producer,
            wake: wake_follower,
        },
        FollowerRings {
            rank,
            commands: consumer,
            parker,
            wake: WakeOnDrop::new(wake_leader),
        },
    )
}

impl FollowerFeed {
    #[must_use]
    pub fn rank(&self) -> Rank {
        self.rank
    }

    /// Pushes `command` for the follower and wakes it, handing the command back when the feed
    /// is full.
    ///
    /// # Errors
    ///
    /// Returns the command when [`FEED_CAPACITY`] steps are already with the follower.
    pub fn push(&mut self, command: Arc<StepCommand>) -> Result<(), Arc<StepCommand>> {
        self.commands
            .push(command)
            .map_err(|PushError::Full(command)| command)?;
        self.wake.wake();
        Ok(())
    }

    /// Whether the follower dropped its ends: it will never run a command again.
    #[must_use]
    pub fn follower_gone(&self) -> bool {
        self.commands.is_abandoned()
    }
}

impl FollowerRings {
    #[must_use]
    pub fn rank(&self) -> Rank {
        self.rank
    }

    /// The next step command, if the leader has fed one; taking it wakes the leader, which may
    /// be waiting on the feed for room.
    pub fn pop_command(&mut self) -> Option<Arc<StepCommand>> {
        let command = self.commands.pop().ok()?;
        self.wake.wake();
        Some(command)
    }

    /// Whether the leader dropped its end: no command will ever arrive again.
    #[must_use]
    pub fn leader_gone(&self) -> bool {
        self.commands.is_abandoned()
    }

    /// Parks until the leader feeds a command or drops its end.
    pub fn park(&self) {
        self.parker.park();
    }
}

/// A follower rank's executor: it runs the forward for every command fed to it and produces
/// nothing.
pub struct Follower<F> {
    rings: FollowerRings,
    forward: F,
    block_size: usize,
}

impl<F: Forward> Follower<F> {
    /// A follower serving `rings` through `forward`, over `block_size`-token KV blocks.
    pub fn new(rings: FollowerRings, forward: F, block_size: TokenCount) -> Self {
        Self {
            rings,
            forward,
            block_size: block_size.get(),
        }
    }

    fn serve(&mut self, command: &StepCommand) -> Result<(), ExecutorError> {
        let layout = BatchLayout::lay_out(command, self.block_size)?;
        self.forward
            .forward(command, &layout)
            .map_err(|cause| ExecutorError::Forward(Box::new(cause)))?;
        debug!(
            rank = %self.rings.rank(),
            step = command.step.get(),
            entries = command.entries.len(),
            "step followed"
        );
        Ok(())
    }
}

impl<F: Forward> ExecutorLoop for Follower<F> {
    /// Runs commands as they are fed, parking between them, until the leader is gone.
    fn run(mut self) -> Result<(), ExecutorError> {
        let rank = self.rings.rank();
        info!(%rank, "follower running");
        loop {
            if let Some(command) = self.rings.pop_command() {
                self.serve(&command)?;
            } else if self.rings.leader_gone() {
                info!(%rank, "leader gone; follower returning");
                return Ok(());
            } else {
                self.rings.park();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    use atoma_core::engine::rings;
    use atoma_core::step::StepCommand;
    use crossbeam_utils::sync::Parker;

    use super::{feed, FEED_CAPACITY};
    use crate::config::Rank;
    use crate::test_support::{command, entry};

    fn step() -> Arc<StepCommand> {
        Arc::new(command(vec![entry(1, 0, vec![1, 2], &[10], true)], 0))
    }

    #[test]
    fn a_feed_carries_commands_in_order_and_bounds_what_is_with_the_follower() {
        let parker = Parker::new();
        let (mut leader, mut follower) = feed(Rank::new(1), parker.unparker().clone());
        assert_eq!(follower.rank(), Rank::new(1));
        assert_eq!(leader.rank(), Rank::new(1));
        assert!(follower.pop_command().is_none());
        for _ in 0..FEED_CAPACITY {
            leader.push(step()).unwrap();
        }
        assert!(leader.push(step()).is_err(), "a full feed hands it back");
        assert!(follower.pop_command().is_some());
        leader.push(step()).expect("popping frees a slot");
    }

    #[test]
    fn a_fed_command_wakes_a_parked_follower() {
        let parker = Parker::new();
        let (mut leader, mut follower) = feed(Rank::new(1), parker.unparker().clone());
        let follower_thread = thread::spawn(move || {
            follower.park();
            follower.pop_command().is_some()
        });
        thread::sleep(Duration::from_millis(10));
        assert!(!follower_thread.is_finished(), "parked until fed");
        leader.push(step()).unwrap();
        assert!(
            follower_thread.join().unwrap(),
            "woken, and the command is there"
        );
    }

    #[test]
    fn the_leader_leaving_wakes_a_parked_follower_which_sees_it() {
        let parker = Parker::new();
        let (leader, follower) = feed(Rank::new(1), parker.unparker().clone());
        let follower_thread = thread::spawn(move || {
            follower.park();
            follower.leader_gone()
        });
        thread::sleep(Duration::from_millis(10));
        assert!(!follower_thread.is_finished(), "parked until woken");
        drop(leader);
        assert!(
            follower_thread.join().unwrap(),
            "woken, and the loss is visible"
        );
    }

    #[test]
    fn a_follower_taking_a_command_wakes_the_parked_leader() {
        let engine_parker = Parker::new();
        let (_engine, executor) = rings(engine_parker.unparker().clone());
        let (mut leader, mut follower) = feed(Rank::new(1), executor.unparker());
        leader.push(step()).unwrap();
        let leader_thread = thread::spawn(move || executor.park());
        thread::sleep(Duration::from_millis(10));
        assert!(!leader_thread.is_finished(), "parked until woken");
        assert!(follower.pop_command().is_some());
        leader_thread.join().unwrap();
    }

    #[test]
    fn a_follower_leaving_wakes_the_parked_leader_which_sees_it() {
        let engine_parker = Parker::new();
        let (_engine, executor) = rings(engine_parker.unparker().clone());
        let (leader, follower) = feed(Rank::new(1), executor.unparker());
        let leader_thread = thread::spawn(move || {
            executor.park();
            leader.follower_gone()
        });
        thread::sleep(Duration::from_millis(10));
        assert!(!leader_thread.is_finished(), "parked until woken");
        drop(follower);
        assert!(
            leader_thread.join().unwrap(),
            "woken, and the loss is visible"
        );
    }
}
