//! The heartbeat: the pass counter and timestamp the engine thread publishes every pass.
//!
//! Liveness is read from the thread that could wedge, not from the API in front of it. Two
//! atomics, written with relaxed stores on the engine thread and read anywhere; no lock.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

#[derive(Debug, Default)]
struct Cell {
    pass: AtomicU64,
    at_unix_nanos: AtomicU64,
}

/// The engine thread's end: publishes once per pass.
#[derive(Debug)]
pub struct HeartbeatPublisher {
    cell: Arc<Cell>,
}

/// Anyone's end: reads the latest beat.
#[derive(Debug, Clone)]
pub struct HeartbeatReader {
    cell: Arc<Cell>,
}

/// One published beat.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Heartbeat {
    /// Passes completed so far.
    pub pass: u64,
    /// When the last pass completed.
    pub at: SystemTime,
}

impl Heartbeat {
    /// How long ago the last pass completed, as of `now`.
    #[must_use]
    pub fn age(&self, now: SystemTime) -> Duration {
        now.duration_since(self.at).unwrap_or(Duration::ZERO)
    }
}

/// Opens a heartbeat: the engine thread publishes, anyone reads.
#[must_use]
pub fn heartbeat() -> (HeartbeatPublisher, HeartbeatReader) {
    let cell = Arc::new(Cell::default());
    (
        HeartbeatPublisher { cell: cell.clone() },
        HeartbeatReader { cell },
    )
}

impl HeartbeatPublisher {
    /// Records that pass `pass` completed now.
    pub fn publish(&self, pass: u64) {
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map_or(0, |since| {
                u64::try_from(since.as_nanos()).unwrap_or(u64::MAX)
            });
        self.cell.at_unix_nanos.store(nanos, Ordering::Relaxed);
        self.cell.pass.store(pass, Ordering::Release);
    }
}

impl HeartbeatReader {
    /// The latest beat; pass zero at an epoch timestamp before the first pass completes.
    #[must_use]
    pub fn read(&self) -> Heartbeat {
        let pass = self.cell.pass.load(Ordering::Acquire);
        let nanos = self.cell.at_unix_nanos.load(Ordering::Relaxed);
        Heartbeat {
            pass,
            at: SystemTime::UNIX_EPOCH + Duration::from_nanos(nanos),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, SystemTime};

    use super::heartbeat;

    #[test]
    fn a_published_pass_is_read_with_its_time() {
        let (publisher, reader) = heartbeat();
        assert_eq!(reader.read().pass, 0, "nothing published yet");

        let before = SystemTime::now();
        publisher.publish(7);
        let beat = reader.read();
        assert_eq!(beat.pass, 7);
        assert!(beat.at >= before - Duration::from_millis(1));
        assert!(beat.age(SystemTime::now()) < Duration::from_secs(1));
        assert_eq!(beat.age(before), Duration::ZERO, "never negative");
    }
}
