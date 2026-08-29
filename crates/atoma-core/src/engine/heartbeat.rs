//! The heartbeat: the pass counter and timestamp the engine thread publishes every pass.
//!
//! Liveness is read from the thread that could wedge, not from the API in front of it. Two
//! atomics, written on the engine thread and read anywhere, with no lock: the timestamp is
//! published before the pass counter that releases it, so a reader that sees a pass sees the
//! time it completed.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

/// A [`Heartbeat`] in the form both ends share: the publisher stores into it, every reader loads
/// from it.
#[derive(Debug, Default)]
struct AtomicHeartbeat {
    pass: AtomicU64,
    at_unix_nanos: AtomicU64,
}

/// The engine thread's end: publishes once per pass.
#[derive(Debug)]
pub struct HeartbeatPublisher {
    beat: Arc<AtomicHeartbeat>,
}

/// Anyone's end: reads the latest beat.
#[derive(Debug, Clone)]
pub struct HeartbeatReader {
    beat: Arc<AtomicHeartbeat>,
}

/// One published beat.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Heartbeat {
    /// Passes completed so far.
    pub pass: u64,
    /// When the last pass completed.
    pub at: SystemTime,
}

/// Opens a heartbeat: the engine thread publishes, anyone reads.
#[must_use]
pub fn heartbeat() -> (HeartbeatPublisher, HeartbeatReader) {
    let beat = Arc::new(AtomicHeartbeat::default());
    (
        HeartbeatPublisher { beat: beat.clone() },
        HeartbeatReader { beat },
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
        self.beat.at_unix_nanos.store(nanos, Ordering::Relaxed);
        self.beat.pass.store(pass, Ordering::Release);
    }
}

impl HeartbeatReader {
    /// The latest beat; pass zero at an epoch timestamp before the first pass completes.
    #[must_use]
    pub fn read(&self) -> Heartbeat {
        let pass = self.beat.pass.load(Ordering::Acquire);
        let nanos = self.beat.at_unix_nanos.load(Ordering::Relaxed);
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
        assert!(beat.at <= SystemTime::now());
    }
}
