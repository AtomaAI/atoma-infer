//! What the engine is built from: one value, fixed for the process lifetime, carrying the
//! scheduler's and the dispatcher's own configurations along with the sizes the thread itself
//! needs.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::dispatch::DispatchConfig;
use crate::scheduler::SchedulerConfig;
use crate::types::RequestCount;

/// Everything the engine is built from, fixed for the process lifetime.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EngineConfig {
    pub scheduler: SchedulerConfig,
    pub dispatch: DispatchConfig,
    /// Blocks in the pool, before the padding reservation.
    pub block_count: u32,
    /// Requests ingress holds beyond what the slab has room for; the burst buffer behind the
    /// overload signal.
    pub ingress_capacity: RequestCount,
    /// How long the thread parks with nothing to do before re-examining its queues, so an
    /// empty schedule can never wedge it. Written in milliseconds wherever configuration is.
    #[serde(rename = "idle_deadline_millis", with = "millis")]
    pub idle_deadline: Duration,
}

/// A duration in memory, milliseconds on the wire: a deadline is a duration, and configuration
/// should not have to spell one out as seconds and nanoseconds.
mod millis {
    use std::time::Duration;

    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub(super) fn serialize<S: Serializer>(
        duration: &Duration,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        u64::try_from(duration.as_millis())
            .unwrap_or(u64::MAX)
            .serialize(serializer)
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Duration, D::Error> {
        u64::deserialize(deserializer).map(Duration::from_millis)
    }
}
