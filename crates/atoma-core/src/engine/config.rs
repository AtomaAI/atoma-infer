//! What the engine is built from: one value, fixed for the process lifetime, carrying the
//! scheduler's and the dispatcher's own configurations along with the sizes the thread itself
//! needs.

use std::time::Duration;

use serde::{Deserialize, Serialize};
use validator::{Validate, ValidationError};

use crate::dispatch::DispatchConfig;
use crate::scheduler::SchedulerConfig;
use crate::types::RequestCount;

/// Everything the engine is built from, fixed for the process lifetime.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
#[validate(schema(function = "every_full_batch_has_a_captured_graph"))]
pub struct EngineConfig {
    #[validate(nested)]
    pub scheduler: SchedulerConfig,
    #[validate(nested)]
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
    /// How long a step may be out with the executor before the executor is treated as lost and
    /// every live request fails. An executor held inside a step never drops its rings, so this
    /// is what ends the wait. Written in milliseconds wherever configuration is.
    #[serde(rename = "step_deadline_millis", with = "millis")]
    #[validate(custom(function = "step_deadline_is_positive"))]
    pub step_deadline: Duration,
}

/// A zero step deadline would fail every step before its result could arrive.
fn step_deadline_is_positive(deadline: &Duration) -> Result<(), ValidationError> {
    if !deadline.is_zero() {
        return Ok(());
    }
    let mut error = ValidationError::new("zero_step_deadline");
    error.message = Some(
        "step_deadline_millis is 0, which would fail every step before its result arrived; \
         set it well above the longest step"
            .into(),
    );
    Err(error)
}

/// Dispatch falls back to eager execution for any batch holding more requests than the captured
/// graphs serve, so a `max_batch` above `captured_max_requests` would send every full batch down
/// the eager path.
fn every_full_batch_has_a_captured_graph(config: &EngineConfig) -> Result<(), ValidationError> {
    if config.scheduler.max_batch <= config.dispatch.captured_max_requests {
        return Ok(());
    }
    let mut error = ValidationError::new("max_batch_over_captured_max_requests");
    error.message = Some(
        format!(
            "scheduler.max_batch is {} but the captured graphs serve at most {} requests, so \
             every full batch would fall back to eager execution; capture a larger bucket or \
             lower scheduler.max_batch",
            config.scheduler.max_batch.get(),
            config.dispatch.captured_max_requests.get()
        )
        .into(),
    );
    Err(error)
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
