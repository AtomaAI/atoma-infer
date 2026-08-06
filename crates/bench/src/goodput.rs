//! Goodput at a fixed service level.
//!
//! Throughput counts every request an engine answers, including the ones it answered so slowly
//! that a user would have left. Goodput counts only the requests that met the service level, which
//! is why it is the headline number of the benchmark protocol: an engine can raise throughput
//! indefinitely by letting latency go, and goodput is what stops that trade from looking like a
//! win.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::record::RequestRecord;

/// The service level a request has to meet to count towards goodput.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct Slo {
    /// The slowest first token a user is expected to accept.
    pub ttft_ms: f64,
    /// The slowest mean gap between tokens a user is expected to accept.
    pub itl_ms: f64,
}

impl Default for Slo {
    /// The rung-0 service level: a first token within 2 s and tokens at ≥ 20/s thereafter.
    fn default() -> Self {
        Self {
            ttft_ms: 2_000.0,
            itl_ms: 50.0,
        }
    }
}

impl Slo {
    /// Whether a request met the service level.
    ///
    /// The inter-token side is judged on the request's *mean* gap, the rate a reader actually
    /// experiences over a session; the p99 of the same gaps is reported separately in
    /// [`crate::record::RunSummary::itl_ms`] so a stall-prone engine is still visible.
    pub fn attained(&self, record: &RequestRecord) -> bool {
        if !record.is_completed() {
            return false;
        }
        let Some(ttft) = record.ttft else {
            return false;
        };
        if ttft.as_secs_f64() * 1_000.0 > self.ttft_ms {
            return false;
        }

        match record.mean_inter_token_latency() {
            Some(itl) => itl.as_secs_f64() * 1_000.0 <= self.itl_ms,
            None => true,
        }
    }
}

/// Requests per second that met the service level over a run of `duration`.
///
/// Returns zero for a run of no length rather than an infinity that would poison the table.
pub fn goodput_per_second(records: &[RequestRecord], slo: &Slo, duration: Duration) -> f64 {
    let seconds = duration.as_secs_f64();
    if seconds <= 0.0 {
        return 0.0;
    }

    let attained = records.iter().filter(|record| slo.attained(record)).count();
    attained as f64 / seconds
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(milliseconds: u64) -> Duration {
        Duration::from_millis(milliseconds)
    }

    fn slo() -> Slo {
        Slo {
            ttft_ms: 500.0,
            itl_ms: 50.0,
        }
    }

    /// Builds a completed request whose first token lands at `ttft` and whose remaining tokens
    /// arrive `gap` apart.
    fn record(ttft: Duration, gap: Duration, num_tokens: usize) -> RequestRecord {
        let offsets = (0..num_tokens)
            .map(|index| ttft + gap * index as u32)
            .collect::<Vec<_>>();
        let last = *offsets.last().unwrap_or(&ttft);
        RequestRecord::completed("request".to_string(), 32, &offsets, last)
    }

    #[test]
    fn test_a_request_inside_both_service_levels_is_attained() {
        assert!(slo().attained(&record(ms(200), ms(20), 16)));
    }

    #[test]
    fn test_a_slow_first_token_misses_the_service_level() {
        assert!(!slo().attained(&record(ms(700), ms(20), 16)));
    }

    #[test]
    fn test_slow_decoding_misses_the_service_level() {
        assert!(!slo().attained(&record(ms(100), ms(80), 16)));
    }

    /// The inter-token service level is written against the mean gap, because that is what a
    /// reader sees: one stalled token inside an otherwise fast stream is not a missed SLO.
    #[test]
    fn test_one_stalled_token_inside_a_fast_stream_still_attains() {
        // Gaps of 10, 90, 10, 10 ms: one stall well over the 50 ms service level, a mean of 30 ms
        // under it.
        let offsets = [ms(100), ms(110), ms(200), ms(210), ms(220)];
        let record = RequestRecord::completed("request".to_string(), 32, &offsets, ms(220));

        assert!(slo().attained(&record));
        assert!(
            !Slo {
                ttft_ms: 500.0,
                itl_ms: 20.0,
            }
            .attained(&record),
            "a service level under the 30 ms mean must not be attained"
        );
    }

    #[test]
    fn test_a_failed_request_is_never_attained() {
        let record =
            RequestRecord::failed("request".to_string(), 32, "HTTP 500".to_string(), ms(5));

        assert!(!slo().attained(&record));
    }

    /// A one-token answer has no inter-token latency to judge, so only its TTFT is held to the
    /// service level.
    #[test]
    fn test_a_single_token_answer_is_judged_on_its_first_token_alone() {
        assert!(slo().attained(&record(ms(200), ms(0), 1)));
        assert!(!slo().attained(&record(ms(900), ms(0), 1)));
    }

    #[test]
    fn test_goodput_is_attained_requests_over_the_wall_clock() {
        let records = vec![
            record(ms(200), ms(20), 16),
            record(ms(200), ms(20), 16),
            record(ms(900), ms(20), 16),
            RequestRecord::failed("request".to_string(), 32, "timeout".to_string(), ms(5)),
        ];

        let goodput = goodput_per_second(&records, &slo(), Duration::from_secs(4));

        assert!((goodput - 0.5).abs() < 1e-9, "{goodput}");
    }

    #[test]
    fn test_goodput_of_an_empty_run_is_zero_rather_than_nan() {
        assert_eq!(goodput_per_second(&[], &slo(), Duration::ZERO), 0.0);
        assert_eq!(
            goodput_per_second(&[record(ms(10), ms(1), 4)], &slo(), Duration::ZERO),
            0.0
        );
    }
}
