//! What a run measured.
//!
//! One [`RequestRecord`] per offered request, and a [`RunSummary`] over all of them. Latencies are
//! kept as distributions ([`Distribution`], backed by hdrhistogram) rather than means: the tail is
//! the number an SLO is written against, and a mean hides it.

use std::time::Duration;

use hdrhistogram::Histogram;
use serde::{Deserialize, Serialize};

use crate::goodput::{goodput_per_second, Slo};

/// The widest latency an hdrhistogram in this module records, in microseconds. A request slower
/// than an hour is a hung run, not a data point.
const MAX_RECORDED_MICROS: u64 = 3_600_000_000;

/// Significant digits kept by the histograms: 1 µs resolution up to 1 ms, 1 ms up to 1 s.
const HISTOGRAM_PRECISION: u8 = 3;

/// How one offered request ended.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum Outcome {
    /// The engine streamed a complete answer.
    Completed,
    /// The request did not produce an answer.
    Failed {
        /// Why it did not.
        reason: String,
    },
}

/// Everything one request contributed to the run.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct RequestRecord {
    /// The workload's id for this request.
    pub id: String,
    /// Prompt length measured with the model's tokenizer.
    pub prompt_tokens: usize,
    /// Number of tokens the engine streamed back.
    pub output_tokens: usize,
    /// Time to first token, absent when nothing was streamed.
    pub ttft: Option<Duration>,
    /// Gaps between consecutive streamed tokens.
    pub inter_token_latencies: Vec<Duration>,
    /// Time from sending the request to the end of the response.
    pub end_to_end: Duration,
    /// Whether the request produced an answer.
    pub outcome: Outcome,
}

impl RequestRecord {
    /// Records an answered request from the offsets, relative to the request being sent, at which
    /// each streamed token arrived.
    ///
    /// A response that streamed no tokens is recorded as a failure: a 200 with an empty body is
    /// not a fast request.
    pub fn completed(
        id: String,
        prompt_tokens: usize,
        token_offsets: &[Duration],
        end_to_end: Duration,
    ) -> Self {
        let Some(ttft) = token_offsets.first().copied() else {
            return Self::failed(
                id,
                prompt_tokens,
                "the response streamed no tokens".to_string(),
                end_to_end,
            );
        };

        Self {
            id,
            prompt_tokens,
            output_tokens: token_offsets.len(),
            ttft: Some(ttft),
            inter_token_latencies: token_offsets
                .windows(2)
                .map(|pair| pair[1].saturating_sub(pair[0]))
                .collect(),
            end_to_end,
            outcome: Outcome::Completed,
        }
    }

    /// Records a request that did not produce an answer.
    pub fn failed(id: String, prompt_tokens: usize, reason: String, end_to_end: Duration) -> Self {
        Self {
            id,
            prompt_tokens,
            output_tokens: 0,
            ttft: None,
            inter_token_latencies: Vec::new(),
            end_to_end,
            outcome: Outcome::Failed { reason },
        }
    }

    /// Whether the engine answered this request.
    pub fn is_completed(&self) -> bool {
        matches!(self.outcome, Outcome::Completed)
    }

    /// Why the request failed, if it did.
    pub fn failure_reason(&self) -> Option<&str> {
        match &self.outcome {
            Outcome::Completed => None,
            Outcome::Failed { reason } => Some(reason),
        }
    }

    /// The mean gap between streamed tokens, absent for answers of a single token.
    pub fn mean_inter_token_latency(&self) -> Option<Duration> {
        if self.inter_token_latencies.is_empty() {
            return None;
        }
        let total: Duration = self.inter_token_latencies.iter().sum();
        Some(total / self.inter_token_latencies.len() as u32)
    }
}

/// A latency distribution, in milliseconds.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Distribution {
    /// Number of recorded values.
    pub count: u64,
    /// Arithmetic mean.
    pub mean_ms: f64,
    /// Median.
    pub p50_ms: f64,
    /// 90th percentile.
    pub p90_ms: f64,
    /// 99th percentile.
    pub p99_ms: f64,
    /// Largest recorded value.
    pub max_ms: f64,
}

impl Distribution {
    /// Records `values` into an hdrhistogram and reads the distribution back.
    ///
    /// Values are recorded in microseconds, so a sub-millisecond inter-token latency keeps its
    /// resolution instead of flooring to zero.
    pub fn of(values: impl Iterator<Item = Duration>) -> Self {
        let mut histogram =
            Histogram::<u64>::new_with_bounds(1, MAX_RECORDED_MICROS, HISTOGRAM_PRECISION)
                .expect("The histogram bounds are constants and are valid");

        for value in values {
            let micros = u64::try_from(value.as_micros()).unwrap_or(MAX_RECORDED_MICROS);
            histogram
                .record(micros.clamp(1, MAX_RECORDED_MICROS))
                .expect("Recorded values are clamped into the histogram's bounds");
        }

        if histogram.is_empty() {
            return Self::default();
        }

        Self {
            count: histogram.len(),
            mean_ms: histogram.mean() / 1_000.0,
            p50_ms: micros_to_ms(histogram.value_at_quantile(0.50)),
            p90_ms: micros_to_ms(histogram.value_at_quantile(0.90)),
            p99_ms: micros_to_ms(histogram.value_at_quantile(0.99)),
            max_ms: micros_to_ms(histogram.max()),
        }
    }
}

/// Converts a recorded microsecond value to milliseconds.
fn micros_to_ms(micros: u64) -> f64 {
    micros as f64 / 1_000.0
}

/// What one run of one engine on one workload measured.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RunSummary {
    /// Requests offered.
    pub num_requests: usize,
    /// Requests the engine answered.
    pub completed: usize,
    /// Requests that failed.
    pub failed: usize,
    /// Tokens streamed across all answered requests.
    pub output_tokens: usize,
    /// Wall-clock length of the run.
    pub duration_seconds: f64,
    /// The headline metric: answered requests per second that met the SLO.
    pub goodput_per_second: f64,
    /// Answered requests per second, whether or not they met the SLO.
    pub request_throughput_per_second: f64,
    /// Streamed tokens per second.
    pub output_token_throughput_per_second: f64,
    /// Time to first token, over answered requests.
    pub ttft_ms: Distribution,
    /// Inter-token latency, over every gap in every answered request.
    pub itl_ms: Distribution,
    /// End-to-end latency, over answered requests.
    pub end_to_end_ms: Distribution,
}

impl RunSummary {
    /// Summarizes the records of one run, measured over `duration` of wall clock.
    ///
    /// Latency distributions cover answered requests only. A failed request contributes to the
    /// failure count and to nothing else — an engine that sheds load fast would otherwise report
    /// the best latencies in the table.
    pub fn summarize(records: &[RequestRecord], duration: Duration, slo: &Slo) -> Self {
        let completed = records.iter().filter(|record| record.is_completed());
        let num_completed = completed.clone().count();
        let output_tokens = completed
            .clone()
            .map(|record| record.output_tokens)
            .sum::<usize>();
        let seconds = duration.as_secs_f64();

        Self {
            num_requests: records.len(),
            completed: num_completed,
            failed: records.len() - num_completed,
            output_tokens,
            duration_seconds: seconds,
            goodput_per_second: goodput_per_second(records, slo, duration),
            request_throughput_per_second: per_second(num_completed as f64, seconds),
            output_token_throughput_per_second: per_second(output_tokens as f64, seconds),
            ttft_ms: Distribution::of(completed.clone().filter_map(|record| record.ttft)),
            itl_ms: Distribution::of(
                completed
                    .clone()
                    .flat_map(|record| record.inter_token_latencies.iter().copied()),
            ),
            end_to_end_ms: Distribution::of(completed.map(|record| record.end_to_end)),
        }
    }
}

/// A rate, or zero when no time passed.
fn per_second(count: f64, seconds: f64) -> f64 {
    if seconds <= 0.0 {
        return 0.0;
    }
    count / seconds
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(milliseconds: u64) -> Duration {
        Duration::from_millis(milliseconds)
    }

    fn completed_record(token_offsets: &[Duration]) -> RequestRecord {
        RequestRecord::completed(
            "request-0".to_string(),
            16,
            token_offsets,
            *token_offsets.last().unwrap_or(&Duration::ZERO),
        )
    }

    #[test]
    fn test_a_completed_request_splits_its_timeline_into_ttft_and_inter_token_latencies() {
        let record = completed_record(&[ms(120), ms(140), ms(150), ms(200)]);

        assert_eq!(record.ttft, Some(ms(120)));
        assert_eq!(record.inter_token_latencies, vec![ms(20), ms(10), ms(50)]);
        assert_eq!(record.output_tokens, 4);
        assert_eq!(record.end_to_end, ms(200));
        assert!(record.is_completed());
    }

    /// A one-token answer has a TTFT and no inter-token latency at all; averaging over an empty
    /// list is what turns that into a NaN further down the pipeline.
    #[test]
    fn test_a_single_token_answer_has_no_inter_token_latency() {
        let record = completed_record(&[ms(80)]);

        assert_eq!(record.ttft, Some(ms(80)));
        assert!(record.inter_token_latencies.is_empty());
        assert_eq!(record.mean_inter_token_latency(), None);
    }

    #[test]
    fn test_mean_inter_token_latency_averages_the_gaps() {
        let record = completed_record(&[ms(100), ms(110), ms(130)]);

        assert_eq!(record.mean_inter_token_latency(), Some(ms(15)));
    }

    #[test]
    fn test_a_failed_request_carries_its_reason_and_no_latencies() {
        let record =
            RequestRecord::failed("request-1".to_string(), 16, "HTTP 503".to_string(), ms(30));

        assert!(!record.is_completed());
        assert_eq!(record.failure_reason(), Some("HTTP 503"));
        assert_eq!(record.ttft, None);
        assert_eq!(record.output_tokens, 0);
    }

    /// An answer that streamed nothing is a failure, not a fast request: counting it as completed
    /// would let an engine that returns empty bodies win the benchmark.
    #[test]
    fn test_a_completed_request_without_tokens_is_recorded_as_a_failure() {
        let record = completed_record(&[]);

        assert!(!record.is_completed());
        assert_eq!(
            record.failure_reason(),
            Some("the response streamed no tokens")
        );
    }

    #[test]
    fn test_a_distribution_reports_percentiles_over_the_recorded_values() {
        let values = (1..=100).map(ms).collect::<Vec<_>>();

        let distribution = Distribution::of(values.iter().copied());

        assert_eq!(distribution.count, 100);
        assert!(
            (distribution.mean_ms - 50.5).abs() < 0.5,
            "{distribution:?}"
        );
        assert!((distribution.p50_ms - 50.0).abs() < 1.0, "{distribution:?}");
        assert!((distribution.p99_ms - 99.0).abs() < 1.0, "{distribution:?}");
        assert!(
            (distribution.max_ms - 100.0).abs() < 1.0,
            "{distribution:?}"
        );
    }

    #[test]
    fn test_an_empty_distribution_is_all_zeroes_rather_than_nan() {
        let distribution = Distribution::of(std::iter::empty());

        assert_eq!(distribution.count, 0);
        assert_eq!(distribution.mean_ms, 0.0);
        assert_eq!(distribution.p99_ms, 0.0);
    }

    /// Sub-millisecond latencies are real on a decode step; a distribution that floors them at
    /// zero would report a p50 of 0 ms for an engine that is simply fast.
    #[test]
    fn test_a_distribution_keeps_sub_millisecond_resolution() {
        let distribution = Distribution::of(std::iter::repeat_n(Duration::from_micros(250), 10));

        assert!(
            (distribution.p50_ms - 0.25).abs() < 0.01,
            "{distribution:?}"
        );
    }

    #[test]
    fn test_a_run_summary_counts_completions_and_failures_separately() {
        let records = vec![
            completed_record(&[ms(100), ms(120)]),
            completed_record(&[ms(90), ms(100), ms(110)]),
            RequestRecord::failed("request-2".to_string(), 16, "timeout".to_string(), ms(50)),
        ];

        let summary = RunSummary::summarize(&records, Duration::from_secs(10), &Slo::default());

        assert_eq!(summary.num_requests, 3);
        assert_eq!(summary.completed, 2);
        assert_eq!(summary.failed, 1);
        assert_eq!(summary.output_tokens, 5);
        assert!((summary.request_throughput_per_second - 0.2).abs() < 1e-9);
        assert!((summary.output_token_throughput_per_second - 0.5).abs() < 1e-9);
    }

    /// Failed requests must not enter the latency distributions: an engine that refuses load
    /// quickly would otherwise report the best TTFT in the table.
    #[test]
    fn test_run_summary_distributions_cover_completed_requests_only() {
        let records = vec![
            completed_record(&[ms(100), ms(120)]),
            RequestRecord::failed("request-1".to_string(), 16, "timeout".to_string(), ms(1)),
        ];

        let summary = RunSummary::summarize(&records, Duration::from_secs(1), &Slo::default());

        assert_eq!(summary.ttft_ms.count, 1);
        assert!((summary.ttft_ms.p50_ms - 100.0).abs() < 1.0);
        assert_eq!(summary.end_to_end_ms.count, 1);
    }
}
