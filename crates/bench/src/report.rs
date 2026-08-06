//! The results artifact and the table rendered from it.
//!
//! One JSON artifact per comparison, holding every run of both engines, both engines' full
//! configurations, the pinned baseline version, and the KV-probe verdicts. The markdown table is
//! rendered from that artifact and nothing else, so the published table and the recorded data
//! cannot drift apart.

use std::fmt::Write;

use serde::{Deserialize, Serialize};

use crate::{
    error::{BenchError, Result},
    goodput::Slo,
    kv_probe::KvProbeVerdict,
    record::RunSummary,
    workload::LoadPlan,
};

/// `writeln!` into a `String` cannot fail — `String`'s `fmt::Write` never errors — so the render
/// helpers state that rather than discarding the result.
const WRITE_TO_STRING: &str = "Writing into a String cannot fail";

/// Runs the protocol requires before a number may be reported.
const MIN_RUNS: usize = 3;

/// What one run measured, and whether its KV pool held.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RunResult {
    /// The run's numbers.
    pub summary: RunSummary,
    /// What the KV-leak probe concluded, or `None` for an engine that publishes no free-block
    /// gauge. The baseline is one such engine: vLLM does not expose the counter, so its runs are
    /// unwatched rather than unjudgeable.
    pub kv_probe: Option<KvProbeVerdict>,
}

/// Every run of one engine, with the configuration it ran under.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct EngineResults {
    /// The engine's name, as it appears in the table.
    pub name: String,
    /// The version that ran: a release for the baseline, a commit for the engine under test.
    pub version: String,
    /// The full configuration the engine ran under.
    pub config: serde_json::Value,
    /// One entry per run.
    pub runs: Vec<RunResult>,
}

impl EngineResults {
    /// The run the table reports: the median by goodput.
    ///
    /// For an even number of runs this is the lower of the two middle runs, which is the
    /// conservative choice for the engine under test.
    ///
    /// Returns `None` for an engine with no runs, which [`BenchmarkResults::validate`] reports.
    pub fn median_run(&self) -> Option<&RunResult> {
        let mut ordered = self.runs.iter().collect::<Vec<_>>();
        ordered.sort_by(|left, right| {
            left.summary
                .goodput_per_second
                .total_cmp(&right.summary.goodput_per_second)
        });

        ordered.get(ordered.len().checked_sub(1)? / 2).copied()
    }

    /// The problems that stop this engine's runs from being reportable.
    fn problems(&self, probe: ProbeExpectation) -> Vec<String> {
        let mut problems = Vec::new();
        if self.runs.len() < MIN_RUNS {
            problems.push(format!(
                "{} has {} runs; the protocol reports the median of at least {MIN_RUNS}",
                self.name,
                self.runs.len()
            ));
        }

        for (index, run) in self.runs.iter().enumerate() {
            match (&run.kv_probe, probe) {
                (Some(KvProbeVerdict::Pass { .. }), _)
                | (None, ProbeExpectation::PublishesNoGauge) => {}
                (None, ProbeExpectation::Required) => problems.push(format!(
                    "{} run {index} was not watched for KV leaks; set `engine.metrics_url` so the \
                     probe can sample the pool",
                    self.name
                )),
                (Some(KvProbeVerdict::Leak { reason, .. }), _) => problems.push(format!(
                    "{} run {index} leaked KV blocks: {reason}",
                    self.name
                )),
                (Some(KvProbeVerdict::Inconclusive { reason, .. }), _) => problems.push(format!(
                    "{} run {index} could not be checked for KV leaks: {reason}",
                    self.name
                )),
            }
        }

        problems
    }
}

/// Whether an engine's runs are expected to carry a KV-probe verdict.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProbeExpectation {
    /// The engine under test publishes a free-block gauge, so every run must have been watched and
    /// must have passed. An unwatched run would retire the leak guard without anyone noticing.
    Required,
    /// The pinned baseline publishes no such gauge, so its runs are unwatched. Holding it to a
    /// probe it cannot answer would fail every comparison.
    PublishesNoGauge,
}

/// Whether a larger number is a better one.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub enum Direction {
    /// Goodput, throughput.
    HigherIsBetter,
    /// Latency.
    LowerIsBetter,
}

impl Direction {
    /// How the direction is rendered in the table.
    fn label(self) -> &'static str {
        match self {
            Self::HigherIsBetter => "higher",
            Self::LowerIsBetter => "lower",
        }
    }
}

/// One metric, both engines' numbers, and the ratio between them.
#[derive(Clone, Debug, PartialEq)]
pub struct Comparison {
    /// The metric's name.
    pub metric: String,
    /// The engine under test.
    pub subject: f64,
    /// The pinned baseline.
    pub baseline: f64,
    /// Subject over baseline, whatever the metric's direction.
    pub ratio: f64,
    /// Which way is better.
    pub direction: Direction,
}

impl Comparison {
    /// Builds a comparison of one metric.
    fn new(metric: &str, subject: f64, baseline: f64, direction: Direction) -> Self {
        Self {
            metric: metric.to_string(),
            subject,
            baseline,
            ratio: subject / baseline,
            direction,
        }
    }

    /// The ratio as the table shows it, or an em dash when there is nothing to divide by.
    pub fn rendered_ratio(&self) -> String {
        if self.ratio.is_finite() {
            format!("{:.2}×", self.ratio)
        } else {
            "—".to_string()
        }
    }
}

/// One comparison of one engine against the pinned baseline, on one workload.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct BenchmarkResults {
    /// When the comparison ran.
    pub generated_at: String,
    /// The hardware both engines ran on, as declared by the operator.
    pub hardware: String,
    /// The workload both engines were offered.
    pub workload: String,
    /// The load that workload was offered at.
    pub plan: LoadPlan,
    /// The service level goodput is measured against.
    pub slo: Slo,
    /// The engine under test.
    pub subject: EngineResults,
    /// The pinned baseline.
    pub baseline: EngineResults,
}

impl BenchmarkResults {
    /// Checks the artifact meets the benchmark protocol.
    ///
    /// # Errors
    ///
    /// Returns [`BenchError::Results`] listing every reason the numbers may not be published: too
    /// few runs of either engine, or a run of the engine under test whose KV pool was not watched
    /// or did not hold.
    pub fn validate(&self) -> Result<()> {
        let problems = self
            .subject
            .problems(ProbeExpectation::Required)
            .into_iter()
            .chain(self.baseline.problems(ProbeExpectation::PublishesNoGauge))
            .collect::<Vec<_>>();

        if problems.is_empty() {
            return Ok(());
        }

        Err(BenchError::Results(problems.join("; ")))
    }

    /// The metrics the table compares, taken from each engine's median run.
    ///
    /// Empty when either engine has no runs to take a median of, which [`Self::validate`] reports
    /// as the problem it is.
    pub fn comparisons(&self) -> Vec<Comparison> {
        let (Some(subject), Some(baseline)) =
            (self.subject.median_run(), self.baseline.median_run())
        else {
            return Vec::new();
        };
        let (subject, baseline) = (&subject.summary, &baseline.summary);

        vec![
            Comparison::new(
                "Goodput at SLO (req/s)",
                subject.goodput_per_second,
                baseline.goodput_per_second,
                Direction::HigherIsBetter,
            ),
            Comparison::new(
                "Request throughput (req/s)",
                subject.request_throughput_per_second,
                baseline.request_throughput_per_second,
                Direction::HigherIsBetter,
            ),
            Comparison::new(
                "Output throughput (tok/s)",
                subject.output_token_throughput_per_second,
                baseline.output_token_throughput_per_second,
                Direction::HigherIsBetter,
            ),
            Comparison::new(
                "TTFT p50 (ms)",
                subject.ttft_ms.p50_ms,
                baseline.ttft_ms.p50_ms,
                Direction::LowerIsBetter,
            ),
            Comparison::new(
                "TTFT p99 (ms)",
                subject.ttft_ms.p99_ms,
                baseline.ttft_ms.p99_ms,
                Direction::LowerIsBetter,
            ),
            Comparison::new(
                "ITL p50 (ms)",
                subject.itl_ms.p50_ms,
                baseline.itl_ms.p50_ms,
                Direction::LowerIsBetter,
            ),
            Comparison::new(
                "ITL p99 (ms)",
                subject.itl_ms.p99_ms,
                baseline.itl_ms.p99_ms,
                Direction::LowerIsBetter,
            ),
            Comparison::new(
                "End-to-end p99 (ms)",
                subject.end_to_end_ms.p99_ms,
                baseline.end_to_end_ms.p99_ms,
                Direction::LowerIsBetter,
            ),
        ]
    }

    /// Renders the whole document that is committed under `docs/plan/`: the table, plus the
    /// command that regenerates it from `results_path`.
    pub fn render_document(&self, results_path: &std::path::Path) -> String {
        format!(
            "# Rung-0 baseline table\n\n\
             Generated by `atoma-bench`. Regenerate from the artifact with:\n\n\
             ```shell\n\
             cargo run -p atoma-bench -- render --results {} --table <this file>\n\
             ```\n\n{}",
            results_path.display(),
            self.render_markdown()
        )
    }

    /// Renders the baseline table: absolute numbers, ratios, both configurations, and what the
    /// KV-leak probe concluded about every run.
    pub fn render_markdown(&self) -> String {
        let mut table = String::new();
        self.render_header(&mut table);
        self.render_comparison(&mut table);
        self.render_runs(&mut table);
        self.render_configs(&mut table);
        table
    }

    /// The run's provenance: what ran, where, against what, under which load.
    fn render_header(&self, table: &mut String) {
        writeln!(
            table,
            "## {} vs {} — {} workload\n\n\
             - **Generated:** {}\n\
             - **Hardware:** {}\n\
             - **Load:** {} requests, Poisson arrivals at {:.1} req/s, seed {}\n\
             - **SLO:** first token ≤ {:.0} ms, mean inter-token ≤ {:.0} ms\n\
             - **Runs:** {} runs per engine, median reported\n\
             - **Engine under test:** {} `{}`\n\
             - **Pinned baseline:** {} `{}`\n",
            self.subject.name,
            self.baseline.name,
            self.workload,
            self.generated_at,
            self.hardware,
            self.plan.num_requests,
            self.plan.request_rate_per_second,
            self.plan.seed,
            self.slo.ttft_ms,
            self.slo.itl_ms,
            self.subject.runs.len(),
            self.subject.name,
            self.subject.version,
            self.baseline.name,
            self.baseline.version,
        )
        .expect(WRITE_TO_STRING);
    }

    /// The headline table: absolute numbers alongside the ratio.
    fn render_comparison(&self, table: &mut String) {
        writeln!(
            table,
            "| Metric | {} | {} | Ratio | Better |\n|---|---|---|---|---|",
            self.subject.name, self.baseline.name
        )
        .expect(WRITE_TO_STRING);

        for comparison in self.comparisons() {
            writeln!(
                table,
                "| {} | {:.2} | {:.2} | {} | {} |",
                comparison.metric,
                comparison.subject,
                comparison.baseline,
                comparison.rendered_ratio(),
                comparison.direction.label(),
            )
            .expect(WRITE_TO_STRING);
        }
    }

    /// Every run behind the medians, so the spread is visible rather than summarized away.
    fn render_runs(&self, table: &mut String) {
        writeln!(
            table,
            "\n### Runs\n\n\
             | Engine | Run | Goodput (req/s) | Completed | Failed | TTFT p99 (ms) | ITL p99 (ms) \
             | KV-leak probe |\n|---|---|---|---|---|---|---|---|"
        )
        .expect(WRITE_TO_STRING);

        for engine in [&self.subject, &self.baseline] {
            for (index, run) in engine.runs.iter().enumerate() {
                writeln!(
                    table,
                    "| {} | {} | {:.2} | {} | {} | {:.2} | {:.2} | {} |",
                    engine.name,
                    index,
                    run.summary.goodput_per_second,
                    run.summary.completed,
                    run.summary.failed,
                    run.summary.ttft_ms.p99_ms,
                    run.summary.itl_ms.p99_ms,
                    rendered_verdict(run.kv_probe.as_ref()),
                )
                .expect(WRITE_TO_STRING);
            }
        }
    }

    /// Both engines' full configurations, which the protocol publishes with the numbers.
    fn render_configs(&self, table: &mut String) {
        for engine in [&self.subject, &self.baseline] {
            let config = serde_json::to_string_pretty(&engine.config)
                .unwrap_or_else(|error| format!("<unserializable configuration: {error}>"));
            writeln!(
                table,
                "\n### {} configuration\n\n```json\n{config}\n```",
                engine.name
            )
            .expect(WRITE_TO_STRING);
        }
    }
}

/// How a probe verdict reads in the table.
fn rendered_verdict(verdict: Option<&KvProbeVerdict>) -> String {
    match verdict {
        None => "not watched (no free-block gauge)".to_string(),
        Some(KvProbeVerdict::Pass {
            baseline_free_blocks,
            final_free_blocks,
            ..
        }) => format!("pass ({final_free_blocks}/{baseline_free_blocks} blocks free at drain)"),
        Some(KvProbeVerdict::Leak { reason, .. }) => format!("**leak** — {reason}"),
        Some(KvProbeVerdict::Inconclusive { reason, .. }) => format!("inconclusive — {reason}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::Distribution;

    fn run(goodput: f64, ttft_p50: f64) -> RunResult {
        RunResult {
            summary: RunSummary {
                num_requests: 100,
                completed: 100,
                failed: 0,
                output_tokens: 12_800,
                duration_seconds: 20.0,
                goodput_per_second: goodput,
                request_throughput_per_second: 5.0,
                output_token_throughput_per_second: 640.0,
                ttft_ms: Distribution {
                    count: 100,
                    mean_ms: ttft_p50,
                    p50_ms: ttft_p50,
                    p90_ms: ttft_p50 * 1.5,
                    p99_ms: ttft_p50 * 2.0,
                    max_ms: ttft_p50 * 3.0,
                },
                itl_ms: Distribution::default(),
                end_to_end_ms: Distribution::default(),
            },
            kv_probe: Some(KvProbeVerdict::Pass {
                baseline_free_blocks: 4_096,
                final_free_blocks: 4_096,
                samples: 20,
            }),
        }
    }

    fn engine(name: &str, goodputs: &[f64]) -> EngineResults {
        EngineResults {
            name: name.to_string(),
            version: "0.1.0".to_string(),
            config: serde_json::json!({ "model": "meta-llama/Meta-Llama-3-8B-Instruct" }),
            runs: goodputs
                .iter()
                .map(|goodput| run(*goodput, 100.0))
                .collect(),
        }
    }

    fn results(subject: EngineResults, baseline: EngineResults) -> BenchmarkResults {
        BenchmarkResults {
            generated_at: "2026-08-06T00:00:00Z".to_string(),
            hardware: "1×H100 SXM 80GB".to_string(),
            workload: "sharegpt".to_string(),
            plan: LoadPlan {
                num_requests: 100,
                request_rate_per_second: 8.0,
                seed: 1,
            },
            slo: Slo {
                ttft_ms: 2_000.0,
                itl_ms: 50.0,
            },
            subject,
            baseline,
        }
    }

    /// The protocol reports the median run, not the best one: reporting the best turns run-to-run
    /// noise into a result.
    #[test]
    fn test_the_median_run_is_reported() {
        let engine = engine("atoma-infer", &[3.0, 9.0, 5.0]);

        assert_eq!(
            engine
                .median_run()
                .expect("Three runs have a median")
                .summary
                .goodput_per_second,
            5.0
        );
    }

    #[test]
    fn test_the_median_of_an_even_number_of_runs_is_the_lower_middle() {
        let engine = engine("atoma-infer", &[3.0, 5.0, 7.0, 9.0]);

        assert_eq!(
            engine
                .median_run()
                .expect("Three runs have a median")
                .summary
                .goodput_per_second,
            5.0
        );
    }

    /// Fewer than three runs is not a measurement under the protocol, and the artifact must not
    /// pretend otherwise.
    #[test]
    fn test_fewer_than_three_runs_fails_validation() {
        let results = results(
            engine("atoma-infer", &[3.0, 5.0]),
            engine("vllm", &[6.0, 6.0, 6.0]),
        );

        let error = results.validate().expect_err("Two runs is not a median");

        assert!(format!("{error}").contains("atoma-infer"), "{error}");
    }

    /// A leaking run's numbers are not comparable — the engine was in a different state at the end
    /// of the run than at the start.
    #[test]
    fn test_a_failed_kv_probe_fails_validation() {
        let mut subject = engine("atoma-infer", &[3.0, 5.0, 7.0]);
        subject.runs[1].kv_probe = Some(KvProbeVerdict::Leak {
            reason: "128 of 4096 blocks had not returned once the run drained".to_string(),
            baseline_free_blocks: 4_096,
            final_free_blocks: 3_968,
            samples: 20,
        });

        let error = results(subject, engine("vllm", &[6.0, 6.0, 6.0]))
            .validate()
            .expect_err("A leaking run must not be reported");

        assert!(
            format!("{error}").contains("blocks had not returned"),
            "{error}"
        );
    }

    /// vLLM publishes no free-block gauge, so its runs carry no verdict. Failing the comparison
    /// for that would fail every comparison there is.
    #[test]
    fn test_an_unwatched_baseline_validates() {
        let mut baseline = engine("vllm", &[6.0, 6.0, 6.0]);
        for run in baseline.runs.iter_mut() {
            run.kv_probe = None;
        }

        let results = results(engine("atoma-infer", &[3.0, 5.0, 7.0]), baseline);

        assert!(results.validate().is_ok(), "{:?}", results.validate());
    }

    /// The engine under test is the one the probe guards; a run nobody watched must not be
    /// reported as a clean one.
    #[test]
    fn test_an_unwatched_engine_under_test_fails_validation() {
        let mut subject = engine("atoma-infer", &[3.0, 5.0, 7.0]);
        subject.runs[2].kv_probe = None;

        let error = results(subject, engine("vllm", &[6.0, 6.0, 6.0]))
            .validate()
            .expect_err("An unwatched run must not be reported");

        assert!(format!("{error}").contains("metrics_url"), "{error}");
    }

    #[test]
    fn test_a_clean_three_run_comparison_validates() {
        let results = results(
            engine("atoma-infer", &[3.0, 5.0, 7.0]),
            engine("vllm", &[6.0, 6.0, 6.0]),
        );

        assert!(results.validate().is_ok());
    }

    #[test]
    fn test_ratios_are_the_subject_over_the_baseline() {
        let results = results(
            engine("atoma-infer", &[3.0, 5.0, 7.0]),
            engine("vllm", &[10.0, 10.0, 10.0]),
        );

        let goodput = results
            .comparisons()
            .into_iter()
            .find(|comparison| comparison.metric.starts_with("Goodput"))
            .expect("The table must compare goodput");

        assert!((goodput.subject - 5.0).abs() < 1e-9);
        assert!((goodput.baseline - 10.0).abs() < 1e-9);
        assert!((goodput.ratio - 0.5).abs() < 1e-9);
        assert_eq!(goodput.direction, Direction::HigherIsBetter);
    }

    #[test]
    fn test_a_ratio_against_a_zero_baseline_is_reported_as_absent() {
        let results = results(
            engine("atoma-infer", &[3.0, 5.0, 7.0]),
            engine("vllm", &[0.0, 0.0, 0.0]),
        );

        let goodput = results
            .comparisons()
            .into_iter()
            .find(|comparison| comparison.metric.starts_with("Goodput"))
            .expect("The table must compare goodput");

        assert!(goodput.ratio.is_nan() || goodput.ratio.is_infinite());
        assert_eq!(goodput.rendered_ratio(), "—");
    }

    /// The table is the rung-0 deliverable: it has to carry the absolute numbers, the ratios, both
    /// engines' configurations and the pinned baseline version, or it is not evidence.
    #[test]
    fn test_the_rendered_table_carries_the_numbers_the_protocol_asks_for() {
        let results = results(
            engine("atoma-infer", &[3.0, 5.0, 7.0]),
            engine("vllm", &[10.0, 10.0, 10.0]),
        );

        let table = results.render_markdown();

        assert!(
            table.contains("| Goodput at SLO (req/s) | 5.00 | 10.00 | 0.50× |"),
            "{table}"
        );
        assert!(table.contains("atoma-infer"), "{table}");
        assert!(table.contains("vllm"), "{table}");
        assert!(table.contains("1×H100 SXM 80GB"), "{table}");
        assert!(table.contains("3 runs"), "{table}");
        assert!(
            table.contains("meta-llama/Meta-Llama-3-8B-Instruct"),
            "{table}"
        );
        assert!(table.contains("TTFT p50"), "{table}");
        assert!(table.contains("KV-leak probe"), "{table}");
    }

    /// The committed document has to say how to regenerate itself, or the next run produces a
    /// second file next to a stale one.
    #[test]
    fn test_the_document_names_the_artifact_it_was_rendered_from() {
        let results = results(
            engine("atoma-infer", &[3.0, 5.0, 7.0]),
            engine("vllm", &[6.0, 6.0, 6.0]),
        );

        let document = results.render_document(std::path::Path::new("results/rung0.json"));

        assert!(
            document.starts_with("# Rung-0 baseline table"),
            "{document}"
        );
        assert!(document.contains("results/rung0.json"), "{document}");
        assert!(
            document.contains("| Goodput at SLO (req/s) |"),
            "{document}"
        );
    }

    #[test]
    fn test_results_round_trip_through_json() {
        let results = results(
            engine("atoma-infer", &[3.0, 5.0, 7.0]),
            engine("vllm", &[6.0, 6.0, 6.0]),
        );

        let json = serde_json::to_string(&results).expect("Failed to serialize the results");
        let parsed: BenchmarkResults =
            serde_json::from_str(&json).expect("Failed to deserialize the results");

        assert_eq!(parsed, results);
    }
}
