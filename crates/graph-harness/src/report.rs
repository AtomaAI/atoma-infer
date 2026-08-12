//! The findings note: capture matrix table, host timings, and the per-graph memory measurement
//! (the calibration constant for bucket-ladder memory budgeting), rendered as markdown plus raw
//! JSON.

use crate::compare::Bf16Divergence;
use serde::Serialize;

/// Order statistics of one timing series, in microseconds.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Stats {
    pub samples: usize,
    pub median_us: f64,
    pub p99_us: f64,
    pub min_us: f64,
    pub max_us: f64,
}

impl Stats {
    /// Computes the statistics, or `None` for an empty series.
    pub fn from_micros(mut samples: Vec<f64>) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }
        samples.sort_unstable_by(|a, b| a.partial_cmp(b).expect("timings are finite"));
        let rank = |q: f64| {
            let position = (q * samples.len() as f64).ceil() as usize;
            samples[position.clamp(1, samples.len()) - 1]
        };
        Some(Self {
            samples: samples.len(),
            median_us: rank(0.5),
            p99_us: rank(0.99),
            min_us: samples[0],
            max_us: samples[samples.len() - 1],
        })
    }
}

/// A bit-identity failure: which step diverged and how.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DivergenceReport {
    pub step: usize,
    pub divergence: Bf16Divergence,
}

/// Everything one capture-matrix cell produced. Fields are `Option` because a failure in any
/// phase leaves the later phases unmeasured; `failure` carries the classification text.
#[derive(Debug, Clone, Serialize)]
pub struct CellReport {
    pub label: String,
    /// Classification of the first failure (kernel capture-illegal / cudarc behavior / FFI
    /// hidden sync / NCCL), with the phase it happened in. `None` when every phase passed.
    pub failure: Option<String>,
    pub capture_ms: Option<f64>,
    pub graph_node_count: Option<usize>,
    /// Steps whose replayed logits and argmax matched the eager run byte-for-byte.
    pub identity_steps: usize,
    pub divergence: Option<DivergenceReport>,
    pub soak_replays: usize,
    /// `cuMemGetInfo` free-memory delta across the soak, after the first replay warmed the
    /// graph; the acceptance criterion is 0.
    pub soak_mem_delta_bytes: Option<i64>,
    /// Free-memory drop from just before capture to after instantiate+upload: the per-graph
    /// dedicated overhead — the A4 calibration constant.
    pub graph_dedicated_bytes: Option<i64>,
    pub eager_enqueue: Option<Stats>,
    pub replay_enqueue: Option<Stats>,
    pub eager_step: Option<Stats>,
    pub replay_step: Option<Stats>,
}

impl CellReport {
    /// An empty report for a cell that has not run any phase yet.
    pub fn new(label: String) -> Self {
        Self {
            label,
            failure: None,
            capture_ms: None,
            graph_node_count: None,
            identity_steps: 0,
            divergence: None,
            soak_replays: 0,
            soak_mem_delta_bytes: None,
            graph_dedicated_bytes: None,
            eager_enqueue: None,
            replay_enqueue: None,
            eager_step: None,
            replay_step: None,
        }
    }
}

fn fmt_stats(stats: Option<Stats>) -> String {
    match stats {
        Some(s) => format!("{:.1} / {:.1}", s.median_us, s.p99_us),
        None => "—".into(),
    }
}

fn fmt_bytes(bytes: Option<i64>) -> String {
    match bytes {
        Some(b) => format!("{b}"),
        None => "—".into(),
    }
}

/// Renders the findings note. `header_lines` carries run metadata (device, dims, seed) the
/// harness knows and this module does not.
pub fn render_markdown(header_lines: &[String], reports: &[CellReport]) -> String {
    let mut out = String::from("# #143 graph harness findings\n\n");
    for line in header_lines {
        out.push_str(&format!("- {line}\n"));
    }

    out.push_str(
        "\n## Capture matrix\n\n\
         | cell | capture | nodes | bit-identity steps | soak replays | soak mem delta (B) | \
         dedicated graph bytes |\n|---|---|---|---|---|---|---|\n",
    );
    for report in reports {
        let capture = match (&report.failure, report.capture_ms) {
            (None, Some(ms)) => format!("ok ({ms:.0} ms)"),
            (None, None) => "not run".into(),
            (Some(_), Some(ms)) => format!("ok ({ms:.0} ms), later phase failed"),
            (Some(_), None) => "FAILED".into(),
        };
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} |\n",
            report.label,
            capture,
            report
                .graph_node_count
                .map_or_else(|| "—".into(), |n| n.to_string()),
            report.identity_steps,
            report.soak_replays,
            fmt_bytes(report.soak_mem_delta_bytes),
            fmt_bytes(report.graph_dedicated_bytes),
        ));
    }

    out.push_str(
        "\n## Host time per step, µs (median / p99)\n\n\
         | cell | eager enqueue | replay enqueue | eager step | replay step |\n\
         |---|---|---|---|---|\n",
    );
    for report in reports {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} |\n",
            report.label,
            fmt_stats(report.eager_enqueue),
            fmt_stats(report.replay_enqueue),
            fmt_stats(report.eager_step),
            fmt_stats(report.replay_step),
        ));
    }

    let failures: Vec<&CellReport> = reports.iter().filter(|r| r.failure.is_some()).collect();
    if !failures.is_empty() {
        out.push_str("\n## Failures (spec findings, not substrate verdicts)\n\n");
        for report in &failures {
            let failure = report.failure.as_deref().unwrap_or_default();
            out.push_str(&format!("- **{}**: {}\n", report.label, failure));
            if let Some(divergence) = &report.divergence {
                out.push_str(&format!(
                    "  - first divergence at step {}, element {}: replay {} (0x{:04x}) vs eager \
                     {} (0x{:04x})\n",
                    divergence.step,
                    divergence.divergence.element_index,
                    divergence.divergence.replay_value,
                    divergence.divergence.replay_bits,
                    divergence.divergence.eager_value,
                    divergence.divergence.eager_bits,
                ));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_report_the_right_order_statistics() {
        let samples: Vec<f64> = (1..=100).map(f64::from).collect();
        let stats = Stats::from_micros(samples).unwrap();
        assert_eq!(stats.median_us, 50.0);
        assert_eq!(stats.p99_us, 99.0);
        assert_eq!(stats.min_us, 1.0);
        assert_eq!(stats.max_us, 100.0);
    }

    #[test]
    fn stats_of_a_single_sample_are_that_sample() {
        let stats = Stats::from_micros(vec![7.5]).unwrap();
        assert_eq!((stats.median_us, stats.p99_us), (7.5, 7.5));
    }

    #[test]
    fn empty_series_produce_no_stats() {
        assert_eq!(Stats::from_micros(Vec::new()), None);
    }

    #[test]
    fn markdown_carries_every_cell_and_flags_failures() {
        let ok = CellReport {
            capture_ms: Some(120.0),
            graph_node_count: Some(64),
            identity_steps: 32,
            soak_replays: 1000,
            soak_mem_delta_bytes: Some(0),
            graph_dedicated_bytes: Some(2_097_152),
            ..CellReport::new("bs64".into())
        };
        let mut failed = CellReport::new("bs1+nccl".into());
        failed.failure = Some("NCCL capture issue: ncclInvalidUsage during begin".into());

        let markdown = render_markdown(&["device: H100".into()], &[ok, failed]);
        assert!(markdown.contains("| bs64 | ok (120 ms) | 64 | 32 | 1000 | 0 | 2097152 |"));
        assert!(markdown.contains("- device: H100"));
        assert!(markdown.contains("**bs1+nccl**: NCCL capture issue"));
    }
}
