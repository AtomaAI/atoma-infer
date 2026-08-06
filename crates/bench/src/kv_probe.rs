//! The KV-leak probe.
//!
//! The engine publishes its free KV block count as a gauge; the probe samples it across a run and
//! judges the series afterwards. Two things fail a run:
//!
//! 1. **A falling ceiling** — the maximum the pool recovers to falls window after window. Under
//!    steady load the count oscillates as batches are admitted and retire, so the ceiling, not the
//!    instantaneous value, is the signal.
//! 2. **A shortfall at drain** — the run ends with fewer free blocks than it started with.
//!
//! This is the standing regression guard for the rung-0 P0 in which finished sequences never
//! returned their blocks: the pool drained monotonically until the engine wedged, and every
//! latency number taken before the wedge looked fine.

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::{sync::oneshot, task::JoinHandle};
use tracing::warn;

/// The gauge the engine publishes its free GPU KV block count on.
pub const FREE_GPU_BLOCKS_METRIC: &str = "atoma_kv_free_gpu_blocks";

/// How the probe samples and how it judges what it sampled.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct KvProbeConfig {
    /// Name of the gauge carrying the free block count.
    pub metric: String,
    /// How often the gauge is sampled.
    pub interval_ms: u64,
    /// Blocks the pool may be short by at the end of a run without failing it.
    pub tolerance_blocks: u64,
    /// Number of windows the series is split into when looking for a falling ceiling.
    pub num_windows: usize,
}

impl Default for KvProbeConfig {
    fn default() -> Self {
        Self {
            metric: FREE_GPU_BLOCKS_METRIC.to_string(),
            interval_ms: 1_000,
            tolerance_blocks: 0,
            num_windows: 4,
        }
    }
}

impl KvProbeConfig {
    /// The sampling interval.
    pub fn interval(&self) -> Duration {
        Duration::from_millis(self.interval_ms)
    }

    /// Samples needed before the series can be judged: enough for every window to hold at least a
    /// dip and a recovery.
    fn min_samples(&self) -> usize {
        self.num_windows.max(1) * 2
    }
}

/// One reading of the engine's free KV block count.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct FreeBlockSample {
    /// Offset from the start of the probe.
    pub at: Duration,
    /// Free blocks at that moment.
    pub free_blocks: u64,
}

/// What the probe concluded about a run.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub enum KvProbeVerdict {
    /// The pool held.
    Pass {
        /// Free blocks before load was offered.
        baseline_free_blocks: u64,
        /// Free blocks after the run drained.
        final_free_blocks: u64,
        /// Samples the verdict was drawn from.
        samples: usize,
    },
    /// Blocks went missing.
    Leak {
        /// What the series showed.
        reason: String,
        /// Free blocks before load was offered.
        baseline_free_blocks: u64,
        /// Free blocks after the run drained.
        final_free_blocks: u64,
        /// Samples the verdict was drawn from.
        samples: usize,
    },
    /// The pool was not measured well enough to judge.
    Inconclusive {
        /// Why it could not be judged.
        reason: String,
        /// Samples that were taken.
        samples: usize,
    },
}

impl KvProbeVerdict {
    /// Whether the run may be reported. An unmeasured pool does not pass: a probe that quietly
    /// scraped nothing would retire the guard without anyone noticing.
    pub fn run_passes(&self) -> bool {
        matches!(self, Self::Pass { .. })
    }
}

/// Judges a sampled free-block series.
pub fn verdict(samples: &[FreeBlockSample], config: &KvProbeConfig) -> KvProbeVerdict {
    if samples.len() < config.min_samples() {
        return KvProbeVerdict::Inconclusive {
            reason: format!(
                "{} samples of `{}` is fewer than the {} needed to judge the pool; is the engine \
                 publishing the gauge?",
                samples.len(),
                config.metric,
                config.min_samples()
            ),
            samples: samples.len(),
        };
    }

    let baseline = samples[0].free_blocks;
    let last = samples[samples.len() - 1].free_blocks;

    if last + config.tolerance_blocks < baseline {
        return KvProbeVerdict::Leak {
            reason: format!(
                "{} of {baseline} blocks had not returned once the run drained",
                baseline - last
            ),
            baseline_free_blocks: baseline,
            final_free_blocks: last,
            samples: samples.len(),
        };
    }

    if let Some(ceilings) = falling_ceiling(samples, config) {
        return KvProbeVerdict::Leak {
            reason: format!(
                "the free-block ceiling fell window after window across the run: {ceilings:?}"
            ),
            baseline_free_blocks: baseline,
            final_free_blocks: last,
            samples: samples.len(),
        };
    }

    KvProbeVerdict::Pass {
        baseline_free_blocks: baseline,
        final_free_blocks: last,
        samples: samples.len(),
    }
}

/// The per-window maxima, when they never rise and end more than the tolerance below where they
/// started — the shape of a pool that is being drained rather than cycled.
fn falling_ceiling(samples: &[FreeBlockSample], config: &KvProbeConfig) -> Option<Vec<u64>> {
    let window = samples.len() / config.num_windows.max(1);
    let ceilings = samples
        .chunks(window.max(1))
        .map(|window| {
            window
                .iter()
                .map(|sample| sample.free_blocks)
                .max()
                .unwrap_or_default()
        })
        .collect::<Vec<_>>();

    let never_rises = ceilings.windows(2).all(|pair| pair[1] <= pair[0]);
    let fall = ceilings.first()?.saturating_sub(*ceilings.last()?);
    (never_rises && fall > config.tolerance_blocks).then_some(ceilings)
}

/// Reads one gauge out of a Prometheus text-format exposition.
///
/// Returns `None` when the metric is absent or its value is not a number, which the probe treats
/// as a missed sample rather than an error: one unreadable scrape should not end a run.
pub fn parse_gauge(exposition: &str, metric: &str) -> Option<f64> {
    exposition
        .lines()
        .filter(|line| !line.starts_with('#'))
        .find_map(|line| gauge_value(line, metric))
}

/// The value of `line`, if the line carries `metric` rather than one whose name merely starts the
/// same way.
fn gauge_value(line: &str, metric: &str) -> Option<f64> {
    let rest = line.strip_prefix(metric)?;
    let value = match rest.split_once('}') {
        Some((labels, after_labels)) if labels.starts_with('{') => after_labels,
        Some(_) | None => rest,
    };
    value.trim().parse().ok()
}

/// Samples an engine's free-block gauge over the length of a run.
pub struct KvProbe {
    /// The engine's Prometheus endpoint.
    metrics_url: String,
    /// What to sample and how often.
    config: KvProbeConfig,
    /// The scraping client.
    client: reqwest::Client,
}

impl KvProbe {
    /// Builds a probe against an engine's metrics endpoint.
    pub fn new(metrics_url: String, config: KvProbeConfig) -> Self {
        Self {
            metrics_url,
            config,
            client: reqwest::Client::new(),
        }
    }

    /// Starts sampling. The first sample is taken immediately, so it should be started before load
    /// is offered — that reading is the baseline the run is judged against.
    pub fn start(self) -> RunningKvProbe {
        let (stop, stopped) = oneshot::channel();
        let task = tokio::spawn(self.sample_until_stopped(stopped));
        RunningKvProbe { stop, task }
    }

    /// Samples on the configured interval until asked to stop, then takes the drain sample.
    async fn sample_until_stopped(
        self,
        mut stopped: oneshot::Receiver<()>,
    ) -> Vec<FreeBlockSample> {
        let started = Instant::now();
        let mut samples = Vec::new();
        let mut ticks = tokio::time::interval(self.config.interval());

        loop {
            tokio::select! {
                _ = &mut stopped => break,
                _ = ticks.tick() => self.sample_into(&mut samples, started).await,
            }
        }

        self.sample_into(&mut samples, started).await;
        samples
    }

    /// Takes one sample, keeping it only if the gauge could be read.
    async fn sample_into(&self, samples: &mut Vec<FreeBlockSample>, started: Instant) {
        match self.read_gauge().await {
            Some(free_blocks) => samples.push(FreeBlockSample {
                at: started.elapsed(),
                free_blocks,
            }),
            None => warn!(
                metric = self.config.metric,
                url = self.metrics_url,
                "Could not read the free-block gauge; skipping this sample"
            ),
        }
    }

    /// Scrapes the endpoint and reads the gauge out of it.
    async fn read_gauge(&self) -> Option<u64> {
        let response = self
            .client
            .get(&self.metrics_url)
            .send()
            .await
            .and_then(reqwest::Response::error_for_status);
        let exposition = match response {
            Ok(response) => response.text().await.ok()?,
            Err(error) => {
                warn!(url = self.metrics_url, %error, "Failed to scrape the engine's metrics");
                return None;
            }
        };

        parse_gauge(&exposition, &self.config.metric).map(|value| value.max(0.0) as u64)
    }
}

/// A probe that is sampling.
pub struct RunningKvProbe {
    /// Tells the sampling task to take its last sample and finish.
    stop: oneshot::Sender<()>,
    /// The sampling task.
    task: JoinHandle<Vec<FreeBlockSample>>,
}

impl RunningKvProbe {
    /// Stops sampling and returns the series, ending with the sample taken after the run drained.
    pub async fn finish(self) -> Vec<FreeBlockSample> {
        let _ = self.stop.send(());
        self.task.await.unwrap_or_else(|error| {
            warn!(%error, "The KV probe task did not finish");
            Vec::new()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> KvProbeConfig {
        KvProbeConfig {
            tolerance_blocks: 0,
            ..KvProbeConfig::default()
        }
    }

    /// Builds a sample series one second apart from a free-block series.
    fn samples(free_blocks: &[u64]) -> Vec<FreeBlockSample> {
        free_blocks
            .iter()
            .enumerate()
            .map(|(index, free_blocks)| FreeBlockSample {
                at: Duration::from_secs(index as u64),
                free_blocks: *free_blocks,
            })
            .collect()
    }

    /// A busy engine's free-block count oscillates: it falls as a batch is admitted and comes back
    /// as the batch retires. Only the ceiling falling is a leak.
    fn sawtooth(cycles: usize, ceiling: u64, floor: u64) -> Vec<u64> {
        (0..cycles)
            .flat_map(|_| [ceiling, floor, floor + 5, ceiling])
            .collect()
    }

    #[test]
    fn test_a_gauge_is_read_from_the_prometheus_text_format() {
        let text = "\
# HELP atoma_kv_free_gpu_blocks Free KV blocks
# TYPE atoma_kv_free_gpu_blocks gauge
atoma_kv_free_gpu_blocks 118
llm_service_requests_total 42
";

        assert_eq!(
            parse_gauge(text, "atoma_kv_free_gpu_blocks"),
            Some(118.0),
            "the gauge line must be read, not the HELP or TYPE lines"
        );
    }

    #[test]
    fn test_a_labelled_gauge_is_read() {
        let text = "atoma_kv_free_gpu_blocks{device=\"gpu\"} 96.0\n";

        assert_eq!(parse_gauge(text, "atoma_kv_free_gpu_blocks"), Some(96.0));
    }

    /// `atoma_kv_free_gpu_blocks_total` is a different series; a prefix match would silently
    /// probe the wrong one.
    #[test]
    fn test_a_metric_whose_name_is_a_prefix_is_not_mistaken_for_the_gauge() {
        let text = "atoma_kv_free_gpu_blocks_total 7\n";

        assert_eq!(parse_gauge(text, "atoma_kv_free_gpu_blocks"), None);
    }

    #[test]
    fn test_a_missing_or_unreadable_gauge_reads_as_absent() {
        assert_eq!(parse_gauge("", "atoma_kv_free_gpu_blocks"), None);
        assert_eq!(
            parse_gauge(
                "atoma_kv_free_gpu_blocks NaN-ish\n",
                "atoma_kv_free_gpu_blocks"
            ),
            None
        );
        assert_eq!(
            parse_gauge("atoma_kv_free_gpu_blocks\n", "atoma_kv_free_gpu_blocks"),
            None
        );
    }

    #[test]
    fn test_a_pool_that_ends_where_it_started_passes() {
        let verdict = verdict(&samples(&sawtooth(4, 128, 40)), &config());

        assert!(verdict.run_passes(), "{verdict:?}");
    }

    #[test]
    fn test_a_pool_under_steady_load_passes_while_it_recovers() {
        let series = [128, 40, 128, 32, 128, 60, 128, 128];

        let verdict = verdict(&samples(&series), &config());

        assert!(verdict.run_passes(), "{verdict:?}");
    }

    /// This is P0-2 reintroduced: blocks are handed out and never returned, so the ceiling the
    /// pool recovers to falls step by step until the engine wedges.
    #[test]
    fn test_a_monotonically_falling_ceiling_fails_the_run() {
        let series = [128, 100, 120, 90, 112, 80, 104, 70, 96, 60, 88, 84];

        let verdict = verdict(&samples(&series), &config());

        assert!(!verdict.run_passes(), "{verdict:?}");
        assert!(
            matches!(verdict, KvProbeVerdict::Leak { .. }),
            "{verdict:?}"
        );
    }

    /// The pool may dip during the run, but once the last request has drained every block has to
    /// be back, whatever the shape in between.
    #[test]
    fn test_blocks_missing_after_the_run_drains_fail_the_run() {
        let series = [128, 40, 128, 40, 128, 40, 128, 40, 128, 120];

        let verdict = verdict(&samples(&series), &config());

        assert!(!verdict.run_passes(), "{verdict:?}");
        match verdict {
            KvProbeVerdict::Leak {
                baseline_free_blocks,
                final_free_blocks,
                ..
            } => {
                assert_eq!(baseline_free_blocks, 128);
                assert_eq!(final_free_blocks, 120);
            }
            other => panic!("expected a leak verdict, got {other:?}"),
        }
    }

    /// The drain check only catches a run that ends below where the *first sample* found the pool,
    /// and that sample can understate it — a warmup request still in flight holds blocks. The
    /// ceiling check is what catches a leak the baseline sample hid: here the pool recovers to 100
    /// at the end, exactly where sampling started, while its ceiling fell from 120 all the way
    /// down.
    #[test]
    fn test_a_falling_ceiling_fails_a_run_that_ends_where_sampling_started() {
        let series = [100, 120, 118, 116, 114, 112, 110, 108, 106, 104, 102, 100];

        let verdict = verdict(&samples(&series), &config());

        match verdict {
            KvProbeVerdict::Leak { ref reason, .. } => assert!(
                reason.contains("ceiling"),
                "the ceiling check, not the drain check, must be what fails this run: {reason}"
            ),
            other => panic!("expected a leak verdict, got {other:?}"),
        }
    }

    /// Block accounting is exact, so the default tolerance is zero; a deployment that legitimately
    /// parks blocks can widen it, and the widened tolerance has to actually apply.
    #[test]
    fn test_a_shortfall_inside_the_tolerance_passes() {
        let series = [128, 40, 128, 40, 128, 40, 128, 40, 128, 126];
        let tolerant = KvProbeConfig {
            tolerance_blocks: 4,
            ..KvProbeConfig::default()
        };

        assert!(verdict(&samples(&series), &tolerant).run_passes());
        assert!(!verdict(&samples(&series), &config()).run_passes());
    }

    #[test]
    fn test_too_few_samples_are_inconclusive_rather_than_a_pass() {
        let verdict = verdict(&samples(&[128, 128, 128]), &config());

        assert!(
            matches!(verdict, KvProbeVerdict::Inconclusive { .. }),
            "{verdict:?}"
        );
        assert!(
            !verdict.run_passes(),
            "an unmeasured pool must not be reported as a clean one"
        );
    }

    #[test]
    fn test_a_run_with_no_samples_is_inconclusive() {
        let verdict = verdict(&[], &config());

        assert!(
            matches!(verdict, KvProbeVerdict::Inconclusive { .. }),
            "{verdict:?}"
        );
    }
}
