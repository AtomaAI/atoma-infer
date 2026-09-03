//! The KV-leak probe.
//!
//! The engine publishes the size of its KV pool as a gauge; the probe samples it across a run and
//! judges the series afterwards. Two things fail a run:
//!
//! 1. **A falling ceiling** — the maximum the pool recovers to falls window after window. Under
//!    steady load the count oscillates as batches are admitted and retire, so the ceiling, not the
//!    instantaneous value, is the signal.
//! 2. **A shortfall at drain** — the run ends with fewer blocks than it started with.
//!
//! This is the standing regression guard for the rung-0 P0 in which finished sequences never
//! returned their blocks: the pool drained monotonically until the engine wedged, and every
//! latency number taken before the wedge looked fine.

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::{sync::oneshot, task::JoinHandle};
use tracing::warn;

/// How the probe samples and how it judges what it sampled.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct KvProbeConfig {
    /// Name of the gauge carrying the count of blocks the pool can still hand out.
    ///
    /// The engine under test owns this name, so the harness takes it as configuration rather than
    /// compiling in a copy that could drift: `atoma-infer` publishes
    /// `atoma_engine_available_blocks`, the `AVAILABLE_BLOCKS_GAUGE` of its API crate, and another
    /// engine publishes whatever it publishes. It must be the count including any blocks the pool
    /// holds cached for reuse — `atoma_engine_free_blocks` counts the free list alone, which a
    /// healthy engine under load drains to zero. Naming no gauge is only allowed when no
    /// `metrics_url` is configured and the probe therefore never runs.
    #[serde(default)]
    pub metric: Option<String>,
    /// How often the gauge is sampled.
    #[serde(default = "default_interval_ms")]
    pub interval_ms: u64,
    /// Blocks the pool may be short by at the end of a run without failing it.
    #[serde(default)]
    pub tolerance_blocks: u64,
    /// Number of windows the series is split into when looking for a falling ceiling.
    #[serde(default = "default_num_windows")]
    pub num_windows: usize,
}

/// A sample a second: fine enough to see a batch retire, coarse enough not to perturb the run.
fn default_interval_ms() -> u64 {
    1_000
}

/// Four windows over a run: enough for a ceiling to fall three times before the run is judged.
fn default_num_windows() -> usize {
    4
}

/// How often a stopped pool is read while it settles.
///
/// Deliberately not the sampling interval: an engine republishes its gauge on a tick of its own,
/// and reading faster than it publishes would read the same stale value twice and call a pool
/// settled that is still moving. A second covers the half-second tick `atoma-infer` publishes on.
const SETTLE_INTERVAL: Duration = Duration::from_secs(1);

/// Readings that have to pass without the pool growing before it counts as settled.
const SETTLED_READINGS: usize = 3;

/// The most readings spent waiting for a pool to settle. A pool still climbing after this many is
/// not lagging behind a run, and the drain check judges whatever it had reached.
const SETTLE_READING_LIMIT: usize = 15;

impl Default for KvProbeConfig {
    fn default() -> Self {
        Self {
            metric: None,
            interval_ms: default_interval_ms(),
            tolerance_blocks: 0,
            num_windows: default_num_windows(),
        }
    }
}

impl KvProbeConfig {
    /// The gauge to sample, if one is named. A name of nothing but whitespace names no gauge.
    pub fn gauge(&self) -> Option<&str> {
        self.metric
            .as_deref()
            .map(str::trim)
            .filter(|gauge| !gauge.is_empty())
    }

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

/// One reading of the engine's KV pool.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct BlockSample {
    /// Offset from the start of the probe.
    pub at: Duration,
    /// Blocks the pool could still hand out at that moment.
    pub blocks: u64,
}

/// What the probe concluded about a run.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub enum KvProbeVerdict {
    /// The pool held.
    Pass {
        /// Blocks the pool could hand out before load was offered.
        baseline_blocks: u64,
        /// Blocks the pool could hand out after the run drained.
        final_blocks: u64,
        /// Samples the verdict was drawn from.
        samples: usize,
    },
    /// Blocks went missing.
    Leak {
        /// What the series showed.
        reason: String,
        /// Blocks the pool could hand out before load was offered.
        baseline_blocks: u64,
        /// Blocks the pool could hand out after the run drained.
        final_blocks: u64,
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

    /// Why the run did not pass, if it did not.
    pub fn failure_reason(&self) -> Option<&str> {
        match self {
            Self::Pass { .. } => None,
            Self::Leak { reason, .. } | Self::Inconclusive { reason, .. } => Some(reason),
        }
    }
}

/// Judges a sampled block-count series.
pub fn verdict(samples: &[BlockSample], config: &KvProbeConfig) -> KvProbeVerdict {
    if samples.len() < config.min_samples() {
        return KvProbeVerdict::Inconclusive {
            reason: format!(
                "{} samples is fewer than the {} needed to judge the pool; is the engine \
                 publishing {}?",
                samples.len(),
                config.min_samples(),
                config.gauge().map_or_else(
                    || "a block-count gauge".to_string(),
                    |gauge| format!("`{gauge}`")
                )
            ),
            samples: samples.len(),
        };
    }

    let baseline = samples[0].blocks;
    let last = samples[samples.len() - 1].blocks;

    if last + config.tolerance_blocks < baseline {
        return KvProbeVerdict::Leak {
            reason: format!(
                "{} of {baseline} blocks had not returned once the run drained",
                baseline - last
            ),
            baseline_blocks: baseline,
            final_blocks: last,
            samples: samples.len(),
        };
    }

    if let Some(ceilings) = falling_ceiling(samples, config) {
        return KvProbeVerdict::Leak {
            reason: format!(
                "the block-count ceiling fell window after window across the run: {ceilings:?}"
            ),
            baseline_blocks: baseline,
            final_blocks: last,
            samples: samples.len(),
        };
    }

    KvProbeVerdict::Pass {
        baseline_blocks: baseline,
        final_blocks: last,
        samples: samples.len(),
    }
}

/// The per-window maxima, when they never rise and end more than the tolerance below where they
/// started — the shape of a pool that is being drained rather than cycled.
fn falling_ceiling(samples: &[BlockSample], config: &KvProbeConfig) -> Option<Vec<u64>> {
    let window = samples.len() / config.num_windows.max(1);
    let ceilings = samples
        .chunks(window.max(1))
        .map(|window| {
            window
                .iter()
                .map(|sample| sample.blocks)
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

/// Samples an engine's block-count gauge over the length of a run.
pub struct KvProbe {
    /// The engine's Prometheus endpoint.
    metrics_url: String,
    /// The gauge the engine publishes its block count on.
    gauge: String,
    /// How often the gauge is sampled.
    interval: Duration,
    /// The scraping client.
    client: reqwest::Client,
}

impl KvProbe {
    /// Builds a probe against an engine's metrics endpoint.
    pub fn new(metrics_url: String, gauge: String, interval: Duration) -> Self {
        Self {
            metrics_url,
            gauge,
            interval,
            client: reqwest::Client::new(),
        }
    }

    /// Takes the baseline sample, then samples on the interval until stopped.
    ///
    /// The baseline is read before this returns, so it measures the pool as it stood before the
    /// caller offered any load — the reading the drain check is judged against. Leaving it to the
    /// spawned task would race the run's first arrivals, and a baseline that already counted
    /// allocated blocks understates the pool.
    pub async fn start(self) -> RunningKvProbe {
        let started = Instant::now();
        let mut samples = Vec::new();
        self.sample_into(&mut samples, started).await;

        let (stop, stopped) = oneshot::channel();
        let task = tokio::spawn(self.sample_until_stopped(stopped, samples, started));
        RunningKvProbe { stop, task }
    }

    /// Samples on the configured interval until asked to stop, then takes the drain sample.
    async fn sample_until_stopped(
        self,
        mut stopped: oneshot::Receiver<()>,
        mut samples: Vec<BlockSample>,
        started: Instant,
    ) -> Vec<BlockSample> {
        // The first tick is skipped: `start` has already taken the baseline sample.
        let first_tick = tokio::time::Instant::now() + self.interval;
        let mut ticks = tokio::time::interval_at(first_tick, self.interval);

        loop {
            tokio::select! {
                _ = &mut stopped => break,
                _ = ticks.tick() => self.sample_into(&mut samples, started).await,
            }
        }

        self.sample_settled(&mut samples, started).await;
        samples
    }

    /// Reads past the end of the run until the pool stops growing, leaving the settled reading
    /// last for the drain check to judge.
    ///
    /// The last response reaching the harness is not the last block returning to the pool: the
    /// request retires on the engine's own thread afterwards, and the engine republishes the gauge
    /// on a tick of its own. Reading the count the instant the run ends therefore misses the
    /// blocks of the batch that just finished, and reports a leak that settles moments later.
    /// Waiting for the count to stop rising measures the pool instead of that lag, and cannot hide
    /// a real leak: blocks that never come back never make the count rise, so the wait runs out at
    /// [`SETTLE_READING_LIMIT`] with the shortfall still there for the drain check to fail on.
    async fn sample_settled(&self, samples: &mut Vec<BlockSample>, started: Instant) {
        let mut highest = 0;
        let mut steady = 0;
        for _ in 0..SETTLE_READING_LIMIT {
            let taken = samples.len();
            self.sample_into(samples, started).await;
            match samples.last() {
                Some(sample) if samples.len() > taken && sample.blocks > highest => {
                    highest = sample.blocks;
                    steady = 0;
                }
                Some(_) if samples.len() > taken => {
                    steady += 1;
                    if steady == SETTLED_READINGS {
                        return;
                    }
                }
                Some(_) | None => {}
            }
            tokio::time::sleep(SETTLE_INTERVAL).await;
        }
    }

    /// Takes one sample, keeping it only if the gauge could be read.
    async fn sample_into(&self, samples: &mut Vec<BlockSample>, started: Instant) {
        match self.read_gauge().await {
            Some(blocks) => samples.push(BlockSample {
                at: started.elapsed(),
                blocks,
            }),
            None => warn!(
                metric = self.gauge,
                url = self.metrics_url,
                "Could not read the block-count gauge; skipping this sample"
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

        parse_gauge(&exposition, &self.gauge).map(|value| value.max(0.0) as u64)
    }
}

/// A probe that is sampling.
pub struct RunningKvProbe {
    /// Tells the sampling task to take its last sample and finish.
    stop: oneshot::Sender<()>,
    /// The sampling task.
    task: JoinHandle<Vec<BlockSample>>,
}

impl RunningKvProbe {
    /// Stops sampling and returns the series, ending with the sample taken after the run drained.
    pub async fn finish(self) -> Vec<BlockSample> {
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

    /// Builds a sample series one second apart from a block-count series.
    fn samples(blocks: &[u64]) -> Vec<BlockSample> {
        blocks
            .iter()
            .enumerate()
            .map(|(index, blocks)| BlockSample {
                at: Duration::from_secs(index as u64),
                blocks: *blocks,
            })
            .collect()
    }

    /// A busy engine's block count oscillates: it falls as a batch is admitted and comes back
    /// as the batch retires. Only the ceiling falling is a leak.
    fn sawtooth(cycles: usize, ceiling: u64, floor: u64) -> Vec<u64> {
        (0..cycles)
            .flat_map(|_| [ceiling, floor, floor + 5, ceiling])
            .collect()
    }

    /// The probe only runs against a named gauge, and a name of nothing but whitespace names
    /// nothing — an operator who leaves the key blank must not get a leak-free verdict on a pool
    /// nobody sampled.
    #[test]
    fn test_a_blank_gauge_name_names_no_gauge() {
        let named = |metric: Option<&str>| KvProbeConfig {
            metric: metric.map(str::to_string),
            ..KvProbeConfig::default()
        };

        assert_eq!(
            named(Some("atoma_engine_available_blocks")).gauge(),
            Some("atoma_engine_available_blocks")
        );
        assert_eq!(
            named(Some("  atoma_engine_available_blocks  ")).gauge(),
            Some("atoma_engine_available_blocks")
        );
        assert_eq!(named(Some("   ")).gauge(), None);
        assert_eq!(named(Some("")).gauge(), None);
        assert_eq!(named(None).gauge(), None);
    }

    #[test]
    fn test_a_gauge_is_read_from_the_prometheus_text_format() {
        let text = "\
# HELP atoma_engine_available_blocks Blocks the KV pool can hand out
# TYPE atoma_engine_available_blocks gauge
atoma_engine_available_blocks 118
llm_service_requests_total 42
";

        assert_eq!(
            parse_gauge(text, "atoma_engine_available_blocks"),
            Some(118.0),
            "the gauge line must be read, not the HELP or TYPE lines"
        );
    }

    #[test]
    fn test_a_labelled_gauge_is_read() {
        let text = "atoma_engine_available_blocks{device=\"gpu\"} 96.0\n";

        assert_eq!(
            parse_gauge(text, "atoma_engine_available_blocks"),
            Some(96.0)
        );
    }

    /// `atoma_engine_available_blocks_total` is a different series; a prefix match would silently
    /// probe the wrong one.
    #[test]
    fn test_a_metric_whose_name_is_a_prefix_is_not_mistaken_for_the_gauge() {
        let text = "atoma_engine_available_blocks_total 7\n";

        assert_eq!(parse_gauge(text, "atoma_engine_available_blocks"), None);
    }

    #[test]
    fn test_a_missing_or_unreadable_gauge_reads_as_absent() {
        assert_eq!(parse_gauge("", "atoma_engine_available_blocks"), None);
        assert_eq!(
            parse_gauge(
                "atoma_engine_available_blocks NaN-ish\n",
                "atoma_engine_available_blocks"
            ),
            None
        );
        assert_eq!(
            parse_gauge(
                "atoma_engine_available_blocks\n",
                "atoma_engine_available_blocks"
            ),
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
                baseline_blocks,
                final_blocks,
                ..
            } => {
                assert_eq!(baseline_blocks, 128);
                assert_eq!(final_blocks, 120);
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
