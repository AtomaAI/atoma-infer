//! Request arrival process.
//!
//! Load is offered as an open-loop Poisson process at a target rate: the next request is sent when
//! its arrival time comes, whether or not the previous one has answered. A closed-loop harness (a
//! fixed number of workers looping) cannot show queueing delay, because it stops offering load
//! exactly when the engine slows down — the metric it produces is the engine's own pace, not its
//! goodput at a rate.

use std::time::Duration;

use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::error::{BenchError, Result};

/// Draws the arrival offsets of a request stream from a Poisson process.
///
/// Inter-arrival times of a Poisson process with rate λ are exponentially distributed, drawn here
/// by inverse transform: `-ln(1 - u) / λ` for `u` uniform on [0, 1).
pub struct PoissonArrivals {
    /// Target arrival rate λ, in requests per second.
    rate_per_second: f64,
    /// Seeded so a run's arrival pattern is reproducible.
    rng: StdRng,
}

impl PoissonArrivals {
    /// Creates an arrival process at `rate_per_second`, drawing from `seed`.
    ///
    /// # Errors
    ///
    /// Returns [`BenchError::Config`] if the rate is not a positive, finite number.
    pub fn new(rate_per_second: f64, seed: u64) -> Result<Self> {
        if !rate_per_second.is_finite() || rate_per_second <= 0.0 {
            return Err(BenchError::Config(format!(
                "Request rate must be a positive, finite number of requests per second, got \
                 {rate_per_second}"
            )));
        }
        Ok(Self {
            rate_per_second,
            rng: StdRng::seed_from_u64(seed),
        })
    }

    /// Draws `count` arrival offsets, measured from the start of the run and non-decreasing.
    pub fn offsets(&mut self, count: usize) -> Vec<Duration> {
        let mut elapsed = 0.0_f64;
        (0..count)
            .map(|_| {
                elapsed += self.next_interval();
                Duration::from_secs_f64(elapsed)
            })
            .collect()
    }

    /// Draws one inter-arrival time, in seconds.
    fn next_interval(&mut self) -> f64 {
        let uniform: f64 = self.rng.random();
        -(1.0 - uniform).ln() / self.rate_per_second
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mean inter-arrival time of a Poisson process is 1/λ; over many draws the sample mean has to
    /// land near it, or the process is not Poisson at the requested rate.
    #[test]
    fn test_mean_inter_arrival_time_matches_the_requested_rate() {
        const RATE: f64 = 20.0;
        const COUNT: usize = 20_000;

        let offsets = PoissonArrivals::new(RATE, 7)
            .expect("Failed to build the arrival process")
            .offsets(COUNT);

        let mean = offsets.last().expect("No arrivals drawn").as_secs_f64() / COUNT as f64;
        assert!(
            (mean - 1.0 / RATE).abs() < 0.1 / RATE,
            "mean inter-arrival time {mean} is not within 10% of {}",
            1.0 / RATE
        );
    }

    #[test]
    fn test_arrivals_are_non_decreasing() {
        let offsets = PoissonArrivals::new(50.0, 1)
            .expect("Failed to build the arrival process")
            .offsets(1_000);

        assert!(
            offsets.windows(2).all(|pair| pair[0] <= pair[1]),
            "arrival offsets must be non-decreasing"
        );
    }

    /// Two runs at the same seed must offer the same load, or run-to-run comparisons measure the
    /// workload rather than the engine.
    #[test]
    fn test_the_same_seed_draws_the_same_arrivals() {
        let first = PoissonArrivals::new(10.0, 99)
            .expect("Failed to build the arrival process")
            .offsets(64);
        let second = PoissonArrivals::new(10.0, 99)
            .expect("Failed to build the arrival process")
            .offsets(64);
        let other_seed = PoissonArrivals::new(10.0, 100)
            .expect("Failed to build the arrival process")
            .offsets(64);

        assert_eq!(first, second);
        assert_ne!(first, other_seed);
    }

    #[test]
    fn test_a_non_positive_rate_is_rejected() {
        for rate in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(
                PoissonArrivals::new(rate, 0).is_err(),
                "rate {rate} should be rejected"
            );
        }
    }

    #[test]
    fn test_no_arrivals_are_drawn_for_an_empty_stream() {
        let offsets = PoissonArrivals::new(10.0, 0)
            .expect("Failed to build the arrival process")
            .offsets(0);

        assert!(offsets.is_empty());
    }
}
