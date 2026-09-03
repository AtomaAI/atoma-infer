//! The rotary embedding's cosine and sine tables, computed on the host once.
//!
//! The kernel reads `cos[position][pair]` and `sin[position][pair]` rather than computing them,
//! because the frequency ladder a model scales is not a closed form the kernel should carry: Llama
//! 3 stretches the low-frequency end of the ladder and blends a band in the middle, and that
//! belongs in one place, on the host, evaluated once at Allocation.
//!
//! Every value is f32, one row per position up to the model's maximum, `head_dim / 2` columns.
//! The frequencies are the ones candle's rotary cache is built from. Candle rounds its tables to
//! the model's dtype and rotates in it, while these stay f32 and the kernel rotates in f32, so
//! the two rotations agree to the model's precision rather than bit for bit.

use std::f32::consts::PI;

use crate::dims::{Llama3RopeScaling, LlamaDims, RopeParams};

/// The cosine and sine tables of one model, row-major `[max_position, head_dim / 2]`.
#[derive(Debug, Clone, PartialEq)]
pub struct RotaryTables {
    cos: Vec<f32>,
    sin: Vec<f32>,
    pairs: usize,
    max_position: usize,
}

impl RotaryTables {
    /// Builds the tables for `dims`: the frequency ladder, scaled if the model scales it, times
    /// every position the model serves.
    #[must_use]
    pub fn new(dims: &LlamaDims) -> Self {
        let frequencies = inverse_frequencies(dims.head_dim, &dims.rope);
        let pairs = frequencies.len();
        let max_position = dims.rope.max_position;
        let mut cos = Vec::with_capacity(max_position * pairs);
        let mut sin = Vec::with_capacity(max_position * pairs);
        for position in 0..max_position {
            #[allow(clippy::cast_precision_loss)]
            let position = position as f32;
            for &frequency in &frequencies {
                let angle = position * frequency;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
        Self {
            cos,
            sin,
            pairs,
            max_position,
        }
    }

    /// The cosine table, row-major.
    #[must_use]
    pub fn cos(&self) -> &[f32] {
        &self.cos
    }

    /// The sine table, row-major.
    #[must_use]
    pub fn sin(&self) -> &[f32] {
        &self.sin
    }

    /// Columns per row: the rotary pairs of one head.
    #[must_use]
    pub fn pairs(&self) -> usize {
        self.pairs
    }

    /// Rows: the positions the tables cover.
    #[must_use]
    pub fn max_position(&self) -> usize {
        self.max_position
    }

    /// Values in either table.
    #[must_use]
    pub fn len(&self) -> usize {
        self.cos.len()
    }

    /// Whether the tables cover no position at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cos.is_empty()
    }
}

/// The inverse frequency of each rotary pair: `theta^(-2i / head_dim)` for pair `i`, scaled when
/// the model scales it.
fn inverse_frequencies(head_dim: usize, rope: &RopeParams) -> Vec<f32> {
    let ladder = (0..head_dim).step_by(2).map(|i| {
        #[allow(clippy::cast_precision_loss)]
        let exponent = i as f32 / head_dim as f32;
        1.0 / rope.theta.powf(exponent)
    });
    match rope.scaling {
        None => ladder.collect(),
        Some(scaling) => ladder.map(|frequency| scale(frequency, scaling)).collect(),
    }
}

/// Llama 3's scaling of one frequency: short wavelengths are left alone, long ones are divided by
/// the factor, and the band between is blended between the two.
#[allow(clippy::cast_precision_loss)]
fn scale(frequency: f32, scaling: Llama3RopeScaling) -> f32 {
    let original = scaling.original_max_position_embeddings as f32;
    let low_wavelength = original / scaling.low_freq_factor;
    let high_wavelength = original / scaling.high_freq_factor;
    let wavelength = 2.0 * PI / frequency;
    if wavelength < high_wavelength {
        return frequency;
    }
    if wavelength > low_wavelength {
        return frequency / scaling.factor;
    }
    let smooth = (original / wavelength - scaling.low_freq_factor)
        / (scaling.high_freq_factor - scaling.low_freq_factor);
    (1.0 - smooth) * frequency / scaling.factor + smooth * frequency
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dims::test_support::llama_8b;

    /// Llama 3.1's declared scaling.
    fn llama3_scaling() -> Llama3RopeScaling {
        Llama3RopeScaling {
            factor: 8.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position_embeddings: 8192,
        }
    }

    fn with_rope(rope: RopeParams) -> LlamaDims {
        LlamaDims {
            rope,
            ..llama_8b(1)
        }
    }

    #[test]
    fn the_tables_are_one_row_per_position_of_one_column_per_pair() {
        let dims = with_rope(RopeParams {
            theta: 500_000.0,
            scaling: None,
            max_position: 16,
        });
        let tables = RotaryTables::new(&dims);
        assert_eq!(tables.pairs(), 64);
        assert_eq!(tables.max_position(), 16);
        assert_eq!(tables.len(), 16 * 64);
        assert_eq!(tables.cos().len(), tables.sin().len());
        assert!(!tables.is_empty());
    }

    #[test]
    fn position_zero_is_the_identity_rotation() {
        let tables = RotaryTables::new(&with_rope(RopeParams {
            theta: 10_000.0,
            scaling: None,
            max_position: 4,
        }));
        for pair in 0..tables.pairs() {
            assert!((tables.cos()[pair] - 1.0).abs() < 1e-6);
            assert!(tables.sin()[pair].abs() < 1e-6);
        }
    }

    #[test]
    fn each_row_is_its_position_times_the_frequency_ladder() {
        let theta = 10_000.0f32;
        let tables = RotaryTables::new(&with_rope(RopeParams {
            theta,
            scaling: None,
            max_position: 8,
        }));
        let pairs = tables.pairs();
        for position in [1usize, 3, 7] {
            for pair in [0usize, 1, 31, 63] {
                #[allow(clippy::cast_precision_loss)]
                let angle = position as f32 * (1.0 / theta.powf(2.0 * pair as f32 / 128.0));
                let at = position * pairs + pair;
                assert!(
                    (tables.cos()[at] - angle.cos()).abs() < 1e-5,
                    "cos at position {position}, pair {pair}"
                );
                assert!(
                    (tables.sin()[at] - angle.sin()).abs() < 1e-5,
                    "sin at position {position}, pair {pair}"
                );
            }
        }
    }

    #[test]
    fn the_first_pair_is_the_fastest_and_the_last_the_slowest() {
        // Frequency falls with the pair index, so at position one the angle does too.
        let tables = RotaryTables::new(&with_rope(RopeParams {
            theta: 10_000.0,
            scaling: None,
            max_position: 2,
        }));
        let row = &tables.sin()[tables.pairs()..2 * tables.pairs()];
        for pair in 1..row.len() {
            assert!(
                row[pair] < row[pair - 1],
                "pair {pair} turns faster than pair {}",
                pair - 1
            );
        }
    }

    #[test]
    fn llama3_scaling_leaves_short_wavelengths_and_divides_long_ones() {
        let scaling = llama3_scaling();
        // A short wavelength: the fastest pair, well under 8192/4.
        let fast = 1.0 / 500_000f32.powf(0.0);
        assert!(
            (scale(fast, scaling) - fast).abs() < 1e-12,
            "a short wavelength is left alone"
        );
        // A long wavelength: 2 pi / frequency above 8192/1, so the frequency is divided.
        let slow = 2.0 * PI / 20_000.0;
        assert!((scale(slow, scaling) - slow / 8.0).abs() < 1e-12);
    }

    #[test]
    fn the_blended_band_lies_between_the_scaled_and_unscaled_frequencies() {
        let scaling = llama3_scaling();
        // Wavelength 4096 sits between 8192/4 and 8192/1, so this frequency is blended.
        let frequency = 2.0 * PI / 4096.0;
        let blended = scale(frequency, scaling);
        assert!(
            blended > frequency / scaling.factor && blended < frequency,
            "blended {blended} is not between {} and {frequency}",
            frequency / scaling.factor
        );
    }

    #[test]
    fn scaling_slows_the_low_frequency_end_of_the_ladder() {
        let unscaled = RotaryTables::new(&with_rope(RopeParams {
            theta: 500_000.0,
            scaling: None,
            max_position: 2,
        }));
        let scaled = RotaryTables::new(&with_rope(RopeParams {
            theta: 500_000.0,
            scaling: Some(llama3_scaling()),
            max_position: 2,
        }));
        let pairs = unscaled.pairs();
        let last = 2 * pairs - 1;
        assert!(
            scaled.sin()[last].abs() < unscaled.sin()[last].abs(),
            "the slowest pair should turn even more slowly once scaled"
        );
        assert!(
            (scaled.sin()[pairs] - unscaled.sin()[pairs]).abs() < 1e-9,
            "the fastest pair is short-wavelength and is left alone"
        );
    }
}
