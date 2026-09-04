//! The host statement of what the sampler computes for one row: the definition the kernel is
//! held to, written for clarity rather than speed.
//!
//! Logits are ordered by a key that is monotone in their value and puts a not-a-number last,
//! so the largest is well defined and ties go to the first index. A greedy record takes the
//! largest logit and draws nothing. A drawn record admits the `top_k` largest logits, every
//! logit tied with the k-th included, then weights each admitted token by `exp((logit - max) /
//! temperature)` quantised to a 32-bit fixed-point integer, so the mass arithmetic that follows
//! is exact and the same on the host and on the device; the `top_p` cutoff keeps the smallest
//! set of heaviest tokens whose mass reaches `top_p` of the total, every token tied with the
//! last included. One 64-bit uniform then picks a token in index order by its share of the
//! admitted mass.
//!
//! The kernel computes the same steps with block-wide reductions and radix selection instead of
//! sorts. Its only sources of difference are the exponential and the division in the weights,
//! which the device rounds within an ulp or two of the host: a weight can differ by a unit of the
//! fixed point, and the total by a unit per token. The point the uniform names is its share of
//! the total, which moves by no more than the total does, so a row's cutoff or pick differs only
//! when it turns on those last units. Reducing the uniform modulo the total would instead move
//! the point by the total's difference times the quotient, which is millions.

use crate::sampling::philox;
use crate::sampling::record::SlotRecord;

/// The weight of the largest logit: one, in 32-bit fixed point.
pub const UNIT_WEIGHT: u64 = 1 << 32;

/// A logit's order among its row: monotone in the value, with a not-a-number below every number.
#[must_use]
pub fn key(logit: f32) -> u32 {
    if logit.is_nan() {
        return key(f32::NEG_INFINITY);
    }
    let bits = logit.to_bits();
    if bits & 0x8000_0000 == 0 {
        bits | 0x8000_0000
    } else {
        !bits
    }
}

/// The first largest logit's index, and its value.
#[must_use]
pub fn argmax(logits: &[f32]) -> (usize, f32) {
    let mut best = 0;
    let mut best_key = key(logits.first().copied().unwrap_or(f32::NEG_INFINITY));
    for (index, &logit) in logits.iter().enumerate().skip(1) {
        let candidate = key(logit);
        if candidate > best_key {
            best = index;
            best_key = candidate;
        }
    }
    (best, logits.get(best).copied().unwrap_or(f32::NEG_INFINITY))
}

/// The tokens the `top_k` filter admits: the `top_k` largest, and every token tied with the
/// k-th. A `top_k` of zero, or one at or past the vocabulary, admits every token.
#[must_use]
pub fn admitted_by_top_k(logits: &[f32], top_k: u32) -> Vec<bool> {
    let vocab = logits.len();
    let top_k = top_k as usize;
    if top_k == 0 || top_k >= vocab {
        return vec![true; vocab];
    }
    let mut keys: Vec<u32> = logits.iter().copied().map(key).collect();
    keys.sort_unstable_by(|a, b| b.cmp(a));
    let threshold = keys[top_k - 1];
    logits
        .iter()
        .map(|&logit| key(logit) >= threshold)
        .collect()
}

/// Each admitted token's weight in 32-bit fixed point: `exp((logit - max) / temperature)`
/// truncated to a multiple of 2^-32, so the largest weighs exactly [`UNIT_WEIGHT`]. A token not
/// admitted weighs nothing.
#[must_use]
pub fn weights(logits: &[f32], admitted: &[bool], max: f32, temperature: f32) -> Vec<u64> {
    logits
        .iter()
        .zip(admitted)
        .map(|(&logit, &admitted)| {
            if !admitted || logit.is_nan() {
                return 0;
            }
            let weight = ((logit - max) / temperature).exp();
            // Truncation is the quantisation; a weight is in [0, 1], so the product fits, and
            // the unit is a power of two the double holds exactly.
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                clippy::cast_precision_loss
            )]
            {
                (f64::from(weight) * (UNIT_WEIGHT as f64)) as u64
            }
        })
        .collect()
}

/// The mass the `top_p` cutoff must reach: `top_p` of `total`, rounded up, and at least one
/// unit of the fixed point so the heaviest token is always kept.
#[must_use]
pub fn target_mass(top_p: f32, total: u64) -> u64 {
    // `total` is below 2^53 for any vocabulary, so the double is exact; the product is rounded
    // once and its ceiling fits.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let target = (f64::from(top_p) * total as f64).ceil() as u64;
    target.max(1)
}

/// The weights the `top_p` cutoff keeps: the heaviest tokens whose mass reaches [`target_mass`],
/// and every token tied with the last included. A `top_p` at or above one keeps every weight.
#[must_use]
pub fn admitted_by_top_p(weights: &[u64], top_p: f32) -> Vec<u64> {
    if top_p >= 1.0 {
        return weights.to_vec();
    }
    let total: u64 = weights.iter().sum();
    let target = target_mass(top_p, total);
    let mut sorted: Vec<u64> = weights.iter().copied().filter(|&w| w > 0).collect();
    sorted.sort_unstable_by(|a, b| b.cmp(a));
    let mut mass = 0;
    let mut threshold = 0;
    for weight in sorted {
        mass += weight;
        threshold = weight;
        if mass >= target {
            break;
        }
    }
    weights
        .iter()
        .map(|&weight| if weight >= threshold { weight } else { 0 })
        .collect()
}

/// The token `uniform` picks in index order by its share of `weights`: the first whose cumulative
/// weight exceeds the [`point`] `uniform` names in the total.
#[must_use]
pub fn pick(weights: &[u64], uniform: u64) -> usize {
    let total: u64 = weights.iter().sum();
    let point = point(uniform, total);
    let mut cumulative = 0;
    for (index, &weight) in weights.iter().enumerate() {
        cumulative += weight;
        if cumulative > point {
            return index;
        }
    }
    unreachable!("the cumulative weight reaches the total, which is past the point")
}

/// The point `uniform` names in `total`: `uniform / 2^64` of it, below `total`. A difference in
/// `total` moves the point by no more than that difference, so a device whose weights sum a few
/// units apart from the host's picks the same token unless the point sits at a boundary.
#[must_use]
pub fn point(uniform: u64, total: u64) -> u64 {
    // The share is below `total`, so the cast back is exact.
    #[allow(clippy::cast_possible_truncation)]
    {
        ((u128::from(uniform) * u128::from(total)) >> 64) as u64
    }
}

/// One row's token under `record`: the largest logit for a greedy record, and otherwise the
/// draw numbered by the record's counter. The counter is the caller's to advance.
///
/// A row whose largest logit is not finite has no distribution to draw from and takes the
/// largest logit whatever the record says.
#[must_use]
pub fn sample(logits: &[f32], record: &SlotRecord) -> u32 {
    let (best, max) = argmax(logits);
    if record.is_greedy() || !max.is_finite() {
        return fits(best);
    }
    let admitted = admitted_by_top_k(logits, record.top_k);
    let weights = weights(logits, &admitted, max, record.temperature);
    let weights = admitted_by_top_p(&weights, record.top_p);
    fits(pick(&weights, philox::draw(record.seed, record.draws)))
}

fn fits(index: usize) -> u32 {
    u32::try_from(index).expect("a vocabulary index fits u32")
}

#[cfg(test)]
mod tests {
    // The tests quantise and count the way the reference does.
    #![allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]

    use atoma_core::request::SamplingParams;

    use super::{
        admitted_by_top_k, admitted_by_top_p, argmax, key, pick, point, sample, target_mass,
        weights, UNIT_WEIGHT,
    };
    use crate::sampling::record::SlotRecord;

    fn drawn(temperature: f32, top_k: u32, top_p: f32, seed: u64) -> SlotRecord {
        SlotRecord::new(&SamplingParams {
            temperature,
            top_k,
            top_p,
            do_sample: true,
            seed,
        })
    }

    /// Logits whose softmax is exactly `probabilities`.
    fn logits_of(probabilities: &[f32]) -> Vec<f32> {
        probabilities.iter().map(|p| p.ln()).collect()
    }

    #[test]
    fn the_key_orders_logits_by_value_with_nan_last_and_negative_zero_below_zero() {
        let ordered = [
            f32::NAN,
            f32::NEG_INFINITY,
            -3.5,
            -0.0,
            0.0,
            1.0e-30,
            2.0,
            f32::INFINITY,
        ];
        for pair in ordered.windows(2) {
            assert!(key(pair[0]) <= key(pair[1]), "{pair:?}");
        }
        assert_eq!(key(f32::NAN), key(f32::NEG_INFINITY));
        assert!(key(-0.0) < key(0.0));
    }

    #[test]
    fn argmax_is_the_first_largest_and_never_a_nan() {
        assert_eq!(argmax(&[0.5, 2.0, 2.0, 1.0]), (1, 2.0));
        assert_eq!(argmax(&[f32::NAN, 1.0, f32::NAN]), (1, 1.0));
        let (index, value) = argmax(&[f32::NAN, f32::NAN]);
        assert_eq!(index, 0);
        assert!(value.is_nan());
        assert_eq!(argmax(&[]), (0, f32::NEG_INFINITY));
    }

    #[test]
    fn top_k_admits_the_k_largest_and_everything_tied_with_the_kth() {
        assert_eq!(
            admitted_by_top_k(&[1.0, 3.0, 3.0, 0.0], 2),
            [false, true, true, false]
        );
        assert_eq!(
            admitted_by_top_k(&[3.0, 1.0, 3.0, 3.0], 2),
            [true, false, true, true],
            "the third three is tied with the second"
        );
        assert_eq!(
            admitted_by_top_k(&[1.0, 3.0, 3.0, 0.0], 0),
            [true; 4],
            "zero is unset"
        );
        assert_eq!(
            admitted_by_top_k(&[1.0, 3.0, 3.0, 0.0], 4),
            [true; 4],
            "the whole vocabulary"
        );
        assert_eq!(
            admitted_by_top_k(&[f32::NAN, 1.0, 0.0], 1),
            [false, true, false]
        );
    }

    #[test]
    fn weights_are_the_softmax_numerators_in_fixed_point_with_the_largest_at_one() {
        let logits = [0.0, -1.0, f32::NAN, 5.0];
        let admitted = [true, true, true, false];
        let warm = weights(&logits, &admitted, 0.0, 1.0);
        assert_eq!(warm[0], UNIT_WEIGHT);
        assert_eq!(
            warm[1],
            (f64::from((-1.0f32).exp()) * UNIT_WEIGHT as f64) as u64
        );
        assert_eq!(warm[2], 0, "a nan weighs nothing");
        assert_eq!(warm[3], 0, "not admitted");
        let cold = weights(&logits, &admitted, 0.0, 0.1);
        assert!(cold[1] < warm[1], "a lower temperature sharpens");
    }

    #[test]
    fn the_target_mass_is_top_p_of_the_total_rounded_up_and_at_least_one() {
        assert_eq!(target_mass(0.5, 100), 50);
        assert_eq!(target_mass(0.501, 100), 51);
        assert_eq!(target_mass(0.0, 100), 1);
        assert_eq!(target_mass(1.0, 100), 100);
    }

    #[test]
    fn top_p_keeps_the_smallest_set_of_heaviest_tokens_reaching_the_mass_and_its_ties() {
        let weights = [50, 30, 20, 30];
        assert_eq!(admitted_by_top_p(&weights, 0.38), [50, 0, 0, 0]);
        assert_eq!(
            admitted_by_top_p(&weights, 0.385),
            [50, 30, 0, 30],
            "the crossing thirty brings the tied thirty with it"
        );
        assert_eq!(
            admitted_by_top_p(&weights, 0.84),
            [50, 30, 0, 30],
            "110 of 130 reaches a target of 110"
        );
        assert_eq!(
            admitted_by_top_p(&weights, 0.85),
            [50, 30, 20, 30],
            "a target of 111 needs the twenty"
        );
        assert_eq!(admitted_by_top_p(&weights, 1.0), weights, "unset");
        assert_eq!(
            admitted_by_top_p(&[50, 0, 30], 0.0),
            [50, 0, 0],
            "at least the heaviest"
        );
    }

    #[test]
    fn the_admitted_mass_reaches_the_target_and_the_lightest_class_is_needed_to() {
        let weights: Vec<u64> = (0..64).map(|i| (i * 7919 + 13) % 97 + 1).collect();
        let total: u64 = weights.iter().sum();
        for top_p in [0.05, 0.3, 0.5, 0.73, 0.99] {
            let kept = admitted_by_top_p(&weights, top_p);
            let target = target_mass(top_p, total);
            let mass: u64 = kept.iter().sum();
            assert!(mass >= target, "top_p {top_p}: {mass} < {target}");
            let lightest = kept.iter().copied().filter(|&w| w > 0).min().unwrap();
            let without_lightest: u64 = kept.iter().filter(|&&w| w > lightest).sum();
            assert!(
                without_lightest < target,
                "top_p {top_p}: the lightest class is not needed"
            );
        }
    }

    /// The smallest uniform whose point in `total` is `at`.
    fn naming(at: u64, total: u64) -> u64 {
        let uniform = (u128::from(at) << 64).div_ceil(u128::from(total));
        u64::try_from(uniform).expect("a point below the total is named by some uniform")
    }

    #[test]
    fn the_point_is_the_uniforms_share_of_the_total_and_moves_no_more_than_the_total() {
        assert_eq!(point(0, 10), 0);
        assert_eq!(point(1 << 63, 10), 5);
        assert_eq!(point(u64::MAX, 10), 9, "below the total");
        assert_eq!(point(naming(3, 10), 10), 3);
        assert_eq!(point(naming(3, 10) - 1, 10), 2);
        let uniform = naming(700, 1000);
        assert!(point(uniform, 1003).abs_diff(700) <= 3);
        assert!(point(uniform, 997).abs_diff(700) <= 3);
    }

    #[test]
    fn the_pick_walks_the_weights_in_index_order_by_share() {
        let weights = [3, 0, 5, 2];
        let at = |point| naming(point, 10);
        assert_eq!(pick(&weights, 0), 0);
        assert_eq!(pick(&weights, at(2)), 0);
        assert_eq!(pick(&weights, at(3)), 2, "a zero weight is never picked");
        assert_eq!(pick(&weights, at(7)), 2);
        assert_eq!(pick(&weights, at(8)), 3);
        assert_eq!(pick(&weights, at(9)), 3);
        assert_eq!(
            pick(&weights, u64::MAX),
            3,
            "the largest uniform names the last unit"
        );
    }

    #[test]
    fn a_greedy_record_takes_the_largest_logit_and_so_does_a_row_with_no_finite_one() {
        let greedy = SlotRecord::new(&SamplingParams::default());
        assert_eq!(sample(&[0.1, 2.0, 0.3, 1.5], &greedy), 1);
        let drawn = drawn(1.0, 0, 1.0, 3);
        assert_eq!(
            sample(&[f32::NEG_INFINITY, f32::INFINITY, 0.0], &drawn),
            1,
            "an infinite logit is taken, not drawn"
        );
        assert_eq!(sample(&[f32::NAN, f32::NAN], &drawn), 0);
    }

    #[test]
    fn draws_follow_the_softmax_of_the_admitted_logits() {
        let logits = logits_of(&[0.5, 0.3, 0.15, 0.05]);
        let draws = 20_000;
        let mut counts = [0u32; 4];
        for draw in 0..draws {
            let mut record = drawn(1.0, 0, 1.0, 11);
            record.draws = draw;
            counts[sample(&logits, &record) as usize] += 1;
        }
        let frequencies: Vec<f32> = counts.iter().map(|&c| c as f32 / draws as f32).collect();
        for (frequency, expected) in frequencies.iter().zip([0.5, 0.3, 0.15, 0.05]) {
            assert!(
                (frequency - expected).abs() < 0.015,
                "{frequencies:?} against the softmax"
            );
        }
    }

    #[test]
    fn top_k_and_top_p_confine_the_draws_and_the_rest_is_renormalised() {
        let logits = logits_of(&[0.5, 0.3, 0.15, 0.05]);
        let draws = 20_000;
        let mut counts = [0u32; 4];
        for draw in 0..draws {
            let mut record = drawn(1.0, 3, 0.7, 5);
            record.draws = draw;
            counts[sample(&logits, &record) as usize] += 1;
        }
        assert_eq!(counts[2], 0, "top_p 0.7 keeps 0.5 and 0.3 alone");
        assert_eq!(counts[3], 0, "top_k 3 drops the smallest");
        let first = counts[0] as f32 / draws as f32;
        assert!((first - 0.625).abs() < 0.015, "renormalised: {first}");
    }

    #[test]
    fn a_seeded_draw_is_a_function_of_the_seed_and_the_counter_alone() {
        let logits = logits_of(&[0.4, 0.3, 0.2, 0.1]);
        let sequence = |seed: u64| -> Vec<u32> {
            (0..32)
                .map(|draw| {
                    let mut record = drawn(0.8, 0, 0.95, seed);
                    record.draws = draw;
                    sample(&logits, &record)
                })
                .collect()
        };
        assert_eq!(sequence(9), sequence(9));
        assert_ne!(sequence(9), sequence(10));
    }
}
