//! Philox4x32-10, the counter-based generator behind every draw the sampler makes.
//!
//! A counter-based generator has no state to carry between draws: the n-th draw under a seed is
//! the block cipher applied to the counter n under the key the seed opens. That is what makes a
//! seeded request reproducible regardless of the batch it was sampled in or the slot it occupied,
//! and what lets the device keep one draw counter per slot instead of a generator state. The
//! kernel computes the same function, and the known-answer vectors here are Random123's, so the
//! host and the device are held to one definition.

/// The Philox4x32 multipliers and Weyl-sequence key increments, as Random123 defines them.
const MULTIPLIER_0: u32 = 0xD251_1F53;
const MULTIPLIER_1: u32 = 0xCD9E_8D57;
const WEYL_0: u32 = 0x9E37_79B9;
const WEYL_1: u32 = 0xBB67_AE85;
const ROUNDS: usize = 10;

/// The Philox4x32-10 block: four output words for `counter` under `key`.
#[must_use]
pub fn philox4x32(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let mut counter = counter;
    let mut key = key;
    for round in 0..ROUNDS {
        counter = one_round(counter, key);
        if round + 1 < ROUNDS {
            key = [key[0].wrapping_add(WEYL_0), key[1].wrapping_add(WEYL_1)];
        }
    }
    counter
}

fn one_round(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let product_0 = u64::from(MULTIPLIER_0) * u64::from(counter[0]);
    let product_1 = u64::from(MULTIPLIER_1) * u64::from(counter[2]);
    let (high_0, low_0) = split(product_0);
    let (high_1, low_1) = split(product_1);
    [
        high_1 ^ counter[1] ^ key[0],
        low_1,
        high_0 ^ counter[3] ^ key[1],
        low_0,
    ]
}

/// The high and low words of a 64-bit product.
fn split(product: u64) -> (u32, u32) {
    // Both halves of a u64 fit u32 by construction.
    #[allow(clippy::cast_possible_truncation)]
    ((product >> 32) as u32, product as u32)
}

/// The 64-bit uniform for draw number `draw` under `seed`: the key is the seed's two words, the
/// counter is the draw index alone, and the uniform is the block's first two words, the first
/// low. The kernel derives its draws the same way.
#[must_use]
pub fn draw(seed: u64, draw: u32) -> u64 {
    // The seed's two halves fit u32 by construction.
    #[allow(clippy::cast_possible_truncation)]
    let key = [seed as u32, (seed >> 32) as u32];
    let words = philox4x32([draw, 0, 0, 0], key);
    u64::from(words[0]) | (u64::from(words[1]) << 32)
}

#[cfg(test)]
mod tests {
    use super::{draw, philox4x32};

    /// Random123's `kat_vectors` for philox4x32 at ten rounds.
    #[test]
    fn the_block_matches_the_published_known_answers() {
        assert_eq!(
            philox4x32([0, 0, 0, 0], [0, 0]),
            [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]
        );
        assert_eq!(
            philox4x32([u32::MAX; 4], [u32::MAX; 2]),
            [0x408f_276d, 0x41c8_3b0e, 0xa20b_c7c6, 0x6d54_51fd]
        );
        assert_eq!(
            philox4x32(
                [0x243f_6a88, 0x85a3_08d3, 0x1319_8a2e, 0x0370_7344],
                [0xa409_3822, 0x299f_31d0]
            ),
            [0xd16c_fe09, 0x94fd_cceb, 0x5001_e420, 0x2412_6ea1]
        );
    }

    #[test]
    fn a_draw_is_the_blocks_first_two_words_under_the_seed_and_the_draw_index() {
        assert_eq!(draw(0, 0), 0xe169_c58d_6627_e8d5);
        assert_eq!(draw(7, 0), 0xc009_f9dc_f460_7a2d);
        assert_eq!(draw(7, 1), 0xcb97_bc13_682e_8e9b);
        assert_eq!(draw(0x0123_4567_89ab_cdef, 42), 0x6195_85ba_36e4_badc);
    }

    #[test]
    fn draws_differ_across_seeds_and_indices_and_repeat_exactly() {
        let under_seven: Vec<u64> = (0..64).map(|index| draw(7, index)).collect();
        let again: Vec<u64> = (0..64).map(|index| draw(7, index)).collect();
        assert_eq!(under_seven, again, "a draw is a pure function");
        let under_eight: Vec<u64> = (0..64).map(|index| draw(8, index)).collect();
        assert!(
            under_seven.iter().zip(&under_eight).all(|(a, b)| a != b),
            "another seed is another sequence"
        );
        let mut sorted = under_seven.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), under_seven.len(), "no two draws collide");
    }
}
