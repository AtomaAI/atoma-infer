//! Bit-identity comparison: the spike's acceptance criterion is byte equality between replayed
//! and eager outputs, with a bf16-aware description of the first divergence when they differ.

/// The first differing element between two bf16 buffers, with both values decoded.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct Bf16Divergence {
    pub element_index: usize,
    pub replay_bits: u16,
    pub eager_bits: u16,
    pub replay_value: f32,
    pub eager_value: f32,
}

/// Decodes a bf16 bit pattern; bf16 is the upper half of an f32.
pub fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Compares two bf16 buffers byte-for-byte, returning the first divergence.
///
/// # Panics
/// When the buffers differ in length or have an odd byte count — a harness bug, not a
/// comparison result.
pub fn first_bf16_divergence(replay: &[u8], eager: &[u8]) -> Option<Bf16Divergence> {
    assert_eq!(
        replay.len(),
        eager.len(),
        "comparison buffers differ in length; the harness copied mismatched outputs"
    );
    assert_eq!(replay.len() % 2, 0, "bf16 buffer has an odd byte count");
    let byte_index = replay.iter().zip(eager).position(|(a, b)| a != b)?;
    let element_index = byte_index / 2;
    let at = element_index * 2;
    let replay_bits = u16::from_le_bytes([replay[at], replay[at + 1]]);
    let eager_bits = u16::from_le_bytes([eager[at], eager[at + 1]]);
    Some(Bf16Divergence {
        element_index,
        replay_bits,
        eager_bits,
        replay_value: bf16_bits_to_f32(replay_bits),
        eager_value: bf16_bits_to_f32(eager_bits),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_buffers_have_no_divergence() {
        let buf = [0x00, 0x3F, 0x80, 0xBF];
        assert_eq!(first_bf16_divergence(&buf, &buf), None);
    }

    #[test]
    fn the_first_differing_element_is_reported_with_decoded_values() {
        // Element 0 equal; element 1 differs: 0x3F80 is 1.0, 0xBF80 is -1.0.
        let replay = [0x11, 0x22, 0x80, 0x3F];
        let eager = [0x11, 0x22, 0x80, 0xBF];
        let divergence = first_bf16_divergence(&replay, &eager).unwrap();
        assert_eq!(divergence.element_index, 1);
        assert_eq!(divergence.replay_value, 1.0);
        assert_eq!(divergence.eager_value, -1.0);
    }

    #[test]
    fn a_divergence_in_the_low_byte_maps_to_the_right_element() {
        let replay = [0x00, 0x00, 0x01, 0x00];
        let eager = [0x00, 0x00, 0x02, 0x00];
        assert_eq!(
            first_bf16_divergence(&replay, &eager)
                .unwrap()
                .element_index,
            1
        );
    }

    #[test]
    #[should_panic(expected = "differ in length")]
    fn mismatched_lengths_are_a_harness_bug() {
        first_bf16_divergence(&[0x00], &[0x00, 0x01]);
    }
}
