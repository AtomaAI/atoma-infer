//! The sampling record one request slot holds on the device, written once when the slot changes
//! hands and read by the sampler every step the slot samples.

use std::mem::{align_of, offset_of, size_of};

use atoma_core::request::SamplingParams;

/// What the device sampler reads for one slot, laid out as the kernel declares it.
///
/// Greedy is a temperature of zero, and the kernel takes the largest logit for it without a draw:
/// a request that opts out of sampling or asks for a temperature of zero, which is how an API
/// client asks for determinism and the only reading of a temperature the sampler would otherwise
/// divide by. A `top_k` of zero and a `top_p` at or above one are the unset values, as in the
/// parameters. The draw counter starts at zero with the record and is advanced by the kernel
/// alone, once per draw, so where the request's draws have got to lives on the device with the
/// rest of its sampling state.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlotRecord {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    /// How many draws the slot's request has made; the counter of its next draw.
    pub draws: u32,
    pub seed: u64,
}

/// The record's size in bytes, as the kernel indexes the array of them.
pub const RECORD_BYTES: usize = 24;

impl SlotRecord {
    /// The record a request sampling under `params` starts with.
    #[must_use]
    pub fn new(params: &SamplingParams) -> Self {
        let greedy = !params.do_sample || params.temperature <= 0.0;
        if greedy {
            return Self {
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                draws: 0,
                seed: params.seed,
            };
        }
        Self {
            temperature: params.temperature,
            top_p: params.top_p,
            top_k: params.top_k,
            draws: 0,
            seed: params.seed,
        }
    }

    /// Whether the slot takes the largest logit without a draw.
    #[must_use]
    pub fn is_greedy(&self) -> bool {
        self.temperature == 0.0
    }
}

/// The layout the kernel declares, checked at compile time.
const _: () = {
    assert!(size_of::<SlotRecord>() == RECORD_BYTES);
    assert!(align_of::<SlotRecord>() == 8);
    assert!(offset_of!(SlotRecord, temperature) == 0);
    assert!(offset_of!(SlotRecord, top_p) == 4);
    assert!(offset_of!(SlotRecord, top_k) == 8);
    assert!(offset_of!(SlotRecord, draws) == 12);
    assert!(offset_of!(SlotRecord, seed) == 16);
};

#[cfg(test)]
mod tests {
    use atoma_core::request::SamplingParams;

    use super::SlotRecord;

    #[test]
    fn a_request_that_does_not_sample_is_a_greedy_record_whatever_else_it_asks() {
        let record = SlotRecord::new(&SamplingParams {
            do_sample: false,
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            seed: 5,
        });
        assert!(record.is_greedy());
        assert_eq!(
            record,
            SlotRecord {
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                draws: 0,
                seed: 5,
            }
        );
        assert!(SlotRecord::new(&SamplingParams::default()).is_greedy());
    }

    #[test]
    fn a_temperature_of_zero_is_greedy_and_a_negative_one_is_too() {
        for temperature in [0.0, -1.0] {
            let record = SlotRecord::new(&SamplingParams {
                do_sample: true,
                temperature,
                top_k: 2,
                top_p: 0.5,
                seed: 3,
            });
            assert!(record.is_greedy(), "temperature {temperature}");
            assert_eq!(record.top_k, 0);
        }
    }

    #[test]
    fn a_drawn_request_keeps_its_parameters_and_starts_at_draw_zero() {
        let record = SlotRecord::new(&SamplingParams {
            do_sample: true,
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            seed: 9,
        });
        assert!(!record.is_greedy());
        assert_eq!(
            record,
            SlotRecord {
                temperature: 0.7,
                top_p: 0.9,
                top_k: 40,
                draws: 0,
                seed: 9,
            }
        );
    }
}
