//! Sampling each selected row's next token on the host from the logits the forward read back.
//!
//! One state per request slot outlives the step: the random number generator the request's seed
//! opened. The state is replaced when a slot changes hands — the request id in the row is not
//! the one the state was built for. A greedy row takes the largest logit straight off its row;
//! the rest go through candle's `LogitsProcessor` for temperature, top-k and top-p.

use atoma_core::request::SamplingParams;
use atoma_core::types::{RequestId, RequestSlot};
use candle_core::{Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use thiserror::Error;

use crate::batch::RowSampling;
use crate::logits::Logits;

/// Why a step's tokens could not be sampled.
#[derive(Debug, Error)]
pub enum SampleError {
    /// The logits rows read back do not pair up with the rows the layout selected.
    #[error("{expected} rows were selected this step but {got} logits rows came back")]
    RowsMismatch { expected: usize, got: usize },
    #[error(transparent)]
    Candle(#[from] candle_core::Error),
}

/// The per-slot sampling state of every request the executor has sampled for.
#[derive(Default)]
pub struct Sampler {
    /// Indexed by request slot; a slot that never sampled holds nothing.
    slots: Vec<Option<SlotState>>,
}

impl Sampler {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Samples one token per selected row into `sampled`, in batch order: row `i` of `logits`
    /// under `rows[i]`.
    ///
    /// # Errors
    ///
    /// Returns [`SampleError`] when the logits are not one row per selected row, or candle
    /// cannot draw from a row.
    pub fn sample(
        &mut self,
        rows: &[RowSampling],
        logits: Logits<'_>,
        sampled: &mut Vec<u32>,
    ) -> Result<(), SampleError> {
        if logits.rows() != rows.len() {
            return Err(SampleError::RowsMismatch {
                expected: rows.len(),
                got: logits.rows(),
            });
        }
        sampled.clear();
        for (row, sampling) in rows.iter().enumerate() {
            let Some(row_logits) = logits.row(row) else {
                unreachable!("the logits hold one row per selected row")
            };
            let state = self.state_for(sampling.slot, sampling.request, &sampling.params);
            sampled.push(state.sample(row_logits)?);
        }
        Ok(())
    }

    /// The slot's state for `request`: what it holds when that is who it was built for, and a
    /// fresh one otherwise.
    fn state_for(
        &mut self,
        slot: RequestSlot,
        request: RequestId,
        params: &SamplingParams,
    ) -> &mut SlotState {
        let index = slot.get() as usize;
        if index >= self.slots.len() {
            self.slots.resize_with(index + 1, || None);
        }
        let state = &mut self.slots[index];
        if state.as_ref().is_none_or(|held| held.request != request) {
            *state = Some(SlotState::new(request, params));
        }
        state.as_mut().expect("just set")
    }
}

/// What one slot keeps between steps for the request it holds.
struct SlotState {
    request: RequestId,
    choice: Choice,
}

/// How a request's next token is chosen.
enum Choice {
    /// The largest logit, off the row itself.
    Greedy,
    /// Drawn by candle under the request's seed.
    Drawn(Box<LogitsProcessor>),
}

impl SlotState {
    fn new(request: RequestId, params: &SamplingParams) -> Self {
        let choice = match strategy(params) {
            Sampling::ArgMax => Choice::Greedy,
            drawn => Choice::Drawn(Box::new(LogitsProcessor::from_sampling(params.seed, drawn))),
        };
        Self { request, choice }
    }

    /// One token off `row`.
    fn sample(&mut self, row: &[f32]) -> Result<u32, candle_core::Error> {
        match &mut self.choice {
            Choice::Greedy => Ok(argmax(row)),
            Choice::Drawn(processor) => {
                processor.sample(&Tensor::from_slice(row, row.len(), &Device::Cpu)?)
            }
        }
    }
}

/// Greedy when the request opts out of sampling or asks for a temperature of zero, which is how
/// an API client asks for determinism and the only reading of a temperature the sampler would
/// otherwise divide by. A temperature of one is the default and means the model's own
/// distribution; `top_k` zero and `top_p` one are the unset values.
fn strategy(params: &SamplingParams) -> Sampling {
    if !params.do_sample || params.temperature <= 0.0 {
        return Sampling::ArgMax;
    }
    let temperature = f64::from(params.temperature);
    match (params.top_k, params.top_p) {
        (0, top_p) if top_p >= 1.0 => Sampling::All { temperature },
        (0, top_p) => Sampling::TopP {
            p: f64::from(top_p),
            temperature,
        },
        (top_k, top_p) if top_p >= 1.0 => Sampling::TopK {
            k: top_k as usize,
            temperature,
        },
        (top_k, top_p) => Sampling::TopKThenTopP {
            k: top_k as usize,
            p: f64::from(top_p),
            temperature,
        },
    }
}

/// The first largest logit's index. A not-a-number logit is never the largest.
fn argmax(row: &[f32]) -> u32 {
    let mut best = 0;
    for (index, &logit) in row.iter().enumerate() {
        if logit > row[best] {
            best = index;
        }
    }
    u32::try_from(best).expect("a vocabulary index fits u32")
}

#[cfg(test)]
mod tests {
    use atoma_core::request::SamplingParams;
    use atoma_core::types::{RequestId, RequestSlot};

    use super::{SampleError, Sampler};
    use crate::batch::RowSampling;
    use crate::logits::Logits;

    const VOCAB: usize = 4;
    /// Token 1 leads.
    const ROW: [f32; VOCAB] = [0.1, 2.0, 0.3, 1.5];

    fn logits(rows: &[[f32; VOCAB]]) -> Vec<f32> {
        rows.iter().flatten().copied().collect()
    }

    fn drawn(seed: u64) -> SamplingParams {
        SamplingParams {
            do_sample: true,
            temperature: 1.0,
            seed,
            ..SamplingParams::default()
        }
    }

    /// The sampling of one row for `request` in slot `slot`, under `params`.
    fn row(request: u64, slot: u32, params: SamplingParams) -> RowSampling {
        RowSampling {
            request: RequestId::new(request),
            slot: RequestSlot::new(slot),
            params,
        }
    }

    /// Samples one row of `data` for `request` in slot `slot`.
    fn one(sampler: &mut Sampler, request: u64, slot: u32, params: SamplingParams) -> u32 {
        let data = logits(&[ROW]);
        let mut sampled = Vec::new();
        sampler
            .sample(
                &[row(request, slot, params)],
                Logits::new(&data, VOCAB),
                &mut sampled,
            )
            .unwrap();
        sampled[0]
    }

    #[test]
    fn greedy_rows_take_the_largest_logit_off_their_row_in_batch_order() {
        let mut sampler = Sampler::new();
        let rows = [
            row(1, 3, SamplingParams::default()),
            row(3, 5, SamplingParams::default()),
        ];
        let data = logits(&[[0.0, 0.0, 9.0, 0.0], [5.0, 0.0, 0.0, 0.0]]);
        let mut sampled = vec![7, 7, 7];
        sampler
            .sample(&rows, Logits::new(&data, VOCAB), &mut sampled)
            .unwrap();
        assert_eq!(sampled, [2, 0], "one token per row, and nothing left over");
    }

    #[test]
    fn a_temperature_of_zero_is_greedy_whatever_else_is_asked() {
        let params = SamplingParams {
            do_sample: true,
            temperature: 0.0,
            top_k: 2,
            top_p: 0.5,
            seed: 3,
        };
        let mut sampler = Sampler::new();
        for _ in 0..8 {
            assert_eq!(one(&mut sampler, 1, 0, params), 1);
        }
    }

    #[test]
    fn the_same_seed_draws_the_same_tokens_step_after_step() {
        let draw = |seed: u64| -> Vec<u32> {
            let mut sampler = Sampler::new();
            (0..16)
                .map(|_| one(&mut sampler, 1, 0, drawn(seed)))
                .collect()
        };
        assert_eq!(draw(7), draw(7));
        assert!(
            draw(7).iter().all(|&token| (token as usize) < VOCAB),
            "every draw is a vocabulary index"
        );
    }

    #[test]
    fn a_slot_changing_hands_reseeds_the_draw() {
        let mut fresh = Sampler::new();
        let expected: Vec<u32> = (0..8).map(|_| one(&mut fresh, 2, 0, drawn(5))).collect();

        let mut sampler = Sampler::new();
        for _ in 0..8 {
            one(&mut sampler, 1, 0, drawn(5));
        }
        let after_handover: Vec<u32> = (0..8).map(|_| one(&mut sampler, 2, 0, drawn(5))).collect();
        assert_eq!(
            after_handover, expected,
            "the seed was reopened for request 2"
        );
    }

    #[test]
    fn logits_that_are_not_one_row_per_selected_row_are_refused() {
        let mut sampler = Sampler::new();
        let data = logits(&[ROW]);
        let rows = [
            row(1, 0, SamplingParams::default()),
            row(2, 1, SamplingParams::default()),
        ];
        let mut sampled = Vec::new();
        assert!(matches!(
            sampler
                .sample(&rows, Logits::new(&data, VOCAB), &mut sampled)
                .unwrap_err(),
            SampleError::RowsMismatch {
                expected: 2,
                got: 1
            }
        ));
    }
}
