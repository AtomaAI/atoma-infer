//! Sampling each entry's next token on the host from the logits the forward selected.
//!
//! One state per request slot outlives the step: the random number generator the request's seed
//! opened, and the tokens sampled for the request so far, which its repetition penalty ranges
//! over. The state is replaced when a slot changes hands — the request id in the entry is not the
//! one the state was built for. A greedy entry takes the largest logit straight off its row; the
//! rest go through candle's `LogitsProcessor` for temperature, top-k and top-p. A request's
//! `typical_p` and `frequency_penalty` are carried but not applied here.

use std::borrow::Cow;
use std::collections::HashSet;

use atoma_core::request::SamplingParams;
use atoma_core::step::StepCommand;
use atoma_core::types::{RequestId, RequestSlot};
use candle_core::{Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use thiserror::Error;

use crate::logits::Logits;

/// Why a step's tokens could not be sampled.
#[derive(Debug, Error)]
pub enum SampleError {
    /// The rows handed over do not pair up with the command's sampling entries.
    #[error("{expected} entries sample this step but {got} logits rows were selected for them")]
    RowsMismatch { expected: usize, got: usize },
    /// A selected row lies past the logits the forward produced.
    #[error("row {row} was selected but only {rows} logits rows came back")]
    RowOutOfRange { row: usize, rows: usize },
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

    /// Samples one token per sampling entry of `command`, in entry order. `rows[i]` is the row of
    /// `logits` the command's `i`-th sampling entry reads.
    ///
    /// # Errors
    ///
    /// Returns [`SampleError`] when `rows` does not pair up with the sampling entries, a row lies
    /// past the logits, or candle cannot draw from a row.
    pub fn sample(
        &mut self,
        command: &StepCommand,
        rows: &[usize],
        logits: Logits<'_>,
    ) -> Result<Vec<u32>, SampleError> {
        let expected = command.sampling_count();
        if rows.len() != expected {
            return Err(SampleError::RowsMismatch {
                expected,
                got: rows.len(),
            });
        }
        let sampling = command
            .entries
            .iter()
            .filter_map(|entry| entry.sampling.map(|params| (entry, params)));
        let mut sampled = Vec::with_capacity(expected);
        for ((entry, params), &row) in sampling.zip(rows) {
            let row_logits = logits.row(row).ok_or(SampleError::RowOutOfRange {
                row,
                rows: logits.rows(),
            })?;
            let state = self.state_for(entry.slot, entry.request, &params);
            let token = state.sample(row_logits, &params)?;
            state.generated.push(token);
            sampled.push(token);
        }
        Ok(sampled)
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
    /// Every token sampled for the request so far, in order.
    generated: Vec<u32>,
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
        Self {
            request,
            choice,
            generated: Vec::new(),
        }
    }

    /// One token off `row`, penalised over the last `repeat_last_n` tokens generated.
    fn sample(&mut self, row: &[f32], params: &SamplingParams) -> Result<u32, candle_core::Error> {
        let window = penalty_window(&self.generated, params);
        let row = if window.is_empty() || !penalises(params) {
            Cow::Borrowed(row)
        } else {
            let mut penalised = row.to_vec();
            apply_repeat_penalty(&mut penalised, params.repetition_penalty, window);
            Cow::Owned(penalised)
        };
        match &mut self.choice {
            Choice::Greedy => Ok(argmax(&row)),
            Choice::Drawn(processor) => {
                processor.sample(&Tensor::from_slice(&row, row.len(), &Device::Cpu)?)
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

fn penalises(params: &SamplingParams) -> bool {
    params.repetition_penalty.to_bits() != 1.0_f32.to_bits()
}

/// The last `repeat_last_n` tokens generated; none when that is zero.
fn penalty_window<'a>(generated: &'a [u32], params: &SamplingParams) -> &'a [u32] {
    let window = params.repeat_last_n as usize;
    &generated[generated.len().saturating_sub(window)..]
}

/// Divides a positive logit by `penalty` and multiplies a negative one by it, once per distinct
/// token in `context`: candle's formula, applied in place on the host copy.
fn apply_repeat_penalty(logits: &mut [f32], penalty: f32, context: &[u32]) {
    let mut seen = HashSet::with_capacity(context.len());
    for &token in context {
        if !seen.insert(token) {
            continue;
        }
        if let Some(logit) = logits.get_mut(token as usize) {
            if *logit >= 0.0 {
                *logit /= penalty;
            } else {
                *logit *= penalty;
            }
        }
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
    use atoma_core::step::StepCommand;

    use super::{SampleError, Sampler};
    use crate::logits::Logits;
    use crate::test_support::{command, entry, sampling_entry};

    const VOCAB: usize = 4;
    /// Token 1 leads, token 3 is second; penalising 1 by two hands the lead to 3.
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

    fn penalised(repeat_last_n: u32) -> SamplingParams {
        SamplingParams {
            repetition_penalty: 2.0,
            repeat_last_n,
            ..SamplingParams::default()
        }
    }

    /// One sampling entry for `request` in slot `slot`, under `params`.
    fn one(request: u64, slot: u32, params: SamplingParams) -> StepCommand {
        command(vec![sampling_entry(request, slot, params)], 0)
    }

    #[test]
    fn greedy_entries_take_the_largest_logit_off_their_row_in_entry_order() {
        let mut sampler = Sampler::new();
        let command = command(
            vec![
                entry(1, 3, vec![9], &[10], true),
                entry(2, 0, vec![1, 2, 3], &[20], false),
                entry(3, 5, vec![9], &[30], true),
            ],
            0,
        );
        let data = logits(&[[0.0, 0.0, 9.0, 0.0], [5.0, 0.0, 0.0, 0.0]]);
        let sampled = sampler
            .sample(&command, &[1, 0], Logits::new(&data, VOCAB))
            .unwrap();
        assert_eq!(
            sampled,
            [0, 2],
            "request 1 reads row 1, request 3 reads row 0"
        );
    }

    #[test]
    fn a_temperature_of_zero_is_greedy_whatever_else_is_asked() {
        let params = SamplingParams {
            do_sample: true,
            temperature: 0.0,
            top_k: 2,
            top_p: 0.5,
            seed: 3,
            ..SamplingParams::default()
        };
        let mut sampler = Sampler::new();
        let data = logits(&[ROW]);
        for _ in 0..8 {
            let sampled = sampler
                .sample(&one(1, 0, params), &[0], Logits::new(&data, VOCAB))
                .unwrap();
            assert_eq!(sampled, [1]);
        }
    }

    #[test]
    fn the_same_seed_draws_the_same_tokens_step_after_step() {
        let data = logits(&[ROW]);
        let draw = |seed: u64| -> Vec<u32> {
            let mut sampler = Sampler::new();
            (0..16)
                .map(|_| {
                    sampler
                        .sample(&one(1, 0, drawn(seed)), &[0], Logits::new(&data, VOCAB))
                        .unwrap()[0]
                })
                .collect()
        };
        assert_eq!(draw(7), draw(7));
        assert!(
            draw(7).iter().all(|&token| (token as usize) < VOCAB),
            "every draw is a vocabulary index"
        );
    }

    #[test]
    fn the_penalty_ranges_over_generated_tokens_only_never_the_prompt() {
        let mut sampler = Sampler::new();
        let data = logits(&[ROW]);
        // The prompt holds token 1, which leads: the first draw is still 1, unpenalised.
        let first = sampler
            .sample(&one(1, 0, penalised(8)), &[0], Logits::new(&data, VOCAB))
            .unwrap();
        assert_eq!(first, [1], "the prompt's token is not penalised");
        let second = sampler
            .sample(&one(1, 0, penalised(8)), &[0], Logits::new(&data, VOCAB))
            .unwrap();
        assert_eq!(second, [3], "the generated 1 is halved, and 3 leads");
        let third = sampler
            .sample(&one(1, 0, penalised(8)), &[0], Logits::new(&data, VOCAB))
            .unwrap();
        assert_eq!(third, [1], "both are halved and 1 leads again");
    }

    #[test]
    fn the_penalty_window_is_the_last_repeat_last_n_tokens_and_zero_is_none() {
        let mut sampler = Sampler::new();
        let data = logits(&[ROW]);
        let mut draw = |params: SamplingParams| {
            sampler
                .sample(&one(1, 0, params), &[0], Logits::new(&data, VOCAB))
                .unwrap()[0]
        };
        assert_eq!(draw(penalised(1)), 1);
        assert_eq!(draw(penalised(1)), 3, "1 is in the window");
        assert_eq!(draw(penalised(1)), 1, "only 3 is in the window now");
        assert_eq!(draw(penalised(0)), 1, "no window, no penalty");
        assert_eq!(draw(penalised(0)), 1);
    }

    #[test]
    fn a_slot_changing_hands_starts_the_new_request_afresh() {
        let mut sampler = Sampler::new();
        let data = logits(&[ROW]);
        let mut draw = |request: u64| {
            sampler
                .sample(
                    &one(request, 0, penalised(8)),
                    &[0],
                    Logits::new(&data, VOCAB),
                )
                .unwrap()[0]
        };
        assert_eq!(draw(1), 1);
        assert_eq!(draw(1), 3, "request 1 is penalised on what it generated");
        assert_eq!(
            draw(2),
            1,
            "request 2 took over the slot and owes nothing to request 1"
        );
        assert_eq!(draw(2), 3);
    }

    #[test]
    fn a_slot_changing_hands_reseeds_the_draw() {
        let data = logits(&[ROW]);
        let mut fresh = Sampler::new();
        let expected: Vec<u32> = (0..8)
            .map(|_| {
                fresh
                    .sample(&one(2, 0, drawn(5)), &[0], Logits::new(&data, VOCAB))
                    .unwrap()[0]
            })
            .collect();

        let mut sampler = Sampler::new();
        for _ in 0..8 {
            sampler
                .sample(&one(1, 0, drawn(5)), &[0], Logits::new(&data, VOCAB))
                .unwrap();
        }
        let after_handover: Vec<u32> = (0..8)
            .map(|_| {
                sampler
                    .sample(&one(2, 0, drawn(5)), &[0], Logits::new(&data, VOCAB))
                    .unwrap()[0]
            })
            .collect();
        assert_eq!(
            after_handover, expected,
            "the seed was reopened for request 2"
        );
    }

    #[test]
    fn rows_that_do_not_pair_up_with_the_entries_are_refused() {
        let mut sampler = Sampler::new();
        let data = logits(&[ROW]);
        let command = one(1, 0, SamplingParams::default());
        assert!(matches!(
            sampler
                .sample(&command, &[0, 1], Logits::new(&data, VOCAB))
                .unwrap_err(),
            SampleError::RowsMismatch {
                expected: 1,
                got: 2
            }
        ));
        assert!(matches!(
            sampler
                .sample(&command, &[1], Logits::new(&data, VOCAB))
                .unwrap_err(),
            SampleError::RowOutOfRange { row: 1, rows: 1 }
        ));
    }
}
