//! The device sampler against its host reference, on a device.
//!
//! Needs a device and the CUDA toolkit; no checkpoint and no model. The sampler is built over
//! synthetic logits uploaded to a buffer of its own, so what is measured is the kernel and
//! nothing else: greedy rows must match the reference exactly, drawn rows must match it token for
//! token wherever the two agree on the admitted set, and a seeded request must produce the same
//! tokens whatever batch it is sampled in and whatever slot it occupies.
//!
//! Run through `scripts/sampler-parity.sh`.

#![cfg(feature = "cuda")]
// The evidence block is this test's product; it goes to stdout on purpose.
#![allow(clippy::print_stdout)]

use std::slice;
use std::sync::Arc;

use atoma_core::dispatch::{DispatchDecision, EagerReason};
use atoma_core::request::SamplingParams;
use atoma_core::step::{CommandEntry, StepCommand};
use atoma_core::types::RequestCount;
use atoma_core::types::{BlockId, RequestId, RequestSlot, SequenceIndex, StepId, TokenCount};
use atoma_engine::batch::BatchLayout;
use atoma_engine::device::sampler::DeviceSampler;
use atoma_engine::sampling::record::SlotRecord;
use atoma_engine::sampling::reference;
use atoma_runtime::context::RuntimeContext;
use atoma_runtime::session::Allocation;
use cudarc::driver::{CudaSlice, CudaStream};

/// Small enough to upload per step, wide enough for the kernel's block-wide reductions to span
/// several strides.
const VOCAB: usize = 4096;
const SLOTS: usize = 16;
const MAX_ROWS: usize = 8;
const BLOCK_SIZE: TokenCount = TokenCount::new(16).expect("nonzero");
/// Draws per distribution when a frequency is measured.
const DRAWS: usize = 4096;

/// A small deterministic generator, so a run is reproducible from its seed alone.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    /// A logit in [-8, 8).
    fn logit(&mut self) -> f32 {
        let thousandths = u16::try_from(self.next() % 16_000).expect("below sixteen thousand");
        f32::from(thousandths) / 1000.0 - 8.0
    }
}

/// The device, its stream and the sampler under test, with a buffer for the logits.
struct Rig {
    sampler: DeviceSampler,
    stream: Arc<CudaStream>,
    logits: CudaSlice<f32>,
    _allocation: Allocation,
}

impl Rig {
    fn open() -> Self {
        let context = RuntimeContext::new(0).expect("device 0 opens");
        let allocation = Allocation::new(&context).expect("the session opens");
        let stream = context.cuda().default_stream();
        let sampler = DeviceSampler::new(&allocation, &stream, SLOTS, MAX_ROWS, VOCAB)
            .expect("the sampler allocates");
        let logits = stream
            .alloc_zeros::<f32>(MAX_ROWS * VOCAB)
            .expect("the logits allocate");
        Self {
            sampler,
            stream,
            logits,
            _allocation: allocation,
        }
    }

    /// Samples `rows` under `layout` over the uploaded `logits`, and returns the tokens.
    fn sample(&mut self, layout: &BatchLayout, rows: &[Vec<f32>]) -> Vec<u32> {
        let mut flat: Vec<f32> = Vec::with_capacity(rows.len() * VOCAB);
        for row in rows {
            flat.extend_from_slice(row);
        }
        self.stream
            .memcpy_htod(&flat, &mut self.logits)
            .expect("the logits upload");
        self.sampler.stage(layout, None).expect("the layout stages");
        let view = self.logits.slice(0..rows.len() * VOCAB);
        self.sampler
            .run_on(&self.stream, &view)
            .expect("the sampler runs")
            .to_vec()
    }
}

/// One decode entry for `request` in `slot`, sampling under `params`.
fn entry(request: u64, slot: u32, params: SamplingParams) -> CommandEntry {
    CommandEntry {
        request: RequestId::new(request),
        slot: RequestSlot::new(slot),
        sequence: SequenceIndex::new(0),
        context_len: 0,
        input_tokens: vec![1],
        block_table: vec![BlockId::new(0)],
        sampling: Some(params),
    }
}

/// The layout of a batch of `entries`, run eagerly so no bucket is implied.
fn layout(entries: Vec<CommandEntry>) -> BatchLayout {
    let command = StepCommand {
        step: StepId::new(1),
        entries,
        padding_count: 0,
        dispatch: DispatchDecision::Eager(EagerReason::NotUniformDecode {
            token_count: TokenCount::new(1).expect("nonzero"),
            request_count: RequestCount::new(1).expect("nonzero"),
        }),
    };
    BatchLayout::lay_out(&command, BLOCK_SIZE).expect("the command lays out")
}

fn drawn(temperature: f32, top_k: u32, top_p: f32, seed: u64) -> SamplingParams {
    SamplingParams {
        temperature,
        top_k,
        top_p,
        do_sample: true,
        seed,
    }
}

/// A row of random logits.
fn random_row(random: &mut Lcg) -> Vec<f32> {
    (0..VOCAB).map(|_| random.logit()).collect()
}

/// `count` out of `total`, as a fraction.
fn ratio(count: usize, total: usize) -> f32 {
    let count = u16::try_from(count).expect("a draw count fits u16");
    let total = u16::try_from(total).expect("a draw count fits u16");
    f32::from(count) / f32::from(total)
}

/// What the reference samples for the `draw`-th draw of `params` over `row`.
fn expected(row: &[f32], params: &SamplingParams, draw: u32) -> u32 {
    let mut record = SlotRecord::new(params);
    record.draws = draw;
    reference::sample(row, &record)
}

#[test]
#[ignore = "needs a device and the CUDA toolkit; run scripts/sampler-parity.sh"]
fn the_device_sampler_matches_its_host_reference() {
    let mut rig = Rig::open();
    let mut random = Lcg(0x5EED_2026_0904);
    let mut greedy_rows = 0;
    let mut drawn_rows = 0;
    let mut disagreements = 0;

    // Greedy rows, exactly: the largest logit, ties to the first index.
    for step in 0..16 {
        let rows: Vec<Vec<f32>> = (0..4).map(|_| random_row(&mut random)).collect();
        let entries = (0..4u32)
            .map(|index| entry(100 + u64::from(index), index, SamplingParams::default()))
            .collect();
        let sampled = rig.sample(&layout(entries), &rows);
        for (row, token) in rows.iter().zip(&sampled) {
            let want = expected(row, &SamplingParams::default(), 0);
            greedy_rows += 1;
            if *token != want {
                disagreements += 1;
                println!("greedy step {step}: sampled {token}, the reference {want}");
            }
        }
    }
    assert_eq!(disagreements, 0, "greedy sampling matches the reference");

    // A row with tied maxima: the first index wins on both sides.
    let mut tied = vec![0.0f32; VOCAB];
    tied[7] = 5.0;
    tied[9] = 5.0;
    let sampled = rig.sample(
        &layout(vec![entry(1, 0, SamplingParams::default())]),
        &[tied.clone()],
    );
    assert_eq!(sampled, [7], "the first of two equal maxima");

    // Drawn rows, against the reference draw for draw, under several filters.
    for (temperature, top_k, top_p) in [
        (1.0, 0, 1.0),
        (0.7, 40, 1.0),
        (1.0, 0, 0.9),
        (0.5, 64, 0.95),
        (1.3, 8, 0.5),
    ] {
        let row = random_row(&mut random);
        let params = drawn(temperature, top_k, top_p, 0x5EED);
        for draw in 0..32u32 {
            // A new request each time, so its slot is claimed afresh and the seed under test
            // reaches the record; its counter is then zero, which is the draw compared.
            let slot = draw % u32::try_from(SLOTS).expect("the slot count fits u32");
            let mut params = params;
            params.seed = 0x5EED + u64::from(draw);
            let sampled = rig.sample(
                &layout(vec![entry(u64::from(draw) + 1, slot, params)]),
                slice::from_ref(&row),
            );
            let want = expected(&row, &params, 0);
            drawn_rows += 1;
            if sampled[0] != want {
                disagreements += 1;
                println!(
                    "drawn (T {temperature}, k {top_k}, p {top_p}) draw {draw}: sampled {}, the \
                     reference {want}",
                    sampled[0]
                );
            }
        }
    }

    println!("=============== sampler parity evidence ===============");
    println!("vocabulary:           {VOCAB}");
    println!("greedy rows:          {greedy_rows}");
    println!("drawn rows:           {drawn_rows}");
    println!("disagreements:        {disagreements}");
    assert_eq!(
        disagreements, 0,
        "every row matches the host reference token for token"
    );
}

#[test]
#[ignore = "needs a device and the CUDA toolkit; run scripts/sampler-parity.sh"]
fn a_seeded_request_draws_the_same_tokens_in_any_batch_and_any_slot() {
    let mut rig = Rig::open();
    let mut random = Lcg(0x11ED_2026_0904);
    let row = random_row(&mut random);
    let params = drawn(0.8, 50, 0.95, 4242);
    let steps = 24;

    // Alone, in slot zero.
    let alone: Vec<u32> = (0..steps)
        .map(|_| rig.sample(&layout(vec![entry(1, 0, params)]), slice::from_ref(&row))[0])
        .collect();

    // The same request in a different slot, sharing every batch with other requests whose rows
    // are different and whose count changes step to step.
    let slot = 11;
    let mut together = Vec::with_capacity(steps);
    for step in 0..steps {
        let others = u32::try_from(step % 3).expect("below three");
        let mut entries = vec![entry(1, slot, params)];
        let mut rows = vec![row.clone()];
        for other in 0..others {
            entries.push(entry(
                200 + u64::from(other),
                other + 1,
                drawn(1.0, 0, 1.0, 7 + u64::from(other)),
            ));
            rows.push(random_row(&mut random));
        }
        let sampled = rig.sample(&layout(entries), &rows);
        together.push(sampled[0]);
    }

    println!("=============== seeded reproducibility ===============");
    println!("steps:                {steps}");
    println!("alone, slot 0:        {alone:?}");
    println!("in company, slot {slot}:  {together:?}");
    assert_eq!(
        alone, together,
        "a seeded request's tokens do not depend on its batch or its slot"
    );
}

#[test]
#[ignore = "needs a device and the CUDA toolkit; run scripts/sampler-parity.sh"]
fn drawn_tokens_follow_the_distribution_the_filters_leave() {
    let mut rig = Rig::open();
    // Four tokens carry all the mass; the rest are far below and top_p drops them.
    let mut row = vec![-30.0f32; VOCAB];
    let heavy = [11usize, 222, 3333, 4000];
    let probabilities = [0.5f32, 0.3, 0.15, 0.05];
    for (&token, &probability) in heavy.iter().zip(&probabilities) {
        row[token] = probability.ln();
    }
    let params = drawn(1.0, 0, 0.999, 9090);

    // One request in one slot throughout, so its record is written once and the kernel advances
    // its draw counter: what is measured is one seeded request's stream, which is what a client
    // gets.
    let mut counts = [0usize; 4];
    for draw in 0..DRAWS {
        let token = rig.sample(&layout(vec![entry(1, 0, params)]), slice::from_ref(&row))[0];
        let index = heavy
            .iter()
            .position(|&heavy| heavy == token as usize)
            .unwrap_or_else(|| {
                panic!("draw {draw} sampled {token}, which top_p should have dropped")
            });
        counts[index] += 1;
    }

    println!("=============== draw frequencies ===============");
    println!("draws:                {DRAWS}");
    for (index, (&count, &probability)) in counts.iter().zip(&probabilities).enumerate() {
        let frequency = ratio(count, DRAWS);
        println!(
            "token {}: {frequency:.4} against {probability:.4}",
            heavy[index]
        );
        assert!(
            (frequency - probability).abs() < 0.03,
            "token {} drawn {frequency:.4} of the time, not {probability:.4}",
            heavy[index]
        );
    }
}
