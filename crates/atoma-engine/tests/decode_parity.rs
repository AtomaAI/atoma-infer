//! Decode parity and capture cleanliness on a device.
//!
//! Builds the decode step over runtime tensors beside the candle forward on the same weights and
//! KV cache, records the step under capture to show the driver accepts it, then runs decode steps
//! of varying ids, lengths and block tables through both forwards and compares their logits: the
//! argmax of every live row must agree, and the largest absolute difference is reported against a
//! bound.
//!
//! Needs a device, the CUDA toolkit and a Llama checkpoint loadable in bf16; run through
//! `scripts/decode-parity.sh`. Under NCCL the decode step stays on candle and there is nothing
//! to compare.

#![cfg(all(feature = "cuda", not(feature = "nccl")))]
// The evidence block is this test's product; it goes to stdout on purpose.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::env;

use atoma_core::attention::{CaptureContract, ModelDeclaration};
use atoma_core::dispatch::{
    BucketLadder, DispatchConfig, DispatchDecision, Dispatcher, EagerReason, LiveBatch,
};
use atoma_core::request::{SamplingParams, PADDING_TOKEN};
use atoma_core::step::{CommandEntry, StepCommand};
use atoma_core::types::{
    BlockId, RequestCount, RequestId, RequestSlot, SequenceIndex, StepId, TokenCount,
};
use atoma_engine::batch::BatchLayout;
use atoma_engine::config::{DeviceOrdinal, Dtype, ModelConfig};
use atoma_engine::decode::batch::Checked;
use atoma_engine::decode::declaration;
use atoma_engine::device::decode::{DecodeStep, DecodeStepPlan};
use atoma_engine::device::forward::{Allocated, CudaForward};
use atoma_engine::device::{Checkpoint, KvCache, KvGeometry, RankDevice, Weights};
use atoma_engine::forward::Forward;
use atoma_engine::model::{fetch, llama_config};
use atoma_engine::readback::Readback;
use atoma_runtime::arena::BucketIdx;
use atoma_runtime::context::RuntimeContext;
use atoma_runtime::session::{Allocation, BakedBuffers, Replay};

const DEFAULT_MODEL: &str = "NousResearch/Meta-Llama-3.1-8B-Instruct";
const BLOCK_SIZE: usize = 16;
const BLOCK_COUNT: usize = 512;
const MAX_MODEL_LEN: usize = 512;
const SEQUENCES: usize = 4;
const STEPS: usize = 32;
const LADDER: [usize; 4] = [1, 2, 4, 8];
const MAX_BATCH: usize = 8;
/// The largest absolute difference on the f32 logits accepted unless `ATOMA_PARITY_MAX_ABS_DIFF`
/// says otherwise; the measured value is printed either way.
const DEFAULT_MAX_ABS_DIFF: f32 = 0.25;

fn tokens(value: usize) -> TokenCount {
    TokenCount::new(value).expect("nonzero")
}

fn requests(value: usize) -> RequestCount {
    RequestCount::new(value).expect("nonzero")
}

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

    fn below(&mut self, bound: usize) -> usize {
        usize::try_from(self.next()).expect("fits") % bound
    }
}

/// One sequence under test: its tokens so far, the blocks it owns, and how many tokens the
/// cache holds.
struct Sequence {
    tokens: Vec<u32>,
    blocks: Vec<u32>,
    context_len: usize,
}

impl Sequence {
    fn entry(&self, index: usize, input: Vec<u32>) -> CommandEntry {
        CommandEntry {
            request: RequestId::new(index as u64 + 1),
            slot: RequestSlot::new(u32::try_from(index).expect("fits")),
            sequence: SequenceIndex::new(0),
            context_len: self.context_len,
            input_tokens: input,
            block_table: self
                .blocks
                .iter()
                .map(|&block| BlockId::new(block))
                .collect(),
            sampling: Some(SamplingParams::default()),
        }
    }

    /// The token this sequence decodes next: the first one the cache does not hold.
    fn next_token(&self) -> u32 {
        self.tokens[self.context_len]
    }
}

fn dummy(index: usize, block: u32) -> CommandEntry {
    CommandEntry {
        request: RequestId::new(1000 + index as u64),
        slot: RequestSlot::new(u32::try_from(1000 + index).expect("fits")),
        sequence: SequenceIndex::new(0),
        context_len: 0,
        input_tokens: vec![PADDING_TOKEN],
        block_table: vec![BlockId::new(block)],
        sampling: None,
    }
}

fn dispatch_config() -> DispatchConfig {
    DispatchConfig {
        bucket_ladder: BucketLadder::new(LADDER.to_vec()).expect("nonzero buckets"),
        captured_max_requests: requests(MAX_BATCH),
    }
}

fn eager() -> DispatchDecision {
    DispatchDecision::Eager(EagerReason::NotUniformDecode {
        token_count: tokens(1),
        request_count: requests(1),
    })
}

fn lay_out(command: &StepCommand) -> BatchLayout {
    BatchLayout::lay_out(command, tokens(BLOCK_SIZE)).expect("the command lays out")
}

/// A decode step over the `live` sequences: the keyed command the engine would issue, padded to
/// its bucket with dummies over `dummy_block`, and the same command marked eager.
fn decode_commands(
    live: &[(usize, &Sequence)],
    dispatcher: &mut Dispatcher,
    dummy_block: u32,
    step: u64,
) -> (StepCommand, StepCommand) {
    let dispatch = dispatcher.dispatch(LiveBatch {
        token_count: tokens(live.len()),
        request_count: requests(live.len()),
        uniform_decode: true,
    });
    let DispatchDecision::FullReplay(key) = dispatch else {
        panic!("a uniform decode of {} is keyed: {dispatch:?}", live.len());
    };
    let padding_count = key.padded_token_count().get() - live.len();
    let mut entries: Vec<CommandEntry> = live
        .iter()
        .map(|&(index, sequence)| sequence.entry(index, vec![sequence.next_token()]))
        .collect();
    entries.extend((0..padding_count).map(|index| dummy(index, dummy_block)));
    let keyed = StepCommand {
        step: StepId::new(step),
        entries: entries.clone(),
        padding_count,
        dispatch,
    };
    let on_candle = StepCommand {
        step: StepId::new(step),
        entries,
        padding_count,
        dispatch: eager(),
    };
    (keyed, on_candle)
}

/// Everything the Allocation phase produced for the device under test.
struct Rig {
    allocation: Allocation,
    allocated: Allocated,
    decode_step: DecodeStep,
    vocab: usize,
}

/// Opens device zero, loads `model` in bf16 and builds both forwards over it.
fn open(model: &ModelConfig) -> Rig {
    let files = fetch(model).expect("the checkpoint fetches");
    let config = llama_config(&files.config).expect("the config reads");
    let context = RuntimeContext::new(0).expect("device 0 opens");
    let allocation = Allocation::new(&context).expect("the session opens");
    let device = RankDevice::open(&allocation, DeviceOrdinal::new(0)).expect("candle opens");
    let checkpoint = Checkpoint {
        files: &files,
        config: &config,
        dtype: model.dtype.into(),
    };
    let weights = Weights::load(&allocation, &device, checkpoint).expect("the weights load");
    let geometry =
        KvGeometry::new(&config, BLOCK_COUNT, tokens(BLOCK_SIZE), 1).expect("the geometry");
    let kv_cache = KvCache::allocate(&allocation, &device, &config, geometry, model.dtype.into())
        .expect("the cache allocates");
    let readback = Readback::new(
        &allocation,
        device.stream().context(),
        MAX_BATCH,
        config.vocab_size,
    )
    .expect("the readback pins");
    let plan = DecodeStepPlan {
        dispatch: dispatch_config(),
        max_model_len: tokens(MAX_MODEL_LEN),
        block_size: tokens(BLOCK_SIZE),
        dtype: model.dtype,
    };
    let decode_step = DecodeStep::build(&allocation, &device, &weights, &kv_cache, &plan)
        .expect("the decode step builds");
    Rig {
        allocation,
        allocated: Allocated {
            device,
            weights,
            kv_cache,
            readback: Some(readback),
            vocab: config.vocab_size,
        },
        decode_step,
        vocab: config.vocab_size,
    }
}

/// Capture cleanliness: the bucket-of-one step, over a staged one-token batch, warms up and
/// records without the driver invalidating it. Returns the Replay phase and the graph's node
/// count.
fn record_bucket_of_one(
    allocation: Allocation,
    decode_step: &mut DecodeStep,
    dispatcher: &mut Dispatcher,
    dummy_block: u32,
) -> (Replay, usize) {
    let first = Sequence {
        tokens: vec![1],
        blocks: vec![0],
        context_len: 0,
    };
    let (keyed, _) = decode_commands(&[(0, &first)], dispatcher, dummy_block, 1);
    let layout = lay_out(&keyed);
    let DispatchDecision::FullReplay(key) = layout.dispatch else {
        panic!("keyed");
    };
    let Checked::Step(batch) = decode_step.check(&layout, key).expect("checks") else {
        panic!("the bucket of one serves one decode");
    };
    decode_step.stage(&layout, &batch).expect("stages");
    let mut capture = allocation.into_capture();
    capture
        .warm_up(&mut decode_step.upload(&batch))
        .expect("the upload runs eagerly");
    capture
        .warm_up(&mut decode_step.descriptor(BucketIdx(0)).expect("bucket 0"))
        .expect("the step warms up");
    let graph = capture
        .record(
            &mut decode_step.descriptor(BucketIdx(0)).expect("bucket 0"),
            BakedBuffers::default(),
        )
        .expect("the step records without invalidating the capture");
    let nodes = capture
        .entry(graph)
        .graph()
        .node_count()
        .expect("the graph reports its nodes");
    (capture.into_replay(), nodes)
}

/// Prefills every sequence through candle over its own blocks, and appends the token each
/// prefill sampled.
fn prefill(forward: &mut CudaForward, sequences: &mut [Sequence]) {
    for (index, sequence) in sequences.iter_mut().enumerate() {
        let command = StepCommand {
            step: StepId::new(100 + index as u64),
            entries: vec![sequence.entry(index, sequence.tokens.clone())],
            padding_count: 0,
            dispatch: eager(),
        };
        let logits = forward
            .forward(&lay_out(&command))
            .expect("the prefill runs on candle");
        let next = argmax(logits.row(0).expect("one row"));
        sequence.context_len = sequence.tokens.len();
        sequence.tokens.push(next);
    }
}

/// What the comparison of every decode step found.
#[derive(Default)]
struct Parity {
    rows: usize,
    argmax_disagreements: usize,
    max_abs_diff: f32,
}

/// Runs one decode step over `chosen` through both forwards and compares them row by row, then
/// advances each chosen sequence by the token candle sampled.
fn compare_step(
    forward: &mut CudaForward,
    sequences: &mut [Sequence],
    chosen: &[usize],
    dispatcher: &mut Dispatcher,
    parity: &mut Parity,
    step: usize,
) {
    let dummy_block = u32::try_from(BLOCK_COUNT - 1).expect("fits");
    let live: Vec<(usize, &Sequence)> = chosen
        .iter()
        .map(|&index| (index, &sequences[index]))
        .collect();
    let (keyed, on_candle) = decode_commands(&live, dispatcher, dummy_block, 200 + step as u64);
    let tensor_logits: Vec<Vec<f32>> = {
        let logits = forward
            .forward(&lay_out(&keyed))
            .expect("the keyed batch runs on the decode step");
        (0..logits.rows())
            .map(|row| logits.row(row).expect("row").to_vec())
            .collect()
    };
    let candle_logits = forward
        .forward(&lay_out(&on_candle))
        .expect("the eager step runs on candle");
    assert_eq!(tensor_logits.len(), chosen.len(), "one row per live entry");
    assert_eq!(candle_logits.rows(), chosen.len());
    let mut sampled = Vec::with_capacity(chosen.len());
    for (row, tensor_row) in tensor_logits.iter().enumerate() {
        let candle_row = candle_logits.row(row).expect("row");
        let (ours, theirs) = (argmax(tensor_row), argmax(candle_row));
        if ours != theirs {
            parity.argmax_disagreements += 1;
            eprintln!("step {step} row {row}: decode step argmax {ours}, candle argmax {theirs}");
        }
        let diff = tensor_row
            .iter()
            .zip(candle_row)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        parity.max_abs_diff = parity.max_abs_diff.max(diff);
        parity.rows += 1;
        sampled.push(theirs);
    }
    for (&index, next) in chosen.iter().zip(sampled) {
        let sequence = &mut sequences[index];
        sequence.context_len += 1;
        sequence.tokens.push(next);
    }
}

#[test]
#[ignore = "needs a device, the CUDA toolkit and a Llama checkpoint; run scripts/decode-parity.sh"]
fn the_two_forwards_agree_on_every_decode_and_the_step_records_under_capture() {
    let model = ModelConfig {
        id: env::var("ATOMA_PARITY_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_owned()),
        revision: "main".to_owned(),
        cache_dir: None,
        dtype: Dtype::Bf16,
    };
    let bound: f32 = env::var("ATOMA_PARITY_MAX_ABS_DIFF")
        .ok()
        .and_then(|bound| bound.parse().ok())
        .unwrap_or(DEFAULT_MAX_ABS_DIFF);
    let Rig {
        allocation,
        allocated,
        mut decode_step,
        vocab,
    } = open(&model);
    let contract = CaptureContract::resolve(&[declaration()], &ModelDeclaration::new("llama"));
    let mut dispatcher = Dispatcher::new(&dispatch_config(), &contract);
    let dummy_block = u32::try_from(BLOCK_COUNT - 1).expect("fits");

    let (session, nodes) =
        record_bucket_of_one(allocation, &mut decode_step, &mut dispatcher, dummy_block);
    println!("capture: the bucket-of-one step recorded as a graph of {nodes} nodes");
    let mut forward = CudaForward::new(allocated, decode_step, session);

    let mut random = Lcg(0x5EED_2026_0903);
    let blocks_each = MAX_MODEL_LEN.div_ceil(BLOCK_SIZE);
    let mut sequences: Vec<Sequence> = (0..SEQUENCES)
        .map(|index| Sequence {
            tokens: (0..8 + random.below(40))
                .map(|_| u32::try_from(random.below(vocab.min(120_000))).expect("fits"))
                .collect(),
            blocks: (0..blocks_each)
                .map(|block| u32::try_from(index * blocks_each + block).expect("fits"))
                .collect(),
            context_len: 0,
        })
        .collect();
    prefill(&mut forward, &mut sequences);

    let mut parity = Parity::default();
    for step in 0..STEPS {
        let mut chosen: Vec<usize> = (0..SEQUENCES).collect();
        for index in (1..SEQUENCES).rev() {
            chosen.swap(index, random.below(index + 1));
        }
        chosen.truncate(1 + random.below(SEQUENCES));
        chosen.sort_unstable();
        compare_step(
            &mut forward,
            &mut sequences,
            &chosen,
            &mut dispatcher,
            &mut parity,
            step,
        );
    }

    println!("=============== decode parity evidence ===============");
    println!("model:                {}", model.id);
    println!("decode steps:         {STEPS}");
    println!("rows compared:        {}", parity.rows);
    println!("argmax disagreements: {}", parity.argmax_disagreements);
    println!("max |logit diff|:     {:.6}", parity.max_abs_diff);
    println!("bound:                {bound}");
    println!("capture graph nodes:  {nodes}");
    assert_eq!(
        parity.argmax_disagreements, 0,
        "every live row's argmax agrees"
    );
    assert!(
        parity.max_abs_diff <= bound,
        "the largest logit difference {} is above the bound {bound}",
        parity.max_abs_diff
    );
}

fn argmax(row: &[f32]) -> u32 {
    let (index, _) = row.iter().enumerate().fold(
        (0, f32::NEG_INFINITY),
        |(best, best_value), (index, &value)| {
            if value > best_value {
                (index, value)
            } else {
                (best, best_value)
            }
        },
    );
    u32::try_from(index).expect("fits")
}
