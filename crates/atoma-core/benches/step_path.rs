//! The step-path latency gate: schedule, command build and result apply, measured on the host
//! with no GPU, at a representative batch.
//!
//! Sixty-four requests decode at a 1024-token context while eight prompts wait in the admission
//! window under longest prefix match. Each measured pass applies the previous step's result,
//! schedules the next step, builds its command and pushes it to the ring; the mock executor
//! answers between passes, untimed. The gate fails when the p99 pass reaches 200 microseconds.

use std::process::ExitCode;
use std::time::Instant;

use atoma_core::dispatch::{BucketLadder, CaptureKind, DispatchConfig, SupportLevel};
use atoma_core::engine::{Engine, EngineConfig, ExecutorRings, Pass};
use atoma_core::kv::HashAlgorithm;
use atoma_core::request::{egress, EgressReceiver, NewRequest, SamplingParams, StopCriteria};
use atoma_core::scheduler::{AdmissionPolicy, SchedulerConfig};
use atoma_core::step::StepResult;
use atoma_core::types::{RequestCount, TokenCount};
use hdrhistogram::Histogram;
use tracing::{error, info};

const BATCH: usize = 64;
const CONTEXT_TOKENS: usize = 1024;
const WAITING: usize = 8;
const BLOCK_SIZE: usize = 16;
const WARMUP_PASSES: usize = 200;
const MEASURED_PASSES: usize = 2000;
const P99_LIMIT_MICROS: u64 = 200;

fn count(value: usize) -> TokenCount {
    TokenCount::new(value).expect("bench counts are nonzero")
}

fn requests(value: usize) -> RequestCount {
    RequestCount::new(value).expect("bench counts are nonzero")
}

fn config() -> EngineConfig {
    let decode_headroom = MEASURED_PASSES + WARMUP_PASSES + BLOCK_SIZE;
    EngineConfig {
        scheduler: SchedulerConfig {
            token_budget: count(BATCH * CONTEXT_TOKENS),
            max_batch: requests(BATCH),
            max_model_len: count(CONTEXT_TOKENS + decode_headroom),
            block_size: count(BLOCK_SIZE),
            window: requests(WAITING),
            admission: AdmissionPolicy::LongestPrefixMatch,
            max_requests: requests(BATCH + WAITING),
            eos_token_ids: Vec::new(),
            hash_algorithm: HashAlgorithm::Sha256V1,
        },
        dispatch: DispatchConfig {
            bucket_ladder: BucketLadder::new(vec![1, 2, 4, 8, 16, 32, 64]).expect("nonzero"),
            captured_max_requests: requests(BATCH),
            support_level: SupportLevel::Always,
            capture_kind: CaptureKind::Full,
        },
        block_count: u32::try_from(
            (BATCH + WAITING) * (CONTEXT_TOKENS + decode_headroom) / BLOCK_SIZE + BATCH,
        )
        .expect("fits u32"),
        ingress_capacity: requests(BATCH + WAITING),
        idle_deadline_millis: 1,
    }
}

/// A prompt of `len` tokens unique to request `index`, so nothing hits the prefix cache.
fn request(index: u32, len: usize) -> (NewRequest, EgressReceiver) {
    let (sender, receiver) = egress();
    let base = index * 1_000_000;
    let request = NewRequest {
        prompt: (0..u32::try_from(len).expect("fits u32"))
            .map(|offset| base + offset)
            .collect(),
        sampling: SamplingParams::default(),
        stop: StopCriteria {
            max_new_tokens: count(1_000_000),
            ignore_eos: true,
        },
        egress: sender,
    };
    (request, receiver)
}

/// Answers the step in flight, if one is, with one token per sampling entry.
fn answer(rings: &mut ExecutorRings) {
    if let Some(command) = rings.pop_command() {
        let sampled = vec![7; command.sampling_count()];
        rings
            .push_result(StepResult {
                step: command.step,
                sampled,
            })
            .expect("one step in flight");
    }
}

fn main() -> ExitCode {
    tracing_subscriber::fmt().with_target(false).init();
    let (mut engine, handle, mut rings) =
        Engine::new(&config()).expect("the bench configuration is valid");
    let mut clients = Vec::with_capacity(BATCH + WAITING);
    for index in 0..BATCH {
        let (request, receiver) = request(u32::try_from(index).expect("fits"), CONTEXT_TOKENS);
        handle.ingress.try_send(request).expect("ingress has room");
        clients.push(receiver);
    }
    // The first pass takes every prompt in and prefills them all under the wide budget.
    assert_eq!(engine.pass(), Pass::Continue);
    answer(&mut rings);
    for index in BATCH..BATCH + WAITING {
        let (request, receiver) = request(u32::try_from(index).expect("fits"), CONTEXT_TOKENS);
        handle.ingress.try_send(request).expect("ingress has room");
        clients.push(receiver);
    }

    let mut histogram: Histogram<u64> =
        Histogram::new_with_bounds(1, 10_000_000, 3).expect("valid histogram bounds");
    for pass in 0..WARMUP_PASSES + MEASURED_PASSES {
        let started = Instant::now();
        assert_eq!(engine.pass(), Pass::Continue);
        let elapsed = started.elapsed();
        answer(&mut rings);
        for client in &clients {
            while client.try_recv().is_ok() {}
        }
        if pass >= WARMUP_PASSES {
            histogram
                .record(u64::try_from(elapsed.as_micros()).expect("fits u64"))
                .expect("within bounds");
        }
    }
    let state = engine.state();
    assert_eq!(state.running, BATCH, "every decode is still running");
    assert_eq!(state.waiting, WAITING, "the window is still full");

    let p99 = histogram.value_at_quantile(0.99);
    info!(
        batch = BATCH,
        context_tokens = CONTEXT_TOKENS,
        waiting = WAITING,
        passes = MEASURED_PASSES,
        p50_us = histogram.value_at_quantile(0.5),
        p90_us = histogram.value_at_quantile(0.9),
        p99_us = p99,
        max_us = histogram.max(),
        "step path: schedule, command build and result apply"
    );
    if p99 >= P99_LIMIT_MICROS {
        error!(
            p99_us = p99,
            limit_us = P99_LIMIT_MICROS,
            "step-path p99 is over the limit"
        );
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
