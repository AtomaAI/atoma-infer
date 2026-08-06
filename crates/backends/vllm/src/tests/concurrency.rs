//! Concurrency harness.
//!
//! Drives many requests at once through the path a request really takes on the host: admission
//! (`RequestAdmitter`), the scheduler, the block manager, and the client channels
//! (`ResponseSenders`). The model is stood in for by a KV pool that is written through the block
//! tables the scheduler hands out, which is what makes a block table collision visible — a stubbed
//! model would happily produce correct-looking output while two requests overwrote each other's
//! cache.

use std::{
    collections::{HashMap, HashSet},
    time::{Duration, Instant},
};

use atoma_bench::kv_probe::{FreeBlockSample, KvProbeConfig, KvProbeVerdict};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};

use crate::{
    admission::RequestAdmitter,
    config::{CacheConfig, SchedulerConfig},
    egress::{ClientState, ResponseSenders},
    output::{GenerateStreamingOutput, StreamResponse},
    policy::FcfsPolicy,
    scheduler::Scheduler,
    sequence::{LogProb, SequenceGroup, SequenceStatus},
    tests::fixtures::{prompt_token_offset, valid_request},
    types::WriteLock,
};

const BLOCK_SIZE: usize = 4;
const GPU_MEMORY_UTILIZATION: f32 = 1.0;
const SWAP_SPACE_FRACTION: f32 = 0.4;
const MAX_NUM_BATCHED_TOKENS: usize = 4096;
const MAX_MODEL_LEN: usize = 512;

const NUM_REQUESTS: usize = 32;
const NUM_GPU_BLOCKS: usize = 128;
const PROMPT_LEN: usize = 4;
const MAX_NEW_TOKENS: u32 = 4;

/// Stands in for the KV cache: one slot per physical block, holding the request that last wrote
/// it.
struct KvPool {
    /// Owner of each physical block, by block number.
    owners: Vec<Option<String>>,
    /// Requests that have finished or been aborted, whose blocks may legitimately be reused.
    retired: HashSet<String>,
}

impl KvPool {
    fn new(num_blocks: usize) -> Self {
        Self {
            owners: vec![None; num_blocks],
            retired: HashSet::new(),
        }
    }

    /// Writes a request's KV through its block table, panicking if the blocks are also owned by
    /// another request that has not finished.
    fn write(&mut self, request_id: &str, block_table: &[u32]) {
        for block_number in block_table {
            let owner = &mut self.owners[*block_number as usize];
            if let Some(owner) = owner {
                assert!(
                    owner == request_id || self.retired.contains(owner),
                    "block {block_number} is written by two live requests: {owner} and {request_id}"
                );
            }
            *owner = Some(request_id.to_string());
        }
    }

    /// Records that a request is done, so its blocks may be handed to another request.
    fn retire(&mut self, request_id: &str) {
        self.retired.insert(request_id.to_string());
    }

    /// Block numbers currently owned by requests that have not been retired.
    fn live_blocks(&self) -> usize {
        self.owners
            .iter()
            .filter(|owner| {
                owner
                    .as_ref()
                    .is_some_and(|owner| !self.retired.contains(owner))
            })
            .count()
    }
}

/// The host side of the engine loop: admission, scheduling, and a KV pool in place of the model.
struct Harness {
    scheduler: Scheduler<FcfsPolicy>,
    request_admitter: RequestAdmitter,
    response_senders: ResponseSenders,
    kv_pool: KvPool,
    /// The admitted requests, so their final state can be inspected after they leave the queues.
    admitted: HashMap<String, SequenceGroup>,
    /// The block table each request was last scheduled with.
    block_tables: HashMap<String, Vec<u32>>,
}

impl Harness {
    fn new(num_gpu_blocks: usize, max_num_sequences: usize) -> Self {
        let scheduler_config = SchedulerConfig::new(
            MAX_NUM_BATCHED_TOKENS,
            max_num_sequences,
            MAX_MODEL_LEN,
            0.0,
            false,
        )
        .expect("Failed to generate `SchedulerConfig`");
        let cache_config = CacheConfig::new_from_blocks(
            BLOCK_SIZE,
            GPU_MEMORY_UTILIZATION,
            SWAP_SPACE_FRACTION,
            None,
            None,
            num_gpu_blocks,
            num_gpu_blocks,
        )
        .expect("Failed to generate `CacheConfig`");

        Self {
            scheduler: Scheduler::new(cache_config, scheduler_config)
                .expect("Failed to generate `Scheduler`"),
            request_admitter: RequestAdmitter::new(BLOCK_SIZE),
            response_senders: ResponseSenders::default(),
            kv_pool: KvPool::new(num_gpu_blocks),
            admitted: HashMap::new(),
            block_tables: HashMap::new(),
        }
    }

    /// Admits a request the way `LlmService` does, queues it, and registers the streaming client
    /// waiting for its output.
    fn admit(&mut self, request_id: &str) -> flume::Receiver<StreamResponse> {
        let request = valid_request(request_id, PROMPT_LEN, MAX_NEW_TOKENS);
        let sequence_group = self
            .request_admitter
            .admit(request_id.to_string(), &request, Instant::now())
            .expect("Failed to admit request");

        self.admitted
            .insert(request_id.to_string(), sequence_group.clone());
        self.scheduler.add_sequence_group(sequence_group);

        let (sender, receiver) = flume::unbounded();
        self.response_senders
            .register_stream(request_id.to_string(), sender);
        receiver
    }

    /// Runs one engine step, then retires whatever finished — the seam the KV-leak regression
    /// tests suppress.
    fn step(&mut self) {
        let disconnected = self.step_without_retiring();
        self.scheduler
            .remove_finished_sequences()
            .expect("Failed to remove finished sequences");

        for request_id in disconnected {
            self.abort(&request_id);
        }
    }

    /// Runs one engine step: schedule, write KV and sample one token per scheduled sequence, and
    /// apply the stopping criteria.
    ///
    /// Returns the requests whose clients hung up during the step, which the engine retires and
    /// this seam deliberately does not.
    fn step_without_retiring(&mut self) -> Vec<String> {
        let mut disconnected = Vec::new();
        let (sequence_groups_metadata, scheduler_outputs) =
            self.scheduler.schedule().expect("Failed to schedule");

        for (metadata, scheduled_sequence_group) in sequence_groups_metadata
            .iter()
            .zip(scheduler_outputs.scheduled_sequence_groups.iter())
        {
            scheduled_sequence_group
                .scheduled_group
                .update_num_computed_tokens(scheduled_sequence_group.token_chunk_size)
                .expect("Failed to update the number of computed tokens");

            let max_new_tokens = scheduled_sequence_group
                .scheduled_group
                .stopping_params()
                .max_new_tokens;

            for (sequence_id, sequence) in scheduled_sequence_group.scheduled_group.sequences.iter()
            {
                let block_table = metadata
                    .block_tables
                    .get(sequence_id)
                    .unwrap_or_else(|| {
                        panic!("Sequence {sequence_id} was scheduled without a block table")
                    })
                    .clone();
                self.kv_pool.write(&metadata.request_id, &block_table);
                self.block_tables
                    .insert(metadata.request_id.clone(), block_table);

                if !metadata.do_sample {
                    continue;
                }

                let mut sequence = sequence.write_lock().expect("Sequence lock is poisoned");
                let token_id = generated_token_id(&metadata.request_id);
                sequence
                    .add_token_id(
                        token_id,
                        HashMap::from_iter([(token_id, LogProb::new(0.5, None, None))]),
                    )
                    .expect("Failed to add token id");

                if sequence.get_output_len() >= max_new_tokens as usize {
                    sequence.set_sequence_status(SequenceStatus::FinishedLengthCapped);
                }

                // The engine streams every generated token, which is how it learns that a client
                // has gone away.
                if self
                    .response_senders
                    .send_chunk(&metadata.request_id, streamed_token(&metadata.request_id))
                    == ClientState::Disconnected
                {
                    disconnected.push(metadata.request_id.clone());
                }
            }
        }

        for scheduled_sequence_group in scheduler_outputs.scheduled_sequence_groups.iter() {
            if scheduled_sequence_group.scheduled_group.is_finished() {
                self.kv_pool
                    .retire(&scheduled_sequence_group.scheduled_group.request_id);
            }
        }

        disconnected
    }

    /// Retires a request, as the engine does when its client disconnects.
    fn abort(&mut self, request_id: &str) {
        self.response_senders.remove(request_id);
        self.scheduler
            .abort_sequence_group(request_id.to_string())
            .expect("Failed to abort sequence group");
        self.kv_pool.retire(request_id);
    }

    /// Steps until every queued request has left the scheduler, with a bound so a wedged engine
    /// fails the test instead of hanging it.
    fn run_to_completion(&mut self) {
        const MAX_STEPS: usize = 512;

        for _ in 0..MAX_STEPS {
            if !self.scheduler.has_unfinished_sequences() {
                return;
            }
            self.step();
        }
        panic!("The scheduler still had unfinished requests after {MAX_STEPS} steps");
    }

    fn free_gpu_blocks(&self) -> usize {
        self.scheduler.num_free_gpu_blocks()
    }

    /// The token ids of a request's sequence, prompt followed by generated tokens.
    fn token_ids(&self, request_id: &str) -> Vec<u32> {
        self.admitted[request_id]
            .sequences
            .values()
            .next()
            .expect("Request has no sequence")
            .read()
            .unwrap()
            .get_token_ids()
    }
}

/// The token a request "generates", distinct per request so that a crossed output is visible in
/// the token ids.
fn generated_token_id(request_id: &str) -> u32 {
    prompt_token_offset(request_id) + 500
}

/// The message the engine streams to a client for one generated token.
fn streamed_token(request_id: &str) -> StreamResponse {
    StreamResponse::Chunk(GenerateStreamingOutput {
        request_id: request_id.to_string(),
        created: 0,
        finish_reason: None,
        logprobs: vec![],
        num_prompt_tokens: PROMPT_LEN,
        num_completion_tokens: 1,
        output_text: format!("token of {request_id}"),
    })
}

fn request_ids(count: usize) -> Vec<String> {
    (0..count).map(|i| format!("request-{i}")).collect()
}

#[test]
fn test_concurrent_requests_get_disjoint_block_tables() {
    let mut harness = Harness::new(NUM_GPU_BLOCKS, NUM_REQUESTS);
    let _clients = request_ids(NUM_REQUESTS)
        .iter()
        .map(|request_id| harness.admit(request_id))
        .collect::<Vec<_>>();

    let (sequence_groups_metadata, _) = harness
        .scheduler
        .schedule()
        .expect("Failed to schedule the first step");

    assert_eq!(
        sequence_groups_metadata.len(),
        NUM_REQUESTS,
        "every request should be scheduled in the first prefill batch"
    );

    let mut sequence_ids = HashSet::new();
    let mut blocks = HashSet::new();
    let mut num_blocks = 0;
    for metadata in sequence_groups_metadata.iter() {
        for (sequence_id, block_table) in metadata.block_tables.iter() {
            assert!(
                !block_table.is_empty(),
                "{} was scheduled with an empty block table",
                metadata.request_id
            );
            sequence_ids.insert(*sequence_id);
            num_blocks += block_table.len();
            blocks.extend(block_table.iter().copied());
        }
    }

    assert_eq!(
        sequence_ids.len(),
        NUM_REQUESTS,
        "sequence ids must be unique"
    );
    assert_eq!(
        blocks.len(),
        num_blocks,
        "no physical block may appear in two block tables"
    );
}

#[test]
fn test_free_blocks_return_to_baseline_once_every_request_finishes() {
    let mut harness = Harness::new(NUM_GPU_BLOCKS, NUM_REQUESTS);
    let baseline_free_blocks = harness.free_gpu_blocks();

    let _clients = request_ids(NUM_REQUESTS)
        .iter()
        .map(|request_id| harness.admit(request_id))
        .collect::<Vec<_>>();
    harness.run_to_completion();

    assert!(!harness.scheduler.has_unfinished_sequences());
    assert_eq!(
        harness.free_gpu_blocks(),
        baseline_free_blocks,
        "every block held by a finished request must return to the pool"
    );
}

#[test]
fn test_concurrent_outputs_do_not_cross() {
    let mut harness = Harness::new(NUM_GPU_BLOCKS, NUM_REQUESTS);
    let request_ids = request_ids(NUM_REQUESTS);
    let _clients = request_ids
        .iter()
        .map(|request_id| harness.admit(request_id))
        .collect::<Vec<_>>();

    harness.run_to_completion();

    for request_id in request_ids.iter() {
        let prompt_offset = prompt_token_offset(request_id);
        let expected = (0..PROMPT_LEN as u32)
            .map(|i| prompt_offset + i)
            .chain((0..MAX_NEW_TOKENS).map(|_| generated_token_id(request_id)))
            .collect::<Vec<_>>();

        assert_eq!(
            harness.token_ids(request_id),
            expected,
            "{request_id} did not keep its own prompt and output"
        );
    }
}

/// One client dropping mid-generation retires that request only: the rest of the batch finishes,
/// and the dropped request's blocks come back.
#[test]
fn test_a_disconnected_client_does_not_disturb_the_others() {
    const DISCONNECTED: usize = 7;

    let mut harness = Harness::new(NUM_GPU_BLOCKS, NUM_REQUESTS);
    let baseline_free_blocks = harness.free_gpu_blocks();
    let request_ids = request_ids(NUM_REQUESTS);
    let mut clients = request_ids
        .iter()
        .map(|request_id| harness.admit(request_id))
        .collect::<Vec<_>>();

    // Two steps in, the request holds blocks and has generated part of its output; then its
    // client hangs up, which the engine only finds out about on the next step.
    harness.step();
    harness.step();
    assert!(harness.kv_pool.live_blocks() > 0);
    drop(clients.remove(DISCONNECTED));

    harness.run_to_completion();

    let disconnected_id = &request_ids[DISCONNECTED];
    for request_id in request_ids.iter().filter(|id| *id != disconnected_id) {
        assert_eq!(
            harness.token_ids(request_id).len(),
            PROMPT_LEN + MAX_NEW_TOKENS as usize,
            "{request_id} did not finish after {disconnected_id} disconnected"
        );
    }
    assert!(
        harness.token_ids(disconnected_id).len() < PROMPT_LEN + MAX_NEW_TOKENS as usize,
        "the disconnected request should have stopped generating"
    );
    assert_eq!(
        harness.free_gpu_blocks(),
        baseline_free_blocks,
        "the disconnected request's blocks must return to the pool"
    );
}

/// Samples the free-block gauge the way the benchmark harness does: scrape the Prometheus text
/// with the harness's own parser, looking for the name the engine says it publishes under. A
/// rename on either side of that seam fails every test that samples the gauge.
fn sample_gauge(handle: &PrometheusHandle, at: Duration) -> FreeBlockSample {
    let free_blocks = atoma_bench::kv_probe::parse_gauge(
        &handle.render(),
        crate::scheduler::FREE_GPU_BLOCKS_METRIC,
    )
    .expect("The scheduler did not publish the free-block gauge");

    FreeBlockSample {
        at,
        free_blocks: free_blocks as u64,
    }
}

/// Whether a run retires its finished requests — the P0-2 seam.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Retirement {
    /// Finished requests give their blocks back, as the fixed engine does.
    Frees,
    /// Finished requests are never retired, as the engine did before the P0 fix.
    Leaks,
}

/// Drives `steps` steps of a run, sampling the gauge before load, after every step, and once the
/// run has drained.
fn sampled_run(retirement: Retirement, steps: usize) -> Vec<FreeBlockSample> {
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();

    metrics::with_local_recorder(&recorder, || {
        let mut harness = Harness::new(NUM_GPU_BLOCKS, NUM_REQUESTS);
        let mut samples = vec![sample_gauge(&handle, Duration::ZERO)];
        let mut clients = Vec::new();

        for step in 0..steps {
            clients.push(harness.admit(&format!("request-{step}")));
            match retirement {
                Retirement::Frees => harness.step(),
                Retirement::Leaks => {
                    harness.step_without_retiring();
                }
            }
            samples.push(sample_gauge(
                &handle,
                Duration::from_millis(step as u64 + 1),
            ));
        }

        if retirement == Retirement::Frees {
            harness.run_to_completion();
        }
        samples.push(sample_gauge(
            &handle,
            Duration::from_millis(steps as u64 + 1),
        ));
        samples
    })
}

/// The gauge an untouched engine publishes has to satisfy the probe, or the guard would fail every
/// clean run and get switched off.
#[test]
fn test_the_kv_probe_passes_a_run_that_returns_its_blocks() {
    let samples = sampled_run(Retirement::Frees, 24);

    let verdict = atoma_bench::kv_probe::verdict(&samples, &KvProbeConfig::default());

    assert!(verdict.run_passes(), "{verdict:?}");
}

/// P0-2 reintroduced: finished requests are never retired, so their blocks never return. The probe
/// has to fail the run — this is the guard that keeps the fix from regressing unnoticed.
#[test]
fn test_the_kv_probe_fails_a_run_that_never_frees_finished_requests() {
    let samples = sampled_run(Retirement::Leaks, 24);

    let verdict = atoma_bench::kv_probe::verdict(&samples, &KvProbeConfig::default());

    assert!(
        matches!(verdict, KvProbeVerdict::Leak { .. }),
        "a pool that never gets its blocks back must fail the run as a leak, not for want of \
         samples: {verdict:?}"
    );
}

/// The pool is smaller than the request stream, so the requests only all complete if finished ones
/// return their blocks.
#[test]
fn test_requests_keep_flowing_through_a_pool_smaller_than_the_batch() {
    const NUM_SMALL_POOL_BLOCKS: usize = 8;
    const NUM_STREAMED_REQUESTS: usize = 64;

    let mut harness = Harness::new(NUM_SMALL_POOL_BLOCKS, 2);
    let baseline_free_blocks = harness.free_gpu_blocks();
    let request_ids = request_ids(NUM_STREAMED_REQUESTS);
    let _clients = request_ids
        .iter()
        .map(|request_id| harness.admit(request_id))
        .collect::<Vec<_>>();

    harness.run_to_completion();

    for request_id in request_ids.iter() {
        assert_eq!(
            harness.token_ids(request_id).len(),
            PROMPT_LEN + MAX_NEW_TOKENS as usize,
            "{request_id} never completed"
        );
    }
    assert_eq!(harness.free_gpu_blocks(), baseline_free_blocks);
}
