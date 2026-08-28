//! Scheduler tests, including the port of the vLLM-derived scheduler tests.
//!
//! Port ledger against `crates/backends/vllm/src/scheduler.rs`:
//!
//! - Ported with semantics intact: `add_sequence_group`, `schedule_simple`, `max_seqs`,
//!   `prefill_schedule_max_prompt_len`, `prefill_schedule_max_seqs`,
//!   `prefill_schedule_no_block_manager_capacity`, `scheduling_budget` (in `budget.rs`),
//!   `finished_sequence_groups_return_their_blocks`,
//!   `sequential_requests_do_not_exhaust_the_block_pool`,
//!   `abort_removes_the_sequence_group_from_every_queue` and `scheduler_abort_sequence_group`
//!   (abort is the client dropping its egress receiver). Where the old tests counted free blocks
//!   after a finish, these count `available` blocks: a finished request's full blocks stay resident
//!   as evictable cache, which the old block manager had no notion of.
//! - Rewritten, divergence documented on the test: `prefill_prioritized` (decode-first, mixed
//!   batches), `prefill_schedule_token_budget` (a prompt over the remaining budget is chunked, not
//!   refused), `scheduler_schedule_preempt_abort` and `decode_schedule_preempted` (preemption is
//!   recompute-only from the tail of the running queue; nothing is swapped).
//! - Dropped, with the machinery they tested: every `schedule_swapped_*`, `infeasible_swap`,
//!   `decode_swap_beam_search` and `schedule_decode_blocks_to_copy_update` (swap and copy-on-write
//!   do not exist; preemption recomputes), `scheduler_delay_factor` (no delay factor), and
//!   `abort_of_unknown_request` (there is no abort by id; a dropped egress receiver is the cancel).

use std::collections::VecDeque;

use crate::kv::{BlockPool, HashAlgorithm};
use crate::request::{
    egress, EgressReceiver, FinishReason, NewRequest, RequestEvent, RequestPhase, SamplingParams,
    StopCriteria, Usage,
};
use crate::scheduler::{AdmissionPolicy, Scheduled, Scheduler, SchedulerConfig, SchedulerError};
use crate::test_support::{requests, tokens};
use crate::types::{RequestSlot, StepId};

const BLOCK_SIZE: usize = 4;
const WINDOW: usize = 8;
const MAX_REQUESTS: usize = 64;
/// The end-of-sequence token; prompts count up from one and never reach it.
const EOS: u32 = 99;

/// A scheduler over `blocks` blocks whose max model length is exactly what the pool holds.
fn scheduler(blocks: u32, token_budget: usize, max_batch: usize) -> Scheduler {
    Scheduler::new(
        config(blocks, token_budget, max_batch),
        BlockPool::new(blocks),
    )
    .expect("the pool holds one maximum-length request")
}

fn config(blocks: u32, token_budget: usize, max_batch: usize) -> SchedulerConfig {
    SchedulerConfig {
        token_budget: tokens(token_budget),
        max_batch: requests(max_batch),
        max_model_len: tokens(blocks as usize * BLOCK_SIZE),
        block_size: tokens(BLOCK_SIZE),
        window: requests(WINDOW),
        admission: AdmissionPolicy::Fcfs,
        max_requests: requests(MAX_REQUESTS),
        eos_token_ids: vec![EOS],
        hash_algorithm: HashAlgorithm::Sha256V1,
    }
}

/// A scheduler admitting by longest prefix match over a window of `window` candidates.
fn lpm_scheduler(blocks: u32, max_batch: usize, window: usize) -> Scheduler {
    Scheduler::new(
        SchedulerConfig {
            admission: AdmissionPolicy::LongestPrefixMatch,
            window: requests(window),
            ..config(blocks, 1000, max_batch)
        },
        BlockPool::new(blocks),
    )
    .expect("the pool holds one maximum-length request")
}

/// The requests a test submits, with their clients' receivers kept alive.
#[derive(Default)]
struct Clients {
    receivers: Vec<EgressReceiver>,
}

impl Clients {
    /// Submits a `prompt_len`-token prompt allowed `max_new_tokens` generated tokens. Each
    /// submission gets its own token range, so requests never share a prefix unless a test
    /// submits an explicit prompt through [`Clients::submit_prompt`].
    fn submit(
        &mut self,
        scheduler: &mut Scheduler,
        prompt_len: usize,
        max_new_tokens: usize,
    ) -> RequestSlot {
        let base = u32::try_from(1000 * (self.receivers.len() + 1)).unwrap();
        let prompt = (base + 1..=base + u32::try_from(prompt_len).unwrap()).collect();
        self.submit_prompt(scheduler, prompt, max_new_tokens)
    }

    /// Submits `prompt` exactly, for tests about shared prefixes.
    fn submit_prompt(
        &mut self,
        scheduler: &mut Scheduler,
        prompt: Vec<u32>,
        max_new_tokens: usize,
    ) -> RequestSlot {
        let (sender, receiver) = egress();
        self.receivers.push(receiver);
        scheduler
            .intake(NewRequest {
                prompt,
                ..new_request(1, max_new_tokens, sender)
            })
            .expect("the prompt fits the model")
    }
}

fn new_request(
    prompt_len: usize,
    max_new_tokens: usize,
    egress: crate::request::EgressSender,
) -> NewRequest {
    NewRequest {
        prompt: (1..=u32::try_from(prompt_len).unwrap()).collect(),
        sampling: SamplingParams::default(),
        stop: StopCriteria {
            max_new_tokens: tokens(max_new_tokens),
            ignore_eos: false,
        },
        egress,
    }
}

/// Schedules one pass and applies it with token `1` sampled for every sampling entry.
fn step(scheduler: &mut Scheduler) -> Scheduled {
    step_sampling(scheduler, 1)
}

/// Schedules one pass and applies it with `token` sampled for every sampling entry.
fn step_sampling(scheduler: &mut Scheduler, token: u32) -> Scheduled {
    let scheduled = scheduler.schedule();
    let sampled = vec![token; scheduled.sampling_entries().count()];
    scheduler.apply(&scheduled, &sampled);
    scheduled
}

/// Every event a client has received so far.
fn events(receiver: &EgressReceiver) -> Vec<RequestEvent> {
    receiver.try_iter().collect()
}

fn finish_reason(receiver: &EgressReceiver) -> Option<FinishReason> {
    events(receiver).into_iter().find_map(|event| match event {
        RequestEvent::Finished { reason, .. } => Some(reason),
        RequestEvent::Token { .. } => None,
    })
}

/// A prompt of exactly `blocks` full blocks, the same for every caller.
fn shared_prompt(blocks: usize) -> Vec<u32> {
    (1..=u32::try_from(blocks * BLOCK_SIZE).unwrap()).collect()
}

fn slots(scheduled: &Scheduled) -> Vec<RequestSlot> {
    scheduled.entries.iter().map(|entry| entry.slot).collect()
}

/// Port of `test_scheduler_add_sequence_group`.
#[test]
fn intake_queues_requests_in_arrival_order() {
    let mut scheduler = scheduler(8, 100, 64);
    let mut clients = Clients::default();
    let mut submitted = Vec::new();
    for i in 0..4 {
        submitted.push(clients.submit(&mut scheduler, BLOCK_SIZE, 16));
        assert_eq!(scheduler.request_count(), i + 1);
    }
    assert_eq!(scheduler.waiting(), &VecDeque::from(submitted.clone()));
    assert!(scheduler.running().is_empty());
    for (i, slot) in submitted.iter().enumerate() {
        let request = scheduler.request(*slot).unwrap();
        assert_eq!(
            request.id().get(),
            i as u64,
            "ids are minted in intake order"
        );
        assert!(matches!(request.phase(), RequestPhase::Waiting(_)));
    }
}

/// Port of `test_scheduler_schedule_simple`: four one-block prompts prefill together, then decode
/// together one token each.
#[test]
fn prompts_are_scheduled_together_then_decode_one_token_each() {
    let mut scheduler = scheduler(8, 100, 4);
    let mut clients = Clients::default();
    let submitted: Vec<_> = (0..4)
        .map(|_| clients.submit(&mut scheduler, BLOCK_SIZE, 16))
        .collect();

    let prefill = step(&mut scheduler);
    assert_eq!(prefill.step, StepId::new(1));
    assert_eq!(slots(&prefill), submitted);
    assert_eq!(prefill.token_count(), BLOCK_SIZE * 4);
    assert!(!prefill.is_uniform_decode());
    for entry in &prefill.entries {
        assert_eq!(entry.context_len, 0);
        assert_eq!(entry.query_len, tokens(BLOCK_SIZE));
        assert!(entry.samples, "a whole prompt in one chunk samples");
    }
    assert_eq!(scheduler.running(), submitted.as_slice());
    assert!(scheduler.waiting().is_empty());

    let decode = step(&mut scheduler);
    assert_eq!(decode.step, StepId::new(2));
    assert_eq!(slots(&decode), submitted);
    assert_eq!(decode.token_count(), 4);
    assert!(decode.is_uniform_decode());
    for entry in &decode.entries {
        assert_eq!(entry.context_len, BLOCK_SIZE, "the prompt is resident");
        assert_eq!(entry.query_len, tokens(1), "the sampled token is the query");
        assert!(entry.samples);
        assert_eq!(entry.sequence_len(), BLOCK_SIZE + 1);
    }
    for slot in submitted {
        let sequence = &scheduler.request(slot).unwrap().sequences()[0];
        assert_eq!(sequence.computed(), BLOCK_SIZE + 1);
        assert_eq!(&sequence.tokens()[BLOCK_SIZE..], &[1, 1]);
        assert!(sequence.is_decoding());
    }
}

/// Port of `test_scheduler_max_seqs`: with a cap of two entries and one request decoding, only one
/// of two new prompts is admitted.
#[test]
fn the_request_cap_bounds_the_batch() {
    let mut scheduler = scheduler(8, 64, 2);
    let mut clients = Clients::default();
    let first = clients.submit(&mut scheduler, BLOCK_SIZE, 16);
    assert_eq!(slots(&step(&mut scheduler)), [first]);
    assert_eq!(slots(&step(&mut scheduler)), [first]);

    let second = clients.submit(&mut scheduler, BLOCK_SIZE, 16);
    let third = clients.submit(&mut scheduler, BLOCK_SIZE, 16);
    let scheduled = step(&mut scheduler);
    assert_eq!(slots(&scheduled), [first, second]);
    assert_eq!(scheduler.waiting(), &VecDeque::from([third]));
}

/// Port of `test_prefill_schedule_max_prompt_len`: a prompt that leaves no room to generate
/// finishes at intake, and its client hears why. An empty prompt likewise.
#[test]
fn a_prompt_that_cannot_fit_the_model_finishes_at_intake() {
    let mut scheduler = scheduler(8, 1000, 64);
    let max_model_length = 8 * BLOCK_SIZE;

    let (sender, receiver) = egress();
    let error = scheduler
        .intake(new_request(max_model_length, 16, sender))
        .unwrap_err();
    assert_eq!(
        error,
        FinishReason::PromptExceedsMaxModelLength {
            prompt_tokens: max_model_length,
            max_model_length,
        }
    );
    assert!(matches!(
        receiver.recv().unwrap(),
        RequestEvent::Finished {
            reason: FinishReason::PromptExceedsMaxModelLength { .. },
            ..
        }
    ));
    assert!(receiver.recv().is_err(), "nothing follows the finish");
    assert_eq!(
        scheduler.request_count(),
        0,
        "a rejected request takes no slot"
    );

    let (sender, receiver) = egress();
    assert_eq!(
        scheduler.intake(new_request(0, 16, sender)).unwrap_err(),
        FinishReason::EmptyPrompt
    );
    assert!(matches!(
        receiver.recv().unwrap(),
        RequestEvent::Finished {
            reason: FinishReason::EmptyPrompt,
            ..
        }
    ));

    let mut clients = Clients::default();
    clients.submit(&mut scheduler, max_model_length - 1, 16);
    assert_eq!(
        scheduler.request_count(),
        1,
        "one short of the maximum fits"
    );
}

/// Rewrite of `test_prefill_schedule_token_budget`. Divergence: the old scheduler refused a
/// prompt that did not fit the remaining budget; here chunked prefill is the one path, so the
/// prompt takes the remainder as a chunk and finishes over later steps.
#[test]
fn a_prompt_over_the_remaining_budget_is_chunked_not_refused() {
    let mut scheduler = scheduler(40, 60, 64);
    let mut clients = Clients::default();
    let first = clients.submit(&mut scheduler, 60, 16);
    let second = clients.submit(&mut scheduler, 60, 16);

    let scheduled = step(&mut scheduler);
    assert_eq!(
        slots(&scheduled),
        [first],
        "sixty of sixty: one prompt fits whole"
    );
    assert_eq!(scheduled.entries[0].query_len, tokens(60));
    assert!(scheduled.entries[0].samples);
    assert_eq!(scheduler.waiting(), &VecDeque::from([second]));

    let scheduled = step(&mut scheduler);
    assert_eq!(slots(&scheduled), [first, second]);
    assert_eq!(
        scheduled.entries[0].query_len,
        tokens(1),
        "the decode is budgeted first"
    );
    assert_eq!(
        scheduled.entries[1].query_len,
        tokens(59),
        "the remainder is a chunk"
    );
    assert!(
        !scheduled.entries[1].samples,
        "a non-final chunk does not sample"
    );
    assert_eq!(scheduled.token_count(), 60);

    let scheduled = step(&mut scheduler);
    assert_eq!(scheduled.entries[1].context_len, 59);
    assert_eq!(scheduled.entries[1].query_len, tokens(1));
    assert!(scheduled.entries[1].samples, "the final chunk samples");
}

/// Port of `test_prefill_schedule_max_seqs`: two of three prompts admit under a cap of two, and a
/// cap spent by running requests admits nothing.
#[test]
fn admission_stops_at_the_request_cap() {
    let mut scheduler = scheduler(64, 10_000, 2);
    let mut clients = Clients::default();
    let submitted: Vec<_> = (0..3)
        .map(|_| clients.submit(&mut scheduler, 60, 16))
        .collect();

    let scheduled = step(&mut scheduler);
    assert_eq!(slots(&scheduled), submitted[..2]);
    assert_eq!(scheduled.token_count(), 120);
    assert_eq!(scheduler.waiting(), &VecDeque::from([submitted[2]]));

    let scheduled = step(&mut scheduler);
    assert_eq!(
        slots(&scheduled),
        submitted[..2],
        "two decodes spend the cap"
    );
    assert_eq!(scheduler.waiting(), &VecDeque::from([submitted[2]]));
}

/// Port of `test_prefill_schedule_no_block_manager_capacity`. The old `Later` case — the pool is
/// full now — keeps the request waiting; the old `Never` case — the pool could never hold the
/// request — is a configuration the scheduler refuses to start under, since any prompt under
/// the maximum model length fits a pool that holds one maximum-length request.
#[test]
fn a_prompt_the_pool_cannot_hold_now_waits_and_a_pool_too_small_is_refused() {
    let mut scheduler = scheduler(2, 1000, 64);
    let mut clients = Clients::default();
    let first = clients.submit(&mut scheduler, 7, 16);
    let second = clients.submit(&mut scheduler, 7, 16);

    let scheduled = scheduler.schedule();
    assert_eq!(slots(&scheduled), [first]);
    assert_eq!(scheduler.pool().free_count(), 0);
    assert_eq!(
        scheduler.waiting(),
        &VecDeque::from([second]),
        "waits for blocks"
    );
    scheduler.apply(&scheduled, &[1]);

    assert_eq!(
        Scheduler::new(config(4, 1000, 64), BlockPool::new(3)).unwrap_err(),
        SchedulerError::PoolTooSmallForMaxModelLength { needed: 4, free: 3 }
    );
}

/// Rewrite of `test_scheduler_prefill_prioritized`. Divergence: the old scheduler ran prefills
/// ahead of decodes and never mixed them; here running decodes spend the budget first and a new
/// prompt takes the remainder in the same batch.
#[test]
fn running_decodes_are_budgeted_before_a_new_prefill_in_the_same_batch() {
    let mut scheduler = scheduler(32, 30, 64);
    let mut clients = Clients::default();
    let first = clients.submit(&mut scheduler, 1, 16);
    assert_eq!(slots(&step(&mut scheduler)), [first]);

    let second = clients.submit(&mut scheduler, 30, 16);
    let scheduled = step(&mut scheduler);
    assert_eq!(slots(&scheduled), [first, second]);
    assert_eq!(scheduled.entries[0].query_len, tokens(1));
    assert_eq!(scheduled.entries[1].query_len, tokens(29));
    assert_eq!(scheduled.token_count(), 30, "the whole budget is spent");
}

#[test]
fn a_prefill_samples_only_on_the_chunk_that_reaches_the_total() {
    let mut scheduler = scheduler(8, 4, 64);
    let mut clients = Clients::default();
    let slot = clients.submit(&mut scheduler, 10, 16);

    let expected = [(0, 4, false), (4, 4, false), (8, 2, true), (10, 1, true)];
    for (context_len, query_len, samples) in expected {
        let scheduled = step(&mut scheduler);
        let entry = scheduled.entries[0];
        assert_eq!(entry.slot, slot);
        assert_eq!(entry.context_len, context_len);
        assert_eq!(entry.query_len, tokens(query_len));
        assert_eq!(entry.samples, samples);
        let sequence = &scheduler.request(slot).unwrap().sequences()[0];
        assert_eq!(sequence.computed(), context_len + query_len);
    }
    let sequence = &scheduler.request(slot).unwrap().sequences()[0];
    assert_eq!(sequence.generated_count(), 2);
}

#[test]
fn block_tables_are_host_native_and_cover_every_scheduled_token() {
    let mut scheduler = scheduler(16, 100, 64);
    let mut clients = Clients::default();
    let slot = clients.submit(&mut scheduler, 9, 16);
    let baseline = scheduler.pool().free_count();

    for expected_blocks in [3, 3, 3, 3, 4] {
        let scheduled = step(&mut scheduler);
        let entry = scheduled.entries[0];
        let table = scheduler.request(slot).unwrap().sequences()[0].block_table();
        assert_eq!(table.len(), expected_blocks);
        assert!(entry.sequence_len() <= table.len() * BLOCK_SIZE);
        assert!(
            entry.sequence_len() > (table.len() - 1) * BLOCK_SIZE,
            "no spare block"
        );
        assert_eq!(scheduler.pool().free_count(), baseline - expected_blocks);
    }
}

#[test]
#[should_panic(expected = "more sampled tokens")]
fn applying_more_tokens_than_sampling_entries_is_a_protocol_violation() {
    let mut scheduler = scheduler(8, 100, 64);
    let mut clients = Clients::default();
    clients.submit(&mut scheduler, 4, 16);
    let scheduled = scheduler.schedule();
    scheduler.apply(&scheduled, &[1, 2]);
}

#[test]
#[should_panic(expected = "one sampled token per sampling entry")]
fn applying_fewer_tokens_than_sampling_entries_is_a_protocol_violation() {
    let mut scheduler = scheduler(8, 100, 64);
    let mut clients = Clients::default();
    clients.submit(&mut scheduler, 4, 16);
    let scheduled = scheduler.schedule();
    scheduler.apply(&scheduled, &[]);
}

/// Port of `test_finished_sequence_groups_return_their_blocks`.
#[test]
fn finished_requests_return_their_blocks() {
    let mut scheduler = scheduler(8, 100, 4);
    let mut clients = Clients::default();
    let baseline = scheduler.pool().available();
    let submitted: Vec<_> = (0..4)
        .map(|_| clients.submit(&mut scheduler, BLOCK_SIZE, 1))
        .collect();

    let scheduled = scheduler.schedule();
    assert_eq!(
        scheduler.pool().available(),
        baseline - 4,
        "one block per prompt"
    );
    scheduler.apply(&scheduled, &[1; 4]);

    assert!(
        scheduler.running().is_empty(),
        "one token was all each asked for"
    );
    assert_eq!(
        scheduler.request_count(),
        0,
        "finished requests free their slots"
    );
    assert_eq!(
        scheduler.pool().available(),
        baseline,
        "every block came back, free or as cache"
    );
    for (slot, receiver) in submitted.iter().zip(&clients.receivers) {
        assert!(scheduler.request(*slot).is_none());
        let events = events(receiver);
        assert!(matches!(events[0], RequestEvent::Token { token: 1, .. }));
        assert!(matches!(
            events[1],
            RequestEvent::Finished {
                reason: FinishReason::MaxNewTokens,
                usage: Usage {
                    prompt_tokens: BLOCK_SIZE,
                    generated_tokens: 1
                },
                ..
            }
        ));
        assert_eq!(events.len(), 2);
    }
}

/// Port of `test_sequential_requests_do_not_exhaust_the_block_pool`: a pool smaller than the
/// request stream keeps serving because every finished request returns its blocks.
#[test]
fn sequential_requests_do_not_exhaust_the_block_pool() {
    let mut scheduler = scheduler(4, 100, 1);
    let mut clients = Clients::default();
    let baseline = scheduler.pool().available();
    for i in 0..32 {
        let slot = clients.submit(&mut scheduler, BLOCK_SIZE, 1);
        let scheduled = step(&mut scheduler);
        assert_eq!(slots(&scheduled), [slot], "request {i} was never scheduled");
        assert_eq!(
            scheduler.pool().available(),
            baseline,
            "request {i} leaked blocks"
        );
    }
}

/// Port of `test_abort_removes_the_sequence_group_from_every_queue` for the waiting queue: the
/// client dropping its receiver is the abort; the cancelled request never admits and the survivor
/// takes its place.
#[test]
fn a_dropped_receiver_retires_only_that_request_from_waiting() {
    let mut scheduler = scheduler(8, 100, 4);
    let mut clients = Clients::default();
    let cancelled = clients.submit(&mut scheduler, BLOCK_SIZE, 16);
    let survivor = clients.submit(&mut scheduler, BLOCK_SIZE, 16);
    drop(clients.receivers.remove(0));

    let scheduled = step(&mut scheduler);
    assert_eq!(slots(&scheduled), [survivor]);
    assert!(scheduler.request(cancelled).is_none());
    assert_eq!(scheduler.request_count(), 1);
}

/// The same for the running queue: the cancelled request's blocks return and the survivor keeps
/// decoding.
#[test]
fn a_dropped_receiver_retires_only_that_request_from_running() {
    let mut scheduler = scheduler(8, 100, 4);
    let mut clients = Clients::default();
    let baseline = scheduler.pool().available();
    let cancelled = clients.submit(&mut scheduler, BLOCK_SIZE, 16);
    let survivor = clients.submit(&mut scheduler, BLOCK_SIZE, 16);
    assert_eq!(slots(&step(&mut scheduler)), [cancelled, survivor]);
    drop(clients.receivers.remove(0));

    let scheduled = step(&mut scheduler);
    assert_eq!(slots(&scheduled), [survivor]);
    assert_eq!(scheduler.running(), [survivor]);
    assert_eq!(scheduler.pool().available(), baseline - 2);
    assert!(matches!(
        events(&clients.receivers[0]).last(),
        Some(RequestEvent::Token { token: 1, .. })
    ));
}

/// Port of `test_scheduler_abort_sequence_group`: every client hanging up empties the scheduler.
#[test]
fn every_client_hanging_up_empties_the_scheduler() {
    let mut scheduler = scheduler(8, 100, 64);
    let mut clients = Clients::default();
    for _ in 0..4 {
        clients.submit(&mut scheduler, BLOCK_SIZE, 16);
    }
    step(&mut scheduler);
    assert_eq!(scheduler.request_count(), 4);
    clients.receivers.clear();
    let scheduled = step(&mut scheduler);
    assert!(scheduled.is_empty());
    assert_eq!(scheduler.request_count(), 0);
    assert_eq!(scheduler.pool().available(), 8);
}

#[test]
fn an_end_of_sequence_token_finishes_a_request_unless_it_ignores_eos() {
    let mut scheduler = scheduler(8, 100, 64);
    let mut clients = Clients::default();
    let stops = clients.submit(&mut scheduler, BLOCK_SIZE, 16);
    let (sender, receiver) = egress();
    let ignores = scheduler
        .intake(NewRequest {
            stop: StopCriteria {
                max_new_tokens: tokens(16),
                ignore_eos: true,
            },
            ..new_request(BLOCK_SIZE, 16, sender)
        })
        .unwrap();
    clients.receivers.push(receiver);

    let scheduled = step_sampling(&mut scheduler, EOS);
    assert_eq!(slots(&scheduled), [stops, ignores]);
    assert_eq!(scheduler.running(), [ignores]);
    assert_eq!(
        finish_reason(&clients.receivers[0]),
        Some(FinishReason::EndOfSequence)
    );
    assert_eq!(finish_reason(&clients.receivers[1]), None);
    assert_eq!(
        scheduler.request(ignores).unwrap().sequences()[0].tokens(),
        &[1, 2, 3, 4, EOS]
    );
}

#[test]
fn reaching_the_max_model_length_finishes_a_request() {
    let mut scheduler = scheduler(2, 100, 64);
    let mut clients = Clients::default();
    let slot = clients.submit(&mut scheduler, 6, 16);

    step(&mut scheduler);
    assert_eq!(scheduler.request(slot).unwrap().sequences()[0].total(), 7);
    step(&mut scheduler);
    assert!(scheduler.request(slot).is_none(), "eight of eight tokens");
    assert_eq!(
        finish_reason(&clients.receivers[0]),
        Some(FinishReason::MaxModelLength)
    );
    assert_eq!(scheduler.pool().available(), 2);
}

/// Rewrite of `test_scheduler_schedule_preempt_abort`. Divergence: the old scheduler preempted
/// into a swapped queue and asserted the swap lists stayed empty; here there is no swapped
/// queue at all — the victim releases its KV, its computed count resets to zero, and it
/// recomputes prompt and generated tokens together when it re-enters.
#[test]
fn a_decode_the_pool_cannot_grow_preempts_the_newest_request_which_recomputes() {
    let mut scheduler = scheduler(2, 64, 2);
    let mut clients = Clients::default();
    let a = clients.submit(&mut scheduler, BLOCK_SIZE, 16);
    let b = clients.submit(&mut scheduler, BLOCK_SIZE, 16);

    let prefill = step(&mut scheduler);
    assert_eq!(slots(&prefill), [a, b]);
    assert_eq!(prefill.token_count(), BLOCK_SIZE * 2);
    assert!(prefill.preempted.is_empty());
    assert_eq!(scheduler.pool().free_count(), 0);

    // Both need a second block to decode; only one exists once the newest gives its block up.
    let decode = step(&mut scheduler);
    assert_eq!(slots(&decode), [a]);
    assert_eq!(decode.preempted, [b]);
    assert_eq!(decode.token_count(), 1);
    assert_eq!(scheduler.running(), [a]);
    assert_eq!(scheduler.preempted(), [b]);
    assert!(scheduler.waiting().is_empty());
    let displaced = scheduler.request(b).unwrap();
    assert!(matches!(displaced.phase(), RequestPhase::Preempted(_)));
    let sequence = &displaced.sequences()[0];
    assert_eq!(sequence.computed(), 0, "recompute from the start");
    assert!(
        sequence.block_table().is_empty(),
        "its KV went back to the pool"
    );
    assert_eq!(
        sequence.total(),
        BLOCK_SIZE + 1,
        "its sampled token is kept"
    );

    // The client of `a` hangs up; `b` re-enters and recomputes prompt and generated tokens.
    drop(clients.receivers.remove(0));
    let recompute = step(&mut scheduler);
    assert_eq!(slots(&recompute), [b]);
    let entry = recompute.entries[0];
    assert_eq!(entry.context_len, 0);
    assert_eq!(entry.query_len, tokens(BLOCK_SIZE + 1));
    assert!(entry.samples, "the recompute reaches the total and samples");
    assert_eq!(scheduler.running(), [b]);
    assert!(scheduler.preempted().is_empty());
    assert!(scheduler.request(a).is_none());
    assert_eq!(
        scheduler.request(b).unwrap().sequences()[0].total(),
        BLOCK_SIZE + 2
    );
}

/// Rewrite of `test_decode_schedule_preempted`. Divergence: the old test forced two victims
/// through a mock; here victims come off the tail of the running queue one at a time, only as
/// many as the pool needs, and the survivors decode in the same step.
#[test]
fn preemption_takes_victims_from_the_tail_only_as_the_pool_needs_them() {
    let mut scheduler = scheduler(12, 1000, 64);
    let mut clients = Clients::default();
    let submitted: Vec<_> = (0..3)
        .map(|_| clients.submit(&mut scheduler, 4 * BLOCK_SIZE, 16))
        .collect();
    assert_eq!(slots(&step(&mut scheduler)), submitted);
    assert_eq!(scheduler.pool().available(), 0);

    let decode = step(&mut scheduler);
    assert_eq!(slots(&decode), submitted[..2]);
    assert_eq!(
        decode.preempted,
        [submitted[2]],
        "one victim frees four blocks"
    );
    assert_eq!(decode.token_count(), 2);
    assert!(decode.is_uniform_decode());
    assert_eq!(
        scheduler.pool().available(),
        2,
        "two of the victim's four blocks were evicted and taken; two stay cached"
    );
    assert_eq!(scheduler.running(), &submitted[..2]);
    assert_eq!(scheduler.preempted(), [submitted[2]]);
}

/// Preempted requests re-enter last in first out, ahead of every waiting request — even one that
/// would fit when the preempted head does not.
#[test]
fn preempted_requests_re_enter_first_last_in_first_out() {
    let mut scheduler = scheduler(3, 64, 64);
    let mut clients = Clients::default();
    let a = clients.submit(&mut scheduler, BLOCK_SIZE, 16);
    let b = clients.submit(&mut scheduler, BLOCK_SIZE, 16);
    let c = clients.submit(&mut scheduler, BLOCK_SIZE, 16);
    assert_eq!(slots(&step(&mut scheduler)), [a, b, c]);

    // `a` takes `c`'s block; `b` then finds nothing and displaces itself.
    let decode = step(&mut scheduler);
    assert_eq!(slots(&decode), [a]);
    assert_eq!(decode.preempted, [c, b]);
    assert_eq!(scheduler.preempted(), [c, b]);
    let d = clients.submit(&mut scheduler, BLOCK_SIZE, 16);

    // `a` hangs up: `b` re-enters first, then `c` cannot fit, so `d` waits behind it.
    drop(clients.receivers.remove(0));
    let readmit = step(&mut scheduler);
    assert_eq!(slots(&readmit), [b]);
    assert_eq!(scheduler.preempted(), [c]);
    assert_eq!(scheduler.waiting(), &VecDeque::from([d]));
    assert_eq!(
        scheduler.pool().free_count(),
        1,
        "`d` would fit, but `c` is ahead of it"
    );

    // `b` hangs up: `c` re-enters, and `d` admits beside it.
    drop(clients.receivers.remove(0));
    let readmit = step(&mut scheduler);
    assert_eq!(slots(&readmit), [c, d]);
    assert!(scheduler.preempted().is_empty());
    assert!(scheduler.waiting().is_empty());
}

/// Port of the preempted-queue case of `test_abort_removes_the_sequence_group_from_every_queue`.
#[test]
fn a_dropped_receiver_retires_only_that_request_from_the_preempted_stack() {
    let mut scheduler = scheduler(3, 64, 64);
    let mut clients = Clients::default();
    let a = clients.submit(&mut scheduler, BLOCK_SIZE, 16);
    let b = clients.submit(&mut scheduler, BLOCK_SIZE, 16);
    let c = clients.submit(&mut scheduler, BLOCK_SIZE, 16);
    step(&mut scheduler);
    assert_eq!(step(&mut scheduler).preempted, [c, b]);

    drop(clients.receivers.remove(1));
    let scheduled = step(&mut scheduler);
    assert_eq!(slots(&scheduled), [a]);
    assert!(scheduler.request(b).is_none(), "retired at admission");
    assert_eq!(
        scheduler.preempted(),
        [c],
        "the other preempted request is untouched"
    );
    assert_eq!(scheduler.request_count(), 2);
}

#[test]
fn a_request_with_a_cached_prefix_starts_past_it() {
    let mut scheduler = scheduler(16, 100, 64);
    let mut clients = Clients::default();
    let shared = shared_prompt(3);
    let first = clients.submit_prompt(&mut scheduler, shared.clone(), 1);
    let scheduled = scheduler.schedule();
    let first_blocks = scheduler.request(first).unwrap().sequences()[0]
        .block_table()
        .to_vec();
    scheduler.apply(&scheduled, &[1]);
    assert!(scheduler.request(first).is_none());
    assert_eq!(scheduler.index().len(), 3, "three full blocks were hashed");

    let second = clients.submit_prompt(&mut scheduler, shared, 1);
    let scheduled = scheduler.schedule();
    let entry = scheduled.entries[0];
    assert_eq!(entry.slot, second);
    assert_eq!(entry.context_len, 2 * BLOCK_SIZE, "two blocks are hits");
    assert_eq!(
        entry.query_len,
        tokens(BLOCK_SIZE),
        "the last block is computed"
    );
    assert!(entry.samples);
    let table = scheduler.request(second).unwrap().sequences()[0].block_table();
    assert_eq!(table[..2], first_blocks[..2], "hits read the cached blocks");
    assert_ne!(table[2], first_blocks[2], "the computed block is its own");
    scheduler.apply(&scheduled, &[1]);
}

/// A prompt of exactly two full blocks hits only the first: the block holding the last prompt
/// token is always computed, so there is a logit to sample the first token from.
#[test]
fn the_block_holding_the_last_prompt_token_is_never_a_hit() {
    let mut scheduler = scheduler(16, 100, 64);
    let mut clients = Clients::default();
    let shared = shared_prompt(2);
    clients.submit_prompt(&mut scheduler, shared.clone(), 1);
    step(&mut scheduler);
    assert_eq!(scheduler.index().len(), 2);

    let second = clients.submit_prompt(&mut scheduler, shared, 1);
    let scheduled = step(&mut scheduler);
    let entry = scheduled.entries[0];
    assert_eq!(entry.slot, second);
    assert_eq!(entry.context_len, BLOCK_SIZE);
    assert_eq!(entry.query_len, tokens(BLOCK_SIZE));
}

/// The preempted request's computed count resets to zero, and what the index still holds of
/// its prefix is rediscovered when it re-enters Running.
#[test]
fn a_preempted_request_rediscovers_its_cached_prefix_on_re_entry() {
    let mut scheduler = scheduler(12, 1000, 64);
    let mut clients = Clients::default();
    let submitted: Vec<_> = (0..3)
        .map(|_| clients.submit(&mut scheduler, 4 * BLOCK_SIZE, 2))
        .collect();
    step(&mut scheduler);

    // The two survivors each take one of the victim's four cached blocks, then finish.
    let decode = step(&mut scheduler);
    assert_eq!(decode.preempted, [submitted[2]]);
    assert!(
        scheduler.running().is_empty(),
        "two tokens was all they asked for"
    );
    assert_eq!(
        scheduler.request(submitted[2]).unwrap().sequences()[0].computed(),
        0
    );

    let readmit = step(&mut scheduler);
    let entry = readmit.entries[0];
    assert_eq!(entry.slot, submitted[2]);
    assert_eq!(
        entry.context_len,
        2 * BLOCK_SIZE,
        "two of its blocks survived"
    );
    assert_eq!(
        entry.query_len,
        tokens(2 * BLOCK_SIZE + 1),
        "the rest of the prompt and its sampled token recompute"
    );
    assert!(entry.samples);
}

#[test]
fn cache_is_evicted_before_anyone_is_preempted() {
    let mut scheduler = scheduler(4, 100, 64);
    let mut clients = Clients::default();
    clients.submit(&mut scheduler, BLOCK_SIZE, 1);
    step(&mut scheduler);
    assert_eq!(scheduler.index().len(), 1);
    assert_eq!(scheduler.pool().free_count(), 3);

    let b = clients.submit(&mut scheduler, BLOCK_SIZE, 16);
    let c = clients.submit(&mut scheduler, BLOCK_SIZE, 16);
    step(&mut scheduler);
    let decode = step(&mut scheduler);
    assert_eq!(slots(&decode), [b, c]);
    assert!(decode.preempted.is_empty(), "the cached block went first");
    assert_eq!(scheduler.pool().free_count(), 0);
}

#[test]
fn finished_requests_leave_their_full_blocks_as_evictable_cache() {
    let mut scheduler = scheduler(8, 100, 64);
    let mut clients = Clients::default();
    let baseline = scheduler.pool().free_count();
    clients.submit(&mut scheduler, 2 * BLOCK_SIZE, 1);
    step(&mut scheduler);

    assert_eq!(scheduler.request_count(), 0);
    assert_eq!(scheduler.pool().available(), baseline);
    assert_eq!(
        scheduler.pool().free_count(),
        baseline - 2,
        "two full blocks stay"
    );
    assert_eq!(scheduler.index().len(), 2);
}

/// Two requests computing the same prefix at once both compute it; only the first claim is
/// cached, the other block frees on finish, and a third request hits the one copy.
#[test]
fn identical_prompts_admitted_together_share_one_cached_copy() {
    let mut scheduler = scheduler(8, 100, 64);
    let mut clients = Clients::default();
    let baseline = scheduler.pool().available();
    let shared = shared_prompt(2);
    clients.submit_prompt(&mut scheduler, shared.clone(), 1);
    clients.submit_prompt(&mut scheduler, shared.clone(), 1);
    let scheduled = step(&mut scheduler);
    assert_eq!(scheduled.entries.len(), 2);
    assert!(scheduled.entries.iter().all(|entry| entry.context_len == 0));

    assert_eq!(scheduler.index().len(), 2, "one node per block, not two");
    assert_eq!(scheduler.pool().available(), baseline);
    assert_eq!(
        scheduler.pool().free_count(),
        baseline - 2,
        "one cached copy"
    );

    let third = clients.submit_prompt(&mut scheduler, shared, 1);
    let scheduled = step(&mut scheduler);
    assert_eq!(scheduled.entries[0].slot, third);
    assert_eq!(scheduled.entries[0].context_len, BLOCK_SIZE);
}

/// Caches a three-block prompt, then submits three prompts sharing none, two and one of its
/// blocks: longest prefix match admits them best hit first, in a window that holds all three.
#[test]
fn longest_prefix_match_admits_the_best_hit_in_the_window_first() {
    let mut scheduler = lpm_scheduler(32, 1, 8);
    let mut clients = Clients::default();
    let cached = shared_prompt(3);
    clients.submit_prompt(&mut scheduler, cached.clone(), 1);
    step(&mut scheduler);

    let none = clients.submit(&mut scheduler, 3 * BLOCK_SIZE, 1);
    let two = clients.submit_prompt(&mut scheduler, cached.clone(), 1);
    let one = clients.submit_prompt(
        &mut scheduler,
        [&cached[..BLOCK_SIZE], &[7; 2 * BLOCK_SIZE]].concat(),
        1,
    );

    let first = step(&mut scheduler);
    assert_eq!(slots(&first), [two]);
    assert_eq!(first.entries[0].context_len, 2 * BLOCK_SIZE);
    let second = step(&mut scheduler);
    assert_eq!(slots(&second), [one]);
    assert_eq!(second.entries[0].context_len, BLOCK_SIZE);
    let third = step(&mut scheduler);
    assert_eq!(slots(&third), [none]);
    assert_eq!(third.entries[0].context_len, 0);
}

#[test]
fn longest_prefix_match_breaks_ties_by_arrival() {
    let mut scheduler = lpm_scheduler(32, 1, 8);
    let mut clients = Clients::default();
    let cached = shared_prompt(2);
    clients.submit_prompt(&mut scheduler, cached.clone(), 1);
    step(&mut scheduler);

    let earlier = clients.submit_prompt(&mut scheduler, cached.clone(), 1);
    let later = clients.submit_prompt(&mut scheduler, cached, 1);
    assert_eq!(slots(&step(&mut scheduler)), [earlier]);
    assert_eq!(slots(&step(&mut scheduler)), [later]);
}

/// The policy sees only the window: a better hit behind it waits until it enters.
#[test]
fn longest_prefix_match_examines_only_the_window() {
    let mut scheduler = lpm_scheduler(32, 1, 2);
    let mut clients = Clients::default();
    let cached = shared_prompt(2);
    clients.submit_prompt(&mut scheduler, cached.clone(), 1);
    step(&mut scheduler);

    let first = clients.submit(&mut scheduler, 2 * BLOCK_SIZE, 1);
    let second = clients.submit(&mut scheduler, 2 * BLOCK_SIZE, 1);
    let hit = clients.submit_prompt(&mut scheduler, cached, 1);

    assert_eq!(
        slots(&step(&mut scheduler)),
        [first],
        "the hit is outside the window"
    );
    assert_eq!(slots(&step(&mut scheduler)), [hit], "now it is inside");
    assert_eq!(slots(&step(&mut scheduler)), [second]);
}

/// A preempted request re-enters before the policy orders the window, even when a waiting
/// request would match more of the cache.
#[test]
fn preempted_requests_re_enter_before_longest_prefix_match_orders_the_window() {
    let mut scheduler = lpm_scheduler(4, 64, 8);
    let mut clients = Clients::default();
    let b = clients.submit(&mut scheduler, 2 * BLOCK_SIZE, 16);
    let c_prompt = shared_prompt(2);
    let c = clients.submit_prompt(&mut scheduler, c_prompt.clone(), 16);
    assert_eq!(slots(&step(&mut scheduler)), [b, c]);

    // `b` needs a third block; nothing is evictable, so `c` is displaced and its blocks cached.
    let decode = step(&mut scheduler);
    assert_eq!(slots(&decode), [b]);
    assert_eq!(decode.preempted, [c]);

    // A request sharing `c`'s cached prompt arrives; `c` itself is still ahead of it.
    let hit = clients.submit_prompt(&mut scheduler, c_prompt, 16);
    let readmit = step(&mut scheduler);
    assert_eq!(
        slots(&readmit),
        [b],
        "`c` cannot fit yet, so nobody admits behind it"
    );
    assert_eq!(scheduler.preempted(), [c]);
    assert_eq!(scheduler.waiting(), &VecDeque::from([hit]));

    drop(clients.receivers.remove(0));
    let readmit = step(&mut scheduler);
    assert_eq!(slots(&readmit)[0], c, "`c` re-enters first");
}

#[test]
fn a_cancelled_request_inside_the_window_retires_even_when_never_the_best_match() {
    let mut scheduler = lpm_scheduler(32, 1, 8);
    let mut clients = Clients::default();
    let cached = shared_prompt(2);
    clients.submit_prompt(&mut scheduler, cached.clone(), 1);
    step(&mut scheduler);

    let cancelled = clients.submit(&mut scheduler, BLOCK_SIZE, 1);
    let hit = clients.submit_prompt(&mut scheduler, cached, 1);
    drop(clients.receivers.remove(1));

    assert_eq!(slots(&step(&mut scheduler)), [hit]);
    assert!(scheduler.request(cancelled).is_none());
    assert!(scheduler.waiting().is_empty());
}
