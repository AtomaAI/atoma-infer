//! What one pass budgets: which requests share a step, the token budget and request cap that
//! bound it, chunked prefill, the block tables it covers, and the step-result protocol applying
//! it demands.

use std::collections::VecDeque;

use super::{scheduler, slots, step, Clients, BLOCK_SIZE};
use crate::test_support::tokens;
use crate::types::StepId;

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
