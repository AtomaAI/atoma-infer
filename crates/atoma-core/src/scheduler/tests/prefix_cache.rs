//! The prefix cache: what a finished request leaves behind, what a new request starts past, and
//! what is evicted before anyone is preempted.

use super::{events, scheduler, shared_prompt, slots, step, Clients, BLOCK_SIZE};
use crate::request::{FinishReason, RequestEvent, Usage};
use crate::test_support::tokens;

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
        scheduler.live_request_count(),
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

    assert_eq!(scheduler.live_request_count(), 0);
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
