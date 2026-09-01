//! Preemption: a running request the pool cannot grow displaces the most recently admitted one,
//! which releases its KV and computes again when it re-enters.

use std::collections::VecDeque;

use super::{scheduler, slots, step, Clients, BLOCK_SIZE};
use crate::request::RequestPhase;
use crate::test_support::tokens;

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
    assert!(
        entry.samples,
        "computing again reaches the total and samples"
    );
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
