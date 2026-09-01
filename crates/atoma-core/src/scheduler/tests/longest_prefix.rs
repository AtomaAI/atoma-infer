//! The longest-prefix-match policy: which request in the window admits first, and what the
//! window hides from it.

use std::collections::VecDeque;

use super::{lpm_scheduler, shared_prompt, slots, step, Clients, BLOCK_SIZE};

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
