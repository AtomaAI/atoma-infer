//! The priority policy: which request in the window admits first, and what the window hides
//! from it.

use std::collections::VecDeque;

use super::{priority_scheduler, slots, step, Clients, BLOCK_SIZE};
use crate::request::Priority;

/// Three prompts submitted lowest priority first: admission takes the highest in the window each
/// pass, and the unprioritised one last.
#[test]
fn priority_admits_the_highest_in_the_window_first() {
    let mut scheduler = priority_scheduler(32, 1, 8);
    let mut clients = Clients::default();
    let none = clients.submit(&mut scheduler, BLOCK_SIZE, 1);
    let low = clients.submit_at_priority(&mut scheduler, BLOCK_SIZE, 1, Priority::new(1));
    let high = clients.submit_at_priority(&mut scheduler, BLOCK_SIZE, 1, Priority::new(9));

    assert_eq!(slots(&step(&mut scheduler)), [high]);
    assert_eq!(slots(&step(&mut scheduler)), [low]);
    assert_eq!(slots(&step(&mut scheduler)), [none]);
}

/// Equal priorities admit in arrival order, behind everything that outranks them. The two
/// unprioritised requests arrive first and admit last, so the pass is ordered by priority and not
/// by arrival — and they admit between themselves in arrival order, which is what leaves traffic
/// at the default first-come-first-served.
#[test]
fn priority_breaks_ties_by_arrival() {
    let mut scheduler = priority_scheduler(32, 1, 8);
    let mut clients = Clients::default();
    let first = clients.submit(&mut scheduler, BLOCK_SIZE, 1);
    let second = clients.submit(&mut scheduler, BLOCK_SIZE, 1);
    let earlier = clients.submit_at_priority(&mut scheduler, BLOCK_SIZE, 1, Priority::new(5));
    let later = clients.submit_at_priority(&mut scheduler, BLOCK_SIZE, 1, Priority::new(5));

    assert_eq!(slots(&step(&mut scheduler)), [earlier], "priority first");
    assert_eq!(
        slots(&step(&mut scheduler)),
        [later],
        "then its tie, by arrival"
    );
    assert_eq!(slots(&step(&mut scheduler)), [first]);
    assert_eq!(slots(&step(&mut scheduler)), [second]);
}

/// The policy sees only the window: a higher priority behind it waits until it enters.
#[test]
fn priority_examines_only_the_window() {
    let mut scheduler = priority_scheduler(32, 1, 2);
    let mut clients = Clients::default();
    let first = clients.submit(&mut scheduler, BLOCK_SIZE, 1);
    let second = clients.submit(&mut scheduler, BLOCK_SIZE, 1);
    let urgent = clients.submit_at_priority(&mut scheduler, BLOCK_SIZE, 1, Priority::new(9));

    assert_eq!(
        slots(&step(&mut scheduler)),
        [first],
        "the urgent request is outside the window"
    );
    assert_eq!(slots(&step(&mut scheduler)), [urgent], "now it is inside");
    assert_eq!(slots(&step(&mut scheduler)), [second]);
}

/// A preempted request re-enters before the policy orders the window, even when a waiting request
/// outranks it: it is a running request the pool displaced, not a candidate to be ordered.
#[test]
fn preempted_requests_re_enter_before_priority_orders_the_window() {
    let mut scheduler = priority_scheduler(4, 64, 8);
    let mut clients = Clients::default();
    let b = clients.submit(&mut scheduler, 2 * BLOCK_SIZE, 16);
    let c = clients.submit(&mut scheduler, 2 * BLOCK_SIZE, 16);
    assert_eq!(slots(&step(&mut scheduler)), [b, c]);

    // `b` needs a third block; nothing is evictable, so `c` is displaced.
    let decode = step(&mut scheduler);
    assert_eq!(slots(&decode), [b]);
    assert_eq!(decode.preempted, [c]);

    // An urgent request arrives behind the displaced one; `c` is still ahead of it.
    let urgent = clients.submit_at_priority(&mut scheduler, BLOCK_SIZE, 16, Priority::new(9));
    let readmit = step(&mut scheduler);
    assert_eq!(
        slots(&readmit),
        [b],
        "`c` cannot fit yet, so nobody admits behind it"
    );
    assert_eq!(scheduler.preempted(), [c]);
    assert_eq!(scheduler.waiting(), &VecDeque::from([urgent]));

    drop(clients.receivers.remove(0));
    let readmit = step(&mut scheduler);
    assert_eq!(
        slots(&readmit),
        [c, urgent],
        "`c` re-enters ahead of the request that outranks it"
    );
}
