//! A closed admission and a wholesale retirement: what still runs, what re-enters, and what is
//! told it will never run.

use std::collections::VecDeque;

use super::{finish_reason, scheduler, slots, step, Clients, BLOCK_SIZE};
use crate::request::FinishReason;

#[test]
fn closing_admission_lets_running_requests_finish_and_admits_nothing_more() {
    let mut scheduler = scheduler(8, 100, 64);
    let mut clients = Clients::default();
    let running = clients.submit(&mut scheduler, BLOCK_SIZE, 2);
    assert_eq!(slots(&step(&mut scheduler)), [running]);
    let waiting = clients.submit(&mut scheduler, BLOCK_SIZE, 2);

    assert!(scheduler.is_admission_open());
    scheduler.close_admission();
    assert!(!scheduler.is_admission_open());
    let scheduled = step(&mut scheduler);
    assert_eq!(
        slots(&scheduled),
        [running],
        "the running request decodes on"
    );
    assert_eq!(scheduler.waiting(), &VecDeque::from([waiting]));
    assert!(
        scheduler.running().is_empty(),
        "and finishes on its second token"
    );
    assert!(step(&mut scheduler).is_empty(), "nothing is admitted after");
}

/// A closed admission refuses only requests that have never run. A preempted request is a
/// running request the pool displaced, so it re-enters and finishes; without that it would hold
/// a live slot with no way back into Running until shutdown.
#[test]
fn a_closed_admission_still_re_admits_preempted_requests() {
    let mut scheduler = scheduler(2, 64, 2);
    let mut clients = Clients::default();
    let a = clients.submit(&mut scheduler, BLOCK_SIZE, 16);
    let b = clients.submit(&mut scheduler, BLOCK_SIZE, 16);
    assert_eq!(slots(&step(&mut scheduler)), [a, b]);
    assert_eq!(
        step(&mut scheduler).preempted,
        [b],
        "the pool cannot grow both"
    );

    scheduler.close_admission();
    let never_ran = clients.submit(&mut scheduler, BLOCK_SIZE, 16);
    drop(clients.receivers.remove(0));

    let readmit = step(&mut scheduler);
    assert_eq!(slots(&readmit), [b], "the preempted request re-enters");
    assert_eq!(readmit.entries[0].context_len, 0, "and recomputes");
    assert_eq!(scheduler.running(), [b]);
    assert!(scheduler.preempted().is_empty());
    assert_eq!(
        scheduler.waiting(),
        &VecDeque::from([never_ran]),
        "what never ran stays waiting"
    );
}

/// The cancel sweep runs before admission does, so a client that hangs up in Waiting or in the
/// preempted stack is retired even while a drain admits nothing new.
#[test]
fn cancelled_requests_retire_while_admission_is_closed() {
    let mut scheduler = scheduler(2, 64, 2);
    let mut clients = Clients::default();
    let running = clients.submit(&mut scheduler, BLOCK_SIZE, 16);
    let preempted = clients.submit(&mut scheduler, BLOCK_SIZE, 16);
    step(&mut scheduler);
    assert_eq!(step(&mut scheduler).preempted, [preempted]);
    let waiting = clients.submit(&mut scheduler, BLOCK_SIZE, 16);
    scheduler.close_admission();

    clients.receivers.remove(2);
    clients.receivers.remove(1);
    step(&mut scheduler);

    assert!(
        scheduler.request(preempted).is_none(),
        "cancelled in the stack"
    );
    assert!(
        scheduler.request(waiting).is_none(),
        "cancelled in the queue"
    );
    assert!(scheduler.preempted().is_empty());
    assert!(scheduler.waiting().is_empty());
    assert_eq!(scheduler.running(), [running], "the live one runs on");
    assert_eq!(scheduler.live_request_count(), 1);
}

#[test]
fn retiring_everything_tells_every_client_and_keeps_the_dummies() {
    let mut scheduler = scheduler(8, 100, 64);
    let mut clients = Clients::default();
    clients.submit(&mut scheduler, BLOCK_SIZE, 16);
    step(&mut scheduler);
    clients.submit(&mut scheduler, BLOCK_SIZE, 16);

    scheduler.retire_all(FinishReason::Shutdown);
    assert_eq!(scheduler.live_request_count(), 0);
    assert!(scheduler.running().is_empty() && scheduler.waiting().is_empty());
    assert_eq!(scheduler.pool().available(), 8);
    for receiver in &clients.receivers {
        assert_eq!(finish_reason(receiver), Some(FinishReason::Shutdown));
    }
}
