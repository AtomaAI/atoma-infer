//! Clients that hang up or stop reading: the sweep that retires them from every queue before
//! any budget is spent, and the backlog cap that decides when a silent reader counts as lost.

use super::{
    backlog_scheduler, events, finish_reason, scheduler, slots, step, step_sampling, Clients,
    BLOCK_SIZE,
};
use crate::request::{FinishReason, RequestEvent};

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
    assert_eq!(scheduler.live_request_count(), 1);
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
    assert_eq!(scheduler.live_request_count(), 2);
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
    assert_eq!(scheduler.live_request_count(), 4);
    clients.receivers.clear();
    let scheduled = step(&mut scheduler);
    assert!(scheduled.is_empty());
    assert_eq!(scheduler.live_request_count(), 0);
    assert_eq!(scheduler.pool().available(), 8);
}

/// The backlog sweep truncates a stalled client's stream; it never punctures it. Every token
/// generated before the retire is still queued, in order, with the finish behind them.
#[test]
fn a_stalled_client_keeps_every_token_generated_before_its_retire() {
    let mut scheduler = backlog_scheduler(8, 3);
    let mut clients = Clients::default();
    let baseline = scheduler.pool().available();
    let stalled = clients.submit(&mut scheduler, BLOCK_SIZE, 16);

    // The client reads nothing, so one sampled token a pass piles up behind it.
    for token in 1..=4 {
        assert_eq!(slots(&step_sampling(&mut scheduler, token)), [stalled]);
    }
    assert!(
        step_sampling(&mut scheduler, 5).is_empty(),
        "the sweep retires it before the pass spends anything"
    );
    assert!(scheduler.running().is_empty());
    assert_eq!(
        scheduler.pool().available(),
        baseline,
        "its blocks came back"
    );

    let received = events(&clients.receivers[0]);
    let (sampled, finish) = received.split_at(received.len() - 1);
    let tokens: Vec<u32> = sampled
        .iter()
        .map(|event| match event {
            RequestEvent::Token { token, .. } => *token,
            RequestEvent::Finished { .. } => unreachable!("the finish is the last event"),
        })
        .collect();
    assert_eq!(tokens, [1, 2, 3, 4], "in order, with no gap");
    assert!(matches!(
        finish,
        [RequestEvent::Finished {
            reason: FinishReason::ClientBacklogged {
                queued: 4,
                max_backlog: 3
            },
            ..
        }]
    ));
}

#[test]
fn a_client_exactly_at_the_backlog_limit_keeps_running() {
    let mut scheduler = backlog_scheduler(8, 3);
    let mut clients = Clients::default();
    let slot = clients.submit(&mut scheduler, BLOCK_SIZE, 16);
    for token in 1..=3 {
        assert_eq!(slots(&step_sampling(&mut scheduler, token)), [slot]);
    }

    assert_eq!(
        scheduler.request(slot).unwrap().backlog(),
        3,
        "at the limit"
    );
    assert_eq!(
        slots(&step_sampling(&mut scheduler, 4)),
        [slot],
        "the limit is what a client may leave unread, not what retires it"
    );
    assert!(finish_reason(&clients.receivers[0]).is_none());
}

#[test]
fn a_client_that_reads_every_pass_never_backlogs() {
    let mut scheduler = backlog_scheduler(8, 3);
    let mut clients = Clients::default();
    let slot = clients.submit(&mut scheduler, BLOCK_SIZE, 16);
    for token in 1..=8 {
        assert_eq!(slots(&step_sampling(&mut scheduler, token)), [slot]);
        assert_eq!(events(&clients.receivers[0]).len(), 1, "one token a pass");
    }
    assert_eq!(scheduler.running(), [slot]);
}

#[test]
fn a_client_that_backlogs_and_then_hangs_up_retires_once() {
    let mut scheduler = backlog_scheduler(8, 3);
    let mut clients = Clients::default();
    let baseline = scheduler.pool().available();
    let slot = clients.submit(&mut scheduler, BLOCK_SIZE, 16);
    for token in 1..=4 {
        step_sampling(&mut scheduler, token);
    }

    drop(clients.receivers.remove(0));
    assert!(step(&mut scheduler).is_empty());
    assert!(scheduler.running().is_empty());
    assert_eq!(scheduler.live_request_count(), 0);
    assert_eq!(scheduler.pool().available(), baseline);
    assert!(
        !scheduler.waiting().contains(&slot),
        "the slot is gone from every queue"
    );
}
