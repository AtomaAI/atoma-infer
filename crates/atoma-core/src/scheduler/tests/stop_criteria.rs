//! The criteria that finish a request: an end-of-sequence token, and the maximum model length.

use super::{
    finish_reason, new_request, scheduler, slots, step, step_sampling, Clients, BLOCK_SIZE, EOS,
};
use crate::request::{egress, FinishReason, NewRequest, StopCriteria};
use crate::test_support::tokens;

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
