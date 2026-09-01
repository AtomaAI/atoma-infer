//! Intake: the order requests queue in, the prompts that are refused on the spot, and the
//! prompt the pool cannot hold yet.

use std::collections::VecDeque;

use super::{config, new_request, scheduler, slots, Clients, BLOCK_SIZE};
use crate::kv::BlockPool;
use crate::request::{egress, FinishReason, RequestEvent, RequestPhase, Usage};
use crate::scheduler::{Scheduler, SchedulerError};

/// Port of `test_scheduler_add_sequence_group`.
#[test]
fn intake_queues_requests_in_arrival_order() {
    let mut scheduler = scheduler(8, 100, 64);
    let mut clients = Clients::default();
    let mut submitted = Vec::new();
    for i in 0..4 {
        submitted.push(clients.submit(&mut scheduler, BLOCK_SIZE, 16));
        assert_eq!(scheduler.live_request_count(), i + 1);
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
/// Port of `test_prefill_schedule_max_prompt_len`: a prompt that leaves no room to generate
/// finishes at intake, and its client hears why. An empty prompt likewise.
#[test]
fn a_prompt_that_cannot_fit_the_model_finishes_at_intake() {
    let mut scheduler = scheduler(8, 1000, 64);
    let max_model_length = 8 * BLOCK_SIZE;

    let (sender, receiver) = egress();
    assert!(
        scheduler
            .intake(new_request(max_model_length, 16, sender))
            .is_none(),
        "a prompt that leaves no room to generate never waits"
    );
    let RequestEvent::Finished { reason, usage, .. } = receiver.recv().unwrap() else {
        panic!("intake finishes a prompt it can never serve");
    };
    assert_eq!(
        reason,
        FinishReason::PromptExceedsMaxModelLength {
            prompt_tokens: max_model_length,
            max_model_length,
        }
    );
    assert_eq!(
        usage,
        Usage {
            prompt_tokens: max_model_length,
            generated_tokens: 0
        }
    );
    assert!(receiver.recv().is_err(), "nothing follows the finish");
    assert_eq!(
        scheduler.live_request_count(),
        0,
        "a rejected request takes no slot"
    );

    let (sender, receiver) = egress();
    assert!(scheduler.intake(new_request(0, 16, sender)).is_none());
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
        scheduler.live_request_count(),
        1,
        "one short of the maximum fits"
    );
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
