//! Building a step command from request state, with zero device reads.
//!
//! Every length, token and block id the executor needs is host-native in the request slab, so
//! the command is a pure function of a scheduling pass over that state. Nothing here can touch
//! a device: the crate links no driver.

use crate::scheduler::{Scheduled, Scheduler};
use crate::step::{CommandEntry, StepCommand};

/// The command for `scheduled`, read entirely from the scheduler's request state.
///
/// # Panics
///
/// Panics when an entry's block table does not cover its sequence length: the scheduler grew
/// every table before it scheduled the entry, so a gap is a scheduler bug, not a runtime state.
#[must_use]
pub fn build_command(scheduler: &Scheduler, scheduled: &Scheduled) -> StepCommand {
    let block_size = scheduler.config().block_size.get();
    let entries = scheduled
        .entries
        .iter()
        .map(|entry| {
            let request = scheduler
                .request(entry.slot)
                .expect("a scheduled slot is live until its result is applied");
            let sequence = &request.sequences()[entry.sequence.get() as usize];
            let sequence_len = entry.sequence_len();
            assert!(
                sequence.block_table().len() * block_size >= sequence_len,
                "block table of {} blocks does not cover {sequence_len} tokens",
                sequence.block_table().len()
            );
            CommandEntry {
                request: request.id(),
                slot: entry.slot,
                sequence: entry.sequence,
                context_len: entry.context_len,
                input_tokens: sequence.tokens()[entry.context_len..sequence_len].to_vec(),
                block_table: sequence.block_table().to_vec(),
                sampling: entry.samples.then(|| request.sampling()),
            }
        })
        .collect();
    StepCommand {
        step: scheduled.step,
        entries,
    }
}

#[cfg(test)]
mod tests {
    use super::build_command;
    use crate::kv::{BlockPool, HashAlgorithm};
    use crate::request::{egress, EgressReceiver, NewRequest, SamplingParams, StopCriteria};
    use crate::scheduler::{AdmissionPolicy, Scheduled, Scheduler, SchedulerConfig};
    use crate::test_support::{requests, tokens};
    use crate::types::{RequestSlot, StepId};

    const BLOCK_SIZE: usize = 4;

    fn scheduler(token_budget: usize) -> Scheduler {
        Scheduler::new(
            SchedulerConfig {
                token_budget: tokens(token_budget),
                max_batch: requests(8),
                max_model_len: tokens(32),
                block_size: tokens(BLOCK_SIZE),
                window: requests(8),
                admission: AdmissionPolicy::Fcfs,
                max_requests: requests(8),
                eos_token_ids: Vec::new(),
                hash_algorithm: HashAlgorithm::Sha256V1,
            },
            BlockPool::new(8),
        )
        .unwrap()
    }

    fn submit(
        scheduler: &mut Scheduler,
        prompt: Vec<u32>,
        temperature: f32,
    ) -> (RequestSlot, EgressReceiver) {
        let (sender, receiver) = egress();
        let slot = scheduler
            .intake(NewRequest {
                prompt,
                sampling: SamplingParams {
                    temperature,
                    ..SamplingParams::default()
                },
                stop: StopCriteria {
                    max_new_tokens: tokens(8),
                    ignore_eos: false,
                },
                egress: sender,
            })
            .unwrap();
        (slot, receiver)
    }

    fn apply_ones(scheduler: &mut Scheduler, scheduled: &Scheduled) {
        let sampled = vec![1; scheduled.sampling_entries().count()];
        scheduler.apply(scheduled, &sampled);
    }

    #[test]
    fn the_command_mirrors_the_pass_with_host_native_lengths_tokens_and_tables() {
        let mut scheduler = scheduler(100);
        let (first, _a) = submit(&mut scheduler, vec![10, 11, 12, 13, 14], 0.5);
        let (second, _b) = submit(&mut scheduler, vec![20, 21], 0.9);

        let scheduled = scheduler.schedule();
        let command = build_command(&scheduler, &scheduled);
        assert_eq!(command.step, scheduled.step);
        assert_eq!(command.entries.len(), 2);

        let entry = &command.entries[0];
        assert_eq!(entry.slot, first);
        assert_eq!(entry.context_len, 0);
        assert_eq!(entry.input_tokens, [10, 11, 12, 13, 14]);
        assert_eq!(entry.sequence_len(), 5);
        assert_eq!(entry.block_table.len(), 2, "five tokens need two blocks");
        assert_eq!(
            entry.block_table,
            scheduler.request(first).unwrap().sequences()[0].block_table()
        );
        assert_eq!(entry.sampling.map(|s| s.temperature), Some(0.5));

        let entry = &command.entries[1];
        assert_eq!(entry.slot, second);
        assert_eq!(entry.input_tokens, [20, 21]);
        assert_eq!(entry.block_table.len(), 1);
        assert_eq!(entry.sampling.map(|s| s.temperature), Some(0.9));
        assert_eq!(command.sampling_count(), 2);
        apply_ones(&mut scheduler, &scheduled);

        // Decoding: the one input token is the token just sampled, at the context's end.
        let scheduled = scheduler.schedule();
        let command = build_command(&scheduler, &scheduled);
        assert_eq!(command.entries[0].context_len, 5);
        assert_eq!(command.entries[0].input_tokens, [1]);
        assert_eq!(command.entries[0].sequence_len(), 6);
        assert_eq!(command.entries[1].context_len, 2);
        assert_eq!(command.entries[1].input_tokens, [1]);
        apply_ones(&mut scheduler, &scheduled);
    }

    #[test]
    fn a_non_final_prefill_chunk_carries_its_slice_and_no_sampling() {
        let mut scheduler = scheduler(3);
        let (_, _client) = submit(&mut scheduler, vec![10, 11, 12, 13, 14], 0.5);

        let scheduled = scheduler.schedule();
        let command = build_command(&scheduler, &scheduled);
        assert_eq!(command.entries[0].input_tokens, [10, 11, 12]);
        assert_eq!(command.entries[0].sampling, None);
        assert_eq!(command.sampling_count(), 0);
        apply_ones(&mut scheduler, &scheduled);

        let scheduled = scheduler.schedule();
        let command = build_command(&scheduler, &scheduled);
        assert_eq!(command.entries[0].context_len, 3);
        assert_eq!(command.entries[0].input_tokens, [13, 14]);
        assert!(command.entries[0].samples());
        apply_ones(&mut scheduler, &scheduled);
    }

    #[test]
    fn an_empty_pass_builds_an_empty_command() {
        let scheduler = scheduler(100);
        let scheduled = Scheduled {
            step: StepId::new(1),
            entries: Vec::new(),
            preempted: Vec::new(),
        };
        let command = build_command(&scheduler, &scheduled);
        assert!(command.entries.is_empty());
        assert_eq!(command.token_count(), None);
    }
}
