//! The executor: one pinned thread per rank, owning a device and acting on the engine thread's
//! step commands.
//!
//! The engine thread decides everything about a step on the host and hands the executor a step
//! command over a ring; the executor runs the model forward for it, samples the tokens the command
//! asks for, and hands a step result back. It re-derives nothing the command already settled.

pub mod batch;
pub mod config;
pub mod logits;
pub mod sampler;

#[cfg(test)]
pub(crate) mod test_support {
    use atoma_core::dispatch::{DispatchDecision, EagerReason};
    use atoma_core::request::{SamplingParams, PADDING_TOKEN};
    use atoma_core::step::{CommandEntry, StepCommand};
    use atoma_core::types::{BlockId, RequestCount, RequestId, RequestSlot, SequenceIndex, StepId};

    /// An entry for `request` in the slot of the same number, sampling under the default
    /// parameters when `samples`.
    pub(crate) fn entry(
        request: u64,
        context_len: usize,
        input_tokens: Vec<u32>,
        blocks: &[u32],
        samples: bool,
    ) -> CommandEntry {
        let slot = u32::try_from(request).expect("test request numbers fit u32");
        CommandEntry {
            request: RequestId::new(request),
            slot: RequestSlot::new(slot),
            sequence: SequenceIndex::new(0),
            context_len,
            input_tokens,
            block_table: blocks.iter().map(|&block| BlockId::new(block)).collect(),
            sampling: samples.then(SamplingParams::default),
        }
    }

    /// A one-token decode entry for `request` in `slot`, sampling under `params`.
    pub(crate) fn sampling_entry(request: u64, slot: u32, params: SamplingParams) -> CommandEntry {
        CommandEntry {
            slot: RequestSlot::new(slot),
            sampling: Some(params),
            ..entry(request, 0, vec![1], &[10], true)
        }
    }

    /// A padding dummy's entry: one token over its own block, never sampling.
    pub(crate) fn dummy(request: u64, block: u32) -> CommandEntry {
        entry(request, 0, vec![PADDING_TOKEN], &[block], false)
    }

    /// An eager step command over `entries`, the last `padding_count` of which are dummies.
    pub(crate) fn command(entries: Vec<CommandEntry>, padding_count: usize) -> StepCommand {
        StepCommand {
            step: StepId::new(1),
            entries,
            padding_count,
            dispatch: DispatchDecision::Eager(EagerReason::RequestsAboveCapturedMaximum {
                request_count: RequestCount::new(1).expect("nonzero"),
                captured_maximum: RequestCount::new(1).expect("nonzero"),
            }),
        }
    }
}
