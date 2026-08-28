//! The step protocol: what crosses between the engine thread and the executor thread.
//!
//! A step command carries everything the executor acts on for one step, built on the engine
//! thread from host-native request state with zero device reads; a step result carries each
//! sampling entry's token back. Both are plain serializable values — no reference counting, no
//! trait objects, no wall-clock types — so the seam survives being moved to a socket.

use serde::{Deserialize, Serialize};

use crate::dispatch::DispatchDecision;
use crate::request::SamplingParams;
use crate::types::{BlockId, RequestId, RequestSlot, SequenceIndex, StepId, TokenCount};

/// One entry of a step command: a sequence, the tokens it computes and where its KV lives.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommandEntry {
    pub request: RequestId,
    pub slot: RequestSlot,
    pub sequence: SequenceIndex,
    /// Tokens the sequence already holds in KV before the step.
    pub context_len: usize,
    /// The tokens the entry computes this step, in position order from `context_len`.
    pub input_tokens: Vec<u32>,
    /// The ordered block ids the sequence's KV occupies, covering the sequence length.
    pub block_table: Vec<BlockId>,
    /// How to sample this entry's next token; `None` for an entry that does not sample.
    pub sampling: Option<SamplingParams>,
}

impl CommandEntry {
    /// Tokens the entry computes this step.
    #[must_use]
    pub fn query_len(&self) -> usize {
        self.input_tokens.len()
    }

    /// Tokens the sequence's KV holds after the step.
    #[must_use]
    pub fn sequence_len(&self) -> usize {
        self.context_len + self.input_tokens.len()
    }

    /// Whether the step samples a token for this entry.
    #[must_use]
    pub fn samples(&self) -> bool {
        self.sampling.is_some()
    }
}

/// Everything the executor acts on for one step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StepCommand {
    pub step: StepId,
    /// The live entries in batch order, then the padding dummies inserted to reach the bucket.
    pub entries: Vec<CommandEntry>,
    /// How many trailing entries are padding dummies.
    pub padding_count: usize,
    /// Which captured graph serves the batch, or why it runs eagerly. Decided here; the
    /// executor never re-derives it.
    pub dispatch: DispatchDecision,
}

impl StepCommand {
    /// The live entries, without the padding dummies.
    #[must_use]
    pub fn live_entries(&self) -> &[CommandEntry] {
        &self.entries[..self.entries.len() - self.padding_count]
    }

    /// Query tokens summed over entries.
    #[must_use]
    pub fn token_count(&self) -> Option<TokenCount> {
        TokenCount::new(self.entries.iter().map(CommandEntry::query_len).sum())
    }

    /// Entries that sample, in order.
    #[must_use]
    pub fn sampling_count(&self) -> usize {
        self.entries.iter().filter(|entry| entry.samples()).count()
    }
}

/// What the executor returns for one step: one token per sampling entry, in entry order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepResult {
    pub step: StepId,
    pub sampled: Vec<u32>,
}

#[cfg(test)]
mod tests {
    use super::{CommandEntry, StepCommand, StepResult};
    use crate::dispatch::{DispatchDecision, EagerReason};
    use crate::request::SamplingParams;
    use crate::test_support::{requests, tokens};
    use crate::types::{BlockId, RequestId, RequestSlot, SequenceIndex, StepId};

    fn eager() -> DispatchDecision {
        DispatchDecision::Eager(EagerReason::RequestsAboveCapturedMaximum {
            request_count: requests(2),
            captured_maximum: requests(1),
        })
    }

    fn entry(context_len: usize, input_tokens: Vec<u32>, samples: bool) -> CommandEntry {
        CommandEntry {
            request: RequestId::new(1),
            slot: RequestSlot::new(0),
            sequence: SequenceIndex::new(0),
            context_len,
            input_tokens,
            block_table: vec![BlockId::new(3), BlockId::new(9)],
            sampling: samples.then(SamplingParams::default),
        }
    }

    #[test]
    fn a_command_counts_its_query_tokens_and_sampling_entries() {
        let command = StepCommand {
            step: StepId::new(4),
            entries: vec![
                entry(8, vec![1, 2, 3], false),
                entry(20, vec![4], true),
                entry(0, vec![0], false),
            ],
            padding_count: 1,
            dispatch: eager(),
        };
        assert_eq!(command.token_count(), Some(tokens(5)));
        assert_eq!(command.sampling_count(), 1);
        assert_eq!(command.live_entries().len(), 2);
        assert_eq!(command.entries[0].query_len(), 3);
        assert_eq!(command.entries[0].sequence_len(), 11);
        assert!(!command.entries[0].samples());
        assert!(command.entries[1].samples());

        let empty = StepCommand {
            step: StepId::new(5),
            entries: Vec::new(),
            padding_count: 0,
            dispatch: eager(),
        };
        assert_eq!(
            empty.token_count(),
            None,
            "an empty command has no live batch"
        );
    }

    #[test]
    fn commands_and_results_round_trip_through_serde() {
        let command = StepCommand {
            step: StepId::new(4),
            entries: vec![entry(0, vec![1, 2], true)],
            padding_count: 0,
            dispatch: eager(),
        };
        let json = serde_json::to_string(&command).unwrap();
        assert_eq!(serde_json::from_str::<StepCommand>(&json).unwrap(), command);

        let result = StepResult {
            step: StepId::new(4),
            sampled: vec![42],
        };
        let json = serde_json::to_string(&result).unwrap();
        assert_eq!(serde_json::from_str::<StepResult>(&json).unwrap(), result);
    }
}
