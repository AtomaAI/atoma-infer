//! The output of one scheduling pass: which sequences run, how many tokens each computes, and
//! which entries sample — indices and counts, never copied request state.

use crate::dispatch::LiveBatch;
use crate::types::{RequestCount, RequestSlot, SequenceIndex, StepId, TokenCount};

/// One row of a [`Scheduled`]: a sequence, what it computes this step, and whether it samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Entry {
    pub(crate) slot: RequestSlot,
    pub(crate) sequence: SequenceIndex,
    /// Tokens the sequence already holds in KV before the step.
    pub(crate) context_len: usize,
    /// Tokens the entry computes this step.
    pub(crate) query_len: TokenCount,
    /// Whether the step samples a token for this entry: only when the query reaches the
    /// sequence's total, so a non-final prefill chunk never does.
    pub(crate) samples: bool,
}

impl Entry {
    /// Tokens the sequence's KV holds after the step.
    #[must_use]
    pub(crate) fn sequence_len(&self) -> usize {
        self.context_len + self.query_len.get()
    }
}

/// The output of one scheduling pass: indices and counts, never copied request state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Scheduled {
    pub(crate) step: StepId,
    pub(crate) entries: Vec<Entry>,
    /// Requests this pass displaced from Running, most recent last.
    pub(crate) preempted: Vec<RequestSlot>,
}

impl Scheduled {
    #[must_use]
    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Query tokens summed over entries.
    #[must_use]
    pub(crate) fn token_count(&self) -> usize {
        self.entries.iter().map(|entry| entry.query_len.get()).sum()
    }

    /// Whether every entry has query length one: the condition full-graph replay requires.
    #[must_use]
    pub(crate) fn is_uniform_decode(&self) -> bool {
        !self.entries.is_empty() && self.entries.iter().all(|entry| entry.query_len.get() == 1)
    }

    /// Entries that sample, in order.
    pub(crate) fn sampling_entries(&self) -> impl Iterator<Item = &Entry> {
        self.entries.iter().filter(|entry| entry.samples)
    }

    /// Live requests in the batch: entries address sequences, and a request's sequences sit
    /// together in batch order.
    #[must_use]
    pub(crate) fn request_count(&self) -> usize {
        self.entries
            .iter()
            .fold((0, None), |(count, last), entry| {
                if last == Some(entry.slot) {
                    (count, last)
                } else {
                    (count + 1, Some(entry.slot))
                }
            })
            .0
    }

    /// The shape of this pass before padding, or `None` when nothing was scheduled.
    #[must_use]
    pub(crate) fn live_batch(&self) -> Option<LiveBatch> {
        Some(LiveBatch {
            token_count: TokenCount::new(self.token_count())?,
            request_count: RequestCount::new(self.request_count())?,
            uniform_decode: self.is_uniform_decode(),
        })
    }
}
