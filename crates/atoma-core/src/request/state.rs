//! Request and sequence state as the engine thread holds it: host-native, one owner, no locks.

use crate::kv::{BlockLease, ExtraKeys, HashAlgorithm};
use crate::request::{
    Egress, EgressSender, Priority, RequestEvent, RequestPhase, SamplingParams, StopCriteria,
    Usage, Waiting,
};
use crate::types::{BlockHash, BlockId, RequestId, StepId, TokenCount};

/// The one token a padding dummy computes every step it is inserted into.
pub const PADDING_TOKEN: u32 = 0;

/// A request as its client submits it: one prompt, one set of sampling parameters, one priority,
/// one egress sink. Everything the engine needs to give it a slot.
#[derive(Debug)]
pub struct NewRequest {
    pub prompt: Vec<u32>,
    pub sampling: SamplingParams,
    pub stop: StopCriteria,
    /// How urgently the request admits. [`Priority::default()`] is the lowest there is, and asks
    /// for arrival order.
    pub priority: Priority,
    pub egress: EgressSender,
}

/// One token stream inside a request, with its own computed count and block table.
///
/// Prompt and generated tokens share one buffer, so a chunk of computation is always a
/// contiguous slice of it — including what a preempted request computes again, which spans both.
#[derive(Debug)]
pub struct Sequence {
    tokens: Vec<u32>,
    prompt_len: usize,
    /// Tokens whose KV is resident. Resets to zero on preemption.
    computed: usize,
    /// The ordered block ids the sequence's KV occupies. Host-native: a step command is built
    /// from it with no device read.
    pub(crate) block_table: Vec<BlockId>,
    /// The leases behind the blocks this sequence obtained fresh from the pool. Lease `i`
    /// backs `block_table[hits + i]`.
    pub(crate) leases: Vec<BlockLease>,
    /// Leading block-table entries found in the prefix index at admission rather than leased.
    pub(crate) hits: usize,
    /// The chain hash of every full block of `tokens`, in block order. Identity only: it
    /// survives preemption, since the tokens do.
    pub(crate) chain: Vec<BlockHash>,
    /// How many leading entries of `chain` this sequence pins in the prefix index.
    pub(crate) pinned: usize,
}

impl Sequence {
    fn from_prompt(prompt: Vec<u32>) -> Self {
        let prompt_len = prompt.len();
        Self {
            tokens: prompt,
            prompt_len,
            computed: 0,
            block_table: Vec::new(),
            leases: Vec::new(),
            hits: 0,
            chain: Vec::new(),
            pinned: 0,
        }
    }

    /// Extends the chain with the hash of every full block of `tokens` not yet hashed.
    pub(crate) fn extend_chain(&mut self, algorithm: HashAlgorithm, block_size: TokenCount) {
        let block_size = block_size.get();
        let full_blocks = self.tokens.len() / block_size;
        for block in self.chain.len()..full_blocks {
            let run = &self.tokens[block * block_size..(block + 1) * block_size];
            let parent = block.checked_sub(1).map(|parent| self.chain[parent]);
            self.chain
                .push(algorithm.hash_run(parent, run, ExtraKeys::none()));
        }
    }

    /// The chain hashes a prefix lookup may claim: every full block except the one holding the
    /// last token, so at least one token is always computed and a logit exists to sample from.
    #[must_use]
    pub(crate) fn hashable_prefix(&self, block_size: TokenCount) -> &[BlockHash] {
        let blocks = (self.tokens.len().saturating_sub(1) / block_size.get()).min(self.chain.len());
        &self.chain[..blocks]
    }

    /// The ordered block ids the sequence's KV occupies.
    #[must_use]
    pub fn block_table(&self) -> &[BlockId] {
        &self.block_table
    }

    /// Prompt tokens plus every token generated so far.
    #[must_use]
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Prompt tokens plus every token generated so far, as a count.
    #[must_use]
    pub fn total(&self) -> usize {
        self.tokens.len()
    }

    #[must_use]
    pub fn prompt_len(&self) -> usize {
        self.prompt_len
    }

    #[must_use]
    pub fn generated_count(&self) -> usize {
        self.tokens.len() - self.prompt_len
    }

    /// Tokens whose KV is resident.
    #[must_use]
    pub fn computed(&self) -> usize {
        self.computed
    }

    /// Tokens still to compute before the sequence can sample again.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.tokens.len() - self.computed
    }

    /// A running sequence whose computed count is below its prompt length.
    #[must_use]
    pub fn is_prefilling(&self) -> bool {
        self.computed < self.prompt_len
    }

    /// A running sequence whose computed count has reached its prompt length.
    #[must_use]
    pub fn is_decoding(&self) -> bool {
        self.computed >= self.prompt_len
    }

    /// Records that `query_len` more tokens have resident KV.
    ///
    /// # Panics
    ///
    /// Panics when the advance passes the total: a step cannot compute tokens that do not exist.
    pub fn advance(&mut self, query_len: usize) {
        assert!(
            self.computed + query_len <= self.tokens.len(),
            "advance of {query_len} past total {} from computed {}",
            self.tokens.len(),
            self.computed
        );
        self.computed += query_len;
    }

    /// Appends one sampled token.
    pub fn push_token(&mut self, token: u32) {
        self.tokens.push(token);
    }

    /// Forgets that any token has resident KV, so the sequence computes from its first token
    /// again when it next runs.
    pub fn forget_computed(&mut self) {
        self.computed = 0;
    }
}

/// A request as the engine thread holds it.
#[derive(Debug)]
pub struct Request {
    id: RequestId,
    phase: RequestPhase,
    sampling: SamplingParams,
    stop: StopCriteria,
    /// How urgently the request admits; an admission policy that reads it offers the highest
    /// first.
    priority: Priority,
    /// Where this request's output goes: to its client, or nowhere at all for a padding dummy.
    egress: Egress,
    /// Born with one; forking adds more.
    sequences: Vec<Sequence>,
}

impl Request {
    /// Takes `new` in as `id`, Waiting since `arrived_at`, with its one sequence.
    #[must_use]
    pub fn new(id: RequestId, new: NewRequest, arrived_at: StepId) -> Self {
        let NewRequest {
            prompt,
            sampling,
            stop,
            priority,
            egress,
        } = new;
        Self {
            id,
            phase: RequestPhase::Waiting(Waiting::new(arrived_at)),
            sampling,
            stop,
            priority,
            egress: Egress::Client(egress),
            sequences: vec![Sequence::from_prompt(prompt)],
        }
    }

    /// A padding dummy: one sequence of one token over `block`, held for the process lifetime.
    /// It never finishes and never enters admission, and it has no client.
    #[must_use]
    pub fn padding(id: RequestId, block: BlockId) -> Self {
        let mut sequence = Sequence::from_prompt(vec![PADDING_TOKEN]);
        sequence.block_table.push(block);
        Self {
            id,
            phase: RequestPhase::Padding,
            sampling: SamplingParams::default(),
            stop: StopCriteria {
                max_new_tokens: TokenCount::ONE,
                ignore_eos: true,
            },
            priority: Priority::default(),
            egress: Egress::Dummy,
            sequences: vec![sequence],
        }
    }

    /// Whether this is a padding dummy.
    #[must_use]
    pub fn is_padding(&self) -> bool {
        matches!(self.phase, RequestPhase::Padding)
    }

    #[must_use]
    pub fn id(&self) -> RequestId {
        self.id
    }

    #[must_use]
    pub fn phase(&self) -> RequestPhase {
        self.phase
    }

    /// Moves the request into `phase`. Only a value a legal transition produced can be passed,
    /// and only the scheduler that owns the request can pass one.
    pub(crate) fn set_phase(&mut self, phase: RequestPhase) {
        self.phase = phase;
    }

    #[must_use]
    pub fn sampling(&self) -> SamplingParams {
        self.sampling
    }

    #[must_use]
    pub fn stop(&self) -> StopCriteria {
        self.stop
    }

    /// How urgently the request admits, as its client submitted it.
    #[must_use]
    pub fn priority(&self) -> Priority {
        self.priority
    }

    /// Sends `event` to the client.
    ///
    /// # Panics
    ///
    /// Panics in debug builds when the request is a padding dummy: a dummy has no client, it
    /// never finishes and its one token is nobody's, so an event reaching it is a bug in the
    /// caller rather than something to drop quietly.
    pub fn send(&self, event: RequestEvent) {
        match &self.egress {
            Egress::Client(client) => client.send(event),
            Egress::Dummy => {
                debug_assert!(false, "a padding dummy has no client to send {event:?} to");
            }
        }
    }

    #[must_use]
    pub fn sequences(&self) -> &[Sequence] {
        &self.sequences
    }

    pub fn sequences_mut(&mut self) -> &mut [Sequence] {
        &mut self.sequences
    }

    /// Whether the client dropped its egress receiver. A dummy has no client to lose.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        match &self.egress {
            Egress::Client(client) => client.is_cancelled(),
            Egress::Dummy => false,
        }
    }

    /// Events the client has not read yet. A dummy has no client, so none.
    #[must_use]
    pub fn backlog(&self) -> usize {
        match &self.egress {
            Egress::Client(client) => client.backlog(),
            Egress::Dummy => 0,
        }
    }

    /// Token accounting over every sequence.
    #[must_use]
    pub fn usage(&self) -> Usage {
        Usage {
            prompt_tokens: self.sequences.first().map_or(0, Sequence::prompt_len),
            generated_tokens: self.sequences.iter().map(Sequence::generated_count).sum(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{NewRequest, Request, PADDING_TOKEN};
    use crate::request::{
        egress, EgressReceiver, Priority, RequestEvent, RequestPhase, SamplingParams, StopCriteria,
        Usage,
    };
    use crate::test_support::tokens;
    use crate::types::{BlockId, RequestId, SequenceIndex, StepId};

    /// A request over `prompt` with its client's receiver, which the caller keeps alive.
    fn request(prompt: &[u32]) -> (Request, EgressReceiver) {
        prioritised_request(prompt, Priority::default())
    }

    /// The same, submitted at `priority`.
    fn prioritised_request(prompt: &[u32], priority: Priority) -> (Request, EgressReceiver) {
        let (sender, receiver) = egress();
        let request = Request::new(
            RequestId::new(1),
            NewRequest {
                prompt: prompt.to_vec(),
                sampling: SamplingParams::default(),
                stop: StopCriteria {
                    max_new_tokens: tokens(4),
                    ignore_eos: false,
                },
                priority,
                egress: sender,
            },
            StepId::new(3),
        );
        (request, receiver)
    }

    #[test]
    fn a_new_request_waits_with_one_uncomputed_sequence() {
        let (request, _receiver) = request(&[1, 2, 3]);
        assert!(
            matches!(request.phase(), RequestPhase::Waiting(w) if w.arrived_at() == StepId::new(3))
        );
        assert_eq!(request.sequences().len(), 1, "born with one sequence");
        let sequence = &request.sequences()[0];
        assert_eq!(sequence.total(), 3);
        assert_eq!(sequence.computed(), 0);
        assert_eq!(sequence.remaining(), 3);
        assert!(sequence.is_prefilling());
        assert!(!sequence.is_decoding());
        assert_eq!(
            request.usage(),
            Usage {
                prompt_tokens: 3,
                generated_tokens: 0
            }
        );
    }

    #[test]
    fn a_request_carries_the_priority_its_client_submitted() {
        let (submitted, _receiver) = prioritised_request(&[1, 2], Priority::new(4));
        assert_eq!(submitted.priority(), Priority::new(4));

        let (unprioritised, _receiver) = request(&[1, 2]);
        assert_eq!(
            unprioritised.priority(),
            Priority::default(),
            "asking for nothing leaves the request at the lowest priority"
        );

        let dummy = Request::padding(RequestId::new(9), BlockId::new(0));
        assert_eq!(
            dummy.priority(),
            Priority::default(),
            "a dummy never enters admission, so it outranks nobody"
        );
    }

    #[test]
    fn prefilling_and_decoding_are_derived_from_computed_against_prompt_length() {
        let (mut request, _receiver) = request(&[1, 2, 3, 4]);
        let sequence = &mut request.sequences_mut()[0];

        sequence.advance(2);
        assert!(
            sequence.is_prefilling(),
            "a chunk short of the prompt is still prefilling"
        );
        assert_eq!(sequence.remaining(), 2);

        sequence.advance(2);
        sequence.push_token(9);
        assert!(sequence.is_decoding());
        assert_eq!(sequence.total(), 5);
        assert_eq!(
            sequence.remaining(),
            1,
            "the sampled token is the next query"
        );
        assert_eq!(sequence.generated_count(), 1);
        assert_eq!(sequence.tokens(), &[1, 2, 3, 4, 9]);

        sequence.forget_computed();
        assert_eq!(sequence.computed(), 0);
        assert!(
            sequence.is_prefilling(),
            "a preempted request recomputes from the start"
        );
        assert_eq!(
            sequence.remaining(),
            5,
            "prompt and generated tokens both recompute"
        );
    }

    #[test]
    #[should_panic(expected = "past total")]
    fn advancing_past_the_total_is_a_caller_bug() {
        let (mut request, _receiver) = request(&[1, 2]);
        request.sequences_mut()[0].advance(3);
    }

    #[test]
    fn a_padding_dummy_holds_its_block_with_no_client_and_never_cancels() {
        let dummy = Request::padding(RequestId::new(9), BlockId::new(4));
        assert!(dummy.is_padding());
        assert_eq!(dummy.phase(), RequestPhase::Padding);
        assert!(!dummy.is_cancelled(), "it has no client to lose");
        let sequence = &dummy.sequences()[0];
        assert_eq!(sequence.tokens(), &[PADDING_TOKEN]);
        assert_eq!(sequence.block_table(), &[BlockId::new(4)]);
        assert_eq!(sequence.computed(), 0);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "no client to send")]
    fn sending_an_event_to_a_padding_dummy_is_a_caller_bug() {
        let dummy = Request::padding(RequestId::new(9), BlockId::new(4));
        dummy.send(RequestEvent::Token {
            request: RequestId::new(9),
            sequence: SequenceIndex::new(0),
            token: 1,
        });
    }

    #[test]
    fn a_dropped_receiver_cancels_the_request() {
        let (request, receiver) = request(&[1]);
        assert!(!request.is_cancelled());
        drop(receiver);
        assert!(request.is_cancelled());
    }
}
