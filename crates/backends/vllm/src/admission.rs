//! Admission: the step between a validated request and the `SequenceGroup` the engine schedules.
//!
//! Nothing here runs in a build without the `cuda` feature — `LlmService`, its only production
//! caller, is compiled out — but the module still builds and its tests still run, which is what
//! keeps the sequence id invariant under test on a host with no GPU.
#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

use std::time::Instant;

use crate::{
    error::LlmServiceError,
    sampling::logits_processor,
    sequence::{Sequence, SequenceGroup, SequenceIdCounter},
    validation::ValidGenerateRequest,
};

/// Turns validated requests into the `SequenceGroup`s the engine schedules.
///
/// This owns the sequence id counter because those ids key the block manager's block tables: two
/// live sequences sharing an id share one block table and corrupt each other's output, so no
/// caller gets to choose an id.
pub(crate) struct RequestAdmitter {
    /// Source of the sequence ids handed to newly admitted requests.
    sequence_id_counter: SequenceIdCounter,
    /// Number of tokens a KV block holds, which fixes how a sequence is split into blocks.
    block_size: usize,
}

impl RequestAdmitter {
    pub(crate) fn new(block_size: usize) -> Self {
        Self {
            sequence_id_counter: SequenceIdCounter::default(),
            block_size,
        }
    }

    /// Admits an already validated request, giving it a sequence id no other live request holds.
    pub(crate) fn admit(
        &mut self,
        request_id: String,
        valid_request: &ValidGenerateRequest,
        arrival_time: Instant,
    ) -> Result<SequenceGroup, LlmServiceError> {
        let sequence = Sequence::new(
            self.sequence_id_counter.next_id(),
            valid_request.inputs.clone(),
            valid_request.encoding.get_ids().to_vec(),
            self.block_size,
            valid_request.return_full_text,
        )?;

        Ok(SequenceGroup::new(
            request_id,
            vec![sequence],
            arrival_time,
            valid_request.parameters.clone(),
            valid_request.stopping_parameters.clone(),
            logits_processor(&valid_request.parameters),
        )?)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::tests::fixtures::valid_request;

    const BLOCK_SIZE: usize = 4;
    const MAX_NEW_TOKENS: u32 = 16;

    #[test]
    fn test_concurrent_requests_get_distinct_sequence_ids() {
        const NUM_REQUESTS: usize = 32;

        let mut admitter = RequestAdmitter::new(BLOCK_SIZE);
        let mut sequence_ids = HashSet::with_capacity(NUM_REQUESTS);

        for i in 0..NUM_REQUESTS {
            let request_id = format!("request-{i}");
            let request = valid_request(&request_id, 8, MAX_NEW_TOKENS);
            let sequence_group = admitter
                .admit(request_id, &request, Instant::now())
                .expect("Failed to admit request");

            sequence_ids.extend(sequence_group.sequences.keys().copied());
        }

        assert_eq!(
            sequence_ids.len(),
            NUM_REQUESTS,
            "each request must own its sequence id, or its block table is shared"
        );
    }

    #[test]
    fn test_admitted_request_carries_its_own_prompt() {
        let request = valid_request("request-3", 6, MAX_NEW_TOKENS);

        let sequence_group = RequestAdmitter::new(BLOCK_SIZE)
            .admit("request-3".to_string(), &request, Instant::now())
            .expect("Failed to admit request");

        assert_eq!(sequence_group.request_id, "request-3");
        assert_eq!(sequence_group.prompt(), "prompt of request-3");
        assert_eq!(sequence_group.sequences.len(), 1);
        assert_eq!(
            sequence_group.prompt_token_ids(),
            request.encoding.get_ids().to_vec()
        );
    }
}
