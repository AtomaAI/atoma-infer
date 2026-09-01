//! What a request asks for beyond its prompt: how to sample, when to stop, and how urgently it
//! admits.

use serde::{Deserialize, Serialize};

use crate::types::TokenCount;

/// How the executor samples a request's next token. One set per request, copied into every
/// sampling entry of a step command.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SamplingParams {
    /// Scales the logits; higher spreads the distribution.
    pub temperature: f32,
    /// Keeps only the `top_k` most likely tokens; zero disables the filter.
    pub top_k: u32,
    /// Keeps the smallest set of tokens whose probabilities sum to at least `top_p`.
    pub top_p: f32,
    /// Keeps tokens whose probability is close to a uniform distribution's expectation.
    pub typical_p: f32,
    /// Whether to sample at all, or to take the most likely token.
    pub do_sample: bool,
    /// Seed for reproducible sampling.
    pub seed: u64,
    /// Multiplies down the logits of tokens already generated; one is no penalty.
    pub repetition_penalty: f32,
    /// How many recent tokens the repetition penalty considers.
    pub repeat_last_n: u32,
    /// Subtracts from a token's logit in proportion to how often it appeared; zero is no penalty.
    pub frequency_penalty: f32,
}

impl Default for SamplingParams {
    /// Greedy: the most likely token every step, no filtering, no penalty.
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            typical_p: 1.0,
            do_sample: false,
            seed: 0,
            repetition_penalty: 1.0,
            repeat_last_n: 0,
            frequency_penalty: 0.0,
        }
    }
}

/// When a request stops generating. Stop strings need detokenization and live outside the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StopCriteria {
    /// The most tokens the request generates.
    pub max_new_tokens: TokenCount,
    /// Whether an end-of-sequence token is generated through rather than stopped on.
    pub ignore_eos: bool,
}

/// How urgently a request admits: the higher, the sooner an admission policy that reads it
/// offers the request. The default is the lowest there is, so traffic that asks for nothing
/// shares one priority and is ordered by arrival alone.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Priority(u32);

impl Priority {
    #[must_use]
    pub const fn new(priority: u32) -> Self {
        Self(priority)
    }
}

#[cfg(test)]
mod tests {
    use super::{Priority, SamplingParams, StopCriteria};
    use crate::test_support::tokens;

    #[test]
    fn default_sampling_is_greedy_with_no_filter_and_no_penalty() {
        assert_eq!(
            SamplingParams::default(),
            SamplingParams {
                temperature: 1.0,
                top_k: 0,
                top_p: 1.0,
                typical_p: 1.0,
                do_sample: false,
                seed: 0,
                repetition_penalty: 1.0,
                repeat_last_n: 0,
                frequency_penalty: 0.0,
            }
        );
    }

    #[test]
    fn params_round_trip_through_serde_and_reject_unknown_fields() {
        let params = SamplingParams {
            temperature: 0.7,
            top_k: 40,
            do_sample: true,
            seed: 9,
            ..SamplingParams::default()
        };
        let json = serde_json::to_string(&params).unwrap();
        assert_eq!(
            serde_json::from_str::<SamplingParams>(&json).unwrap(),
            params
        );
        assert!(serde_json::from_str::<SamplingParams>(r#"{"temperature": 1.0}"#).is_err());

        let stop = StopCriteria {
            max_new_tokens: tokens(16),
            ignore_eos: true,
        };
        let json = serde_json::to_string(&stop).unwrap();
        assert_eq!(serde_json::from_str::<StopCriteria>(&json).unwrap(), stop);
        assert!(
            serde_json::from_str::<StopCriteria>(r#"{"max_new_tokens": 0, "ignore_eos": false}"#)
                .is_err(),
            "a request must be allowed at least one token"
        );
    }

    #[test]
    fn the_default_priority_is_the_lowest_and_higher_values_outrank_it() {
        assert_eq!(Priority::default(), Priority::new(0));
        assert!(Priority::new(1) > Priority::default());
        assert!(Priority::new(7) > Priority::new(1));
    }

    #[test]
    fn priority_serializes_as_the_bare_integer() {
        assert_eq!(serde_json::to_string(&Priority::new(5)).unwrap(), "5");
        assert_eq!(
            serde_json::from_str::<Priority>("5").unwrap(),
            Priority::new(5)
        );
    }
}
