//! Workload generation.
//!
//! Two workloads, both offered as an open-loop stream (see [`crate::arrival`]):
//!
//! - **ShareGPT-derived** — the conversational shape the plan's gates are written against: short
//!   prompts, short answers, wide length spread. Derived from the public ShareGPT dump, one request
//!   per conversation, the reference answer's length setting `max_tokens`.
//! - **Long-context** — prompts of a target token length, minted from the model's own vocabulary,
//!   which is what moves the pressure from the sampler to the KV cache and attention.
//!
//! Both are reproducible from a seed: at a fixed seed the same run offers the same prompts at the
//! same arrival times, so run-to-run differences are the engine's, not the workload's.

use std::{path::Path, time::Duration};

use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::{
    arrival::PoissonArrivals,
    error::{BenchError, Result},
};

/// One request of a workload, with the offset from run start at which it should be sent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BenchRequest {
    /// Identifies the request in the run's records.
    pub id: String,
    /// The prompt sent to the engine.
    pub prompt: String,
    /// Length of `prompt` measured with the model's tokenizer.
    pub prompt_tokens: usize,
    /// The generation length asked of the engine.
    pub max_tokens: u32,
    /// Offset from the start of the run at which this request is sent.
    pub arrival: Duration,
}

/// The load a run offers: how many requests, how fast, and from which seed.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct LoadPlan {
    /// Number of requests in the run.
    pub num_requests: usize,
    /// Target Poisson arrival rate, in requests per second.
    pub request_rate_per_second: f64,
    /// Seed for prompt selection and arrival times.
    pub seed: u64,
}

/// The model's tokenizer, as the workload needs it: to measure prompt lengths and to mint prompts
/// of a target length.
///
/// A trait rather than a concrete tokenizer so workload generation is testable without a
/// tokenizer file, and so the harness can be pointed at any model's vocabulary.
pub trait Vocabulary {
    /// Number of tokens `text` encodes to.
    ///
    /// # Errors
    ///
    /// Returns [`BenchError::Tokenizer`] if the text cannot be encoded.
    fn count_tokens(&self, text: &str) -> Result<usize>;

    /// Renders token ids back to the text that carries them.
    ///
    /// # Errors
    ///
    /// Returns [`BenchError::Tokenizer`] if the ids cannot be decoded.
    fn decode(&self, token_ids: &[u32]) -> Result<String>;

    /// Number of ids in the vocabulary.
    fn size(&self) -> usize;
}

/// The model's tokenizer, loaded from a `tokenizer.json` file or from a Hugging Face repository.
pub struct HfVocabulary {
    /// The loaded tokenizer.
    tokenizer: tokenizers::Tokenizer,
}

impl HfVocabulary {
    /// Loads a tokenizer from `source`: a path to a `tokenizer.json` if one exists there,
    /// otherwise a Hugging Face model id whose `tokenizer.json` is fetched (and cached) through
    /// the hub.
    ///
    /// # Errors
    ///
    /// Returns [`BenchError::Tokenizer`] if the file cannot be fetched or parsed.
    pub fn load(source: &str) -> Result<Self> {
        let path = if Path::new(source).is_file() {
            std::path::PathBuf::from(source)
        } else {
            hf_hub::api::sync::ApiBuilder::new()
                .build()
                .and_then(|api| api.model(source.to_string()).get("tokenizer.json"))
                .map_err(|error| {
                    BenchError::Tokenizer(format!(
                        "`{source}` is neither a tokenizer file nor a fetchable Hugging Face \
                         model: {error}"
                    ))
                })?
        };

        let tokenizer = tokenizers::Tokenizer::from_file(&path).map_err(|error| {
            BenchError::Tokenizer(format!("Failed to load {}: {error}", path.display()))
        })?;
        Ok(Self { tokenizer })
    }
}

impl Vocabulary for HfVocabulary {
    fn count_tokens(&self, text: &str) -> Result<usize> {
        self.tokenizer
            .encode(text, false)
            .map(|encoding| encoding.len())
            .map_err(|error| BenchError::Tokenizer(format!("Failed to encode a prompt: {error}")))
    }

    fn decode(&self, token_ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(token_ids, false)
            .map_err(|error| BenchError::Tokenizer(format!("Failed to decode a prompt: {error}")))
    }

    fn size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }
}

/// A vocabulary that counts and mints tokens as whitespace-separated words.
///
/// The harness's own tests measure with this rather than a real tokenizer, so neither the unit
/// tests nor the end-to-end tests need a `tokenizer.json` on disk or a Hugging Face fetch.
#[derive(Clone, Copy, Debug, Default)]
pub struct WordVocabulary;

impl Vocabulary for WordVocabulary {
    fn count_tokens(&self, text: &str) -> Result<usize> {
        Ok(text.split_whitespace().count())
    }

    fn decode(&self, token_ids: &[u32]) -> Result<String> {
        Ok(token_ids
            .iter()
            .map(|token_id| format!("w{token_id}"))
            .collect::<Vec<_>>()
            .join(" "))
    }

    fn size(&self) -> usize {
        1_000
    }
}

/// One prompt and the answer it was given, derived from a ShareGPT conversation.
#[derive(Clone, Debug)]
pub struct Conversation {
    /// The first human turn.
    pub prompt: String,
    /// The answer that turn received, whose length becomes the request's `max_tokens`.
    pub completion: String,
}

/// The lengths a ShareGPT conversation must have to be servable as a benchmark request.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct LengthLimits {
    /// Conversations with shorter prompts are dropped; a two-token prompt measures nothing.
    pub min_prompt_tokens: usize,
    /// Conversations with shorter answers are dropped; they never reach steady-state decoding.
    pub min_output_tokens: usize,
    /// Conversations whose prompt plus answer exceed this are dropped rather than truncated.
    pub max_total_tokens: usize,
}

impl Default for LengthLimits {
    fn default() -> Self {
        Self {
            min_prompt_tokens: 4,
            min_output_tokens: 4,
            max_total_tokens: 2_048,
        }
    }
}

/// A long-context workload: prompts of a target token length, minted from the vocabulary.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct LongContextSpec {
    /// Target prompt length, in tokens.
    pub input_tokens: usize,
    /// Generation length asked of the engine.
    pub output_tokens: u32,
    /// Prompt lengths are drawn uniformly from `input_tokens × (1 ± range_ratio)`, so the batch
    /// is not uniform — a batch of identical lengths hides the padding and scheduling costs of a
    /// real one.
    pub range_ratio: f64,
}

/// Which workload a run offers.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WorkloadSpec {
    /// The ShareGPT-derived conversational trace, read from the public dump.
    #[serde(rename = "sharegpt")]
    ShareGpt {
        /// Path to the ShareGPT JSON dump.
        path: std::path::PathBuf,
        /// Lengths a conversation must have to be used.
        #[serde(default)]
        limits: LengthLimits,
    },
    /// Prompts minted at a target length.
    #[serde(rename = "long-context")]
    LongContext(LongContextSpec),
}

impl WorkloadSpec {
    /// The workload's name, as it appears in the results artifact.
    pub fn name(&self) -> &'static str {
        match self {
            Self::ShareGpt { .. } => "sharegpt",
            Self::LongContext(_) => "long-context",
        }
    }

    /// Builds the request stream this workload offers under `plan`.
    ///
    /// # Errors
    ///
    /// Returns [`BenchError::Workload`] if the source data cannot serve the plan, or
    /// [`BenchError::Tokenizer`] if prompts cannot be measured.
    pub fn build(&self, vocabulary: &dyn Vocabulary, plan: &LoadPlan) -> Result<Vec<BenchRequest>> {
        match self {
            Self::ShareGpt { path, limits } => {
                let dump = std::fs::read_to_string(path)
                    .map_err(|error| BenchError::io(path.display(), error))?;
                let conversations = conversations_from_sharegpt(&serde_json::from_str(&dump)?)?;
                sharegpt_requests(&conversations, vocabulary, limits, plan)
            }
            Self::LongContext(spec) => long_context_requests(vocabulary, spec, plan),
        }
    }
}

/// Extracts the answered opening turn of every conversation in a ShareGPT dump.
///
/// Entries without a human turn followed by an answer are skipped: the dump contains truncated and
/// single-turn conversations, and there is nothing to serve in them.
///
/// # Errors
///
/// Returns [`BenchError::Workload`] if `dump` is not a JSON array.
pub fn conversations_from_sharegpt(dump: &serde_json::Value) -> Result<Vec<Conversation>> {
    let entries = dump.as_array().ok_or_else(|| {
        BenchError::Workload("The ShareGPT dump must be a JSON array of conversations".to_string())
    })?;

    Ok(entries.iter().filter_map(opening_exchange).collect())
}

/// The first human turn of a conversation and the answer that follows it, if both are there.
fn opening_exchange(entry: &serde_json::Value) -> Option<Conversation> {
    let turns = entry.get("conversations")?.as_array()?;
    let human = turns
        .iter()
        .position(|turn| turn.get("from").and_then(serde_json::Value::as_str) == Some("human"))?;
    let answer = turns.get(human + 1)?;

    Some(Conversation {
        prompt: turns[human].get("value")?.as_str()?.to_string(),
        completion: answer.get("value")?.as_str()?.to_string(),
    })
}

/// Samples `plan.num_requests` servable conversations and offers them at the plan's arrival rate.
///
/// Sampling is without replacement: a repeated prompt would be answered from the prefix cache and
/// report a TTFT no real conversational stream would see.
///
/// # Errors
///
/// Returns [`BenchError::Workload`] if fewer conversations pass `limits` than the plan needs.
pub fn sharegpt_requests(
    conversations: &[Conversation],
    vocabulary: &dyn Vocabulary,
    limits: &LengthLimits,
    plan: &LoadPlan,
) -> Result<Vec<BenchRequest>> {
    let mut servable = Vec::new();
    for conversation in conversations {
        let prompt_tokens = vocabulary.count_tokens(&conversation.prompt)?;
        let output_tokens = vocabulary.count_tokens(&conversation.completion)?;
        if prompt_tokens >= limits.min_prompt_tokens
            && output_tokens >= limits.min_output_tokens
            && prompt_tokens + output_tokens <= limits.max_total_tokens
        {
            servable.push((conversation, prompt_tokens, output_tokens));
        }
    }

    if servable.len() < plan.num_requests {
        return Err(BenchError::Workload(format!(
            "{} of {} conversations are servable within the length limits, but the run needs {}",
            servable.len(),
            conversations.len(),
            plan.num_requests
        )));
    }

    let mut rng = StdRng::seed_from_u64(plan.seed);
    servable.shuffle(&mut rng);
    servable.truncate(plan.num_requests);

    let arrivals = arrivals(plan)?;
    Ok(servable
        .iter()
        .zip(arrivals)
        .enumerate()
        .map(
            |(index, ((conversation, prompt_tokens, output_tokens), arrival))| BenchRequest {
                id: format!("sharegpt-{index}"),
                prompt: conversation.prompt.clone(),
                prompt_tokens: *prompt_tokens,
                max_tokens: *output_tokens as u32,
                arrival,
            },
        )
        .collect())
}

/// Mints `plan.num_requests` long-context prompts and offers them at the plan's arrival rate.
///
/// Prompts are decoded from token ids drawn uniformly from the vocabulary, so they carry no text
/// the engine could have cached, and their measured length is recorded rather than the requested
/// one — decoding and re-encoding is not always the identity.
///
/// # Errors
///
/// Returns [`BenchError::Workload`] if the spec asks for an impossible length, or
/// [`BenchError::Tokenizer`] if prompts cannot be minted.
pub fn long_context_requests(
    vocabulary: &dyn Vocabulary,
    spec: &LongContextSpec,
    plan: &LoadPlan,
) -> Result<Vec<BenchRequest>> {
    let (shortest, longest) = spec.length_range()?;
    let mut rng = StdRng::seed_from_u64(plan.seed);
    let arrivals = arrivals(plan)?;
    let mut requests = Vec::with_capacity(plan.num_requests);

    for (index, arrival) in arrivals.into_iter().enumerate() {
        let num_tokens = rng.random_range(shortest..=longest);
        let token_ids = (0..num_tokens)
            .map(|_| rng.random_range(0..vocabulary.size() as u32))
            .collect::<Vec<_>>();
        let prompt = vocabulary.decode(&token_ids)?;

        requests.push(BenchRequest {
            prompt_tokens: vocabulary.count_tokens(&prompt)?,
            id: format!("long-context-{index}"),
            prompt,
            max_tokens: spec.output_tokens,
            arrival,
        });
    }

    Ok(requests)
}

impl LongContextSpec {
    /// The inclusive range of prompt lengths this spec draws from.
    fn length_range(&self) -> Result<(usize, usize)> {
        if self.input_tokens == 0 {
            return Err(BenchError::Workload(
                "A long-context workload needs a non-zero prompt length".to_string(),
            ));
        }
        if !(0.0..1.0).contains(&self.range_ratio) {
            return Err(BenchError::Workload(format!(
                "The long-context range ratio must be in [0, 1), got {}",
                self.range_ratio
            )));
        }

        let spread = self.input_tokens as f64 * self.range_ratio;
        let shortest = ((self.input_tokens as f64 - spread).round() as usize).max(1);
        let longest = (self.input_tokens as f64 + spread).round() as usize;
        Ok((shortest, longest))
    }
}

/// The arrival offsets of a plan's requests.
fn arrivals(plan: &LoadPlan) -> Result<Vec<Duration>> {
    Ok(PoissonArrivals::new(plan.request_rate_per_second, plan.seed)?.offsets(plan.num_requests))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sharegpt_json(conversations: &[(&str, &str)]) -> serde_json::Value {
        let entries = conversations
            .iter()
            .enumerate()
            .map(|(index, (prompt, completion))| {
                serde_json::json!({
                    "id": format!("conversation-{index}"),
                    "conversations": [
                        { "from": "human", "value": prompt },
                        { "from": "gpt", "value": completion },
                    ],
                })
            })
            .collect::<Vec<_>>();
        serde_json::Value::Array(entries)
    }

    /// A prompt of `count` words, distinct per conversation so sampling is observable.
    fn words(tag: usize, count: usize) -> String {
        (0..count)
            .map(|index| format!("t{tag}x{index}"))
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn load_plan(count: usize) -> LoadPlan {
        LoadPlan {
            num_requests: count,
            request_rate_per_second: 100.0,
            seed: 11,
        }
    }

    #[test]
    fn test_sharegpt_conversations_are_the_first_human_turn_and_its_answer() {
        let json = serde_json::json!([{
            "conversations": [
                { "from": "human", "value": "first ask" },
                { "from": "gpt", "value": "first answer" },
                { "from": "human", "value": "second ask" },
                { "from": "gpt", "value": "second answer" },
            ],
        }]);

        let conversations = conversations_from_sharegpt(&json).expect("Failed to parse ShareGPT");

        assert_eq!(conversations.len(), 1);
        assert_eq!(conversations[0].prompt, "first ask");
        assert_eq!(conversations[0].completion, "first answer");
    }

    #[test]
    fn test_sharegpt_entries_without_an_answered_human_turn_are_skipped() {
        let json = serde_json::json!([
            { "conversations": [{ "from": "human", "value": "unanswered" }] },
            { "conversations": [{ "from": "gpt", "value": "no question" }] },
            { "conversations": [] },
            { "no_conversations_key": true },
            {
                "conversations": [
                    { "from": "human", "value": "kept" },
                    { "from": "gpt", "value": "answer" },
                ],
            },
        ]);

        let conversations = conversations_from_sharegpt(&json).expect("Failed to parse ShareGPT");

        assert_eq!(conversations.len(), 1);
        assert_eq!(conversations[0].prompt, "kept");
    }

    #[test]
    fn test_sharegpt_rejects_data_that_is_not_a_list_of_conversations() {
        let error = conversations_from_sharegpt(&serde_json::json!({ "conversations": [] }))
            .expect_err("An object is not a ShareGPT dataset");

        assert!(matches!(error, BenchError::Workload(_)), "{error}");
    }

    /// Prompts shorter than the floor, and conversations whose prompt plus answer exceed the
    /// context budget, cannot be served as written and are dropped rather than truncated.
    #[test]
    fn test_sharegpt_requests_drop_conversations_outside_the_length_limits() {
        let json = sharegpt_json(&[
            ("too short", &words(0, 8)),
            (&words(1, 40), &words(1, 40)),
            (&words(2, 900), &words(2, 400)),
        ]);
        let conversations = conversations_from_sharegpt(&json).expect("Failed to parse ShareGPT");
        let limits = LengthLimits {
            min_prompt_tokens: 4,
            min_output_tokens: 4,
            max_total_tokens: 1_024,
        };

        let requests = sharegpt_requests(&conversations, &WordVocabulary, &limits, &load_plan(1))
            .expect("Failed to build the ShareGPT workload");

        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].prompt_tokens, 40);
        assert_eq!(requests[0].max_tokens, 40);
    }

    #[test]
    fn test_sharegpt_requests_are_sampled_without_replacement() {
        let json = sharegpt_json(
            &(0..32)
                .map(|index| (words(index, 20), words(index, 10)))
                .collect::<Vec<_>>()
                .iter()
                .map(|(prompt, completion)| (prompt.as_str(), completion.as_str()))
                .collect::<Vec<_>>(),
        );
        let conversations = conversations_from_sharegpt(&json).expect("Failed to parse ShareGPT");

        let requests = sharegpt_requests(
            &conversations,
            &WordVocabulary,
            &LengthLimits::default(),
            &load_plan(16),
        )
        .expect("Failed to build the ShareGPT workload");

        let distinct = requests
            .iter()
            .map(|request| request.prompt.as_str())
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(distinct.len(), 16, "a prompt must not be offered twice");
    }

    /// Re-serving a prompt would be answered from the prefix cache, so a pool too small for the
    /// run is an error rather than a silent repeat.
    #[test]
    fn test_sharegpt_rejects_a_pool_smaller_than_the_run() {
        let json = sharegpt_json(&[(&words(0, 20), &words(0, 10))]);
        let conversations = conversations_from_sharegpt(&json).expect("Failed to parse ShareGPT");

        let error = sharegpt_requests(
            &conversations,
            &WordVocabulary,
            &LengthLimits::default(),
            &load_plan(4),
        )
        .expect_err("One usable conversation cannot serve four requests");

        assert!(matches!(error, BenchError::Workload(_)), "{error}");
    }

    #[test]
    fn test_long_context_prompts_land_inside_the_requested_range() {
        let spec = LongContextSpec {
            input_tokens: 4_000,
            output_tokens: 128,
            range_ratio: 0.1,
        };

        let requests = long_context_requests(&WordVocabulary, &spec, &load_plan(32))
            .expect("Failed to build the long-context workload");

        assert_eq!(requests.len(), 32);
        for request in requests.iter() {
            assert!(
                (3_600..=4_400).contains(&request.prompt_tokens),
                "prompt of {} tokens is outside the ±10% range of 4000",
                request.prompt_tokens
            );
            assert_eq!(request.max_tokens, 128);
        }
    }

    /// The recorded prompt length is the measured one: sampled token ids are decoded to text, and
    /// what the engine re-encodes is what the results have to report.
    #[test]
    fn test_long_context_prompt_tokens_are_measured_not_requested() {
        let spec = LongContextSpec {
            input_tokens: 512,
            output_tokens: 16,
            range_ratio: 0.0,
        };

        let requests = long_context_requests(&WordVocabulary, &spec, &load_plan(4))
            .expect("Failed to build the long-context workload");

        for request in requests.iter() {
            assert_eq!(
                request.prompt_tokens,
                WordVocabulary
                    .count_tokens(&request.prompt)
                    .expect("Failed to count tokens"),
            );
        }
    }

    #[test]
    fn test_long_context_requests_are_reproducible_from_the_seed() {
        let spec = LongContextSpec {
            input_tokens: 256,
            output_tokens: 16,
            range_ratio: 0.2,
        };

        let first = long_context_requests(&WordVocabulary, &spec, &load_plan(8))
            .expect("Failed to build the long-context workload");
        let again = long_context_requests(&WordVocabulary, &spec, &load_plan(8))
            .expect("Failed to build the long-context workload");
        let other_seed = long_context_requests(
            &WordVocabulary,
            &spec,
            &LoadPlan {
                seed: 12,
                ..load_plan(8)
            },
        )
        .expect("Failed to build the long-context workload");

        assert_eq!(first, again);
        assert_ne!(first, other_seed);
    }

    #[test]
    fn test_requests_arrive_over_time_at_the_requested_rate() {
        let spec = LongContextSpec {
            input_tokens: 64,
            output_tokens: 8,
            range_ratio: 0.0,
        };
        let plan = LoadPlan {
            num_requests: 512,
            request_rate_per_second: 64.0,
            seed: 3,
        };

        let requests = long_context_requests(&WordVocabulary, &spec, &plan)
            .expect("Failed to build the long-context workload");

        assert!(requests
            .windows(2)
            .all(|pair| pair[0].arrival <= pair[1].arrival));
        let span = requests.last().expect("No requests").arrival.as_secs_f64();
        assert!(
            (span - 8.0).abs() < 2.0,
            "512 requests at 64/s should span about 8 s, spanned {span} s"
        );
    }

    #[test]
    fn test_request_ids_are_unique() {
        let requests = long_context_requests(
            &WordVocabulary,
            &LongContextSpec {
                input_tokens: 32,
                output_tokens: 4,
                range_ratio: 0.0,
            },
            &load_plan(64),
        )
        .expect("Failed to build the long-context workload");

        let ids = requests
            .iter()
            .map(|request| request.id.as_str())
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(ids.len(), 64);
    }
}
