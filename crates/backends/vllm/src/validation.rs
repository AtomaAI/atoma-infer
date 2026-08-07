use thiserror::Error;
use tokenizers::Encoding;
use tokio::sync::{mpsc, oneshot};
use tracing::{error, info_span, instrument, trace, Span};

use crate::{
    tokenizer::{EncodeTokenizerRequest, TokenizerError},
    types::{GenerateParameters, GenerateRequest},
};

/// `Validation` - Responsible for validating `Request`/`Response` parameters
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
#[derive(Clone, Debug)]
pub struct Validation {
    /// Maximum number of sequences to generate for ranking (currently unused)
    #[allow(dead_code)]
    best_of: usize,
    /// Maximum number of stop sequences allowed in a request
    max_stop_sequences: usize,
    /// Maximum number of top tokens to return in the response
    max_top_n_tokens: u32,
    /// Maximum allowed length of the input text in tokens
    max_input_length: usize,
    /// Maximum total number of tokens (input + generated) allowed
    max_total_tokens: u32,
    /// Channel to send tokenization requests to the background tokenizer task
    sender: mpsc::UnboundedSender<EncodeTokenizerRequest>,
    /// Tracing span for logging and diagnostics
    span: Span,
}

impl Validation {
    /// Constructor
    pub fn new(
        best_of: usize,
        max_stop_sequences: usize,
        max_top_n_tokens: u32,
        max_input_length: usize,
        max_total_tokens: u32,
        sender: mpsc::UnboundedSender<EncodeTokenizerRequest>,
    ) -> Self {
        Self {
            best_of,
            max_stop_sequences,
            max_top_n_tokens,
            max_input_length,
            max_total_tokens,
            span: info_span!("validation"),
            sender,
        }
    }

    /// Tokenize the input string
    ///
    /// # Arguments
    ///
    /// * `input` - The input string to tokenize
    /// * `truncate` - Optional maximum number of tokens to keep
    ///
    /// # Returns
    ///
    /// A tuple containing the `Encoding` and the potentially truncated input string
    ///
    /// # Errors
    ///
    /// Returns a `ValidationError` if tokenization fails
    #[instrument(skip_all)]
    pub async fn tokenize(
        &self,
        input: String,
        truncate: Option<usize>,
    ) -> Result<(Encoding, String), ValidationError> {
        let _enter = self.span.enter();
        trace!("Tokenizing input: {input}");
        // Response channel
        let (response_sender, response_receiver) = oneshot::channel();
        let request = EncodeTokenizerRequest {
            input,
            truncate,
            sender: response_sender,
            span: Span::current(),
        };
        // Send request to the background tokenization task
        self.sender.send(request).unwrap(); // DON'T PANIC: safe to unwrap here

        let response = response_receiver.await.unwrap();
        Ok(response?)
    }

    /// Validates the input of a received `Request`.
    ///
    /// # Arguments
    ///
    /// * `input` - The input string to be validated and tokenized.
    /// * `truncate` - Optional maximum number of tokens to keep.
    /// * `max_new_tokens` - Optional maximum number of new tokens to generate.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// * The validated input string
    /// * The tokenized encoding of the input
    /// * The maximum number of new tokens to generate
    ///
    /// # Errors
    ///
    /// Returns a `ValidationError` if:
    /// * Tokenization fails
    /// * The total number of tokens (input + new) exceeds the maximum allowed
    /// * The input length exceeds the maximum allowed
    #[instrument(skip_all)]
    async fn validate_input(
        &self,
        input: String,
        truncate: Option<usize>,
        max_new_tokens: Option<u32>,
    ) -> Result<(String, Encoding, u32), ValidationError> {
        let _enter = self.span.enter();
        let (encoding, input) = self.tokenize(input.clone(), truncate).await?;
        let input_len = if let Some(truncate) = truncate {
            std::cmp::min(truncate, encoding.len())
        } else {
            encoding.len()
        };

        if input_len == 0 {
            // TODO: handle the case in which input length == 0
        }

        // Get total number of tokens
        // NOTE: we assume `input_len < 2^32`
        let max_new_tokens = if let Some(max_new_tokens) = max_new_tokens {
            max_new_tokens
        } else {
            self.max_total_tokens.saturating_sub(input_len as u32)
        };
        let total_tokens = input_len as u32 + max_new_tokens;

        // Validate `total_tokens`
        if total_tokens > self.max_total_tokens {
            error!("Max total tokens exceeded by request's total number of tokens ({total_tokens} > {})", self.max_total_tokens);
            return Err(ValidationError::MaxTotalTokens(
                self.max_total_tokens,
                input_len,
                max_new_tokens,
            ));
        }

        // Validate `input_len`
        if input_len > self.max_input_length {
            error!(
                "Input length exceeded by request's input length ({input_len} > {})",
                self.max_input_length
            );
            return Err(ValidationError::InputLength(
                self.max_input_length,
                input_len,
            ));
        }

        let histogram = metrics::histogram!("atoma-vllm_input_length");
        histogram.record(input_len as f64);

        Ok((input, encoding, max_new_tokens))
    }

    /// Validates a payload and gets the number of tokens in the input
    ///
    /// # Arguments
    ///
    /// * `request` - The `GenerateRequest` to validate
    ///
    /// # Returns
    ///
    /// A `Result` containing either a `ValidGenerateRequest` or a `ValidationError`
    ///
    /// # Errors
    ///
    /// This function will return an error if any of the request parameters are invalid,
    /// including but not limited to:
    /// - Invalid sampling parameters (temperature, top_k, top_p, etc.)
    /// - Invalid token limits (max_new_tokens, truncate)
    /// - Empty input
    /// - Too many stop sequences
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    #[instrument(skip_all)]
    pub(crate) async fn validate(
        &self,
        request: GenerateRequest,
    ) -> Result<ValidGenerateRequest, ValidationError> {
        let _enter = self.span.enter();

        let GenerateParameters {
            best_of,
            temperature,
            repetition_penalty,
            frequency_penalty,
            top_k,
            top_p,
            typical_p,
            do_sample,
            max_new_tokens,
            stop: stop_sequences,
            truncate,
            random_seed,
            decoder_input_details,
            top_n_tokens,
            n,
            return_full_text,
            repeat_last_n,
        } = request.parameters;

        // sampling must be true when best_of > 1
        let best_of = best_of.unwrap_or(1);
        let sampling = do_sample
            || temperature.is_some()
            || top_k.is_some()
            || top_p.is_some()
            || typical_p.is_some();

        if best_of > 1 && !sampling {
            error!("Best of is only supported with sampling");
            return Err(ValidationError::BestOfSampling);
        }

        // `temperature == 0` is the OpenAI convention for greedy decoding, which `sampling` maps to
        // `Sampling::ArgMax`. Rejecting it would refuse every deterministic request.
        let temperature = temperature.unwrap_or(1.0);
        if temperature < 0.0 {
            error!("Temperature must not be negative");
            return Err(ValidationError::Temperature);
        }

        let repetition_penalty = repetition_penalty.unwrap_or(1.0);
        if repetition_penalty <= 0.0 {
            error!("Repetition penalty must be greater than 0");
            return Err(ValidationError::RepetitionPenalty);
        }

        let frequency_penalty = frequency_penalty.unwrap_or(0.0);
        if !(-2.0..=2.0).contains(&frequency_penalty) {
            error!("Frequency penalty must be between -2.0 and 2.0");
            return Err(ValidationError::FrequencyPenalty);
        }

        let top_p = top_p
            .map(|value| {
                if value <= 0.0 || value >= 1.0 {
                    error!("Top p must be between 0.0 and 1.0");
                    return Err(ValidationError::TopP);
                }
                Ok(value)
            })
            .unwrap_or(Ok(1.0))?;

        let typical_p = typical_p
            .map(|value| {
                if value <= 0.0 || value >= 1.0 {
                    error!("Typical p must be between 0.0 and 1.0");
                    return Err(ValidationError::TypicalP);
                }
                Ok(value)
            })
            .unwrap_or(Ok(1.0))?;

        let top_k = top_k
            .map(|value| {
                if value == 0 {
                    error!("Top k must be greater than 0");
                    return Err(ValidationError::TopK);
                }
                Ok(value)
            })
            .unwrap_or(Ok(0))?;

        if max_new_tokens == Some(0) {
            error!("Max new tokens must be greater than 0");
            return Err(ValidationError::NegativeMaxNewTokens);
        }

        if stop_sequences.len() > self.max_stop_sequences {
            error!(
                "Stop sequences exceeded by request's stop sequences ({} > {})",
                stop_sequences.len(),
                self.max_stop_sequences
            );
            return Err(ValidationError::StopSequence(
                self.max_stop_sequences,
                stop_sequences.len(),
            ));
        }

        let random_seed = match random_seed {
            None => fresh_random_seed(),
            Some(seed) => {
                if best_of > 1 {
                    error!("Best of is not supported with sampling");
                    return Err(ValidationError::BestOfSampling);
                }
                seed
            }
        };

        let top_n_tokens = top_n_tokens
            .map(|value| {
                if value > self.max_top_n_tokens {
                    error!("`Validation` instance top n tokens exceeded by request's top n tokens ({value} > {})", self.max_top_n_tokens);
                    return Err(ValidationError::TopNTokens(self.max_top_n_tokens, value));
                }
                Ok(value)
            })
            .unwrap_or(Ok(0))?;

        let repeat_last_n = repeat_last_n.unwrap_or(0);

        // Check if inputs is empty
        if request.inputs.is_empty() {
            error!("Empty input");
            return Err(ValidationError::EmptyInput);
        }

        // Check if truncate is strictly positive and less than max_input_length
        let truncate = truncate
            .map(|value| {
                if value == 0 || value > self.max_input_length {
                    error!(
                        "Truncate exceeded by request's truncate ({value} > {})",
                        self.max_input_length
                    );
                    return Err(ValidationError::Truncate(self.max_input_length, value));
                }
                Ok(Some(value))
            })
            .unwrap_or(Ok(None))?;

        // Validate inputs
        let (inputs, encoding, max_new_tokens) = self
            .validate_input(request.inputs, truncate, max_new_tokens)
            .await?;

        let parameters = NextTokenChooserParameters {
            temperature,
            repetition_penalty,
            best_of,
            frequency_penalty,
            top_k,
            top_p,
            typical_p,
            do_sample,
            random_seed,
            repeat_last_n,
            n,
        };
        let stopping_parameters = StoppingCriteriaParameters {
            max_new_tokens,
            stop_sequences,
            ignore_eos_token: false,
        };

        let histogram = metrics::histogram!("tgi_request_max_new_tokens");
        histogram.record(max_new_tokens as f64);

        let input_token_len = encoding.len();
        Ok(ValidGenerateRequest {
            request_id: request.request_id,
            inputs,
            decoder_input_details,
            encoding,
            input_token_len,
            truncate: truncate.unwrap_or(self.max_input_length) as u32,
            parameters,
            stopping_parameters,
            top_n_tokens,
            return_full_text: return_full_text.unwrap_or(false),
        })
    }
}

/// Draws the sampling seed for a request that did not pin one.
///
/// Every unseeded request has to get its own seed: sharing one makes the whole node decode the
/// same continuation for the same prompt, which is a correctness bug for callers who asked for
/// sampling rather than a reproducibility feature.
fn fresh_random_seed() -> u64 {
    rand::random()
}

/// `ValidGenerateRequest` - A validated and processed version of a `GenerateRequest`.
///
/// This struct is created after input validation has taken place, ensuring that all
/// parameters are within acceptable ranges and the input is properly tokenized.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
#[derive(Clone, Debug)]
pub(crate) struct ValidGenerateRequest {
    /// Unique identifier for the request
    #[allow(dead_code)]
    pub request_id: String,
    /// The input text to be processed
    pub inputs: String,
    /// Tokenized representation of the input
    pub encoding: Encoding,
    /// Number of tokens in the input
    #[allow(dead_code)]
    pub input_token_len: usize,
    /// Maximum number of tokens to consider from the input
    #[allow(dead_code)]
    pub truncate: u32,
    /// Flag to include detailed information about decoder input tokens
    #[allow(dead_code)]
    pub decoder_input_details: bool,
    /// Parameters for the next token selection algorithm
    pub parameters: NextTokenChooserParameters,
    /// Criteria for when to stop generating tokens
    pub stopping_parameters: StoppingCriteriaParameters,
    /// Number of top tokens to return in the response
    #[allow(dead_code)]
    pub top_n_tokens: u32,
    /// Whether to include the original prompt in the generated text
    pub return_full_text: bool,
}

/// Parameters for controlling the next token selection process in language model generation.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NextTokenChooserParameters {
    /// Number of alternative sequences to generate
    pub n: usize,
    /// Number of sequences to generate and select the best from
    pub best_of: usize,
    /// Temperature for controlling randomness in sampling (higher values increase diversity)
    pub temperature: f32,
    /// Limits sampling to the k most likely next tokens
    pub top_k: u32,
    /// Limits sampling to the smallest set of most probable tokens with probabilities that add up
    /// to top_p or higher
    pub top_p: f32,
    /// Selects tokens whose probability is close to the expected probability of tokens in a
    /// uniform distribution
    pub typical_p: f32,
    /// Whether to use sampling instead of greedy selection
    pub do_sample: bool,
    /// Seed for reproducible random sampling
    pub random_seed: u64,
    /// Penalizes repeated tokens
    pub repetition_penalty: f32,
    /// Number of previous tokens to consider for repetition penalty
    pub repeat_last_n: u32,
    /// Decreases the model's likelihood to repeat the same line verbatim
    pub frequency_penalty: f32,
}

/// Criteria for stopping token generation in language models.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StoppingCriteriaParameters {
    /// Maximum number of new tokens to generate.
    pub max_new_tokens: u32,
    /// List of sequences that, if generated, will cause the model to stop.
    pub stop_sequences: Vec<String>,
    /// If true, the model will ignore the end-of-sequence token and continue generating.
    /// This is primarily used for benchmarking purposes.
    pub ignore_eos_token: bool,
}

#[derive(Debug, Error)]
pub enum ValidationError {
    #[error("Tokenizer error: `{0}`")]
    TokenizerError(#[from] TokenizerError),
    #[error("Maximum total number error: max_total_tokens = `{0}`, input_len = `{1}`, max_new_tokens = `{2}`")]
    MaxTotalTokens(u32, usize, u32),
    #[error("Input length error: max_input_len = `{0}`, input_len = `{1}`")]
    InputLength(usize, usize),
    #[error("Invalid best of sampling parameter")]
    BestOfSampling,
    #[error("Invalid temperature parameter")]
    Temperature,
    #[error("Invalid repetition parameter")]
    RepetitionPenalty,
    #[error("Invalid frequency penalty parameter")]
    FrequencyPenalty,
    #[error("Invalid top p parameter")]
    TopP,
    #[error("Invalid typical p parameter")]
    TypicalP,
    #[error("Invalid top k parameter")]
    TopK,
    #[error("Negative max new tokens to generate")]
    NegativeMaxNewTokens,
    #[error("Stop sequences size exceeds maximum number of stop sequences allowed: `{0}` < `{1}`")]
    StopSequence(usize, usize),
    #[error("Empty random seed")]
    NullRandomSeed,
    #[error("Invalid top n tokens parameter: `{0}`, `{1}`")]
    TopNTokens(u32, u32),
    #[error("Empty input")]
    EmptyInput,
    #[error("Invalid truncate paremeter: `{0}` < `{1}`")]
    Truncate(usize, usize),
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use tokenizers::{Encoding, Token};

    use super::*;
    use crate::types::GenerateParameters;

    const MAX_INPUT_LENGTH: usize = 128;
    const MAX_TOTAL_TOKENS: u32 = 256;

    /// A `Validation` whose tokenizer is a background task returning one token per whitespace
    /// separated word, so the tests exercise the real request path without loading a tokenizer.
    fn test_validation() -> Validation {
        let (sender, mut receiver) = mpsc::unbounded_channel::<EncodeTokenizerRequest>();
        tokio::spawn(async move {
            while let Some(request) = receiver.recv().await {
                let tokens = request
                    .input
                    .split_whitespace()
                    .enumerate()
                    .map(|(i, word)| Token::new(i as u32, word.to_string(), (0, 0)))
                    .collect::<Vec<_>>();
                let encoding = Encoding::from_tokens(tokens, 0);
                request.sender.send(Ok((encoding, request.input))).ok();
            }
        });

        Validation::new(1, 4, 8, MAX_INPUT_LENGTH, MAX_TOTAL_TOKENS, sender)
    }

    fn generate_request(parameters: GenerateParameters) -> GenerateRequest {
        GenerateRequest {
            request_id: "request".to_string(),
            inputs: "Hello world, from the Caribbean".to_string(),
            parameters,
        }
    }

    fn generate_parameters() -> GenerateParameters {
        GenerateParameters {
            best_of: None,
            temperature: None,
            repetition_penalty: None,
            frequency_penalty: None,
            repeat_last_n: None,
            top_k: None,
            top_p: None,
            typical_p: None,
            do_sample: true,
            max_new_tokens: Some(16),
            return_full_text: None,
            stop: Vec::new(),
            truncate: None,
            decoder_input_details: false,
            random_seed: None,
            top_n_tokens: None,
            n: 1,
        }
    }

    #[tokio::test]
    async fn test_unseeded_requests_do_not_share_a_seed() {
        const NUM_REQUESTS: usize = 64;

        let validation = test_validation();
        let mut seeds = HashSet::with_capacity(NUM_REQUESTS);
        for _ in 0..NUM_REQUESTS {
            let valid_request = validation
                .validate(generate_request(generate_parameters()))
                .await
                .expect("Failed to validate request");
            seeds.insert(valid_request.parameters.random_seed);
        }

        assert_eq!(
            seeds.len(),
            NUM_REQUESTS,
            "unseeded requests must not decode from a shared seed"
        );
    }

    #[tokio::test]
    async fn test_requested_seed_is_preserved() {
        let validation = test_validation();
        let parameters = GenerateParameters {
            random_seed: Some(1234),
            ..generate_parameters()
        };

        let valid_request = validation
            .validate(generate_request(parameters))
            .await
            .expect("Failed to validate request");

        assert_eq!(valid_request.parameters.random_seed, 1234);
    }

    /// The OpenAI defaults have to survive validation unchanged, since they are what selects the
    /// decoding strategy downstream.
    #[tokio::test]
    async fn test_default_sampling_parameters_are_normalised() {
        let validation = test_validation();

        let valid_request = validation
            .validate(generate_request(generate_parameters()))
            .await
            .expect("Failed to validate request");

        assert_eq!(valid_request.parameters.temperature, 1.0);
        assert_eq!(valid_request.parameters.top_k, 0);
        assert_eq!(valid_request.parameters.top_p, 1.0);
        assert!(valid_request.parameters.do_sample);
    }

    #[tokio::test]
    async fn test_empty_input_is_rejected() {
        let validation = test_validation();
        let mut request = generate_request(generate_parameters());
        request.inputs = String::new();

        let error = validation
            .validate(request)
            .await
            .expect_err("Empty inputs must be rejected");

        assert!(matches!(error, ValidationError::EmptyInput));
    }

    #[tokio::test]
    async fn test_negative_temperature_is_rejected() {
        let validation = test_validation();
        let parameters = GenerateParameters {
            temperature: Some(-0.5),
            ..generate_parameters()
        };

        let error = validation
            .validate(generate_request(parameters))
            .await
            .expect_err("Negative temperature must be rejected");

        assert!(matches!(error, ValidationError::Temperature));
    }

    /// `temperature: 0` is how an OpenAI client asks for greedy decoding, and how the benchmark
    /// harness holds the sampler still across engines. Rejecting it refuses every deterministic
    /// request, including every request the harness makes.
    #[tokio::test]
    async fn test_zero_temperature_is_accepted() {
        let validation = test_validation();
        let parameters = GenerateParameters {
            temperature: Some(0.0),
            ..generate_parameters()
        };

        let valid_request = validation
            .validate(generate_request(parameters))
            .await
            .expect("Zero temperature must be accepted as a request for greedy decoding");

        assert_eq!(valid_request.parameters.temperature, 0.0);
        assert!(valid_request.parameters.do_sample);
    }

    #[tokio::test]
    async fn test_too_many_stop_sequences_are_rejected() {
        let validation = test_validation();
        let parameters = GenerateParameters {
            stop: (0..8).map(|i| format!("stop-{i}")).collect(),
            ..generate_parameters()
        };

        let error = validation
            .validate(generate_request(parameters))
            .await
            .expect_err("Too many stop sequences must be rejected");

        assert!(matches!(error, ValidationError::StopSequence(4, 8)));
    }

    #[tokio::test]
    async fn test_request_exceeding_total_tokens_is_rejected() {
        let validation = test_validation();
        let parameters = GenerateParameters {
            max_new_tokens: Some(MAX_TOTAL_TOKENS),
            ..generate_parameters()
        };

        let error = validation
            .validate(generate_request(parameters))
            .await
            .expect_err("Requests over the total token budget must be rejected");

        assert!(matches!(error, ValidationError::MaxTotalTokens(..)));
    }

    #[tokio::test]
    async fn test_unset_max_new_tokens_fills_the_remaining_budget() {
        let validation = test_validation();
        let parameters = GenerateParameters {
            max_new_tokens: None,
            ..generate_parameters()
        };

        let valid_request = validation
            .validate(generate_request(parameters))
            .await
            .expect("Failed to validate request");

        let input_token_len = valid_request.input_token_len as u32;
        assert_eq!(
            valid_request.stopping_parameters.max_new_tokens,
            MAX_TOTAL_TOKENS - input_token_len
        );
    }
}
