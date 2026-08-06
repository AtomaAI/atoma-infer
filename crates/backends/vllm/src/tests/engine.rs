//! Engine harness: the whole service, from `LlmService::start` down to the CUDA kernels, driven at
//! the concurrency it has to survive.
//!
//! The model is a mock, but its KV cache is not: `MockModel::forward` runs the real attention
//! layer, which writes each token's key and value into the block the scheduler allocated for it
//! and reads back over the blocks the scheduler says the sequence owns. A request whose block
//! table overlapped another's would attend over tokens it never saw, and the KV cache would be the
//! thing that noticed.

#[cfg(feature = "nccl")]
use cudarc::nccl::Comm;
#[cfg(feature = "nccl")]
use std::rc::Rc;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use candle_core::{DType, Device, Tensor};
use futures::{stream::FuturesUnordered, StreamExt};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use models::{FlashAttention, FlashAttentionMetadata};
use serde::Deserialize;
use tokio::sync::{mpsc, oneshot};
use tracing::info;

use crate::{
    llm_service::LlmService,
    model_executor::{
        Config, ConfigError, ModelExecutor, ModelExecutorError, ModelFilePaths, ModelLoader,
        ModelLoaderError,
    },
    output::{GenerateRequestOutput, StreamResponse},
    request::ServiceRequest,
    tests::init_tracing,
    types::{GenerateParameters, GenerateRequest},
};

const MAX_ELAPSED_INTERNAL: u64 = 50;
const VOCAB_SIZE: usize = 128;

/// Shape of the model the engine tests pretend to serve.
///
/// The values only have to be self-consistent and cheap, but they shape a real attention layer and
/// the KV cache the worker allocates from them, so `MOCK_HEAD_DIM` has to be one of the head sizes
/// the attention layer supports.
const MOCK_HEAD_DIM: usize = 64;
const MOCK_NUM_ATTENTION_HEADS: usize = 4;
const MOCK_NUM_HIDDEN_LAYERS: usize = 2;
const MOCK_NUM_KV_HEADS: usize = 4;
const MOCK_HIDDEN_SIZE: usize = MOCK_NUM_ATTENTION_HEADS * MOCK_HEAD_DIM;

struct MockModel {
    config: MockConfig,
    /// The attention layer the forward pass runs, so the KV cache is written through the
    /// scheduler's block tables rather than stubbed out.
    attention: FlashAttention,
    /// `[1, hidden_size]` of `0..hidden_size`, added to a token id to spread it over the features.
    feature_ramp: Tensor,
    dtype: DType,
}

/// Configuration of the mock model served by the engine tests.
#[derive(Clone, Debug, Default, Deserialize)]
struct MockConfig {}

impl Config for MockConfig {
    fn alibi_slopes(&self) -> Option<&Tensor> {
        None
    }

    fn eos_token_ids(&self) -> Option<Vec<u32>> {
        None
    }

    fn hidden_dim(&self) -> usize {
        MOCK_HEAD_DIM
    }

    fn num_attention_heads(&self) -> usize {
        MOCK_NUM_ATTENTION_HEADS
    }

    fn num_hidden_layers(&self) -> usize {
        MOCK_NUM_HIDDEN_LAYERS
    }

    fn num_kv_heads(&self) -> usize {
        MOCK_NUM_KV_HEADS
    }

    fn sliding_window(&self) -> Option<usize> {
        None
    }

    fn softmax_scale(&self) -> f32 {
        1f32 / (MOCK_HEAD_DIM as f32).sqrt()
    }

    /// `MockModel::fetch` downloads a tokenizer but no config file, so the shape above is used
    /// instead of reading `path`.
    fn from_file_path(_: &PathBuf) -> Result<Self, ConfigError> {
        Ok(Self::default())
    }
}

impl MockModel {
    /// Builds the mock over an attention layer shaped like the KV cache the worker allocates from
    /// `MockConfig`.
    fn build(config: MockConfig, device: &Device, dtype: DType) -> Result<Self, ModelLoaderError> {
        let attention = FlashAttention::new(
            MOCK_NUM_ATTENTION_HEADS,
            MOCK_NUM_KV_HEADS,
            MOCK_HEAD_DIM,
            config.softmax_scale(),
            None,
            None,
            dtype,
            device.clone(),
        )?;

        let feature_ramp = Tensor::arange(0u32, MOCK_HIDDEN_SIZE as u32, device)?
            .to_dtype(dtype)?
            .reshape((1, MOCK_HIDDEN_SIZE))?;

        Ok(Self {
            config,
            attention,
            feature_ramp,
            dtype,
        })
    }
}

impl ModelLoader for MockModel {
    type C = MockConfig;

    fn fetch<T: AsRef<Path>>(
        api_key: String,
        cache_dir: T,
        model_id: String,
        revision: String,
    ) -> Result<ModelFilePaths, ModelLoaderError> {
        let api = ApiBuilder::new()
            .with_progress(true)
            .with_token(Some(api_key))
            .with_cache_dir(cache_dir.as_ref().to_path_buf())
            .build()?;
        let repo = api.repo(Repo::with_revision(
            model_id.clone(),
            RepoType::Model,
            revision,
        ));
        let tokenizer_file_path = repo.get("tokenizer.json")?;

        Ok(ModelFilePaths {
            config_path: "".into(),
            tokenizer_path: tokenizer_file_path,
            weights_path: vec![],
        })
    }

    #[cfg(not(feature = "nccl"))]
    fn load(
        config: Self::C,
        device: &Device,
        dtype: DType,
        _: &ModelFilePaths,
    ) -> Result<Self, ModelLoaderError> {
        Self::build(config, device, dtype)
    }

    #[cfg(feature = "nccl")]
    fn load(
        config: Self::C,
        device: &Device,
        dtype: DType,
        _: &ModelFilePaths,
        _: &Rc<Comm>,
    ) -> Result<Self, ModelLoaderError> {
        Self::build(config, device, dtype)
    }
}

impl ModelExecutor for MockModel {
    /// Runs attention over the batch, writing every token's key and value into the KV cache
    /// through the slot mapping the worker built from the scheduler's block tables.
    ///
    /// Queries, keys and values are derived from the token ids alone, so a sequence's attention
    /// output depends only on the tokens whose blocks it owns.
    fn forward(
        &mut self,
        input_tensor: &Tensor,
        _: &Tensor,
        selected_token_positions: &Tensor,
        kv_cache: Vec<&mut Tensor>,
        attention_metadata: FlashAttentionMetadata,
    ) -> Result<Tensor, ModelExecutorError> {
        let num_tokens = input_tensor.elem_count();
        // A token embeds to its own id, spread over the features so the attention weights are not
        // uniform. Nothing else feeds the embedding, so what a sequence attends over is decided by
        // which tokens its blocks hold.
        let embeddings = input_tensor
            .flatten_all()?
            .to_dtype(self.dtype)?
            .reshape((num_tokens, 1))?
            .broadcast_add(&self.feature_ramp)?
            .contiguous()?;
        let query = embeddings.reshape((num_tokens, MOCK_NUM_ATTENTION_HEADS, MOCK_HEAD_DIM))?;
        let key_value = embeddings.reshape((num_tokens, MOCK_NUM_KV_HEADS, MOCK_HEAD_DIM))?;

        let mut hidden_states = embeddings;
        for layer_kv_cache in kv_cache {
            hidden_states = self.attention.forward(
                &query,
                &key_value,
                &key_value,
                layer_kv_cache,
                &attention_metadata,
            )?;
        }

        // The vocabulary is a slice of the hidden state, which keeps the mock free of weights
        // while still making the logits a function of what attention read out of the cache.
        let logits = hidden_states
            .index_select(selected_token_positions, 0)?
            .narrow(1, 0, VOCAB_SIZE)?
            .to_dtype(DType::F32)?;

        Ok(logits.unsqueeze(0)?)
    }

    fn config(&self) -> &Self::C {
        &self.config
    }
}

/// The prompt of the request with this index. Prompts differ per request so that a response
/// carrying the wrong prompt is a crossed output rather than a coincidence.
fn prompt_of(request_index: usize) -> String {
    format!("Hello world, from Caribbean island number {request_index}")
}

fn generate_request(request_index: usize) -> GenerateRequest {
    GenerateRequest {
        request_id: format!("{request_index}"),
        inputs: prompt_of(request_index),
        parameters: GenerateParameters {
            best_of: None,
            temperature: Some(1.2),
            repetition_penalty: Some(1.1),
            frequency_penalty: Some(1.1),
            repeat_last_n: Some(8),
            top_k: Some(8),
            top_p: Some(0.8),
            typical_p: None,
            do_sample: true,
            max_new_tokens: Some(16),
            return_full_text: Some(true),
            stop: vec!["STOP".to_string()],
            truncate: None,
            decoder_input_details: true,
            random_seed: Some(42),
            top_n_tokens: None,
            n: 1,
        },
    }
}

fn config_path(file_name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("tests")
        .join(file_name)
}

/// Starts the service under test, returning the channel requests are submitted on and the one that
/// shuts it down.
async fn start_service(
    config_file: &str,
) -> (mpsc::UnboundedSender<ServiceRequest>, mpsc::Sender<()>) {
    let (service_request_sender, service_request_receiver) = mpsc::unbounded_channel();
    let (shutdown_signal_sender, shutdown_signal_receiver) = mpsc::channel(1);

    let service = LlmService::start::<MockModel, PathBuf>(
        service_request_receiver,
        config_path(config_file),
        shutdown_signal_receiver,
    )
    .await
    .expect("Failed to start LLM service");

    tokio::spawn(async move {
        service.run().await.expect("Fail to run llm service");
    });

    (service_request_sender, shutdown_signal_sender)
}

/// Submits `num_requests` requests and waits for all of them, returning each answered request's
/// output by request id, and when each answer arrived.
async fn run_requests(
    service_request_sender: &mpsc::UnboundedSender<ServiceRequest>,
    num_requests: usize,
) -> (HashMap<String, GenerateRequestOutput>, Vec<Duration>) {
    let mut futures = FuturesUnordered::new();
    for request_index in 0..num_requests {
        let (sender, receiver) = oneshot::channel();
        service_request_sender
            .send(ServiceRequest::GenerateRequest(
                generate_request(request_index),
                sender,
            ))
            .expect("Failed to send request");
        futures.push(receiver);
    }

    let start = Instant::now();
    let mut outputs = HashMap::with_capacity(num_requests);
    let mut elapsed_times = Vec::with_capacity(num_requests);
    while let Some(response) = futures.next().await {
        let response = response.expect("The engine dropped a request's response channel");
        elapsed_times.push(start.elapsed());
        info!("Got new response for request {}", response.request_id);
        outputs.insert(response.request_id.clone(), response);
    }

    (outputs, elapsed_times)
}

/// Asserts the engine kept draining the queue: no two consecutive completions are further apart
/// than a single scheduler run, with enough slack for different machines.
fn assert_completions_are_not_stalled(elapsed_times: &[Duration]) {
    let max_elapsed_interval = Duration::from_secs(MAX_ELAPSED_INTERNAL);
    for window in elapsed_times.windows(2) {
        assert!(window[1] - window[0] <= max_elapsed_interval);
    }
}

async fn run_engine_test(config_file: &str) {
    init_tracing();

    const NUM_REQUESTS: usize = 128;

    let (service_request_sender, shutdown_signal_sender) = start_service(config_file).await;
    let (outputs, elapsed_times) = run_requests(&service_request_sender, NUM_REQUESTS).await;

    // Every request is answered exactly once, so one completion time is recorded per request.
    assert_eq!(outputs.len(), NUM_REQUESTS);
    assert_eq!(elapsed_times.len(), NUM_REQUESTS);
    assert_completions_are_not_stalled(&elapsed_times);

    shutdown_signal_sender.send(()).await.unwrap();
}

#[tokio::test]
async fn test_llm_engine() {
    run_engine_test("test_config_disable_chunked_prefill.toml").await;
}

#[tokio::test]
async fn test_llm_engine_with_enable_chunking() {
    run_engine_test("test_config_enable_chunked_prefill.toml").await;
}

/// Concurrency 32 with a distinct prompt per request: every request has to come back with its own
/// prompt and its own output, which it cannot do if two requests share a block table.
#[tokio::test]
async fn test_concurrent_requests_do_not_cross() {
    init_tracing();

    const NUM_REQUESTS: usize = 32;

    let (service_request_sender, shutdown_signal_sender) =
        start_service("test_config_disable_chunked_prefill.toml").await;
    let (outputs, _) = run_requests(&service_request_sender, NUM_REQUESTS).await;

    assert_eq!(
        outputs.len(),
        NUM_REQUESTS,
        "every request must be answered"
    );
    for request_index in 0..NUM_REQUESTS {
        let request_id = format!("{request_index}");
        let output = outputs
            .get(&request_id)
            .unwrap_or_else(|| panic!("request {request_id} was never answered"));

        assert_eq!(
            output.prompt,
            prompt_of(request_index),
            "request {request_id} came back with another request's prompt"
        );
        assert_eq!(output.inference_outputs.len(), 1);
        assert!(
            !output.inference_outputs[0].token_ids.is_empty(),
            "request {request_id} generated nothing"
        );
    }

    shutdown_signal_sender.send(()).await.unwrap();
}

/// A client that hangs up costs its own request and nothing else: the engine keeps stepping for
/// every other request in flight.
#[tokio::test]
async fn test_engine_survives_a_disconnected_client() {
    init_tracing();

    const NUM_REQUESTS: usize = 32;
    const DISCONNECTED: usize = 7;

    let (service_request_sender, shutdown_signal_sender) =
        start_service("test_config_disable_chunked_prefill.toml").await;

    let mut futures = FuturesUnordered::new();
    for request_index in 0..NUM_REQUESTS {
        let (sender, receiver) = oneshot::channel();
        service_request_sender
            .send(ServiceRequest::GenerateRequest(
                generate_request(request_index),
                sender,
            ))
            .expect("Failed to send request");

        if request_index == DISCONNECTED {
            drop(receiver);
        } else {
            futures.push(receiver);
        }
    }

    let mut num_responses = 0;
    while let Some(response) = futures.next().await {
        response.expect("A connected client lost its response when another client disconnected");
        num_responses += 1;
    }

    assert_eq!(num_responses, NUM_REQUESTS - 1);

    shutdown_signal_sender.send(()).await.unwrap();
}

/// The same for a streaming client: its channel fails mid-generation, which retires that request
/// while the rest of the batch keeps streaming.
#[tokio::test]
async fn test_engine_survives_a_disconnected_streaming_client() {
    init_tracing();

    const NUM_REQUESTS: usize = 8;
    const DISCONNECTED: usize = 3;

    let (service_request_sender, shutdown_signal_sender) =
        start_service("test_config_disable_chunked_prefill.toml").await;

    let mut receivers = Vec::with_capacity(NUM_REQUESTS);
    for request_index in 0..NUM_REQUESTS {
        let (sender, receiver) = flume::unbounded();
        service_request_sender
            .send(ServiceRequest::GenerateStreamingRequest(
                generate_request(request_index),
                sender,
            ))
            .expect("Failed to send request");

        if request_index == DISCONNECTED {
            drop(receiver);
        } else {
            receivers.push((request_index, receiver));
        }
    }

    for (request_index, receiver) in receivers {
        let mut num_chunks = 0;
        while let Ok(response) = receiver.recv_async().await {
            match response {
                StreamResponse::Chunk(_) => num_chunks += 1,
                StreamResponse::Finished => break,
                StreamResponse::Error(error) => panic!("Request {request_index} failed: {error}"),
            }
        }
        assert!(
            num_chunks > 0,
            "request {request_index} streamed nothing after another client disconnected"
        );
    }

    shutdown_signal_sender.send(()).await.unwrap();
}
