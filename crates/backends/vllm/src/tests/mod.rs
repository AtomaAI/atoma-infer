#[cfg(feature = "nccl")]
use cudarc::nccl::Comm;
#[cfg(feature = "nccl")]
use std::rc::Rc;
use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
#[cfg(not(feature = "nccl"))]
mod llama;
#[cfg(feature = "nccl")]
mod llama_nccl;

use candle_core::{DType, Device, Tensor};
use futures::{stream::FuturesUnordered, StreamExt};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use models::FlashAttentionMetadata;
use rand::Rng;
use serde::Deserialize;
use tokio::sync::{mpsc, oneshot};
use tracing::info;

use crate::{
    llm_service::LlmService,
    model_executor::{
        Config, ConfigError, ModelExecutor, ModelExecutorError, ModelFilePaths, ModelLoader,
        ModelLoaderError,
    },
    request::ServiceRequest,
    sequence::ExecuteModelRequest,
    types::{GenerateParameters, GenerateRequest},
};

const MAX_ELAPSED_INTERNAL: u64 = 50;
const VOCAB_SIZE: usize = 128;

/// Shape of the model the engine tests pretend to serve.
///
/// The values only have to be self-consistent and cheap: `MockModel::forward` never touches the KV
/// cache, but the worker still builds a real `CacheEngine` from them, so `MOCK_HEAD_DIM` has to be
/// one of the head sizes the attention layer supports.
const MOCK_HEAD_DIM: usize = 64;
const MOCK_NUM_ATTENTION_HEADS: usize = 4;
const MOCK_NUM_HIDDEN_LAYERS: usize = 2;
const MOCK_NUM_KV_HEADS: usize = 4;

struct MockModel {
    config: MockConfig,
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
        _: &Device,
        _: DType,
        _: &ModelFilePaths,
    ) -> Result<Self, ModelLoaderError> {
        Ok(Self { config })
    }

    #[cfg(feature = "nccl")]
    fn load(
        config: Self::C,
        _: &Device,
        _: DType,
        _: &ModelFilePaths,
        _: &Rc<Comm>,
    ) -> Result<Self, ModelLoaderError> {
        Ok(Self { config })
    }
}

impl From<ExecuteModelRequest> for Vec<u32> {
    fn from(value: ExecuteModelRequest) -> Self {
        value
            .sequence_groups_metadata
            .first()
            .unwrap()
            .sequence_data
            .values()
            .next()
            .unwrap()
            .get_token_ids()
    }
}

impl ModelExecutor for MockModel {
    fn forward(
        &mut self,
        _: &Tensor,
        _: &Tensor,
        selected_token_positions: &Tensor,
        _: Vec<&mut Tensor>,
        _: FlashAttentionMetadata,
    ) -> Result<Tensor, ModelExecutorError> {
        let mut rng = rand::rng();
        std::thread::sleep(Duration::from_secs(2)); // mimic forward pass
        let num_selected_tokens = selected_token_positions.dims()[0];
        let logits = (0..(num_selected_tokens * VOCAB_SIZE))
            .map(|_| rng.random_range(0.0..1.0) as f32)
            .collect::<Vec<_>>();

        Ok(Tensor::new(logits, &Device::Cpu)?.reshape((1, num_selected_tokens, VOCAB_SIZE))?)
    }

    fn config(&self) -> &Self::C {
        &self.config
    }
}

#[tokio::test]
#[ignore = "hangs: one closed client output channel aborts the llm_engine step, so remaining in-flight requests never resolve"]
async fn test_llm_engine() {
    init_tracing();

    const NUM_REQUESTS: usize = 128;

    let (shutdown_signal_sender, shutdown_signal_receiver) = mpsc::channel(1);

    let config_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("tests")
        .join("test_config_disable_chunked_prefill.toml");

    let (service_request_sender, service_request_receiver) = mpsc::unbounded_channel();
    let service = LlmService::start::<MockModel, PathBuf>(
        service_request_receiver,
        config_path,
        shutdown_signal_receiver,
    )
    .await
    .expect("Failed to start LLM service");

    tokio::spawn(async move {
        service.run().await.expect("Fail to run llm service");
    });

    info!("Sending request through atoma_event_subscriber_sender");

    let requests = (0..NUM_REQUESTS).map(|i| GenerateRequest {
        request_id: format!("{}", i),
        inputs: "Hello world, from the Caribbean".to_string(),
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
    });

    let mut futures = FuturesUnordered::new();
    for request in requests {
        let (sender, receiver) = oneshot::channel();
        service_request_sender
            .send(ServiceRequest::GenerateRequest(request, sender))
            .expect("Failed to send request");
        futures.push(receiver);
    }

    let mut number_of_responses = 0;

    let start = Instant::now();
    let mut elapsed_times = Vec::with_capacity(100);

    while let Some(responses) = futures.next().await {
        let responses = responses.unwrap();
        elapsed_times.push(start.elapsed());
        for response in responses.inference_outputs.iter() {
            number_of_responses += 1;
            info!("Got new response: {response:?}");
        }
        info!("Number of responses {number_of_responses}")
    }

    info!("Elapsed times: {elapsed_times:?}");

    // Every request is answered exactly once, so one completion time is recorded per request.
    assert_eq!(number_of_responses, NUM_REQUESTS);
    assert_eq!(elapsed_times.len(), NUM_REQUESTS);

    // The engine keeps draining the queue: no two consecutive completions are further apart than a
    // single scheduler run, with enough slack for different machines.
    let max_elapsed_interval = Duration::from_secs(MAX_ELAPSED_INTERNAL);
    for window in elapsed_times.windows(2) {
        assert!(window[1] - window[0] <= max_elapsed_interval);
    }

    shutdown_signal_sender.send(()).await.unwrap();
}

#[tokio::test]
#[ignore = "hangs: one closed client output channel aborts the llm_engine step, so remaining in-flight requests never resolve"]
async fn test_llm_engine_with_enable_chunking() {
    init_tracing();

    const NUM_REQUESTS: usize = 128;

    let (service_request_sender, service_request_receiver) = mpsc::unbounded_channel();
    let (shutdown_signal_sender, shutdown_signal_receiver) = mpsc::channel(1);

    let config_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("tests")
        .join("test_config_enable_chunked_prefill.toml");

    let service = LlmService::start::<MockModel, PathBuf>(
        service_request_receiver,
        config_path,
        shutdown_signal_receiver,
    )
    .await
    .expect("Failed to start LLM service");

    tokio::spawn(async move {
        service.run().await.expect("Fail to run llm service");
    });

    info!("Sending request through atoma_event_subscriber_sender");

    let requests = (0..NUM_REQUESTS).map(|i| GenerateRequest {
        request_id: format!("{}", i),
        inputs: "Hello world, from the Caribbean".to_string(),
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
    });

    let mut futures = FuturesUnordered::new();
    for request in requests {
        let (sender, receiver) = oneshot::channel();
        service_request_sender
            .send(ServiceRequest::GenerateRequest(request, sender))
            .expect("Failed to send request");
        futures.push(receiver);
    }

    let mut number_of_responses = 0;

    let start = Instant::now();
    let mut elapsed_times = Vec::with_capacity(100);

    while let Some(responses) = futures.next().await {
        let responses = responses.unwrap();
        elapsed_times.push(start.elapsed());
        for response in responses.inference_outputs.iter() {
            number_of_responses += 1;
            info!("Got new response: {response:?}");
        }
        info!("Number of responses {number_of_responses}")
    }
    info!("Elapsed times: {elapsed_times:?}");

    // Every request is answered exactly once, so one completion time is recorded per request.
    assert_eq!(number_of_responses, NUM_REQUESTS);
    assert_eq!(elapsed_times.len(), NUM_REQUESTS);

    // The engine keeps draining the queue: no two consecutive completions are further apart than a
    // single scheduler run, with enough slack for different machines.
    let max_elapsed_interval = Duration::from_secs(MAX_ELAPSED_INTERNAL);
    for window in elapsed_times.windows(2) {
        assert!(window[1] - window[0] <= max_elapsed_interval);
    }

    shutdown_signal_sender.send(()).await.unwrap();
}

pub fn init_tracing() {
    let _ = tracing_subscriber::fmt::try_init();
}
