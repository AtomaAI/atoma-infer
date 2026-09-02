#[cfg(feature = "cuda")]
use atoma_backends::{LlamaModel, LlmService};
#[cfg(feature = "cuda")]
use clap::Parser;
#[cfg(feature = "cuda")]
use std::env;
#[cfg(feature = "cuda")]
use tokio::{net::TcpListener, sync::mpsc};

#[cfg(feature = "cuda")]
use server::{run_server, AppState};

pub mod api;
pub mod completion;
pub mod detokenize;
pub mod server;
pub mod stream;

#[cfg(feature = "cuda")]
pub const DEFAULT_SERVER_ADDRESS: &str = "0.0.0.0";
#[cfg(feature = "cuda")]
pub const DEFAULT_SERVER_PORT: &str = "8080";

#[cfg(feature = "cuda")]
#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    #[arg(short, long)]
    config_path: String,
}

#[cfg(feature = "cuda")]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Args::parse();

    // TODO: Write a clap cli for passing arguments
    let address =
        env::var("ATOMA_NODE_INFERENCE_SERVER_ADDRESS").unwrap_or(DEFAULT_SERVER_ADDRESS.into());
    let config_path = cli.config_path;
    let port = env::var("ATOMA_NODE_INFERENCE_SERVER_PORT").unwrap_or(DEFAULT_SERVER_PORT.into());
    let listener = TcpListener::bind(format!("{address}:{port}")).await?;

    let (llm_service_sender, llm_service_receiver) = mpsc::unbounded_channel();
    let (shutdown_signal_sender, shutdown_signal_receiver) = mpsc::channel(1);
    // TODO: Add model dispatcher
    let llm_service = LlmService::start::<LlamaModel, _>(
        llm_service_receiver,
        config_path,
        shutdown_signal_receiver,
    )
    .await
    .map_err(|e| anyhow::anyhow!("Failed to start `LlmService`, with error: {e}"))?;

    let join_handle = tokio::spawn(async move {
        llm_service
            .run()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to run `LlmService`, with error: {e}"))
    });

    let app_state = AppState {
        llm_service_sender,
        shutdown_signal_sender,
        streaming_interval_in_millis: env::var("STREAMING_INTERVAL_IN_MILLIS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(100),
    };
    run_server(listener, app_state, join_handle).await
}

#[cfg(not(feature = "cuda"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("CUDA support is disabled; rebuild with `--features cuda`")
}

#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::Arc;

    use serde_json::json;
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;
    use tokenizers::Tokenizer;

    /// A byte-level BPE tokenizer over every byte, with a few merges, built from the JSON a
    /// `tokenizer.json` holds: no file, no Hub.
    pub(crate) fn tokenizer() -> Arc<Tokenizer> {
        let mut alphabet: Vec<char> = ByteLevel::alphabet().into_iter().collect();
        alphabet.sort_unstable();
        let mut vocab: Vec<(String, u32)> = alphabet
            .iter()
            .enumerate()
            .map(|(id, &c)| (c.to_string(), u32::try_from(id).unwrap()))
            .collect();
        let merges = ["h e", "l l", "Ġ w", "o r"];
        for merge in merges {
            let merged = merge.replace(' ', "");
            let id = u32::try_from(vocab.len()).unwrap();
            vocab.push((merged, id));
        }
        let vocab: serde_json::Map<String, serde_json::Value> = vocab
            .into_iter()
            .map(|(token, id)| (token, json!(id)))
            .collect();
        let byte_level = json!({
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": true,
            "use_regex": true
        });
        let spec = json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": byte_level,
            "post_processor": null,
            "decoder": byte_level,
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": null,
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "byte_fallback": false,
                "ignore_merges": false,
                "vocab": vocab,
                "merges": merges,
            }
        });
        Arc::new(serde_json::from_value(spec).expect("a byte-level BPE tokenizer"))
    }
}
