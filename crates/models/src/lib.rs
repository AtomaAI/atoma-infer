#![cfg(feature = "cuda")]

pub mod flash_attention;
pub mod llama;
#[cfg(feature = "nccl")]
pub mod llama_nccl;
#[cfg(feature = "nccl")]
pub mod multi_gpu;

pub use flash_attention::{
    FlashAttention, FlashAttentionDecodingMetadata, FlashAttentionMetadata,
    FlashAttentionPrefillMetadata,
};

pub use llama::Llama;
