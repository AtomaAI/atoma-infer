pub mod attention_window;
pub mod decode_ops;
pub mod error;
pub mod paged_decode;
pub mod splits;

pub use attention_window::CAUSAL_WINDOW;
pub use error::{check_supported_capabilities, Capability, KernelError};

#[cfg(feature = "cuda")]
pub mod cache_manager;
#[cfg(feature = "cuda")]
mod ffi;
#[cfg(feature = "cuda")]
pub mod flash_attention;
#[cfg(feature = "cuda")]
pub mod ops;

#[cfg(feature = "cuda")]
pub use cache_manager::{copy_blocks, reshape_and_cache_flash, swap_blocks};
#[cfg(feature = "cuda")]
pub use flash_attention::{
    flash_attn, flash_attn_alibi, flash_attn_alibi_windowed,
    flash_attn_alibi_windowed_with_softcap, flash_attn_kv_cache, flash_attn_kv_cache_alibi,
    flash_attn_kv_cache_alibi_windowed, flash_attn_kv_cache_full, flash_attn_kv_cache_windowed,
    flash_attn_varlen, flash_attn_varlen_alibi, flash_attn_varlen_alibi_windowed,
    flash_attn_varlen_full, flash_attn_varlen_windowed, flash_attn_varlen_with_block_table,
    flash_attn_varlen_with_lse, flash_attn_windowed, flash_attn_with_lse, FlashAttention,
};
