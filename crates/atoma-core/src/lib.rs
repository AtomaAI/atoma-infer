//! GPU-free engine core.
//!
//! Hosts the decisions the engine makes on the host side: which captured CUDA graph serves a live
//! batch, and the shared id and count types those decisions are written in. The crate links no
//! driver, and every test runs without a GPU.

pub mod dispatch;
pub mod kv;
pub mod request;
pub mod scheduler;
pub mod types;

#[cfg(test)]
pub(crate) mod test_support {
    use crate::types::{RequestCount, TokenCount};

    pub(crate) fn tokens(value: usize) -> TokenCount {
        TokenCount::new(value).expect("test token counts are nonzero")
    }

    pub(crate) fn requests(value: usize) -> RequestCount {
        RequestCount::new(value).expect("test request counts are nonzero")
    }
}
