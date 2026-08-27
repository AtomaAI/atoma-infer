//! The graph key: the value that selects one captured graph for a padded batch.

use crate::protocol::{RequestCount, TokenCount};

/// Selects one captured graph for a padded batch.
///
/// A key is built by exactly one pure function of the padded batch — the crate-internal
/// constructor that admission owns. There is no default value and no public field, so adding a
/// field later fails to compile at the construction site, and code outside this crate can carry
/// and compare keys but never mint one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphKey {
    padded_token_count: TokenCount,
    request_count: RequestCount,
    uniform_decode: bool,
}

impl GraphKey {
    /// The one pure function of the current padded batch.
    pub(crate) fn from_padded_batch(
        padded_token_count: TokenCount,
        request_count: RequestCount,
        uniform_decode: bool,
    ) -> Self {
        Self {
            padded_token_count,
            request_count,
            uniform_decode,
        }
    }

    /// The batch's token count after padding up to its bucket.
    #[must_use]
    pub fn padded_token_count(self) -> TokenCount {
        self.padded_token_count
    }

    /// The exact number of live requests in the batch, excluding padding.
    #[must_use]
    pub fn request_count(self) -> RequestCount {
        self.request_count
    }

    /// Whether every request in the batch is decoding.
    #[must_use]
    pub fn uniform_decode(self) -> bool {
        self.uniform_decode
    }
}
