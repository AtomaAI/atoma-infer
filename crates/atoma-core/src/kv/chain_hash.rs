//! Chain hashing: the versioned content identity of block-sized token runs.
//!
//! A run's digest commits to its parent's digest, so equal hashes mean equal prefixes and a
//! prefix index can answer in hashes alone. The digest is a pure function of its inputs over a
//! canonical encoding, so it reproduces across process restarts, and the algorithm identifier is
//! versioned in the schema so a cache written by one version is never read by another.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::protocol::{BlockHash, TokenCount};

/// Domain string opening every `sha256_v1` digest, separating this encoding from any other use
/// of SHA-256 over similar bytes. Part of the frozen encoding: changing it orphans every
/// written cache.
const SHA256_V1_DOMAIN: &[u8] = b"atoma-kv-sha256-v1";
/// Presence byte preceding an optional field that is absent.
const FIELD_ABSENT: [u8; 1] = [0];
/// Presence byte preceding an optional field that follows.
const FIELD_PRESENT: [u8; 1] = [1];

/// The fields beyond the token run that namespace a block's identity.
///
/// Both fields are carried in the digest from day one so populating them later never changes the
/// encoding: a per-request cache salt partitions otherwise-equal prefixes, and an adapter slot
/// separates caches per `LoRA` adapter. Nothing populates either yet.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExtraKeys<'a> {
    /// Opaque salt partitioning the cache, e.g. per tenant. `None` until something populates it.
    pub cache_salt: Option<&'a [u8]>,
    /// The `LoRA` adapter slot the run was computed under. `None` until `LoRA` exists.
    pub adapter_slot: Option<u32>,
}

impl ExtraKeys<'_> {
    /// The empty namespace every request hashes under today.
    #[must_use]
    pub const fn none() -> Self {
        Self {
            cache_salt: None,
            adapter_slot: None,
        }
    }
}

/// The versioned chain-hash algorithm identifier.
///
/// Serialized wherever hashes travel, so a reader can reject digests minted under any other
/// algorithm or encoding instead of silently mixing caches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HashAlgorithm {
    /// SHA-256 over the `atoma-kv-sha256-v1` canonical encoding.
    Sha256V1,
}

impl HashAlgorithm {
    /// Digests one token run under `parent`'s chain.
    ///
    /// The canonical encoding is unambiguous by construction — every variable-length field is
    /// length-prefixed and every optional field carries a presence byte:
    /// domain string, parent presence + digest, token count as u64 LE, each token as u32 LE,
    /// salt presence + u64 LE length + bytes, adapter-slot presence + u32 LE.
    #[must_use]
    pub fn hash_run(
        self,
        parent: Option<BlockHash>,
        token_ids: &[u32],
        extra_keys: ExtraKeys<'_>,
    ) -> BlockHash {
        match self {
            HashAlgorithm::Sha256V1 => sha256_v1_run(parent, token_ids, extra_keys),
        }
    }

    /// Digests every full block-sized run of `token_ids`, each chained through the last.
    ///
    /// A trailing partial run has no stable identity — its digest would change as tokens arrive —
    /// so it is excluded, and fewer than `block_size` tokens produce no hashes at all.
    #[must_use]
    pub fn chain(
        self,
        block_size: TokenCount,
        token_ids: &[u32],
        extra_keys: ExtraKeys<'_>,
    ) -> Vec<BlockHash> {
        let mut parent = None;
        token_ids
            .chunks_exact(block_size.get())
            .map(|run| {
                let hash = self.hash_run(parent, run, extra_keys);
                parent = Some(hash);
                hash
            })
            .collect()
    }
}

fn sha256_v1_run(
    parent: Option<BlockHash>,
    token_ids: &[u32],
    extra_keys: ExtraKeys<'_>,
) -> BlockHash {
    let mut hasher = Sha256::new();
    hasher.update(SHA256_V1_DOMAIN);
    match parent {
        None => hasher.update(FIELD_ABSENT),
        Some(parent) => {
            hasher.update(FIELD_PRESENT);
            hasher.update(parent.as_bytes());
        }
    }
    hasher.update((token_ids.len() as u64).to_le_bytes());
    for token in token_ids {
        hasher.update(token.to_le_bytes());
    }
    match extra_keys.cache_salt {
        None => hasher.update(FIELD_ABSENT),
        Some(salt) => {
            hasher.update(FIELD_PRESENT);
            hasher.update((salt.len() as u64).to_le_bytes());
            hasher.update(salt);
        }
    }
    match extra_keys.adapter_slot {
        None => hasher.update(FIELD_ABSENT),
        Some(slot) => {
            hasher.update(FIELD_PRESENT);
            hasher.update(slot.to_le_bytes());
        }
    }
    BlockHash::from_bytes(hasher.finalize().into())
}

#[cfg(test)]
mod tests {
    use std::fmt::Write as _;

    use super::{ExtraKeys, HashAlgorithm};
    use crate::protocol::{BlockHash, TokenCount};

    /// Golden digests computed by an independent Python hashlib implementation of the documented
    /// encoding. They pin both the encoding and cross-restart reproducibility: a digest that
    /// drifts from these means caches written before the change can no longer be read.
    const RUN_1234: &str = "36de4443a5a78b80cc1b4524051a3dbccd2bd1cc52ca8bdc851c235425219383";
    const RUN_5678_UNDER_1234: &str =
        "473fc0b8c38f50a97867838c2442e631e45fb1d283a370d56d3cd7695fed9f82";
    const RUN_1234_SALTED: &str =
        "80b9a1932e2e27c8ef26fa46d060194274e3389e55908b398cf3aaa446876281";
    const RUN_1234_ADAPTER_3: &str =
        "38d200e2b532f9a9ce1d158cffabc9aa0428d8be1457dc42183b38c39b8c0911";

    fn hex(hash: BlockHash) -> String {
        hash.as_bytes().iter().fold(String::new(), |mut out, byte| {
            // Writing into a String cannot fail.
            let _ = write!(out, "{byte:02x}");
            out
        })
    }

    fn block_size(tokens: usize) -> TokenCount {
        TokenCount::new(tokens).expect("test block sizes are nonzero")
    }

    #[test]
    fn digests_match_the_independent_oracle() {
        let algorithm = HashAlgorithm::Sha256V1;
        let root = algorithm.hash_run(None, &[1, 2, 3, 4], ExtraKeys::none());
        assert_eq!(hex(root), RUN_1234);
        assert_eq!(
            hex(algorithm.hash_run(Some(root), &[5, 6, 7, 8], ExtraKeys::none())),
            RUN_5678_UNDER_1234
        );
        assert_eq!(
            hex(algorithm.hash_run(
                None,
                &[1, 2, 3, 4],
                ExtraKeys {
                    cache_salt: Some(b"tenant-a"),
                    adapter_slot: None,
                }
            )),
            RUN_1234_SALTED
        );
        assert_eq!(
            hex(algorithm.hash_run(
                None,
                &[1, 2, 3, 4],
                ExtraKeys {
                    cache_salt: None,
                    adapter_slot: Some(3),
                }
            )),
            RUN_1234_ADAPTER_3
        );
    }

    #[test]
    fn every_input_field_separates_identities() {
        let algorithm = HashAlgorithm::Sha256V1;
        let base = algorithm.hash_run(None, &[1, 2, 3, 4], ExtraKeys::none());
        let parent = algorithm.hash_run(Some(base), &[1, 2, 3, 4], ExtraKeys::none());
        let tokens = algorithm.hash_run(None, &[1, 2, 3, 5], ExtraKeys::none());
        let salt = algorithm.hash_run(
            None,
            &[1, 2, 3, 4],
            ExtraKeys {
                cache_salt: Some(b"tenant-a"),
                adapter_slot: None,
            },
        );
        let adapter = algorithm.hash_run(
            None,
            &[1, 2, 3, 4],
            ExtraKeys {
                cache_salt: None,
                adapter_slot: Some(0),
            },
        );
        let distinct = [base, parent, tokens, salt, adapter];
        for (i, left) in distinct.iter().enumerate() {
            for right in &distinct[i + 1..] {
                assert_ne!(left, right);
            }
        }
    }

    #[test]
    fn chain_hashes_full_runs_and_excludes_the_partial_tail() {
        let algorithm = HashAlgorithm::Sha256V1;
        let chained = algorithm.chain(
            block_size(4),
            &[1, 2, 3, 4, 5, 6, 7, 8, 9],
            ExtraKeys::none(),
        );
        assert_eq!(
            chained.len(),
            2,
            "the trailing run of one token has no identity"
        );
        assert_eq!(hex(chained[0]), RUN_1234);
        assert_eq!(hex(chained[1]), RUN_5678_UNDER_1234);
    }

    #[test]
    fn chain_of_an_exact_multiple_has_no_excluded_tail() {
        let algorithm = HashAlgorithm::Sha256V1;
        let chained = algorithm.chain(block_size(4), &[1, 2, 3, 4, 5, 6, 7, 8], ExtraKeys::none());
        assert_eq!(chained.len(), 2);
        assert_eq!(hex(chained[1]), RUN_5678_UNDER_1234);
    }

    #[test]
    fn chains_over_empty_or_sub_block_tokens_are_empty() {
        let algorithm = HashAlgorithm::Sha256V1;
        assert!(algorithm
            .chain(block_size(4), &[], ExtraKeys::none())
            .is_empty());
        assert!(algorithm
            .chain(block_size(4), &[1, 2, 3], ExtraKeys::none())
            .is_empty());
    }

    #[test]
    fn length_prefixes_keep_field_boundaries_unambiguous() {
        // A token moved into the salt must not collide with the same bytes as tokens.
        let algorithm = HashAlgorithm::Sha256V1;
        let two_tokens = algorithm.hash_run(None, &[1, 2], ExtraKeys::none());
        let one_token_salted = algorithm.hash_run(
            None,
            &[1],
            ExtraKeys {
                cache_salt: Some(&2u32.to_le_bytes()),
                adapter_slot: None,
            },
        );
        assert_ne!(two_tokens, one_token_salted);
    }

    #[test]
    fn algorithm_identifier_serializes_versioned() {
        assert_eq!(
            serde_json::to_string(&HashAlgorithm::Sha256V1).unwrap(),
            "\"sha256_v1\""
        );
        assert_eq!(
            serde_json::from_str::<HashAlgorithm>("\"sha256_v1\"").unwrap(),
            HashAlgorithm::Sha256V1
        );
    }
}
