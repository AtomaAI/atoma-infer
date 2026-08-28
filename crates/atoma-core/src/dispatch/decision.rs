//! The dispatch decision: exactly one graph key for a live batch, or exactly one named reason
//! it runs eagerly.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::dispatch::{GraphKey, PaddingLookup};
use crate::types::{RequestCount, TokenCount};

/// What the scheduler reports about the live batch it wants served.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiveBatch {
    /// Tokens in the batch before padding.
    pub token_count: TokenCount,
    /// Live requests in the batch.
    pub request_count: RequestCount,
    /// Whether every request in the batch is decoding.
    pub uniform_decode: bool,
}

/// The live batches a backend's captured routine is valid for, weakest to strongest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SupportLevel {
    /// The captured routine is never valid.
    Never,
    /// Valid only when every request decodes exactly one token.
    UniformSingleTokenDecode,
    /// Valid for any uniform batch.
    UniformBatch,
    /// Valid for any live batch.
    Always,
}

/// Why a batch fell back to eager execution, carrying the numbers that caused it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EagerReason {
    /// No bucket holds this many tokens.
    #[error(
        "token count {token_count} exceeds every captured bucket; {}",
        bucket_ladder_maximum_clause(*.bucket_ladder_maximum)
    )]
    TokensAboveBucketLadderMaximum {
        /// Tokens in the batch before padding.
        token_count: TokenCount,
        /// The largest captured bucket; `None` for an empty bucket ladder.
        bucket_ladder_maximum: Option<TokenCount>,
    },
    /// More live requests than any captured graph was built to serve.
    #[error("request count {request_count} is above the captured maximum {captured_maximum}")]
    RequestsAboveCapturedMaximum {
        /// Live requests in the batch.
        request_count: RequestCount,
        /// The largest request count any captured graph serves.
        captured_maximum: RequestCount,
    },
    /// The backends' captured routines are not valid for this live batch.
    #[error(
        "backend support level {support_level:?} is insufficient for a batch of {token_count} \
         tokens over {request_count} requests, which requires {required:?}"
    )]
    SupportLevelInsufficient {
        /// The support level the active backends declare.
        support_level: SupportLevel,
        /// The weakest level whose captured routine is valid for this batch.
        required: SupportLevel,
        /// Tokens in the batch before padding.
        token_count: TokenCount,
        /// Live requests in the batch.
        request_count: RequestCount,
    },
    /// Only uniform-decode batches are captured.
    #[error("batch of {token_count} tokens over {request_count} requests is not uniform decode")]
    NotUniformDecode {
        /// Tokens in the batch before padding.
        token_count: TokenCount,
        /// Live requests in the batch.
        request_count: RequestCount,
    },
}

/// Eager fallbacks so far, by eager reason.
///
/// Lives beside [`EagerReason`] so a new variant and its counter are one edit; the running
/// instance is owned by [`crate::dispatch::Dispatcher`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct EagerFallbackCounters {
    /// Batches whose token count exceeded the bucket-ladder maximum.
    pub tokens_above_bucket_ladder_maximum: u64,
    /// Batches with more requests than any captured graph serves.
    pub requests_above_captured_maximum: u64,
    /// Batches the backends' declared support level could not serve.
    pub support_level_insufficient: u64,
    /// Batches that were not uniform decode.
    pub not_uniform_decode: u64,
}

impl EagerFallbackCounters {
    pub(crate) fn count(&mut self, reason: &EagerReason) {
        match reason {
            EagerReason::TokensAboveBucketLadderMaximum {
                token_count: _,
                bucket_ladder_maximum: _,
            } => {
                self.tokens_above_bucket_ladder_maximum += 1;
            }
            EagerReason::RequestsAboveCapturedMaximum {
                request_count: _,
                captured_maximum: _,
            } => {
                self.requests_above_captured_maximum += 1;
            }
            EagerReason::SupportLevelInsufficient {
                support_level: _,
                required: _,
                token_count: _,
                request_count: _,
            } => {
                self.support_level_insufficient += 1;
            }
            EagerReason::NotUniformDecode {
                token_count: _,
                request_count: _,
            } => {
                self.not_uniform_decode += 1;
            }
        }
    }
}

/// Decides `batch`: exactly one graph key, or exactly one reason it runs eagerly.
///
/// Crate-internal: [`crate::dispatch::Dispatcher`] is the only public dispatch surface, so
/// executors cannot re-derive dispatch truth outside this crate.
///
/// Checks run in a fixed order, so a batch failing several lands on the first: token count
/// against the bucket ladder, request count against the captured maximum, uniform decode,
/// then backend support level against the live batch. Uniform decode comes before support so a
/// reason never names a support gap that closing would not actually fix.
///
/// The uniform-decode check trusts no flag alone: a token count that does not divide evenly
/// among the requests cannot be uniform decode, so such a batch runs eagerly whatever it claims.
///
/// # Errors
///
/// Returns the [`EagerReason`] naming the first failed check, carrying the numbers that
/// caused it.
pub(crate) fn decide(
    batch: LiveBatch,
    support_level: SupportLevel,
    captured_max_requests: RequestCount,
    lookup: &PaddingLookup,
) -> Result<GraphKey, EagerReason> {
    let Some(padded_token_count) = lookup.bucket_for(batch.token_count) else {
        return Err(EagerReason::TokensAboveBucketLadderMaximum {
            token_count: batch.token_count,
            bucket_ladder_maximum: lookup.bucket_ladder_maximum(),
        });
    };
    if batch.request_count > captured_max_requests {
        return Err(EagerReason::RequestsAboveCapturedMaximum {
            request_count: batch.request_count,
            captured_maximum: captured_max_requests,
        });
    }
    if !batch.uniform_decode
        || !batch
            .token_count
            .get()
            .is_multiple_of(batch.request_count.get())
    {
        return Err(EagerReason::NotUniformDecode {
            token_count: batch.token_count,
            request_count: batch.request_count,
        });
    }
    let required = required_support(batch);
    if support_level < required {
        return Err(EagerReason::SupportLevelInsufficient {
            support_level,
            required,
            token_count: batch.token_count,
            request_count: batch.request_count,
        });
    }
    Ok(GraphKey::from_padded_batch(
        padded_token_count,
        batch.request_count,
        batch.uniform_decode,
    ))
}

/// Renders the bucket-ladder half of the tokens reason: the maximum, or the empty ladder.
fn bucket_ladder_maximum_clause(bucket_ladder_maximum: Option<TokenCount>) -> String {
    bucket_ladder_maximum.map_or_else(
        || "the bucket ladder is empty".to_owned(),
        |maximum| format!("the bucket-ladder maximum is {maximum}"),
    )
}

/// The weakest support level whose captured routine is valid for a uniform-decode `batch`.
///
/// Non-uniform batches go eager before support is judged, so no batch ever requires
/// [`SupportLevel::Always`] here.
fn required_support(batch: LiveBatch) -> SupportLevel {
    if batch.token_count.get() == batch.request_count.get() {
        SupportLevel::UniformSingleTokenDecode
    } else {
        SupportLevel::UniformBatch
    }
}

#[cfg(test)]
mod tests {
    use super::{decide, EagerReason, SupportLevel};
    use crate::dispatch::test_support::batch;
    use crate::dispatch::{BucketLadder, PaddingLookup, Platform};
    use crate::test_support::{requests, tokens};

    fn hopper_lookup() -> PaddingLookup {
        PaddingLookup::new(&BucketLadder::default_for(Platform::Hopper))
    }

    #[test]
    fn keyed_batch_pads_to_its_bucket_and_binds_its_request_count() {
        let key = decide(
            batch(5, 5, true),
            SupportLevel::Always,
            requests(512),
            &hopper_lookup(),
        )
        .unwrap();
        assert_eq!(key.padded_token_count(), tokens(8));
        assert_eq!(key.request_count(), requests(5));
        assert!(key.uniform_decode());
    }

    #[test]
    fn keys_bind_the_exact_request_count_not_only_the_bucket() {
        let lookup = hopper_lookup();
        let five = decide(
            batch(5, 5, true),
            SupportLevel::Always,
            requests(512),
            &lookup,
        )
        .unwrap();
        let six = decide(
            batch(6, 6, true),
            SupportLevel::Always,
            requests(512),
            &lookup,
        )
        .unwrap();
        assert_eq!(five.padded_token_count(), six.padded_token_count());
        assert_ne!(five, six);
    }

    #[test]
    fn bucket_ladder_boundary_keys_exactly_and_goes_eager_one_past() {
        let lookup = hopper_lookup();
        let at_max = decide(
            batch(512, 512, true),
            SupportLevel::Always,
            requests(512),
            &lookup,
        )
        .unwrap();
        assert_eq!(at_max.padded_token_count(), tokens(512));
        assert_eq!(
            decide(
                batch(513, 513, true),
                SupportLevel::Always,
                requests(1024),
                &lookup
            ),
            Err(EagerReason::TokensAboveBucketLadderMaximum {
                token_count: tokens(513),
                bucket_ladder_maximum: Some(tokens(512)),
            })
        );
    }

    #[test]
    fn requests_above_captured_maximum_go_eager_with_both_numbers() {
        assert_eq!(
            decide(
                batch(8, 8, true),
                SupportLevel::Always,
                requests(4),
                &hopper_lookup()
            ),
            Err(EagerReason::RequestsAboveCapturedMaximum {
                request_count: requests(8),
                captured_maximum: requests(4),
            })
        );
    }

    #[test]
    fn captured_maximum_boundary_keys_exactly() {
        assert!(decide(
            batch(4, 4, true),
            SupportLevel::Always,
            requests(4),
            &hopper_lookup()
        )
        .is_ok());
    }

    #[test]
    fn single_token_decode_needs_at_least_that_support_level() {
        let lookup = hopper_lookup();
        assert!(decide(
            batch(8, 8, true),
            SupportLevel::UniformSingleTokenDecode,
            requests(512),
            &lookup
        )
        .is_ok());
        assert_eq!(
            decide(
                batch(8, 8, true),
                SupportLevel::Never,
                requests(512),
                &lookup
            ),
            Err(EagerReason::SupportLevelInsufficient {
                support_level: SupportLevel::Never,
                required: SupportLevel::UniformSingleTokenDecode,
                token_count: tokens(8),
                request_count: requests(8),
            })
        );
    }

    #[test]
    fn uniform_multi_token_decode_requires_uniform_batch_support() {
        // Four requests decoding four tokens each: uniform, but not single-token.
        let lookup = hopper_lookup();
        assert_eq!(
            decide(
                batch(16, 4, true),
                SupportLevel::UniformSingleTokenDecode,
                requests(512),
                &lookup
            ),
            Err(EagerReason::SupportLevelInsufficient {
                support_level: SupportLevel::UniformSingleTokenDecode,
                required: SupportLevel::UniformBatch,
                token_count: tokens(16),
                request_count: requests(4),
            })
        );
        assert!(decide(
            batch(16, 4, true),
            SupportLevel::UniformBatch,
            requests(512),
            &lookup
        )
        .is_ok());
    }

    #[test]
    fn non_uniform_batch_goes_eager_even_at_full_support() {
        assert_eq!(
            decide(
                batch(16, 4, false),
                SupportLevel::Always,
                requests(512),
                &hopper_lookup()
            ),
            Err(EagerReason::NotUniformDecode {
                token_count: tokens(16),
                request_count: requests(4),
            })
        );
    }

    #[test]
    fn non_uniform_batch_goes_eager_as_not_uniform_regardless_of_support() {
        // Not a support gap: no support level keys a non-uniform batch, so the reason must
        // name the uniformity, not a level that closing would not fix.
        assert_eq!(
            decide(
                batch(16, 4, false),
                SupportLevel::UniformBatch,
                requests(512),
                &hopper_lookup()
            ),
            Err(EagerReason::NotUniformDecode {
                token_count: tokens(16),
                request_count: requests(4),
            })
        );
    }

    #[test]
    fn tokens_that_cannot_divide_evenly_go_eager_as_not_uniform() {
        // Five tokens over two requests, or two tokens over four, cannot be uniform decode
        // whatever the flag claims.
        let lookup = hopper_lookup();
        for (token_count, request_count) in [(5, 2), (2, 4)] {
            assert_eq!(
                decide(
                    batch(token_count, request_count, true),
                    SupportLevel::Always,
                    requests(512),
                    &lookup
                ),
                Err(EagerReason::NotUniformDecode {
                    token_count: tokens(token_count),
                    request_count: requests(request_count),
                })
            );
        }
    }

    #[test]
    fn batch_failing_every_check_lands_on_the_first_reason() {
        // Above the bucket ladder, above the captured maximum, unsupported and not uniform.
        assert_eq!(
            decide(
                batch(600, 600, false),
                SupportLevel::Never,
                requests(4),
                &hopper_lookup()
            ),
            Err(EagerReason::TokensAboveBucketLadderMaximum {
                token_count: tokens(600),
                bucket_ladder_maximum: Some(tokens(512)),
            })
        );
    }

    #[test]
    fn empty_bucket_ladder_sends_everything_eager_with_no_maximum() {
        let lookup = PaddingLookup::new(&BucketLadder::new(Vec::new()).unwrap());
        assert_eq!(
            decide(
                batch(1, 1, true),
                SupportLevel::Always,
                requests(512),
                &lookup
            ),
            Err(EagerReason::TokensAboveBucketLadderMaximum {
                token_count: tokens(1),
                bucket_ladder_maximum: None,
            })
        );
    }

    #[test]
    fn eager_reasons_render_their_numbers() {
        let reason = decide(
            batch(600, 600, true),
            SupportLevel::Always,
            requests(512),
            &hopper_lookup(),
        )
        .unwrap_err();
        assert_eq!(
            reason.to_string(),
            "token count 600 exceeds every captured bucket; the bucket-ladder maximum is 512"
        );
    }

    #[test]
    fn empty_bucket_ladder_reason_names_the_empty_ladder() {
        let lookup = PaddingLookup::new(&BucketLadder::new(Vec::new()).unwrap());
        let reason = decide(
            batch(1, 1, true),
            SupportLevel::Always,
            requests(512),
            &lookup,
        )
        .unwrap_err();
        assert_eq!(
            reason.to_string(),
            "token count 1 exceeds every captured bucket; the bucket ladder is empty"
        );
    }

    #[test]
    fn support_levels_order_weakest_to_strongest() {
        assert!(SupportLevel::Never < SupportLevel::UniformSingleTokenDecode);
        assert!(SupportLevel::UniformSingleTokenDecode < SupportLevel::UniformBatch);
        assert!(SupportLevel::UniformBatch < SupportLevel::Always);
    }
}
