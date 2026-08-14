//! Admission: exactly one graph key for a live batch, or exactly one named rejection reason.

use std::num::NonZeroUsize;

use thiserror::Error;

use crate::dispatch::{GraphKey, PaddingLookup};

/// What the scheduler reports about the live batch it wants served.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchShape {
    /// Tokens in the batch before padding.
    pub token_count: NonZeroUsize,
    /// Live requests in the batch.
    pub request_count: NonZeroUsize,
    /// Whether every request in the batch is decoding.
    pub uniform_decode: bool,
}

/// The batch shapes a backend's captured routine is valid for, weakest to strongest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SupportLevel {
    /// The captured routine is never valid.
    Never,
    /// Valid only when every request decodes exactly one token.
    UniformSingleTokenDecode,
    /// Valid for any uniform batch.
    UniformBatch,
    /// Valid for any batch shape.
    Always,
}

/// Why a batch fell back to eager execution, carrying the numbers that caused it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum RejectionReason {
    /// No bucket holds this many tokens.
    #[error(
        "token count {token_count} is above the bucket-ladder maximum {bucket_ladder_maximum}"
    )]
    TokensAboveBucketLadderMaximum {
        /// Tokens in the rejected batch before padding.
        token_count: NonZeroUsize,
        /// The largest captured bucket; zero for an empty bucket ladder.
        bucket_ladder_maximum: usize,
    },
    /// More live requests than any captured graph was built to serve.
    #[error("request count {request_count} is above the captured maximum {captured_maximum}")]
    RequestsAboveCapturedMaximum {
        /// Live requests in the rejected batch.
        request_count: NonZeroUsize,
        /// The largest request count any captured graph serves.
        captured_maximum: usize,
    },
    /// The backends' captured routines are not valid for this batch shape.
    #[error(
        "backend support level {support_level:?} is insufficient for a batch of {token_count} \
         tokens over {request_count} requests, which requires {required:?}"
    )]
    SupportLevelInsufficient {
        /// The support level the active backends declare.
        support_level: SupportLevel,
        /// The weakest level whose captured routine is valid for this batch.
        required: SupportLevel,
        /// Tokens in the rejected batch before padding.
        token_count: NonZeroUsize,
        /// Live requests in the rejected batch.
        request_count: NonZeroUsize,
    },
    /// Only uniform-decode batches are captured.
    #[error("batch of {token_count} tokens over {request_count} requests is not uniform decode")]
    NotUniformDecode {
        /// Tokens in the rejected batch before padding.
        token_count: NonZeroUsize,
        /// Live requests in the rejected batch.
        request_count: NonZeroUsize,
    },
}

/// Admits `batch` to exactly one graph key, or rejects it for exactly one reason.
///
/// Crate-internal: [`crate::dispatch::Dispatcher`] is the only public admission surface, so
/// executors cannot re-derive dispatch truth outside this crate.
///
/// Checks run in a fixed order, so a batch failing several lands on the first: token count
/// against the bucket ladder, request count against the captured maximum, uniform decode,
/// then backend support level against the batch shape. Uniform decode comes before support so a
/// rejection never names a support gap that closing would not actually fix.
///
/// # Errors
///
/// Returns the [`RejectionReason`] naming the first failed check, carrying the numbers that
/// caused it.
pub(crate) fn admit(
    batch: BatchShape,
    support_level: SupportLevel,
    captured_max_requests: usize,
    lookup: &PaddingLookup,
) -> Result<GraphKey, RejectionReason> {
    let Some(padded_token_count) = lookup.bucket_for(batch.token_count) else {
        return Err(RejectionReason::TokensAboveBucketLadderMaximum {
            token_count: batch.token_count,
            bucket_ladder_maximum: lookup.bucket_ladder_maximum(),
        });
    };
    if batch.request_count.get() > captured_max_requests {
        return Err(RejectionReason::RequestsAboveCapturedMaximum {
            request_count: batch.request_count,
            captured_maximum: captured_max_requests,
        });
    }
    if !batch.uniform_decode {
        return Err(RejectionReason::NotUniformDecode {
            token_count: batch.token_count,
            request_count: batch.request_count,
        });
    }
    let required = required_support(batch);
    if support_level < required {
        return Err(RejectionReason::SupportLevelInsufficient {
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

/// The weakest support level whose captured routine is valid for a uniform-decode `batch`.
///
/// Non-uniform batches are rejected before support is judged, so no batch ever requires
/// [`SupportLevel::Always`] here.
fn required_support(batch: BatchShape) -> SupportLevel {
    if batch.token_count == batch.request_count {
        SupportLevel::UniformSingleTokenDecode
    } else {
        SupportLevel::UniformBatch
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::{admit, BatchShape, RejectionReason, SupportLevel};
    use crate::dispatch::{BucketLadder, PaddingLookup, Platform};

    fn count(value: usize) -> NonZeroUsize {
        NonZeroUsize::new(value).expect("test counts are nonzero")
    }

    fn batch(token_count: usize, request_count: usize, uniform_decode: bool) -> BatchShape {
        BatchShape {
            token_count: count(token_count),
            request_count: count(request_count),
            uniform_decode,
        }
    }

    fn hopper_lookup() -> PaddingLookup {
        PaddingLookup::new(&BucketLadder::default_for(Platform::Hopper))
    }

    #[test]
    fn admitted_batch_pads_to_its_bucket_and_binds_its_request_count() {
        let key = admit(
            batch(5, 5, true),
            SupportLevel::Always,
            512,
            &hopper_lookup(),
        )
        .unwrap();
        assert_eq!(key.padded_token_count(), count(8));
        assert_eq!(key.request_count(), count(5));
        assert!(key.uniform_decode());
    }

    #[test]
    fn keys_bind_the_exact_request_count_not_only_the_bucket() {
        let lookup = hopper_lookup();
        let five = admit(batch(5, 5, true), SupportLevel::Always, 512, &lookup).unwrap();
        let six = admit(batch(6, 6, true), SupportLevel::Always, 512, &lookup).unwrap();
        assert_eq!(five.padded_token_count(), six.padded_token_count());
        assert_ne!(five, six);
    }

    #[test]
    fn bucket_ladder_boundary_admits_exactly_and_rejects_one_past() {
        let lookup = hopper_lookup();
        let at_max = admit(batch(512, 512, true), SupportLevel::Always, 512, &lookup).unwrap();
        assert_eq!(at_max.padded_token_count(), count(512));
        assert_eq!(
            admit(batch(513, 513, true), SupportLevel::Always, 1024, &lookup),
            Err(RejectionReason::TokensAboveBucketLadderMaximum {
                token_count: count(513),
                bucket_ladder_maximum: 512,
            })
        );
    }

    #[test]
    fn requests_above_captured_maximum_reject_with_both_numbers() {
        assert_eq!(
            admit(batch(8, 8, true), SupportLevel::Always, 4, &hopper_lookup()),
            Err(RejectionReason::RequestsAboveCapturedMaximum {
                request_count: count(8),
                captured_maximum: 4,
            })
        );
    }

    #[test]
    fn captured_maximum_boundary_admits_exactly() {
        assert!(admit(batch(4, 4, true), SupportLevel::Always, 4, &hopper_lookup()).is_ok());
    }

    #[test]
    fn single_token_decode_needs_at_least_that_support_level() {
        let lookup = hopper_lookup();
        assert!(admit(
            batch(8, 8, true),
            SupportLevel::UniformSingleTokenDecode,
            512,
            &lookup
        )
        .is_ok());
        assert_eq!(
            admit(batch(8, 8, true), SupportLevel::Never, 512, &lookup),
            Err(RejectionReason::SupportLevelInsufficient {
                support_level: SupportLevel::Never,
                required: SupportLevel::UniformSingleTokenDecode,
                token_count: count(8),
                request_count: count(8),
            })
        );
    }

    #[test]
    fn uniform_multi_token_decode_requires_uniform_batch_support() {
        // Four requests decoding four tokens each: uniform, but not single-token.
        let lookup = hopper_lookup();
        assert_eq!(
            admit(
                batch(16, 4, true),
                SupportLevel::UniformSingleTokenDecode,
                512,
                &lookup
            ),
            Err(RejectionReason::SupportLevelInsufficient {
                support_level: SupportLevel::UniformSingleTokenDecode,
                required: SupportLevel::UniformBatch,
                token_count: count(16),
                request_count: count(4),
            })
        );
        assert!(admit(batch(16, 4, true), SupportLevel::UniformBatch, 512, &lookup).is_ok());
    }

    #[test]
    fn non_uniform_batch_is_rejected_even_at_full_support() {
        assert_eq!(
            admit(
                batch(16, 4, false),
                SupportLevel::Always,
                512,
                &hopper_lookup()
            ),
            Err(RejectionReason::NotUniformDecode {
                token_count: count(16),
                request_count: count(4),
            })
        );
    }

    #[test]
    fn non_uniform_batch_is_rejected_as_not_uniform_regardless_of_support() {
        // Not a support gap: no support level admits a non-uniform batch, so the reason must
        // name the uniformity, not a level that closing would not fix.
        assert_eq!(
            admit(
                batch(16, 4, false),
                SupportLevel::UniformBatch,
                512,
                &hopper_lookup()
            ),
            Err(RejectionReason::NotUniformDecode {
                token_count: count(16),
                request_count: count(4),
            })
        );
    }

    #[test]
    fn batch_failing_every_check_lands_on_the_first_reason() {
        // Above the bucket ladder, above the captured maximum, unsupported and not uniform.
        assert_eq!(
            admit(
                batch(600, 600, false),
                SupportLevel::Never,
                4,
                &hopper_lookup()
            ),
            Err(RejectionReason::TokensAboveBucketLadderMaximum {
                token_count: count(600),
                bucket_ladder_maximum: 512,
            })
        );
    }

    #[test]
    fn empty_bucket_ladder_rejects_everything_with_zero_maximum() {
        let lookup = PaddingLookup::new(&BucketLadder::new(Vec::new()).unwrap());
        assert_eq!(
            admit(batch(1, 1, true), SupportLevel::Always, 512, &lookup),
            Err(RejectionReason::TokensAboveBucketLadderMaximum {
                token_count: count(1),
                bucket_ladder_maximum: 0,
            })
        );
    }

    #[test]
    fn rejection_reasons_render_their_numbers() {
        let reason = admit(
            batch(600, 600, true),
            SupportLevel::Always,
            512,
            &hopper_lookup(),
        )
        .unwrap_err();
        assert_eq!(
            reason.to_string(),
            "token count 600 is above the bucket-ladder maximum 512"
        );
    }

    #[test]
    fn support_levels_order_weakest_to_strongest() {
        assert!(SupportLevel::Never < SupportLevel::UniformSingleTokenDecode);
        assert!(SupportLevel::UniformSingleTokenDecode < SupportLevel::UniformBatch);
        assert!(SupportLevel::UniformBatch < SupportLevel::Always);
    }
}
