//! The bucket ladder: the ordered list of buckets the engine captures.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::types::TokenCount;

/// The GPU platform a default bucket ladder is sized for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    /// Hopper-class parts (SM90), carrying the baseline default.
    Hopper,
    /// Data-center Blackwell parts.
    DataCenterBlackwell,
}

impl Platform {
    /// The largest bucket the default bucket ladder reaches on this platform.
    #[must_use]
    pub fn bucket_ladder_maximum(self) -> usize {
        match self {
            Platform::Hopper => 512,
            Platform::DataCenterBlackwell => 1024,
        }
    }
}

/// The ordered list of buckets the engine captures, in tokens per bucket.
///
/// Config-exposed as a plain list of bucket sizes and used uninterpreted: entries keep their
/// configured order and multiplicity — never sorted, deduplicated, or assumed monotonic. The only
/// rejected configuration is a zero-sized bucket, which could never serve a batch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(into = "Vec<usize>", try_from = "Vec<usize>")]
pub struct BucketLadder {
    buckets: Box<[usize]>,
}

impl BucketLadder {
    /// Builds a bucket ladder from `buckets`, preserving order and duplicates.
    ///
    /// # Errors
    ///
    /// Returns [`BucketLadderError::ZeroSizedBucket`] if any entry is zero.
    pub fn new(buckets: Vec<usize>) -> Result<Self, BucketLadderError> {
        if let Some(index) = buckets.iter().position(|&bucket| bucket == 0) {
            return Err(BucketLadderError::ZeroSizedBucket { index });
        }
        Ok(Self {
            buckets: buckets.into_boxed_slice(),
        })
    }

    /// The default bucket ladder for `platform`: 1, 2 and 4, then 8 to 128 in steps of 8,
    /// then 192 to the platform's maximum in steps of 64.
    #[must_use]
    pub fn default_for(platform: Platform) -> Self {
        let mut buckets = vec![1, 2, 4];
        buckets.extend((8..=128).step_by(8));
        buckets.extend((192..=platform.bucket_ladder_maximum()).step_by(64));
        Self {
            buckets: buckets.into_boxed_slice(),
        }
    }

    /// The buckets in configured order, in tokens per bucket.
    #[must_use]
    pub fn buckets(&self) -> &[usize] {
        &self.buckets
    }

    /// The largest bucket, or `None` for an empty bucket ladder.
    #[must_use]
    pub fn maximum(&self) -> Option<TokenCount> {
        self.buckets.iter().copied().max().and_then(TokenCount::new)
    }
}

impl TryFrom<Vec<usize>> for BucketLadder {
    type Error = BucketLadderError;

    fn try_from(buckets: Vec<usize>) -> Result<Self, Self::Error> {
        Self::new(buckets)
    }
}

impl From<BucketLadder> for Vec<usize> {
    fn from(bucket_ladder: BucketLadder) -> Self {
        bucket_ladder.buckets.into_vec()
    }
}

/// A bucket-ladder configuration the engine could never capture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum BucketLadderError {
    /// A bucket of zero tokens can never serve a batch.
    #[error("bucket at index {index} has zero size; every bucket must hold at least one token")]
    ZeroSizedBucket {
        /// Position of the offending entry in the configured list.
        index: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::{BucketLadder, BucketLadderError, Platform};
    use crate::types::TokenCount;

    #[test]
    fn default_hopper_bucket_ladder_is_the_contract_list() {
        let bucket_ladder = BucketLadder::default_for(Platform::Hopper);
        let expected = [
            1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 192, 256,
            320, 384, 448, 512,
        ];
        assert_eq!(bucket_ladder.buckets(), expected);
        assert_eq!(bucket_ladder.maximum(), TokenCount::new(512));
    }

    #[test]
    fn default_blackwell_bucket_ladder_extends_hopper_to_1024() {
        let bucket_ladder = BucketLadder::default_for(Platform::DataCenterBlackwell);
        let hopper = BucketLadder::default_for(Platform::Hopper);
        let (shared, extension) = bucket_ladder.buckets().split_at(hopper.buckets().len());
        assert_eq!(shared, hopper.buckets());
        assert_eq!(extension, [576, 640, 704, 768, 832, 896, 960, 1024]);
        assert_eq!(bucket_ladder.maximum(), TokenCount::new(1024));
    }

    #[test]
    fn bucket_ladder_preserves_order_and_duplicates() {
        let bucket_ladder = BucketLadder::new(vec![64, 8, 8, 32]).unwrap();
        assert_eq!(bucket_ladder.buckets(), [64, 8, 8, 32]);
        assert_eq!(bucket_ladder.maximum(), TokenCount::new(64));
    }

    #[test]
    fn zero_sized_bucket_is_rejected_with_its_index() {
        assert_eq!(
            BucketLadder::new(vec![8, 0, 32]),
            Err(BucketLadderError::ZeroSizedBucket { index: 1 })
        );
    }

    #[test]
    fn empty_bucket_ladder_has_no_maximum() {
        let bucket_ladder = BucketLadder::new(Vec::new()).unwrap();
        assert!(bucket_ladder.buckets().is_empty());
        assert_eq!(bucket_ladder.maximum(), None);
    }

    #[test]
    fn bucket_ladder_round_trips_through_config_json() {
        let bucket_ladder: BucketLadder = serde_json::from_str("[1, 2, 4, 8]").unwrap();
        assert_eq!(bucket_ladder.buckets(), [1, 2, 4, 8]);
        assert_eq!(serde_json::to_string(&bucket_ladder).unwrap(), "[1,2,4,8]");
    }

    #[test]
    fn zero_sized_bucket_is_rejected_in_config_position_too() {
        let error = serde_json::from_str::<BucketLadder>("[1, 0]").unwrap_err();
        assert!(error.to_string().contains("zero size"));
    }
}
