//! The bucket ladder: the ordered list of buckets the engine captures.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// The GPU platform a default bucket ladder is sized for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    /// Hopper-class parts (SM90), carrying the baseline default.
    Hopper,
    /// Data-center Blackwell parts.
    DataCenterBlackwell,
}

impl Platform {
    /// The largest bucket the default ladder reaches on this platform.
    ///
    /// Contract defaults (#188): 512 by default, 1024 on data-center Blackwell.
    #[must_use]
    pub fn ladder_maximum(self) -> usize {
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
    /// Builds a ladder from `buckets`, preserving order and duplicates.
    ///
    /// # Errors
    ///
    /// Returns [`LadderError::ZeroSizedBucket`] if any entry is zero.
    pub fn new(buckets: Vec<usize>) -> Result<Self, LadderError> {
        if let Some(index) = buckets.iter().position(|&bucket| bucket == 0) {
            return Err(LadderError::ZeroSizedBucket { index });
        }
        Ok(Self {
            buckets: buckets.into_boxed_slice(),
        })
    }

    /// The default ladder for `platform`: 1, 2 and 4, then 8 to 128 in steps of 8, then 192 to
    /// the platform's maximum in steps of 64.
    #[must_use]
    pub fn default_for(platform: Platform) -> Self {
        let mut buckets = vec![1, 2, 4];
        buckets.extend((8..=128).step_by(8));
        buckets.extend((192..=platform.ladder_maximum()).step_by(64));
        Self {
            buckets: buckets.into_boxed_slice(),
        }
    }

    /// The buckets in configured order, in tokens per bucket.
    #[must_use]
    pub fn buckets(&self) -> &[usize] {
        &self.buckets
    }

    /// The largest bucket, or zero for an empty ladder.
    #[must_use]
    pub fn maximum(&self) -> usize {
        self.buckets.iter().copied().max().unwrap_or(0)
    }
}

impl TryFrom<Vec<usize>> for BucketLadder {
    type Error = LadderError;

    fn try_from(buckets: Vec<usize>) -> Result<Self, Self::Error> {
        Self::new(buckets)
    }
}

impl From<BucketLadder> for Vec<usize> {
    fn from(ladder: BucketLadder) -> Self {
        ladder.buckets.into_vec()
    }
}

/// A bucket-ladder configuration the engine could never capture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum LadderError {
    /// A bucket of zero tokens can never serve a batch.
    #[error("bucket at index {index} has zero size; every bucket must hold at least one token")]
    ZeroSizedBucket {
        /// Position of the offending entry in the configured list.
        index: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_hopper_ladder_is_the_contract_list() {
        let ladder = BucketLadder::default_for(Platform::Hopper);
        let expected = [
            1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 192, 256,
            320, 384, 448, 512,
        ];
        assert_eq!(ladder.buckets(), expected);
        assert_eq!(ladder.maximum(), 512);
    }

    #[test]
    fn default_blackwell_ladder_extends_hopper_to_1024() {
        let ladder = BucketLadder::default_for(Platform::DataCenterBlackwell);
        let hopper = BucketLadder::default_for(Platform::Hopper);
        let (shared, extension) = ladder.buckets().split_at(hopper.buckets().len());
        assert_eq!(shared, hopper.buckets());
        assert_eq!(extension, [576, 640, 704, 768, 832, 896, 960, 1024]);
        assert_eq!(ladder.maximum(), 1024);
    }

    #[test]
    fn ladder_preserves_order_and_duplicates() {
        let ladder = BucketLadder::new(vec![64, 8, 8, 32]).unwrap();
        assert_eq!(ladder.buckets(), [64, 8, 8, 32]);
        assert_eq!(ladder.maximum(), 64);
    }

    #[test]
    fn zero_sized_bucket_is_rejected_with_its_index() {
        assert_eq!(
            BucketLadder::new(vec![8, 0, 32]),
            Err(LadderError::ZeroSizedBucket { index: 1 })
        );
    }

    #[test]
    fn empty_ladder_has_zero_maximum() {
        let ladder = BucketLadder::new(Vec::new()).unwrap();
        assert!(ladder.buckets().is_empty());
        assert_eq!(ladder.maximum(), 0);
    }

    #[test]
    fn ladder_round_trips_through_config_json() {
        let ladder: BucketLadder = serde_json::from_str("[1, 2, 4, 8]").unwrap();
        assert_eq!(ladder.buckets(), [1, 2, 4, 8]);
        assert_eq!(serde_json::to_string(&ladder).unwrap(), "[1,2,4,8]");
    }

    #[test]
    fn zero_sized_bucket_is_rejected_in_config_position_too() {
        let error = serde_json::from_str::<BucketLadder>("[1, 0]").unwrap_err();
        assert!(error.to_string().contains("zero size"));
    }
}
