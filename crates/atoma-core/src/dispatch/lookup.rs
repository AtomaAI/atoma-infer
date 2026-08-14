//! Dense lookup from token count to the next captured bucket.

use std::num::NonZeroUsize;

use crate::dispatch::BucketLadder;

/// Maps every token count in range to the next captured bucket — the smallest bucket that holds
/// at least that many tokens.
///
/// Built once from the bucket ladder as a dense table, so a lookup on the step path is one bounds
/// check and one load.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PaddingLookup {
    /// `next_bucket[tokens]` is the smallest bucket holding `tokens`, for counts up to the ladder
    /// maximum.
    next_bucket: Box<[Option<NonZeroUsize>]>,
    ladder_maximum: usize,
}

impl PaddingLookup {
    /// Builds the dense table for `ladder`.
    #[must_use]
    pub fn new(ladder: &BucketLadder) -> Self {
        let ladder_maximum = ladder.maximum();
        let next_bucket = (0..=ladder_maximum)
            .map(|tokens| {
                let smallest_holding = ladder
                    .buckets()
                    .iter()
                    .copied()
                    .filter(|&bucket| bucket >= tokens)
                    .min();
                smallest_holding.and_then(NonZeroUsize::new)
            })
            .collect();
        Self {
            next_bucket,
            ladder_maximum,
        }
    }

    /// The smallest bucket holding `tokens`, or `None` when `tokens` exceeds every bucket.
    #[must_use]
    pub fn bucket_for(&self, tokens: NonZeroUsize) -> Option<NonZeroUsize> {
        self.next_bucket.get(tokens.get()).copied().flatten()
    }

    /// The largest bucket in the ladder this lookup was built from; zero for an empty ladder.
    #[must_use]
    pub fn ladder_maximum(&self) -> usize {
        self.ladder_maximum
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use proptest::prelude::*;

    use super::PaddingLookup;
    use crate::dispatch::{BucketLadder, Platform};

    fn tokens(count: usize) -> NonZeroUsize {
        NonZeroUsize::new(count).expect("test token counts are nonzero")
    }

    /// Test-side oracle: linear scan for the smallest bucket holding `count`.
    fn naive_next_bucket(ladder: &BucketLadder, count: usize) -> Option<usize> {
        ladder
            .buckets()
            .iter()
            .copied()
            .filter(|&bucket| bucket >= count)
            .min()
    }

    #[test]
    fn every_bucket_edge_maps_exactly_for_both_default_ladders() {
        for platform in [Platform::Hopper, Platform::DataCenterBlackwell] {
            let ladder = BucketLadder::default_for(platform);
            let lookup = PaddingLookup::new(&ladder);
            for &bucket in ladder.buckets() {
                // A bucket-sized batch pads to exactly its own bucket.
                assert_eq!(
                    lookup.bucket_for(tokens(bucket)).map(NonZeroUsize::get),
                    Some(bucket)
                );
                // One past a bucket pads to the next bucket up, or to nothing at the top.
                assert_eq!(
                    lookup.bucket_for(tokens(bucket + 1)).map(NonZeroUsize::get),
                    naive_next_bucket(&ladder, bucket + 1)
                );
                // One short of a bucket never overshoots it.
                if bucket > 1 {
                    let below = lookup.bucket_for(tokens(bucket - 1)).map(NonZeroUsize::get);
                    assert!(below.expect("below the maximum, so in range") <= bucket);
                }
            }
        }
    }

    #[test]
    fn hopper_defaults_pad_to_the_documented_buckets() {
        let lookup = PaddingLookup::new(&BucketLadder::default_for(Platform::Hopper));
        let cases = [
            (1, Some(1)),
            (2, Some(2)),
            (3, Some(4)),
            (5, Some(8)),
            (127, Some(128)),
            (128, Some(128)),
            (129, Some(192)),
            (511, Some(512)),
            (512, Some(512)),
            (513, None),
        ];
        for (count, expected) in cases {
            assert_eq!(
                lookup.bucket_for(tokens(count)).map(NonZeroUsize::get),
                expected,
                "token count {count}"
            );
        }
    }

    #[test]
    fn unsorted_ladder_with_duplicates_maps_between_entries() {
        let ladder = BucketLadder::new(vec![64, 8, 8, 32]).unwrap();
        let lookup = PaddingLookup::new(&ladder);
        assert_eq!(lookup.bucket_for(tokens(1)).map(NonZeroUsize::get), Some(8));
        assert_eq!(lookup.bucket_for(tokens(8)).map(NonZeroUsize::get), Some(8));
        assert_eq!(
            lookup.bucket_for(tokens(9)).map(NonZeroUsize::get),
            Some(32)
        );
        assert_eq!(
            lookup.bucket_for(tokens(33)).map(NonZeroUsize::get),
            Some(64)
        );
        assert_eq!(
            lookup.bucket_for(tokens(64)).map(NonZeroUsize::get),
            Some(64)
        );
        assert_eq!(lookup.bucket_for(tokens(65)), None);
        assert_eq!(lookup.ladder_maximum(), 64);
    }

    #[test]
    fn empty_ladder_serves_nothing() {
        let ladder = BucketLadder::new(Vec::new()).unwrap();
        let lookup = PaddingLookup::new(&ladder);
        assert_eq!(lookup.bucket_for(tokens(1)), None);
        assert_eq!(lookup.ladder_maximum(), 0);
    }

    proptest! {
        #[test]
        fn dense_table_agrees_with_naive_scan(
            buckets in proptest::collection::vec(1_usize..=256, 0..=24),
            count in 1_usize..=300,
        ) {
            let ladder = BucketLadder::new(buckets).expect("nonzero buckets are always valid");
            let lookup = PaddingLookup::new(&ladder);
            prop_assert_eq!(
                lookup.bucket_for(tokens(count)).map(NonZeroUsize::get),
                naive_next_bucket(&ladder, count)
            );
        }
    }
}
