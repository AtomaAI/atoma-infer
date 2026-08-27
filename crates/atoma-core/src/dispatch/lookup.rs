//! Dense lookup from token count to the next captured bucket.

use crate::dispatch::BucketLadder;
use crate::protocol::TokenCount;

/// Maps every token count in range to the next captured bucket — the smallest bucket that holds
/// at least that many tokens.
///
/// Built once from the bucket ladder as a dense table, so a lookup on the step path is one
/// bounds check and one load.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PaddingLookup {
    /// `next_bucket[tokens]` is the smallest bucket holding `tokens`, for counts up to the
    /// bucket-ladder maximum.
    next_bucket: Box<[Option<TokenCount>]>,
    bucket_ladder_maximum: Option<TokenCount>,
}

impl PaddingLookup {
    /// Builds the dense table for `bucket_ladder`.
    #[must_use]
    pub fn new(bucket_ladder: &BucketLadder) -> Self {
        let bucket_ladder_maximum = bucket_ladder.maximum();
        let next_bucket = (0..=bucket_ladder_maximum.map_or(0, TokenCount::get))
            .map(|tokens| {
                let smallest_holding = bucket_ladder
                    .buckets()
                    .iter()
                    .copied()
                    .filter(|&bucket| bucket >= tokens)
                    .min();
                smallest_holding.and_then(TokenCount::new)
            })
            .collect();
        Self {
            next_bucket,
            bucket_ladder_maximum,
        }
    }

    /// The smallest bucket holding `tokens`, or `None` when `tokens` exceeds every bucket.
    #[must_use]
    pub fn bucket_for(&self, tokens: TokenCount) -> Option<TokenCount> {
        self.next_bucket.get(tokens.get()).copied().flatten()
    }

    /// The largest bucket in the bucket ladder this lookup was built from; `None` for an empty
    /// bucket ladder.
    #[must_use]
    pub fn bucket_ladder_maximum(&self) -> Option<TokenCount> {
        self.bucket_ladder_maximum
    }
}

#[cfg(test)]
mod tests {
    use proptest::collection::vec;
    use proptest::prelude::*;

    use super::PaddingLookup;
    use crate::dispatch::test_support::tokens;
    use crate::dispatch::{BucketLadder, Platform};
    use crate::protocol::TokenCount;

    /// Test-side oracle, independent of the table's filter-and-min: sort a copy of the ladder
    /// and take the first bucket that holds `token_count`.
    fn sorted_scan_next_bucket(bucket_ladder: &BucketLadder, token_count: usize) -> Option<usize> {
        let mut sorted = bucket_ladder.buckets().to_vec();
        sorted.sort_unstable();
        sorted.into_iter().find(|&bucket| bucket >= token_count)
    }

    #[test]
    fn every_bucket_edge_maps_exactly_for_both_default_bucket_ladders() {
        for platform in [Platform::Hopper, Platform::DataCenterBlackwell] {
            let bucket_ladder = BucketLadder::default_for(platform);
            let lookup = PaddingLookup::new(&bucket_ladder);
            for &bucket in bucket_ladder.buckets() {
                // A bucket-sized batch pads to exactly its own bucket.
                assert_eq!(
                    lookup.bucket_for(tokens(bucket)).map(TokenCount::get),
                    Some(bucket)
                );
                // One past a bucket pads to the next bucket up, or to nothing at the top.
                assert_eq!(
                    lookup.bucket_for(tokens(bucket + 1)).map(TokenCount::get),
                    sorted_scan_next_bucket(&bucket_ladder, bucket + 1)
                );
                // One short of a bucket never overshoots it.
                if bucket > 1 {
                    let below = lookup.bucket_for(tokens(bucket - 1)).map(TokenCount::get);
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
        for (token_count, expected) in cases {
            assert_eq!(
                lookup.bucket_for(tokens(token_count)).map(TokenCount::get),
                expected,
                "token count {token_count}"
            );
        }
    }

    #[test]
    fn unsorted_bucket_ladder_with_duplicates_maps_between_entries() {
        let bucket_ladder = BucketLadder::new(vec![64, 8, 8, 32]).unwrap();
        let lookup = PaddingLookup::new(&bucket_ladder);
        assert_eq!(lookup.bucket_for(tokens(1)).map(TokenCount::get), Some(8));
        assert_eq!(lookup.bucket_for(tokens(8)).map(TokenCount::get), Some(8));
        assert_eq!(lookup.bucket_for(tokens(9)).map(TokenCount::get), Some(32));
        assert_eq!(lookup.bucket_for(tokens(33)).map(TokenCount::get), Some(64));
        assert_eq!(lookup.bucket_for(tokens(64)).map(TokenCount::get), Some(64));
        assert_eq!(lookup.bucket_for(tokens(65)), None);
        assert_eq!(lookup.bucket_ladder_maximum(), TokenCount::new(64));
    }

    #[test]
    fn empty_bucket_ladder_serves_nothing() {
        let bucket_ladder = BucketLadder::new(Vec::new()).unwrap();
        let lookup = PaddingLookup::new(&bucket_ladder);
        assert_eq!(lookup.bucket_for(tokens(1)), None);
        assert_eq!(lookup.bucket_ladder_maximum(), None);
    }

    proptest! {
        #[test]
        fn dense_table_agrees_with_sorted_scan(
            buckets in vec(1_usize..=256, 0..=24),
            token_count in 1_usize..=300,
        ) {
            let bucket_ladder = BucketLadder::new(buckets).expect("nonzero buckets are always valid");
            let lookup = PaddingLookup::new(&bucket_ladder);
            prop_assert_eq!(
                lookup.bucket_for(tokens(token_count)).map(TokenCount::get),
                sorted_scan_next_bucket(&bucket_ladder, token_count)
            );
        }
    }
}
