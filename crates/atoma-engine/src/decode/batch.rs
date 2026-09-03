//! A keyed batch held to its bucket, and the buckets the decode step serves.
//!
//! The engine pads a uniform-decode batch to its bucket and keys it; the executor acts on the key
//! without re-deriving it. What the decode step still has to establish is that the batch it was
//! handed is the shape its graphs bake: one token per entry, as many entries as the bucket, the
//! live entries leading and every one of them sampling, so the logits it reads back are exactly
//! the leading rows. A batch that is keyed but not that shape is not an error: a one-token
//! prefill chunk that does not sample yet, or a bucket the decode step did not resolve, is named
//! and served eagerly. A batch that contradicts its own key is the engine breaking the step
//! protocol, and is refused.

use atoma_core::dispatch::{DispatchConfig, GraphKey};
use atoma_runtime::arena::BucketIdx;
use thiserror::Error;

use crate::batch::BatchLayout;

/// A keyed batch that contradicts its key or the layout protocol: the engine's bug, not a
/// runtime state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum DecodeBatchError {
    #[error("the key is not uniform decode, but only uniform-decode batches are keyed")]
    KeyNotUniformDecode,
    #[error("{entries} entries compute {tokens} tokens; a keyed batch computes one per entry")]
    EntryNotOneToken { entries: usize, tokens: usize },
    #[error("{entries} entries were laid out for a bucket of {bucket}")]
    EntriesNotBucket { entries: usize, bucket: usize },
    #[error("{entries} entries are all {padding} padding dummies")]
    NoLiveEntries { entries: usize, padding: usize },
    #[error("a padding dummy at entry {entry} samples")]
    DummySamples { entry: usize },
    #[error(
        "the block table is {width} blocks wide; the decode step stages {max_width}, the most \
         a sequence can hold"
    )]
    BlockTableTooWide { width: usize, max_width: usize },
}

/// Why a keyed batch runs eagerly after all: a shape the decode step does not serve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum Unservable {
    /// A live entry that does not sample: a prefill chunk whose last token is not the prompt's.
    /// The logits read back are the leading rows, one per live entry, so every live entry must
    /// want one.
    #[error(
        "live entry {entry} does not sample; the decode step reads back one row per live entry"
    )]
    LiveEntryNotSampling { entry: usize },
    /// The key's bucket is above every bucket the decode step resolved.
    #[error("no resolved bucket holds {tokens} tokens; the largest is {largest}")]
    NoBucket { tokens: usize, largest: usize },
}

/// Where a keyed batch runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    /// The decode step, at this bucket.
    Bucket(DecodeBatch),
    /// The eager path, for this reason.
    Eager(Unservable),
}

/// The buckets the decode step serves: the configured ladder's rungs at or below the captured
/// maximum, in configured order. The arena and every slot table are built over exactly these, so
/// a rung's position here is its bucket index everywhere.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeBuckets {
    tokens: Vec<usize>,
}

impl DecodeBuckets {
    /// The rungs of `config`'s ladder a captured decode graph can serve: a uniform-decode batch
    /// has as many tokens as entries, so a rung above the captured request maximum never fills.
    #[must_use]
    pub fn usable(config: &DispatchConfig) -> Self {
        let captured_max = config.captured_max_requests.get();
        Self {
            tokens: config
                .bucket_ladder
                .buckets()
                .iter()
                .copied()
                .filter(|&tokens| tokens <= captured_max)
                .collect(),
        }
    }

    /// The rungs, in bucket-index order.
    #[must_use]
    pub fn tokens(&self) -> &[usize] {
        &self.tokens
    }

    /// The largest rung, or zero when nothing is usable.
    #[must_use]
    pub fn largest(&self) -> usize {
        self.tokens.iter().copied().max().unwrap_or(0)
    }

    /// The bucket serving exactly `tokens`: the first rung of that size.
    #[must_use]
    pub fn index_of(&self, tokens: usize) -> Option<BucketIdx> {
        self.tokens
            .iter()
            .position(|&rung| rung == tokens)
            .map(BucketIdx)
    }
}

/// A keyed batch the decode step serves: its bucket, and how many of the bucket's rows are live.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeBatch {
    pub bucket: BucketIdx,
    /// Entries in the batch, dummies included: the bucket's rung.
    pub tokens: usize,
    /// The leading entries that are live; each samples, so these are the logits rows read back.
    pub live: usize,
    pub key: GraphKey,
}

impl DecodeBatch {
    /// Holds `layout`, keyed by `key`, to the shape the decode step bakes, and finds its bucket
    /// among `buckets`; the block table must fit `max_block_table_width` columns.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeBatchError`] when the layout contradicts its key: an entry computing more
    /// than one token, an entry count that is not the bucket, no live entry, a dummy that
    /// samples, or a block table wider than a sequence can be.
    pub fn route(
        layout: &BatchLayout,
        key: GraphKey,
        buckets: &DecodeBuckets,
        max_block_table_width: usize,
    ) -> Result<Route, DecodeBatchError> {
        if !key.uniform_decode() {
            return Err(DecodeBatchError::KeyNotUniformDecode);
        }
        let entries = layout.entry_count();
        let tokens = layout.token_count();
        if layout.prefill_entries != 0 || tokens != entries {
            return Err(DecodeBatchError::EntryNotOneToken { entries, tokens });
        }
        let bucket = key.padded_token_count().get();
        if entries != bucket {
            return Err(DecodeBatchError::EntriesNotBucket { entries, bucket });
        }
        let padding = layout.padding_count;
        let Some(live) = entries.checked_sub(padding).filter(|&live| live > 0) else {
            return Err(DecodeBatchError::NoLiveEntries { entries, padding });
        };
        if layout.block_table_width > max_block_table_width {
            return Err(DecodeBatchError::BlockTableTooWide {
                width: layout.block_table_width,
                max_width: max_block_table_width,
            });
        }
        // Every entry computes one token, so a selected row is its entry's index, and the live
        // entries lead: the sampling entries are the leading rows exactly when each live entry
        // samples and no dummy does.
        let selected: Vec<usize> = layout.selected.iter().map(|&row| row as usize).collect();
        if let Some(&entry) = selected.iter().find(|&&entry| entry >= live) {
            return Err(DecodeBatchError::DummySamples { entry });
        }
        if let Some(entry) = (0..live).find(|entry| !selected.contains(entry)) {
            return Ok(Route::Eager(Unservable::LiveEntryNotSampling { entry }));
        }
        let Some(index) = buckets.index_of(bucket) else {
            return Ok(Route::Eager(Unservable::NoBucket {
                tokens: bucket,
                largest: buckets.largest(),
            }));
        };
        Ok(Route::Bucket(Self {
            bucket: index,
            tokens: bucket,
            live,
            key,
        }))
    }
}

#[cfg(test)]
mod tests {
    use atoma_core::dispatch::{BucketLadder, DispatchDecision};
    use atoma_core::step::CommandEntry;
    use atoma_core::types::RequestCount;

    use super::*;
    use crate::test_support::{dummy, engine_config, entry, keyed_command, BLOCK_SIZE};

    const MAX_WIDTH: usize = 8;

    fn buckets() -> DecodeBuckets {
        DecodeBuckets::usable(&engine_config().dispatch)
    }

    /// Lays a keyed command out and takes its key.
    fn keyed(live: Vec<CommandEntry>) -> (BatchLayout, GraphKey) {
        let command = keyed_command(live);
        let layout = BatchLayout::lay_out(&command, BLOCK_SIZE).unwrap();
        let DispatchDecision::FullReplay(key) = layout.dispatch else {
            panic!("a uniform-decode batch is keyed: {:?}", layout.dispatch);
        };
        (layout, key)
    }

    #[test]
    fn the_usable_buckets_are_the_ladders_rungs_up_to_the_captured_maximum_in_order() {
        let mut config = engine_config().dispatch;
        config.bucket_ladder = BucketLadder::new(vec![8, 1, 4, 2, 16, 4]).unwrap();
        config.captured_max_requests = RequestCount::new(4).unwrap();

        let buckets = DecodeBuckets::usable(&config);

        assert_eq!(buckets.tokens(), [1, 4, 2, 4]);
        assert_eq!(buckets.largest(), 4);
        assert_eq!(
            buckets.index_of(4),
            Some(BucketIdx(1)),
            "the first rung of four"
        );
        assert_eq!(buckets.index_of(2), Some(BucketIdx(2)));
        assert_eq!(buckets.index_of(8), None);
        assert_eq!(buckets.index_of(3), None);
    }

    #[test]
    fn three_decodes_pad_to_the_bucket_of_four_with_three_live_rows() {
        let (layout, key) = keyed(vec![
            entry(1, 3, vec![9], &[10], true),
            entry(2, 8, vec![9], &[20, 21, 22], true),
            entry(3, 1, vec![9], &[30], true),
        ]);
        assert_eq!(layout.padding_count, 1);

        let route = DecodeBatch::route(&layout, key, &buckets(), MAX_WIDTH).unwrap();

        assert_eq!(
            route,
            Route::Bucket(DecodeBatch {
                bucket: BucketIdx(2),
                tokens: 4,
                live: 3,
                key,
            })
        );
    }

    #[test]
    fn one_decode_fills_the_bucket_of_one_with_no_padding() {
        let (layout, key) = keyed(vec![entry(1, 3, vec![9], &[10], true)]);
        let route = DecodeBatch::route(&layout, key, &buckets(), MAX_WIDTH).unwrap();
        let Route::Bucket(batch) = route else {
            panic!("{route:?}");
        };
        assert_eq!(
            (batch.bucket, batch.tokens, batch.live),
            (BucketIdx(0), 1, 1)
        );
    }

    #[test]
    fn a_one_token_chunk_that_does_not_sample_yet_goes_eager_by_name() {
        let (layout, key) = keyed(vec![
            entry(1, 3, vec![9], &[10], true),
            entry(2, 2, vec![9], &[20, 21], false),
        ]);
        let route = DecodeBatch::route(&layout, key, &buckets(), MAX_WIDTH).unwrap();
        assert_eq!(
            route,
            Route::Eager(Unservable::LiveEntryNotSampling { entry: 1 })
        );
    }

    #[test]
    fn a_bucket_the_decode_step_did_not_resolve_goes_eager_with_the_largest_it_has() {
        let (layout, key) = keyed(vec![
            entry(1, 3, vec![9], &[10], true),
            entry(2, 3, vec![9], &[20], true),
            entry(3, 3, vec![9], &[30], true),
        ]);
        let mut config = engine_config().dispatch;
        config.captured_max_requests = RequestCount::new(2).unwrap();
        let buckets = DecodeBuckets::usable(&config);

        let route = DecodeBatch::route(&layout, key, &buckets, MAX_WIDTH).unwrap();

        assert_eq!(
            route,
            Route::Eager(Unservable::NoBucket {
                tokens: 4,
                largest: 2
            })
        );
    }

    #[test]
    fn a_layout_that_contradicts_its_key_is_refused_by_name() {
        let (mut layout, key) = keyed(vec![
            entry(1, 3, vec![9], &[10], true),
            entry(2, 3, vec![9], &[20], true),
            entry(3, 3, vec![9], &[30], true),
        ]);
        assert_eq!((layout.entry_count(), layout.padding_count), (4, 1));
        let buckets = buckets();

        let mut wide = layout.clone();
        wide.block_table_width = MAX_WIDTH + 1;
        assert_eq!(
            DecodeBatch::route(&wide, key, &buckets, MAX_WIDTH).unwrap_err(),
            DecodeBatchError::BlockTableTooWide {
                width: MAX_WIDTH + 1,
                max_width: MAX_WIDTH
            }
        );

        let mut sampling_dummy = layout.clone();
        sampling_dummy.selected.push(3);
        assert_eq!(
            DecodeBatch::route(&sampling_dummy, key, &buckets, MAX_WIDTH).unwrap_err(),
            DecodeBatchError::DummySamples { entry: 3 }
        );

        let mut all_padding = layout.clone();
        all_padding.padding_count = 4;
        assert_eq!(
            DecodeBatch::route(&all_padding, key, &buckets, MAX_WIDTH).unwrap_err(),
            DecodeBatchError::NoLiveEntries {
                entries: 4,
                padding: 4
            }
        );

        layout.tokens.push(9);
        assert_eq!(
            DecodeBatch::route(&layout, key, &buckets, MAX_WIDTH).unwrap_err(),
            DecodeBatchError::EntryNotOneToken {
                entries: 4,
                tokens: 5
            }
        );
    }

    #[test]
    fn a_batch_laid_out_for_another_bucket_than_its_key_is_refused() {
        let (layout, key) = keyed(vec![
            entry(1, 3, vec![9], &[10], true),
            entry(2, 3, vec![9], &[20], true),
        ]);
        let (_, key_of_one) = keyed(vec![entry(1, 3, vec![9], &[10], true)]);
        assert_ne!(key, key_of_one);
        assert_eq!(
            DecodeBatch::route(&layout, key_of_one, &buckets(), MAX_WIDTH).unwrap_err(),
            DecodeBatchError::EntriesNotBucket {
                entries: 2,
                bucket: 1
            }
        );
    }

    #[test]
    fn a_prefill_in_a_keyed_batch_is_refused_as_not_one_token() {
        let command = keyed_command(vec![entry(1, 3, vec![9], &[10], true)]);
        let mut with_prefill = command.clone();
        with_prefill
            .entries
            .push(entry(2, 0, vec![1, 2, 3], &[20], true));
        with_prefill.entries.push(dummy(3, 30));
        with_prefill.padding_count = 1;
        let layout = BatchLayout::lay_out(&with_prefill, BLOCK_SIZE).unwrap();
        let DispatchDecision::FullReplay(key) = command.dispatch else {
            panic!("keyed");
        };
        assert_eq!(
            DecodeBatch::route(&layout, key, &buckets(), MAX_WIDTH).unwrap_err(),
            DecodeBatchError::EntryNotOneToken {
                entries: 3,
                tokens: 5
            }
        );
    }
}
