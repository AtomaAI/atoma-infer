//! One step's inputs written into staging at full width, ready to upload.
//!
//! The captured graphs bake one device buffer per input, sized at the largest bucket, and read
//! the leading rows of each. Staging is the host mirror of those buffers, written from the batch
//! layout before every step and uploaded as it stands. The block table is staged at the full
//! width a sequence can reach, never at the layout's batch-local width: the width is baked into
//! the attention launch, so it cannot follow the batch.
//!
//! The arrays are borrowed rather than owned so the same fill writes pinned host memory in
//! serving and plain vectors in tests.

use std::fmt;

use thiserror::Error;

use crate::batch::BatchLayout;
use crate::decode::batch::DecodeBatch;

/// One of the five inputs the step reads, as the staging names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StagedInput {
    TokenIds,
    Positions,
    KeyLengths,
    SlotMapping,
    BlockTable,
}

impl fmt::Display for StagedInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            StagedInput::TokenIds => "token ids",
            StagedInput::Positions => "positions",
            StagedInput::KeyLengths => "key lengths",
            StagedInput::SlotMapping => "slot mapping",
            StagedInput::BlockTable => "block table",
        })
    }
}

/// Why the inputs could not be staged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum StagingError {
    #[error("the {input} array holds {len} values; a bucket of {tokens} tokens stages {needed}")]
    ArrayTooShort {
        input: StagedInput,
        len: usize,
        tokens: usize,
        needed: usize,
    },
    #[error("{value} in the {input} does not fit the kernel's 32-bit input")]
    Overflow { input: StagedInput, value: i64 },
    #[error("position {position} is past the rotary tables, which cover {max_position} positions")]
    PositionPastTables {
        position: usize,
        max_position: usize,
    },
}

/// How wide the staged arrays are: the shape every bucket's inputs are cut from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StagingShape {
    /// Rows the arrays hold: the largest bucket.
    pub max_tokens: usize,
    /// Columns of the block table: the blocks a sequence of the model's maximum length holds.
    pub block_table_width: usize,
    /// Positions the rotary tables cover.
    pub max_position: usize,
}

impl StagingShape {
    /// The block-table width `max_model_len` tokens need over `block_size`-token blocks.
    #[must_use]
    pub fn block_table_width(max_model_len: usize, block_size: usize) -> usize {
        max_model_len.div_ceil(block_size)
    }
}

/// The staged arrays, one per input the step reads; each holds at least
/// [`StagingShape::max_tokens`] rows.
pub struct StagingArrays<'a> {
    pub token_ids: &'a mut [u32],
    /// Each token's position: its context length.
    pub positions: &'a mut [i32],
    /// Each sequence's key length after this step's token.
    pub seqlens_k: &'a mut [i32],
    pub slot_mapping: &'a mut [i64],
    /// Row-major, [`StagingShape::block_table_width`] columns per row.
    pub block_table: &'a mut [i32],
}

/// Writes `batch`'s inputs from `layout` into the leading rows of `arrays`, the block table at
/// full width with each row zero-filled past the sequence's blocks.
///
/// # Errors
///
/// Returns [`StagingError`] when an array is shorter than the bucket, a value does not fit the
/// kernel's input, or a position is past the rotary tables.
pub fn stage(
    layout: &BatchLayout,
    batch: &DecodeBatch,
    shape: StagingShape,
    arrays: StagingArrays<'_>,
) -> Result<(), StagingError> {
    let tokens = batch.tokens;
    let width = shape.block_table_width;
    let StagingArrays {
        token_ids,
        positions,
        seqlens_k,
        slot_mapping,
        block_table,
    } = arrays;
    fits(StagedInput::TokenIds, token_ids.len(), tokens, tokens)?;
    fits(StagedInput::Positions, positions.len(), tokens, tokens)?;
    fits(StagedInput::KeyLengths, seqlens_k.len(), tokens, tokens)?;
    fits(StagedInput::SlotMapping, slot_mapping.len(), tokens, tokens)?;
    fits(
        StagedInput::BlockTable,
        block_table.len(),
        tokens,
        tokens * width,
    )?;

    token_ids[..tokens].copy_from_slice(&layout.tokens[..tokens]);
    slot_mapping[..tokens].copy_from_slice(&layout.slot_mapping[..tokens]);
    for (entry, &position) in layout.positions[..tokens].iter().enumerate() {
        let overflow = || StagingError::Overflow {
            input: StagedInput::Positions,
            value: position,
        };
        let index = usize::try_from(position).map_err(|_| overflow())?;
        if index >= shape.max_position {
            return Err(StagingError::PositionPastTables {
                position: index,
                max_position: shape.max_position,
            });
        }
        positions[entry] = i32::try_from(position).map_err(|_| overflow())?;
    }
    for (entry, sequence_len) in layout.sequence_lengths[..tokens].iter().enumerate() {
        seqlens_k[entry] = i32::try_from(*sequence_len).map_err(|_| StagingError::Overflow {
            input: StagedInput::KeyLengths,
            value: i64::from(*sequence_len),
        })?;
    }
    let laid_out = layout.block_table_width;
    for entry in 0..tokens {
        let row = &mut block_table[entry * width..(entry + 1) * width];
        let blocks = &layout.block_tables[entry * laid_out..(entry + 1) * laid_out];
        for (slot, block) in row.iter_mut().zip(blocks) {
            *slot = i32::try_from(*block).map_err(|_| StagingError::Overflow {
                input: StagedInput::BlockTable,
                value: i64::from(*block),
            })?;
        }
        row[laid_out..].fill(0);
    }
    Ok(())
}

/// Holds an array of `len` values to the `needed` a bucket of `tokens` stages.
fn fits(input: StagedInput, len: usize, tokens: usize, needed: usize) -> Result<(), StagingError> {
    if len < needed {
        return Err(StagingError::ArrayTooShort {
            input,
            len,
            tokens,
            needed,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use atoma_core::dispatch::DispatchDecision;
    use atoma_core::step::CommandEntry;

    use super::*;
    use crate::decode::batch::{Checked, DecodeBuckets};
    use crate::test_support::{engine_config, entry, keyed_command, BLOCK_SIZE};

    const MAX_TOKENS: usize = 4;
    const WIDTH: usize = 8;

    fn shape() -> StagingShape {
        StagingShape {
            max_tokens: MAX_TOKENS,
            block_table_width: WIDTH,
            max_position: 32,
        }
    }

    /// Owned arrays of the staging shape, to lend out.
    struct Owned {
        token_ids: Vec<u32>,
        positions: Vec<i32>,
        seqlens_k: Vec<i32>,
        slot_mapping: Vec<i64>,
        block_table: Vec<i32>,
    }

    impl Owned {
        fn new(shape: StagingShape) -> Self {
            Self {
                token_ids: vec![u32::MAX; shape.max_tokens],
                positions: vec![-1; shape.max_tokens],
                seqlens_k: vec![-1; shape.max_tokens],
                slot_mapping: vec![-1; shape.max_tokens],
                block_table: vec![-1; shape.max_tokens * shape.block_table_width],
            }
        }

        fn arrays(&mut self) -> StagingArrays<'_> {
            StagingArrays {
                token_ids: &mut self.token_ids,
                positions: &mut self.positions,
                seqlens_k: &mut self.seqlens_k,
                slot_mapping: &mut self.slot_mapping,
                block_table: &mut self.block_table,
            }
        }
    }

    fn routed(live: Vec<CommandEntry>) -> (BatchLayout, DecodeBatch) {
        let layout = BatchLayout::lay_out(&keyed_command(live), BLOCK_SIZE).unwrap();
        let DispatchDecision::FullReplay(key) = layout.dispatch else {
            panic!("keyed: {:?}", layout.dispatch);
        };
        let buckets = DecodeBuckets::usable(&engine_config().dispatch);
        let Checked::Step(batch) = DecodeBatch::check(&layout, key, &buckets, WIDTH).unwrap()
        else {
            panic!("served by the decode step");
        };
        (layout, batch)
    }

    #[test]
    fn the_block_table_width_is_the_blocks_the_longest_sequence_holds() {
        assert_eq!(StagingShape::block_table_width(32, 4), 8);
        assert_eq!(StagingShape::block_table_width(33, 4), 9);
        assert_eq!(StagingShape::block_table_width(1, 16), 1);
    }

    #[test]
    fn a_padded_batch_stages_its_leading_rows_and_the_block_table_at_full_width() {
        let (layout, batch) = routed(vec![
            entry(1, 3, vec![9], &[10], true),
            entry(2, 8, vec![7], &[20, 21, 22], true),
            entry(3, 1, vec![5], &[30], true),
        ]);
        assert_eq!(batch.tokens, 4, "three live entries and one dummy");
        let mut owned = Owned::new(shape());

        stage(&layout, &batch, shape(), owned.arrays()).unwrap();

        assert_eq!(owned.token_ids[..3], [9, 7, 5]);
        assert_eq!(owned.positions, [3, 8, 1, 0]);
        assert_eq!(owned.seqlens_k, [4, 9, 2, 1]);
        assert_eq!(
            owned.slot_mapping,
            [43, 88, 121, layout.slot_mapping[3]],
            "block times block size plus offset; the dummy's is its own block's first slot"
        );
        let rows: Vec<&[i32]> = owned.block_table.chunks(WIDTH).collect();
        assert_eq!(rows[0], [10, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(rows[1], [20, 21, 22, 0, 0, 0, 0, 0]);
        assert_eq!(rows[2], [30, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(
            rows[3][1..],
            [0; WIDTH - 1],
            "the dummy's row is its block, then zero"
        );
    }

    #[test]
    fn a_smaller_bucket_leaves_the_rows_past_it_untouched() {
        let (layout, batch) = routed(vec![entry(1, 3, vec![9], &[10], true)]);
        let mut owned = Owned::new(shape());

        stage(&layout, &batch, shape(), owned.arrays()).unwrap();

        assert_eq!(owned.token_ids, [9, u32::MAX, u32::MAX, u32::MAX]);
        assert_eq!(owned.positions, [3, -1, -1, -1]);
        assert_eq!(owned.block_table[WIDTH..], vec![-1; 3 * WIDTH]);
    }

    #[test]
    fn an_array_shorter_than_the_bucket_is_refused_by_name() {
        let (layout, batch) = routed(vec![
            entry(1, 3, vec![9], &[10], true),
            entry(2, 3, vec![9], &[20], true),
        ]);
        let mut owned = Owned::new(shape());
        let mut short = vec![0; 1];
        let arrays = StagingArrays {
            token_ids: &mut owned.token_ids,
            positions: &mut owned.positions,
            seqlens_k: &mut short,
            slot_mapping: &mut owned.slot_mapping,
            block_table: &mut owned.block_table,
        };

        assert_eq!(
            stage(&layout, &batch, shape(), arrays).unwrap_err(),
            StagingError::ArrayTooShort {
                input: StagedInput::KeyLengths,
                len: 1,
                tokens: 2,
                needed: 2
            }
        );
    }

    #[test]
    fn a_position_the_rotary_tables_do_not_cover_is_refused() {
        let (layout, batch) = routed(vec![entry(1, 3, vec![9], &[10], true)]);
        let mut owned = Owned::new(shape());
        let short_tables = StagingShape {
            max_position: 3,
            ..shape()
        };

        assert_eq!(
            stage(&layout, &batch, short_tables, owned.arrays()).unwrap_err(),
            StagingError::PositionPastTables {
                position: 3,
                max_position: 3
            }
        );
    }

    #[test]
    fn a_block_id_past_the_kernels_input_width_is_refused() {
        let (mut layout, batch) = routed(vec![entry(1, 3, vec![9], &[10], true)]);
        layout.block_tables[0] = u32::MAX;
        let mut owned = Owned::new(shape());

        assert_eq!(
            stage(&layout, &batch, shape(), owned.arrays()).unwrap_err(),
            StagingError::Overflow {
                input: StagedInput::BlockTable,
                value: i64::from(u32::MAX)
            }
        );
    }
}
