//! Deterministic per-step input variation: new token ids, advancing sequence lengths, and block
//! tables that change as sequences cross page boundaries — the replay-variation axis of the
//! capture matrix.
//!
//! The plan is pure host logic and fully deterministic from its seed, so an eager step and a
//! replayed step consume byte-identical inputs, and a failing step reproduces from
//! `(seed, step index)` alone.

use anyhow::{bail, Result};

/// SplitMix64: a tiny deterministic generator so the probe needs no RNG dependency.
#[derive(Debug, Clone)]
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Host-side inputs of one decode step, written into device staging and copied D2D into the
/// graph's static input buffers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepInputs {
    /// New token id per sequence, `[batch_size]`.
    pub token_ids: Vec<u32>,
    /// Key length per sequence including this step's token, `[batch_size]` — what
    /// `cu_seqlens_k` carries in non-cumulative form.
    pub seqlens_k: Vec<i32>,
    /// Row-major `[batch_size, max_blocks_per_seq]`, unused tail entries zero.
    pub block_table: Vec<i32>,
    /// KV-cache slot of this step's token per sequence, `[batch_size]`.
    pub slot_mapping: Vec<i64>,
}

/// Configuration of a [`VariationPlan`].
#[derive(Debug, Clone, Copy)]
pub struct PlanConfig {
    pub batch_size: usize,
    pub page_block: usize,
    pub max_blocks_per_seq: usize,
    /// Cache blocks the fake pool hands out; exhaustion is a construction-time error.
    pub total_blocks: usize,
    /// Tokens already in the cache at step 0 for sequence 0; sequence `i` starts at
    /// `start_seqlen + (i % page_block)` so page-boundary crossings rotate across sequences and
    /// the block table changes on every step once `batch_size >= page_block`.
    pub start_seqlen: usize,
    /// Steps the plan must be able to run without exceeding `max_blocks_per_seq`.
    pub planned_steps: usize,
    pub vocab: usize,
    pub seed: u64,
}

/// The per-cell input generator: one decode token per sequence per step.
#[derive(Debug, Clone)]
pub struct VariationPlan {
    rng: SplitMix64,
    cfg: PlanConfig,
    next_free_block: i32,
    /// Tokens already in the cache per sequence.
    seqlens: Vec<usize>,
    /// Allocated cache block ids per sequence.
    blocks: Vec<Vec<i32>>,
}

impl VariationPlan {
    /// Builds the plan and pre-allocates each sequence's initial blocks, failing loudly when the
    /// planned steps cannot fit the per-sequence block budget or the pool.
    pub fn new(cfg: PlanConfig) -> Result<Self> {
        let worst_final_len = cfg.start_seqlen + cfg.page_block - 1 + cfg.planned_steps;
        let max_tokens = cfg.max_blocks_per_seq * cfg.page_block;
        if worst_final_len > max_tokens {
            bail!(
                "plan overflows the block table: start_seqlen {} + stagger {} + planned_steps {} \
                 exceeds max_blocks_per_seq {} * page_block {} = {} tokens; raise max_seqlen or \
                 lower the step count",
                cfg.start_seqlen,
                cfg.page_block - 1,
                cfg.planned_steps,
                cfg.max_blocks_per_seq,
                cfg.page_block,
                max_tokens
            );
        }
        let worst_blocks = cfg.batch_size * worst_final_len.div_ceil(cfg.page_block);
        if worst_blocks > cfg.total_blocks {
            bail!(
                "plan overflows the block pool: {} sequences * {} blocks each exceeds the pool \
                 of {} blocks; grow the KV pool or shrink the bucket",
                cfg.batch_size,
                worst_final_len.div_ceil(cfg.page_block),
                cfg.total_blocks
            );
        }

        let mut plan = Self {
            rng: SplitMix64::new(cfg.seed),
            cfg,
            next_free_block: 0,
            seqlens: Vec::with_capacity(cfg.batch_size),
            blocks: vec![Vec::new(); cfg.batch_size],
        };
        for seq in 0..cfg.batch_size {
            let seqlen = cfg.start_seqlen + (seq % cfg.page_block);
            plan.seqlens.push(seqlen);
            for _ in 0..seqlen.div_ceil(cfg.page_block) {
                let block = plan.alloc_block();
                plan.blocks[seq].push(block);
            }
        }
        Ok(plan)
    }

    /// Produces the next step's inputs and advances every sequence by one token.
    pub fn next_step(&mut self) -> StepInputs {
        let cfg = self.cfg;
        let mut inputs = StepInputs {
            token_ids: Vec::with_capacity(cfg.batch_size),
            seqlens_k: Vec::with_capacity(cfg.batch_size),
            block_table: vec![0; cfg.batch_size * cfg.max_blocks_per_seq],
            slot_mapping: Vec::with_capacity(cfg.batch_size),
        };
        for seq in 0..cfg.batch_size {
            let position = self.seqlens[seq];
            if position / cfg.page_block == self.blocks[seq].len() {
                let block = self.alloc_block();
                self.blocks[seq].push(block);
            }
            let slot = i64::from(self.blocks[seq][position / cfg.page_block])
                * cfg.page_block as i64
                + (position % cfg.page_block) as i64;

            inputs
                .token_ids
                .push((self.rng.next_u64() % cfg.vocab as u64) as u32);
            inputs
                .seqlens_k
                .push(i32::try_from(position + 1).expect("seqlen fits i32"));
            inputs.slot_mapping.push(slot);
            let row = &mut inputs.block_table
                [seq * cfg.max_blocks_per_seq..(seq + 1) * cfg.max_blocks_per_seq];
            row[..self.blocks[seq].len()].copy_from_slice(&self.blocks[seq]);

            self.seqlens[seq] = position + 1;
        }
        inputs
    }

    fn alloc_block(&mut self) -> i32 {
        let block = self.next_free_block;
        assert!(
            (block as usize) < self.cfg.total_blocks,
            "block pool exhausted at block {block}: the construction-time capacity check missed \
             a case"
        );
        self.next_free_block += 1;
        block
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> PlanConfig {
        PlanConfig {
            batch_size: 2,
            page_block: 4,
            max_blocks_per_seq: 8,
            total_blocks: 64,
            start_seqlen: 3,
            planned_steps: 8,
            vocab: 1000,
            seed: 7,
        }
    }

    #[test]
    fn slots_and_block_tables_match_the_hand_computed_layout() {
        // Sequence 0 starts at 3 tokens (1 block: id 0); sequence 1 at 4 tokens (1 block: id 1).
        let mut plan = VariationPlan::new(config()).unwrap();

        // Step 1: seq 0 writes position 3 into block 0, slot 0*4+3 = 3. Seq 1 writes position 4,
        // which needs a new block (id 2), slot 2*4+0 = 8.
        let step = plan.next_step();
        assert_eq!(step.seqlens_k, [4, 5]);
        assert_eq!(step.slot_mapping, [3, 8]);
        assert_eq!(&step.block_table[0..8], [0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(&step.block_table[8..16], [1, 2, 0, 0, 0, 0, 0, 0]);

        // Step 2: seq 0 writes position 4 into new block 3, slot 12; seq 1 position 5, slot 9.
        let step = plan.next_step();
        assert_eq!(step.seqlens_k, [5, 6]);
        assert_eq!(step.slot_mapping, [12, 9]);
        assert_eq!(&step.block_table[0..8], [0, 3, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn token_ids_are_deterministic_from_the_seed_and_in_vocab() {
        let mut lhs = VariationPlan::new(config()).unwrap();
        let mut rhs = VariationPlan::new(config()).unwrap();
        for _ in 0..8 {
            let (lhs_step, rhs_step) = (lhs.next_step(), rhs.next_step());
            assert_eq!(lhs_step, rhs_step);
            assert!(lhs_step.token_ids.iter().all(|&t| t < 1000));
        }
    }

    #[test]
    fn a_full_page_of_steps_changes_the_block_table_every_step() {
        let cfg = PlanConfig {
            batch_size: 4,
            page_block: 4,
            ..config()
        };
        let mut plan = VariationPlan::new(cfg).unwrap();
        let mut previous = plan.next_step().block_table;
        for _ in 0..4 {
            let table = plan.next_step().block_table;
            assert_ne!(table, previous, "staggered starts must rotate block growth");
            previous = table;
        }
    }

    #[test]
    fn overflowing_the_block_table_is_a_construction_error() {
        let cfg = PlanConfig {
            planned_steps: 1000,
            ..config()
        };
        let err = VariationPlan::new(cfg).unwrap_err().to_string();
        assert!(err.contains("overflows the block table"));
    }

    #[test]
    fn overflowing_the_pool_is_a_construction_error() {
        let cfg = PlanConfig {
            total_blocks: 2,
            ..config()
        };
        let err = VariationPlan::new(cfg).unwrap_err().to_string();
        assert!(err.contains("overflows the block pool"));
    }
}
