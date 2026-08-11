//! The capture matrix: which (bucket, step-contents) cells the spike captures and replays.

/// What one captured step contains, beyond the full decode step every cell runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum StepContents {
    /// Full decode step including the KV write.
    Decode,
    /// The decode step plus a ws=1 NCCL all-reduce per layer, inside the captured region — the
    /// collective capture-legality check.
    DecodeAllReduce,
}

/// One cell of the capture matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct CaptureCell {
    pub batch_size: usize,
    pub contents: StepContents,
}

impl CaptureCell {
    /// The cell's label in tables and file names, e.g. `bs64` or `bs64+nccl`.
    pub fn label(&self) -> String {
        match self.contents {
            StepContents::Decode => format!("bs{}", self.batch_size),
            StepContents::DecodeAllReduce => format!("bs{}+nccl", self.batch_size),
        }
    }
}

/// Builds the full matrix in execution order: largest bucket first (the harness rule — the
/// largest capture exercises pool high-water first), plain decode before the all-reduce variant
/// within a bucket.
pub fn capture_matrix(buckets: &[usize], include_all_reduce: bool) -> Vec<CaptureCell> {
    let mut buckets = buckets.to_vec();
    buckets.sort_unstable_by(|a, b| b.cmp(a));
    buckets.dedup();
    buckets
        .into_iter()
        .flat_map(|batch_size| {
            let mut cells = vec![CaptureCell {
                batch_size,
                contents: StepContents::Decode,
            }];
            if include_all_reduce {
                cells.push(CaptureCell {
                    batch_size,
                    contents: StepContents::DecodeAllReduce,
                });
            }
            cells
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_runs_largest_bucket_first_with_decode_before_all_reduce() {
        let cells = capture_matrix(&[1, 8, 32, 64], true);
        let labels: Vec<String> = cells.iter().map(CaptureCell::label).collect();
        assert_eq!(
            labels,
            [
                "bs64",
                "bs64+nccl",
                "bs32",
                "bs32+nccl",
                "bs8",
                "bs8+nccl",
                "bs1",
                "bs1+nccl"
            ]
        );
    }

    #[test]
    fn matrix_without_all_reduce_has_one_cell_per_bucket() {
        let cells = capture_matrix(&[8, 1], false);
        assert_eq!(
            cells,
            [
                CaptureCell {
                    batch_size: 8,
                    contents: StepContents::Decode
                },
                CaptureCell {
                    batch_size: 1,
                    contents: StepContents::Decode
                },
            ]
        );
    }

    #[test]
    fn duplicate_buckets_collapse() {
        assert_eq!(capture_matrix(&[8, 8], false).len(), 1);
    }
}
