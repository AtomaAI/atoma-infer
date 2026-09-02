//! The forward on the device: a step command's batch arrays uploaded, the Llama forward over the
//! paged KV cache on candle's stream, and the selected logits read back to the host.

use std::sync::Arc;

use atoma_core::step::StepCommand;
use atoma_runtime::session::Replay;
use candle_core::{Storage, Tensor};
use cudarc::driver::CudaStream;
use models::FlashAttentionMetadata;
use thiserror::Error;

use crate::batch::BatchLayout;
use crate::device::{KvCache, RankDevice, Weights};
use crate::forward::Forward;
use crate::logits::Logits;
use crate::readback::{Readback, ReadbackError};

/// Why a step could not be run on the device.
#[derive(Debug, Error)]
pub enum CudaForwardError {
    #[error(transparent)]
    Candle(#[from] candle_core::Error),
    #[error(transparent)]
    Readback(#[from] ReadbackError),
    /// The forward's logits came back on the host, which no device forward should produce.
    #[error("the logits are not on the device")]
    LogitsNotOnDevice,
}

/// What the Allocation session phase produced for one rank.
pub struct Allocated {
    pub device: RankDevice,
    pub weights: Weights,
    pub kv_cache: KvCache,
    /// Rank zero reads its logits back; a follower's are read by nobody, so it holds no readback
    /// and its forward returns no rows.
    pub readback: Option<Readback>,
    pub vocab: usize,
}

/// The model forward on one rank's device.
///
/// Holds the session's Replay phase for the process lifetime: nothing is captured in this crate,
/// and holding the phase is what keeps the allocation from being reopened.
pub struct CudaForward {
    device: RankDevice,
    weights: Weights,
    kv_cache: KvCache,
    readback: Option<Readback>,
    vocab: usize,
    _session: Replay,
}

impl CudaForward {
    #[must_use]
    pub fn new(allocated: Allocated, session: Replay) -> Self {
        let Allocated {
            device,
            weights,
            kv_cache,
            readback,
            vocab,
        } = allocated;
        Self {
            device,
            weights,
            kv_cache,
            readback,
            vocab,
            _session: session,
        }
    }

    /// The forward's inputs and attention metadata, uploaded from `layout`.
    fn upload(&self, layout: &BatchLayout) -> Result<Uploaded, candle_core::Error> {
        let device = self.device.candle();
        let tokens = layout.token_count();
        let entries = layout.entry_count();
        // A step in which no entry samples still runs, to write its KV; selecting one row keeps
        // every tensor downstream of the selection nonempty, and nobody reads that row.
        let selected: &[u32] = if layout.selected.is_empty() {
            &[0]
        } else {
            &layout.selected
        };
        let metadata = FlashAttentionMetadata::new(
            Tensor::from_slice(&layout.context_lengths, entries, device)?,
            Tensor::from_slice(&layout.slot_mapping, tokens, device)?,
            Tensor::from_slice(&layout.query_start_locations, entries + 1, device)?,
            layout.prefill_tokens,
            layout.decode_tokens,
            layout.max_query_len,
            layout.max_decode_sequence_len,
            layout.max_prefill_sequence_len,
            layout.prefill_entries,
            Tensor::from_slice(&layout.sequence_start_locations, entries + 1, device)?,
            Tensor::from_slice(&layout.sequence_lengths, entries, device)?,
            Tensor::from_slice(
                &layout.block_tables,
                (entries, layout.block_table_width),
                device,
            )?,
        )?;
        Ok(Uploaded {
            tokens: Tensor::from_slice(&layout.tokens, (1, tokens), device)?,
            positions: Tensor::from_slice(&layout.positions, (1, tokens), device)?,
            selected: Tensor::from_slice(selected, selected.len(), device)?,
            metadata,
        })
    }
}

/// One step's inputs on the device.
struct Uploaded {
    tokens: Tensor,
    positions: Tensor,
    selected: Tensor,
    metadata: FlashAttentionMetadata,
}

impl Forward for CudaForward {
    type Error = CudaForwardError;

    fn forward(
        &mut self,
        _command: &StepCommand,
        layout: &BatchLayout,
    ) -> Result<Logits<'_>, CudaForwardError> {
        let Uploaded {
            tokens,
            positions,
            selected,
            metadata,
        } = self.upload(layout)?;
        let kv_caches = self.kv_cache.layers_mut();
        let logits = self
            .weights
            .llama_mut()
            .forward(&tokens, &positions, &selected, &kv_caches, metadata)?;
        let rows = layout.selected.len();
        let Some(readback) = &mut self.readback else {
            return Ok(Logits::new(&[], self.vocab));
        };
        if rows == 0 {
            return Ok(Logits::new(&[], self.vocab));
        }
        read_back(readback, self.device.stream(), &logits, rows)
    }
}

/// Copies the `rows` rows of `logits` back through `readback` on `stream`.
fn read_back<'a>(
    readback: &'a mut Readback,
    stream: &Arc<CudaStream>,
    logits: &Tensor,
    rows: usize,
) -> Result<Logits<'a>, CudaForwardError> {
    let (storage, layout) = logits.storage_and_layout();
    let Storage::Cuda(storage) = &*storage else {
        return Err(CudaForwardError::LogitsNotOnDevice);
    };
    let start = layout.start_offset();
    let device_logits = storage
        .as_cuda_slice::<f32>()?
        .slice(start..start + layout.shape().elem_count());
    Ok(readback.read(stream, &device_logits, rows)?)
}
