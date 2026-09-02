//! Reading a step's selected logits back to the host in one event-fenced copy.
//!
//! The pinned host buffer is allocated once, sized for the most rows a step can select, and every
//! step copies its rows into it with one asynchronous device-to-host copy on the forward's
//! stream. The copy records the buffer's own event on that stream, and reading the buffer waits on
//! that event and nothing else: no stream synchronize and no device-wide wait, so whatever else
//! the stream holds behind the copy is not waited for.

use std::sync::Arc;

use atoma_runtime::error::RuntimeError;
use cudarc::driver::{CudaContext, CudaStream, DevicePtr, PinnedHostSlice};
use thiserror::Error;

use crate::logits::Logits;

/// Why a step's logits could not be read back.
#[derive(Debug, Error)]
pub enum ReadbackError {
    /// The forward selected more rows than the readback was sized for.
    #[error("{rows} logits rows were selected but the readback holds {max_rows} at most")]
    TooManyRows { rows: usize, max_rows: usize },
    /// The device logits are not the rows of the vocabulary the forward said it selected.
    #[error("the device logits hold {len} values, not {rows} rows of {vocab}")]
    Shape {
        len: usize,
        rows: usize,
        vocab: usize,
    },
    #[error(transparent)]
    Driver(#[from] RuntimeError),
}

/// The pinned host buffer a step's selected logits are copied into.
pub struct Readback {
    buffer: PinnedHostSlice<f32>,
    vocab: usize,
    max_rows: usize,
}

impl Readback {
    /// A readback for up to `max_rows` rows of `vocab` logits, pinned in `context`'s host
    /// memory.
    ///
    /// # Errors
    ///
    /// Returns [`ReadbackError::Driver`] when the driver cannot pin the buffer.
    pub fn new(
        context: &Arc<CudaContext>,
        max_rows: usize,
        vocab: usize,
    ) -> Result<Self, ReadbackError> {
        // SAFETY: the memory is unset after the call; it is zeroed on the next line before
        // anything reads it.
        let mut buffer =
            unsafe { context.alloc_pinned::<f32>(max_rows * vocab) }.map_err(RuntimeError::from)?;
        buffer.as_mut_slice().map_err(RuntimeError::from)?.fill(0.0);
        Ok(Self {
            buffer,
            vocab,
            max_rows,
        })
    }

    #[must_use]
    pub fn max_rows(&self) -> usize {
        self.max_rows
    }

    #[must_use]
    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// Copies `rows` rows of `logits` back on `stream` and waits for that copy alone.
    ///
    /// # Errors
    ///
    /// Returns [`ReadbackError`] when `rows` is more than the readback holds, `logits` is not
    /// that many rows of the vocabulary, or the driver fails the copy or the wait.
    pub fn read<S: DevicePtr<f32>>(
        &mut self,
        stream: &Arc<CudaStream>,
        logits: &S,
        rows: usize,
    ) -> Result<Logits<'_>, ReadbackError> {
        let len = selected_len(rows, self.vocab, self.max_rows, logits.len())?;
        stream
            .memcpy_dtoh(logits, &mut self.buffer)
            .map_err(RuntimeError::from)?;
        let host = self.buffer.as_slice().map_err(RuntimeError::from)?;
        Ok(Logits::new(&host[..len], self.vocab))
    }
}

/// How many values `rows` rows of `vocab` are, once they fit the readback and match what the
/// device holds.
fn selected_len(
    rows: usize,
    vocab: usize,
    max_rows: usize,
    device_len: usize,
) -> Result<usize, ReadbackError> {
    if rows > max_rows {
        return Err(ReadbackError::TooManyRows { rows, max_rows });
    }
    let len = rows * vocab;
    if device_len != len {
        return Err(ReadbackError::Shape {
            len: device_len,
            rows,
            vocab,
        });
    }
    Ok(len)
}

#[cfg(test)]
mod tests {
    use super::{selected_len, ReadbackError};

    #[test]
    fn the_copy_is_the_selected_rows_of_the_vocabulary_and_nothing_else() {
        assert_eq!(selected_len(3, 8, 4, 24).unwrap(), 24);
        assert_eq!(
            selected_len(0, 8, 4, 0).unwrap(),
            0,
            "a step selecting nothing"
        );
        assert!(matches!(
            selected_len(5, 8, 4, 40).unwrap_err(),
            ReadbackError::TooManyRows {
                rows: 5,
                max_rows: 4
            }
        ));
        assert!(matches!(
            selected_len(3, 8, 4, 25).unwrap_err(),
            ReadbackError::Shape {
                len: 25,
                rows: 3,
                vocab: 8
            }
        ));
    }
}
