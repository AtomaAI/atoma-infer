//! Reading a step's selected logits back to the host in one event-fenced copy.
//!
//! The pinned host buffer is allocated once, sized for the most rows a step can select, and every
//! step copies its rows into it with one asynchronous device-to-host copy on the forward's
//! stream, records the buffer's own event behind the copy, and waits on that event and nothing
//! else: no stream synchronize and no device-wide wait, so whatever else the stream holds behind
//! the copy is not waited for.
//!
//! The buffer is pinned as cacheable host memory, never write-combined: the sampler reads every
//! value of every row, and reads from write-combined memory are uncached.

use std::ffi::c_void;
use std::mem::size_of;
use std::slice;
use std::sync::Arc;

use atoma_runtime::error::RuntimeError;
use atoma_runtime::session::Allocation;
use cudarc::driver::result::{free_host, malloc_host, memcpy_dtoh_async};
use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{CudaContext, CudaEvent, CudaStream, DevicePtr, DriverError};
use thiserror::Error;

use crate::logits::Logits;

/// `cuMemHostAlloc` flags: pinned, cacheable, mapped for this context alone.
const CACHEABLE_PINNED: u32 = 0;

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
    /// `max_rows * vocab` values of pinned host memory, owned here and freed on drop.
    buffer: *mut f32,
    /// Recorded behind every copy; waited on before the host reads, and before the buffer is
    /// freed.
    event: CudaEvent,
    vocab: usize,
    max_rows: usize,
}

impl Readback {
    /// A readback for up to `max_rows` rows of `vocab` logits, pinned in `context`'s host
    /// memory during the Allocation session phase, which is taken as a witness.
    ///
    /// # Errors
    ///
    /// Returns [`ReadbackError::Driver`] when the driver cannot pin the buffer or create its
    /// event.
    pub fn new(
        _allocation: &Allocation,
        context: &Arc<CudaContext>,
        max_rows: usize,
        vocab: usize,
    ) -> Result<Self, ReadbackError> {
        let event = context
            .new_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))
            .map_err(RuntimeError::from)?;
        context.bind_to_thread().map_err(RuntimeError::from)?;
        // SAFETY: a driver allocation of the size asked for, freed once, in `Drop`, after the
        // event has fenced the last copy into it.
        let buffer = unsafe { malloc_host(max_rows * vocab * size_of::<f32>(), CACHEABLE_PINNED) }
            .map_err(RuntimeError::from)?
            .cast::<f32>();
        Ok(Self {
            buffer,
            event,
            vocab,
            max_rows,
        })
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
            .context()
            .bind_to_thread()
            .map_err(RuntimeError::from)?;
        // SAFETY: `len` values lie within the buffer, and nothing else reads or writes them: the
        // last read's borrow of this readback ended before this call.
        let host = unsafe { slice::from_raw_parts_mut(self.buffer, len) };
        let (device, _reads) = logits.device_ptr(stream);
        // SAFETY: `device` addresses the `len` values the stream's earlier work wrote, `host` is
        // `len` pinned values, and the event recorded next fences the copy before the host reads.
        unsafe { memcpy_dtoh_async(host, device, stream.cu_stream()) }
            .map_err(RuntimeError::from)?;
        self.event.record(stream).map_err(RuntimeError::from)?;
        self.event.synchronize().map_err(RuntimeError::from)?;
        Ok(Logits::new(host, self.vocab))
    }
}

impl Drop for Readback {
    fn drop(&mut self) {
        // The last copy may still be in flight; the event waits for it before the memory goes.
        // Neither failure can be acted on from a destructor.
        match self.event.synchronize() {
            Ok(()) | Err(DriverError(_)) => {}
        }
        // SAFETY: the pointer came from `malloc_host` and is freed here alone.
        match unsafe { free_host(self.buffer.cast::<c_void>()) } {
            Ok(()) | Err(DriverError(_)) => {}
        }
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
