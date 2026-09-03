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
//!
//! Two paths reach the buffer. The candle forward copies from a device tensor on candle's stream
//! and waits in one call. The tensor path enqueues the copy through the descriptor seam, as the
//! last descriptor of a step on the capture stream, and waits on it separately once the step is
//! enqueued.

use std::ffi::c_void;
use std::mem::size_of;
use std::slice;
use std::sync::Arc;

use atoma_runtime::error::RuntimeError;
use atoma_runtime::session::{Allocation, Descriptor};
use cudarc::driver::result::{event, free_host, malloc_host, memcpy_dtoh_async};
use cudarc::driver::sys::{self, CUevent_flags};
use cudarc::driver::{CudaContext, CudaEvent, CudaStream, DevicePtr};
use thiserror::Error;
use tracing::warn;

use crate::logits::Logits;

/// `cuMemHostAlloc` flags: pinned, cacheable, mapped for this context alone.
pub(crate) const CACHEABLE_PINNED: u32 = 0;

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
    /// A wait with no copy described before it.
    #[error("no readback copy is pending; describe one with `copy` before waiting on it")]
    NoCopyPending,
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
    /// Values the copy described by [`Readback::copy`] brings back, until waited on.
    pending: Option<usize>,
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
            pending: None,
        })
    }

    /// The descriptor that copies `rows` rows of the f32 logits at `device` back on the stream
    /// it is enqueued on, and records the event behind the copy for [`Readback::wait`].
    ///
    /// # Errors
    ///
    /// Returns [`ReadbackError::TooManyRows`] when `rows` is more than the readback holds.
    pub fn copy(&mut self, device: u64, rows: usize) -> Result<ReadbackCopy<'_>, ReadbackError> {
        if rows > self.max_rows {
            return Err(ReadbackError::TooManyRows {
                rows,
                max_rows: self.max_rows,
            });
        }
        let len = rows * self.vocab;
        self.pending = Some(len);
        Ok(ReadbackCopy {
            host: self.buffer,
            len,
            device,
            event: &self.event,
        })
    }

    /// Waits for the copy the last [`Readback::copy`] described, and that copy alone, and
    /// returns its rows.
    ///
    /// # Errors
    ///
    /// Returns [`ReadbackError::NoCopyPending`] when no copy was described since the last wait,
    /// or [`ReadbackError::Driver`] when the wait fails.
    pub fn wait(&mut self) -> Result<Logits<'_>, ReadbackError> {
        let Some(len) = self.pending.take() else {
            return Err(ReadbackError::NoCopyPending);
        };
        self.event.synchronize().map_err(RuntimeError::from)?;
        // SAFETY: `len` values lie within the buffer, the copy into them has completed, and
        // nothing writes them while this borrow is live.
        let host = unsafe { slice::from_raw_parts(self.buffer, len) };
        Ok(Logits::new(host, self.vocab))
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
        self.pending = None;
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
        // Neither failure can be acted on from a destructor beyond saying so.
        if let Err(error) = self.event.synchronize() {
            warn!(%error, "the readback's last copy could not be waited on before its buffer goes");
        }
        // SAFETY: the pointer came from `malloc_host` and is freed here alone.
        if let Err(error) = unsafe { free_host(self.buffer.cast::<c_void>()) } {
            warn!(%error, "the readback's pinned buffer could not be freed");
        }
    }
}

/// One step's copy of its selected logits, enqueued on the capture stream as the step's last
/// descriptor.
pub struct ReadbackCopy<'a> {
    host: *mut f32,
    len: usize,
    device: u64,
    event: &'a CudaEvent,
}

impl Descriptor for ReadbackCopy<'_> {
    type Error = ReadbackError;

    unsafe fn enqueue(&mut self, stream: sys::CUstream) -> Result<(), ReadbackError> {
        // SAFETY: `len` values lie within the pinned buffer and nothing else touches them until
        // the wait; `device` addresses the values the stream's earlier work wrote; the session
        // hands a live stream; and the event recorded behind the copy is what the wait fences.
        unsafe {
            let host = slice::from_raw_parts_mut(self.host, self.len);
            memcpy_dtoh_async(host, self.device, stream).map_err(RuntimeError::from)?;
            event::record(self.event.cu_event(), stream).map_err(RuntimeError::from)?;
        }
        Ok(())
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
