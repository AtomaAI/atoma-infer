//! The capture session: one graph set moving through its three session phases — Allocation,
//! Capture, Replay — with consuming transitions between them.
//!
//! Each phase is a type permitting only the operations it is named for. [`Allocation`] fixes every
//! device address and binds every stream, BLAS handle and communicator; it is the only phase from
//! which the owning [`CaptureStream`] reference is reachable, so binding a handle after capture
//! has begun cannot be expressed. [`Capture`] warms up and records. [`Replay`] launches — captured
//! graphs and the eager fallback — and its one consuming transition back to [`Allocation`] is
//! [`Replay::reload_weights`]: new weights mean new baked addresses, so the graph set dies with
//! the phase.
//!
//! The session phase and [`CaptureState`] are different truths, and both stay. The state is what
//! the driver reports the stream to be doing at run time; the phase is what the caller may call at
//! compile time. The lifecycle table in [`crate::capture`] still guards every driver transition
//! inside [`Capture::record`], because the driver can invalidate a recording that no phase type
//! could have forbidden.
//!
//! All enqueue onto the capture stream goes through the [`Descriptor`] seam. Kernels reached
//! through a C ABI take a raw stream in their signature and always will; the seam confines that to
//! one implementation per backend instead of one occurrence per call site. Outside this crate the
//! capture stream's raw handle exists only as the argument of a [`Descriptor::enqueue`]
//! implementation — `sys::CUstream` anywhere else in a consuming crate is a review failure.
//!
//! A transition that does not exist has no method to call, so it does not compile. Beginning a
//! capture from the Replay phase:
//!
//! ```compile_fail
//! use atoma_runtime::session::{BakedBuffers, Descriptor, Replay};
//! fn begin<D: Descriptor>(replay: &mut Replay, work: &mut D, buffers: BakedBuffers) {
//!     replay.record(work, buffers);
//! }
//! ```
//!
//! Reaching the owning stream reference after allocation is sealed:
//!
//! ```compile_fail
//! use atoma_runtime::session::Capture;
//! fn bind_late(capture: &Capture) {
//!     capture.stream();
//! }
//! ```
//!
//! Forging a phase from outside:
//!
//! ```compile_fail
//! use atoma_runtime::session::Replay;
//! fn forge() -> Replay {
//!     Replay { entries: Vec::new() }
//! }
//! ```
//!
//! Returning to Allocation without a weight reload:
//!
//! ```compile_fail
//! use atoma_runtime::session::{Allocation, Replay};
//! fn shortcut(replay: Replay) -> Allocation {
//!     replay.seal()
//! }
//! ```
//!
//! Moving any phase of a session off the thread that owns it:
//!
//! ```compile_fail
//! fn executor_thread_only<T: Send>() {}
//! executor_thread_only::<atoma_runtime::session::Allocation>();
//! ```
//!
//! And the legal sequence compiles on a machine with no GPU:
//!
//! ```no_run
//! use atoma_runtime::context::RuntimeContext;
//! use atoma_runtime::error::RuntimeError;
//! use atoma_runtime::session::{Allocation, BakedBuffers, Descriptor};
//! use cudarc::driver::sys;
//!
//! struct Noop;
//! impl Descriptor for Noop {
//!     type Error = RuntimeError;
//!     unsafe fn enqueue(&mut self, _stream: sys::CUstream) -> Result<(), RuntimeError> {
//!         Ok(())
//!     }
//! }
//!
//! fn one_session() -> Result<(), RuntimeError> {
//!     let ctx = RuntimeContext::new(0)?;
//!     let allocation = Allocation::new(&ctx)?;
//!     let mut capture = allocation.seal();
//!     capture.warm_up(&mut Noop)?;
//!     let graph = capture.record(&mut Noop, BakedBuffers::default())?;
//!     let replay = capture.seal();
//!     replay.run(&mut Noop)?;
//!     replay.replay(graph)?;
//!     replay.synchronize()?;
//!     let _reloading = replay.reload_weights();
//!     Ok(())
//! }
//! ```

use std::marker::PhantomData;

use cudarc::driver::{sys, CudaSlice};
#[cfg(feature = "nccl")]
use cudarc::nccl::Comm;

use crate::capture::{self, CaptureState};
use crate::context::RuntimeContext;
use crate::error::RuntimeError;
use crate::graph_entry::GraphEntry;
use crate::stream::CaptureStream;

/// `!Send`/`!Sync` marker: NVIDIA documents graph objects as not internally synchronized, so a
/// session — warmup, capture and replay alike — lives and dies on the executor thread that owns
/// its stream. A session cannot be built on a setup thread and moved.
type ExecutorThreadOnly = PhantomData<*const ()>;

/// A description of device work, enqueued onto the capture stream by the session — the descriptor
/// seam.
///
/// A backend implements this once; that implementation is the only place its C-ABI kernels
/// receive the raw stream handle. An implementation must only enqueue asynchronous work onto the
/// stream it is handed: during [`Capture::record`] a synchronize, a device allocation or free, or
/// work on any other stream invalidates the recording. A backend's one-time lazy allocations
/// (cuBLAS workspaces, first-call setup) are why [`Capture::warm_up`] runs the same descriptor
/// eagerly first.
pub trait Descriptor {
    /// The implementor's error type; the session's own failures convert into it.
    type Error: From<RuntimeError>;

    /// Enqueues the described work onto `stream`.
    ///
    /// # Safety
    /// `stream` must be a live capture-stream handle whose context is current on the calling
    /// thread; the session upholds both for every call it makes.
    unsafe fn enqueue(&mut self, stream: sys::CUstream) -> Result<(), Self::Error>;
}

/// The device buffers whose addresses one recording bakes, handed to [`Capture::record`] so the
/// resulting entry owns them for exactly the graph's lifetime.
#[derive(Default)]
pub struct BakedBuffers {
    /// Written by the host between launches; the graph reads them.
    pub inputs: Vec<CudaSlice<u8>>,
    /// Read back by the host after launches.
    pub outputs: Vec<CudaSlice<u8>>,
    /// Graph-internal: captured copy destinations and kernel workspaces.
    pub workspaces: Vec<CudaSlice<u8>>,
}

/// One recorded graph in a session's set, minted by [`Capture::record`] and only meaningful to
/// the session that minted it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphIdx(usize);

/// The first session phase: fixes every device address and binds every stream, BLAS handle and
/// communicator. Nothing is captured in it, and nothing may be allocated after it.
pub struct Allocation {
    stream: CaptureStream,
    _executor_thread: ExecutorThreadOnly,
}

impl Allocation {
    /// Opens a session on `ctx`'s device with its own dedicated capture side stream.
    pub fn new(ctx: &RuntimeContext) -> Result<Self, RuntimeError> {
        Ok(Self {
            stream: CaptureStream::new(ctx)?,
            _executor_thread: PhantomData,
        })
    }

    /// The owning reference to the capture stream, for binding communicators. Unreachable from
    /// any later phase, so a handle bound after capture has begun cannot be expressed.
    pub fn stream(&self) -> &CaptureStream {
        &self.stream
    }

    /// Seals allocation: every device address is fixed and every handle bound. The stream
    /// reference does not survive the transition.
    #[must_use]
    pub fn seal(self) -> Capture {
        Capture {
            stream: self.stream,
            entries: Vec::new(),
            warmed: false,
            _executor_thread: PhantomData,
        }
    }
}

/// The second session phase: warms up and records, accumulating the graph set entry by entry.
pub struct Capture {
    stream: CaptureStream,
    entries: Vec<GraphEntry>,
    /// Whether a warmup pass ran since the last recording. A run-time detail, not a phase: the
    /// phase says warming and recording are legal here, and this orders the two.
    warmed: bool,
    _executor_thread: ExecutorThreadOnly,
}

impl Capture {
    /// Runs `work` eagerly on the capture stream and waits for it: the warmup pass that lands a
    /// backend's lazy allocations before any recording. Run the exact step at the graph's exact
    /// shape immediately before each [`Capture::record`] of it.
    pub fn warm_up<D: Descriptor>(&mut self, work: &mut D) -> Result<(), D::Error> {
        // SAFETY: the session's own stream, on the thread that owns it.
        unsafe { work.enqueue(self.stream.cu_stream()) }?;
        synchronize_idle(&self.stream)?;
        self.warmed = true;
        Ok(())
    }

    /// Records `work` as one graph on the dedicated side stream in relaxed mode, instantiates it
    /// with flags zero, pre-uploads the executable, and takes ownership of every buffer whose
    /// address the recording baked.
    ///
    /// Each recording consumes the preceding [`Capture::warm_up`]; recording without one is a
    /// named error before anything reaches the driver. A capture the driver invalidated — by the
    /// enqueue failing or by the recording itself — is discarded without instantiating, and the
    /// stream is left idle for the next warmup.
    pub fn record<D: Descriptor>(
        &mut self,
        work: &mut D,
        buffers: BakedBuffers,
    ) -> Result<GraphIdx, D::Error> {
        if !self.warmed {
            return Err(D::Error::from(RuntimeError::RecordWithoutWarmup));
        }
        self.warmed = false;

        self.stream.begin_capture().map_err(D::Error::from)?;
        // SAFETY: the session's own stream, on the thread that owns it.
        if let Err(err) = unsafe { work.enqueue(self.stream.cu_stream()) } {
            discard_recording(&self.stream);
            return Err(err);
        }
        let graph = match capture::end_capture_instantiate(&self.stream) {
            Ok(graph) => graph,
            Err(err) => {
                discard_recording(&self.stream);
                return Err(D::Error::from(err));
            }
        };
        graph.upload().map_err(D::Error::from)?;
        synchronize_idle(&self.stream)?;

        let BakedBuffers {
            inputs,
            outputs,
            workspaces,
        } = buffers;
        self.entries
            .push(GraphEntry::new(inputs, outputs, workspaces, graph));
        Ok(GraphIdx(self.entries.len() - 1))
    }

    /// Attaches the communicator whose collectives entry `idx` recorded, so the teardown order
    /// [`GraphEntry`] declares holds: abort blocks until no live graph references the
    /// communicator. Attach immediately after the record that captured the collective.
    ///
    /// # Panics
    /// Panics when `idx` was minted by a different session.
    #[cfg(feature = "nccl")]
    pub fn attach_comm(&mut self, idx: GraphIdx, comm: Comm) {
        self.entries[idx.0].attach_comm(comm);
    }

    /// The recorded entry, for topology diagnostics between recordings.
    ///
    /// # Panics
    /// Panics when `idx` was minted by a different session.
    pub fn entry(&self, idx: GraphIdx) -> &GraphEntry {
        &self.entries[idx.0]
    }

    /// Seals the graph set: recording is over and only launches remain.
    #[must_use]
    pub fn seal(self) -> Replay {
        Replay {
            stream: self.stream,
            entries: self.entries,
            _executor_thread: PhantomData,
        }
    }
}

/// The third session phase: launches — captured graphs and the eager fallback. No stream
/// reference escapes it and no method records.
pub struct Replay {
    stream: CaptureStream,
    entries: Vec<GraphEntry>,
    _executor_thread: ExecutorThreadOnly,
}

impl Replay {
    /// Enqueues eager work onto the capture stream: the per-step input uploads and the eager
    /// fallback path. Launch only — pair with [`Replay::synchronize`].
    pub fn run<D: Descriptor>(&self, work: &mut D) -> Result<(), D::Error> {
        // SAFETY: the session's own stream, on the thread that owns it.
        unsafe { work.enqueue(self.stream.cu_stream()) }
    }

    /// Launches entry `idx`'s executable on the stream it was captured from. Launch only — pair
    /// with [`Replay::synchronize`].
    ///
    /// # Panics
    /// Panics when `idx` was minted by a different session.
    pub fn replay(&self, idx: GraphIdx) -> Result<(), RuntimeError> {
        self.entries[idx.0].graph().replay()
    }

    /// Waits for everything enqueued — replays and eager work alike — to finish.
    pub fn synchronize(&self) -> Result<(), RuntimeError> {
        synchronize_idle(&self.stream)
    }

    /// The captured entry: its buffers for address asserts and readback, its communicator for
    /// the eager pass's collectives.
    ///
    /// # Panics
    /// Panics when `idx` was minted by a different session.
    pub fn entry(&self, idx: GraphIdx) -> &GraphEntry {
        &self.entries[idx.0]
    }

    /// The one consuming transition back to [`Allocation`]: an explicit weight reload. Every
    /// recorded graph baked the old weights' addresses, so the whole graph set is torn down with
    /// the phase and the reopened Allocation starts over on the same stream.
    #[must_use]
    pub fn reload_weights(self) -> Allocation {
        let Self {
            stream,
            entries,
            _executor_thread,
        } = self;
        drop(entries);
        Allocation {
            stream,
            _executor_thread: PhantomData,
        }
    }
}

/// Synchronizes the capture stream, refusing while the driver reports a recording in flight — a
/// synchronize would invalidate it. The session's own sequencing never leaves a recording open
/// across a synchronize; this guards the invariant with a named error instead of trusting it.
fn synchronize_idle(stream: &CaptureStream) -> Result<(), RuntimeError> {
    let state = stream.state()?;
    if state != CaptureState::Idle {
        return Err(RuntimeError::SyncWhileCapturing(state));
    }
    stream.cudarc_stream().synchronize()?;
    Ok(())
}

/// Best-effort drain after a failed recording, so the stream is idle for the session's next
/// operation. The originating failure is already on its way to the caller; a discard failure here
/// would have nothing actionable to add.
fn discard_recording(stream: &CaptureStream) {
    if matches!(
        stream.state(),
        Ok(CaptureState::Active | CaptureState::Invalidated)
    ) {
        let _ = capture::end_capture_discard(stream);
    }
}
