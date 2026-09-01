//! Caller-owned workspaces, kept apart by the path they serve.
//!
//! A kernel reachable from a captured step may not allocate: an allocation inside a captured
//! region invalidates the capture, or bakes a pool-owned address into the graph. So every such
//! kernel declares how many bytes one invocation needs through [`WorkspaceRequirement`], and the
//! caller hands it a buffer allocated before capture.
//!
//! Captured and eager execution never share one. A captured graph baked its workspace's address,
//! so those bytes must stay put and must not be written by anything running between replays;
//! eager execution wants a buffer it can resize as shapes change. [`Workspace<Captured, B>`] and
//! [`Workspace<Eager, B>`] are separate types with no conversion between them, so the two cannot
//! be swapped by mistake.

use std::marker::PhantomData;

/// The path whose workspace a captured graph baked the address of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Captured {}

/// The path whose workspace is used only outside captured regions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Eager {}

/// A caller-owned workspace for one execution path.
///
/// `Path` is [`Captured`] or [`Eager`] and appears in no field, so the two are distinct types
/// carrying the same buffer kind. Handing an eager workspace to a call that records into a graph
/// does not compile:
///
/// ```compile_fail
/// use atoma_core::attention::{Captured, Eager, Workspace};
///
/// fn record(_workspace: &mut Workspace<Captured, Vec<u8>>) {}
///
/// let mut eager: Workspace<Eager, Vec<u8>> = Workspace::new(vec![0; 64], 64);
/// record(&mut eager);
/// ```
///
/// The captured workspace does:
///
/// ```
/// use atoma_core::attention::{Captured, Workspace};
///
/// fn record(_workspace: &mut Workspace<Captured, Vec<u8>>) {}
///
/// let mut captured: Workspace<Captured, Vec<u8>> = Workspace::new(vec![0; 64], 64);
/// record(&mut captured);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Workspace<Path, Buffer> {
    buffer: Buffer,
    bytes: usize,
    path: PhantomData<fn() -> Path>,
}

impl<Path, Buffer> Workspace<Path, Buffer> {
    /// Wraps `buffer`, which the caller allocated and states holds `bytes` bytes.
    ///
    /// The size is stated rather than read: a device buffer's length is not readable through a
    /// bound this crate can require without linking a driver.
    pub fn new(buffer: Buffer, bytes: usize) -> Self {
        Self {
            buffer,
            bytes,
            path: PhantomData,
        }
    }

    /// Bytes the buffer holds.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// The buffer itself, for a call that only reads it.
    #[must_use]
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// The buffer itself, for the call that writes it.
    ///
    /// A captured graph baked this buffer's address, so a caller that writes through it must not
    /// replace or resize the allocation behind it. The production buffer is a device allocation
    /// with no resize to reach for; a host buffer standing in for one in a test has, and must be
    /// left where it is.
    #[must_use]
    pub fn buffer_mut(&mut self) -> &mut Buffer {
        &mut self.buffer
    }

    /// Whether this workspace is large enough for `requirement`.
    #[must_use]
    pub fn covers(&self, requirement: &impl WorkspaceRequirement) -> bool {
        self.bytes >= requirement.workspace_bytes()
    }
}

/// What a kernel reachable from a captured step declares: the caller owns its workspace, and the
/// kernel allocates none of its own.
///
/// A shaped invocation implements this — a prepared plan, or a kernel-call descriptor — so the
/// caller can size one buffer before capture and hand the same bytes to every launch.
pub trait WorkspaceRequirement {
    /// Bytes of caller-owned workspace one invocation of this call needs.
    fn workspace_bytes(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::{Captured, Eager, Workspace, WorkspaceRequirement};

    struct Call(usize);

    impl WorkspaceRequirement for Call {
        fn workspace_bytes(&self) -> usize {
            self.0
        }
    }

    #[test]
    fn a_workspace_covers_exactly_what_it_holds() {
        let workspace: Workspace<Captured, Vec<u8>> = Workspace::new(vec![0; 128], 128);

        assert!(workspace.covers(&Call(127)));
        assert!(workspace.covers(&Call(128)));
        assert!(!workspace.covers(&Call(129)));
    }

    #[test]
    fn a_workspace_hands_out_the_buffer_it_wraps() {
        let mut workspace: Workspace<Eager, Vec<u8>> = Workspace::new(vec![0; 4], 4);
        workspace.buffer_mut()[0] = 7;

        assert_eq!(workspace.buffer(), &[7, 0, 0, 0]);
        assert_eq!(workspace.bytes(), 4);
    }
}
