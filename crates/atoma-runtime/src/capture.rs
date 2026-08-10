//! Capture lifecycle: which operations are legal in which capture state.
//!
//! The driver reports a stream's capture status as none, active, or invalidated. The transition
//! rules — begin only when idle, instantiate only when active, and an invalidated capture can
//! only be discarded — are pure logic, kept out of the driver-calling seams so they are testable
//! on a machine with no GPU.

use cudarc::driver::sys::CUstreamCaptureStatus;

use crate::error::RuntimeError;

/// Capture state of a stream, as the driver reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureState {
    /// No capture is in progress.
    Idle,
    /// A capture is recording.
    Active,
    /// A capture is recording but a previous operation broke it; it can only be discarded.
    Invalidated,
}

impl CaptureState {
    /// The state the driver's capture-status query maps to.
    pub fn from_status(status: CUstreamCaptureStatus) -> Self {
        match status {
            CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE => Self::Idle,
            CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE => Self::Active,
            CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_INVALIDATED => Self::Invalidated,
        }
    }

    /// The state after `op`, or the named error telling the operator what to do instead.
    pub fn apply(self, op: CaptureOp) -> Result<Self, RuntimeError> {
        match (self, op) {
            (Self::Idle, CaptureOp::Begin) => Ok(Self::Active),
            (Self::Active, CaptureOp::EndInstantiate) => Ok(Self::Idle),
            (Self::Active | Self::Invalidated, CaptureOp::Discard) => Ok(Self::Idle),
            (Self::Active, CaptureOp::Begin) => Err(RuntimeError::BeginWhileActive),
            (Self::Invalidated, CaptureOp::Begin) => Err(RuntimeError::BeginAfterInvalidation),
            (Self::Idle, CaptureOp::EndInstantiate) => Err(RuntimeError::EndWithoutCapture),
            (Self::Invalidated, CaptureOp::EndInstantiate) => {
                Err(RuntimeError::EndAfterInvalidation)
            }
            (Self::Idle, CaptureOp::Discard) => Err(RuntimeError::DiscardWithoutCapture),
        }
    }
}

/// An operation the capture substrate can attempt on a stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureOp {
    /// Start recording.
    Begin,
    /// Stop recording and instantiate the recorded graph.
    EndInstantiate,
    /// Stop recording and destroy the recorded graph without instantiating it.
    Discard,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legal_transitions_produce_the_expected_state() {
        assert_eq!(
            CaptureState::Idle.apply(CaptureOp::Begin).unwrap(),
            CaptureState::Active
        );
        assert_eq!(
            CaptureState::Active
                .apply(CaptureOp::EndInstantiate)
                .unwrap(),
            CaptureState::Idle
        );
        assert_eq!(
            CaptureState::Active.apply(CaptureOp::Discard).unwrap(),
            CaptureState::Idle
        );
        assert_eq!(
            CaptureState::Invalidated.apply(CaptureOp::Discard).unwrap(),
            CaptureState::Idle
        );
    }

    #[test]
    fn illegal_transitions_produce_named_errors() {
        assert!(matches!(
            CaptureState::Active.apply(CaptureOp::Begin),
            Err(RuntimeError::BeginWhileActive)
        ));
        assert!(matches!(
            CaptureState::Invalidated.apply(CaptureOp::Begin),
            Err(RuntimeError::BeginAfterInvalidation)
        ));
        assert!(matches!(
            CaptureState::Idle.apply(CaptureOp::EndInstantiate),
            Err(RuntimeError::EndWithoutCapture)
        ));
        assert!(matches!(
            CaptureState::Invalidated.apply(CaptureOp::EndInstantiate),
            Err(RuntimeError::EndAfterInvalidation)
        ));
        assert!(matches!(
            CaptureState::Idle.apply(CaptureOp::Discard),
            Err(RuntimeError::DiscardWithoutCapture)
        ));
    }

    #[test]
    fn an_invalidated_capture_directs_the_operator_to_the_discard_path() {
        let err = CaptureState::Invalidated
            .apply(CaptureOp::EndInstantiate)
            .unwrap_err();
        assert!(err.to_string().contains("end_capture_discard"));
    }

    #[test]
    fn driver_statuses_map_onto_states() {
        assert_eq!(
            CaptureState::from_status(CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE),
            CaptureState::Idle
        );
        assert_eq!(
            CaptureState::from_status(CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE),
            CaptureState::Active
        );
        assert_eq!(
            CaptureState::from_status(CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_INVALIDATED),
            CaptureState::Invalidated
        );
    }
}
