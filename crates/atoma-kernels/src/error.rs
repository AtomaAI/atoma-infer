use candle_core::Error;
use std::fmt::{Display, Formatter, Result as FmtResult};
use thiserror::Error as ThisError;

/// An attention feature the vendored flash-attention kernels are not compiled for.
///
/// `kernels/static_switch.h` compiles the corresponding template parameter out, so the kernels
/// ignore the feature and return an unmodified result rather than failing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Capability {
    /// Logit softcapping, `softcap * tanh(scores / softcap)`, used by Grok and Gemma 2.
    Softcap,
    /// Sliding-window (local) attention, requested through a left or right window size.
    SlidingWindow,
}

impl Capability {
    /// The `static_switch.h` macro that compiles this capability out.
    const fn disabling_macro(self) -> &'static str {
        match self {
            Self::Softcap => "FLASHATTENTION_DISABLE_SOFTCAP",
            Self::SlidingWindow => "FLASHATTENTION_DISABLE_LOCAL",
        }
    }
}

impl Display for Capability {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::Softcap => f.write_str("softcap"),
            Self::SlidingWindow => f.write_str("sliding-window attention"),
        }
    }
}

/// A failure raised by the kernel layer itself, as opposed to a shape or dtype mismatch.
#[derive(Clone, Debug, Eq, PartialEq, ThisError)]
pub enum KernelError {
    /// The caller asked for an attention feature the compiled kernels silently drop.
    #[error(
        "flash-attention was compiled without {capability} support ({}); \
         running this request would return a result computed without it",
        .capability.disabling_macro()
    )]
    UnsupportedCapability {
        /// The feature that was requested.
        capability: Capability,
    },

    /// A CUDA call made while launching a kernel reported a failure.
    #[error("{kernel} failed with CUDA error {code}: {message}")]
    LaunchFailed {
        /// The FFI entry point that was called.
        kernel: &'static str,
        /// The `cudaError_t` value returned by the driver.
        code: i32,
        /// `cudaGetErrorString` for `code`.
        message: String,
    },
}

impl KernelError {
    /// Builds an [`KernelError::UnsupportedCapability`] for `capability`.
    pub const fn unsupported(capability: Capability) -> Self {
        Self::UnsupportedCapability { capability }
    }
}

impl From<KernelError> for Error {
    fn from(error: KernelError) -> Self {
        Self::wrap(error)
    }
}

/// Rejects attention features the compiled kernels drop instead of computing.
///
/// Supported window configurations are full attention (`None`, `None`) and causal attention
/// (`None`, `Some(0)`); every other combination requests local attention.
///
/// # Arguments
///
/// * `softcap` - Requested softcap value; any strictly positive value is a request.
/// * `window_size_left` - Requested left window size.
/// * `window_size_right` - Requested right window size.
pub fn check_supported_capabilities(
    softcap: Option<f32>,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
) -> Result<(), KernelError> {
    if softcap.is_some_and(|softcap| softcap > 0.0) {
        return Err(KernelError::unsupported(Capability::Softcap));
    }
    if window_size_left.is_some() || window_size_right.is_some_and(|window| window != 0) {
        return Err(KernelError::unsupported(Capability::SlidingWindow));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_and_causal_attention_are_supported() {
        assert_eq!(check_supported_capabilities(None, None, None), Ok(()));
        assert_eq!(check_supported_capabilities(None, None, Some(0)), Ok(()));
    }

    #[test]
    fn a_zero_or_absent_softcap_is_not_a_request() {
        assert_eq!(check_supported_capabilities(Some(0.0), None, None), Ok(()));
    }

    #[test]
    fn softcap_is_rejected() {
        assert_eq!(
            check_supported_capabilities(Some(30.0), None, None),
            Err(KernelError::unsupported(Capability::Softcap))
        );
    }

    #[test]
    fn a_left_window_is_rejected() {
        assert_eq!(
            check_supported_capabilities(None, Some(1024), None),
            Err(KernelError::unsupported(Capability::SlidingWindow))
        );
    }

    #[test]
    fn a_non_causal_right_window_is_rejected() {
        assert_eq!(
            check_supported_capabilities(None, None, Some(1024)),
            Err(KernelError::unsupported(Capability::SlidingWindow))
        );
    }

    #[test]
    fn softcap_is_reported_before_the_window() {
        assert_eq!(
            check_supported_capabilities(Some(30.0), Some(1024), None),
            Err(KernelError::unsupported(Capability::Softcap))
        );
    }

    #[test]
    fn the_rejection_names_the_capability_and_the_disabling_macro() {
        let error = KernelError::unsupported(Capability::SlidingWindow).to_string();
        assert!(error.contains("sliding-window attention"), "{error}");
        assert!(error.contains("FLASHATTENTION_DISABLE_LOCAL"), "{error}");
    }
}
