//! The forward seam: what the executor runs a step command through to get its sampled tokens.
//!
//! One implementation runs the model on the device and samples there; tests stand a fake on the
//! same seam and drive the executor without one. What crosses the seam is tokens, never logits:
//! where the logits live and how a token is drawn from them is the implementation's alone.

use std::error::Error;

use crate::batch::BatchLayout;

/// One model forward and sample per step command.
pub trait Forward {
    type Error: Error + Send + Sync + 'static;

    /// Runs the model over `layout` and samples the rows the layout selected: one token per
    /// selected row, in batch order.
    ///
    /// # Errors
    ///
    /// Returns the implementation's error when the step could not be run; the executor treats
    /// it as fatal.
    fn forward(&mut self, layout: &BatchLayout) -> Result<&[u32], Self::Error>;
}
