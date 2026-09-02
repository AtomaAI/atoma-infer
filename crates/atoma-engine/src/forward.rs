//! The forward seam: what the executor runs a step command through to get its logits.
//!
//! One implementation runs the model on the device; tests stand a fake on the same seam and
//! drive the executor without one.

use std::error::Error;

use atoma_core::step::StepCommand;

use crate::batch::BatchLayout;
use crate::logits::Logits;

/// One model forward per step command.
pub trait Forward {
    type Error: Error + Send + Sync + 'static;

    /// Runs the model over `layout`, laid out from `command`, and returns the logits of the rows
    /// the layout selected: one row per sampling entry, in batch order, a vocabulary wide.
    ///
    /// # Errors
    ///
    /// Returns the implementation's error when the step could not be run; the executor treats
    /// it as fatal.
    fn forward(
        &mut self,
        command: &StepCommand,
        layout: &BatchLayout,
    ) -> Result<Logits<'_>, Self::Error>;
}
