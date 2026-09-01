//! The level a backend declares its captured routine is valid at, and the graph mode the engine
//! settles on across every active backend.

/// The live batches a backend's captured routine is valid for, weakest to strongest.
///
/// A level states when the routine that was *captured* stays correct, not what the backend's
/// kernels can compute. A backend whose kernels serve any batch still declares
/// [`SupportLevel::UniformSingleTokenDecode`] when the work it records is only valid while every
/// request decodes exactly one token — because the graph baked that step's shape and scheduling,
/// and replaying it for a different shape would run the wrong work rather than fail. Eager
/// execution is unaffected by the level: a batch the level does not cover falls back to it.
///
/// Not serializable on purpose: a level is a backend's statement, and nothing an operator writes
/// down should be able to stand in for one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SupportLevel {
    /// The captured routine is never valid.
    Never,
    /// Valid only when every request decodes exactly one token.
    UniformSingleTokenDecode,
    /// Valid for any uniform batch.
    UniformBatch,
    /// Valid for any live batch.
    Always,
}

/// The level every captured routine runs under: the minimum across the active backends.
///
/// Settled once at startup by [`GraphMode::resolve`] and never raised, since one backend whose
/// captured routine is only valid for single-token decode makes the whole captured step only
/// valid there. Resolving is the only way to build a graph mode and the level it settled is
/// private, so a graph mode is always the minimum of levels that were declared, and nothing can
/// widen one afterwards. What the levels are worth is the declaring backends' business.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct GraphMode(SupportLevel);

impl GraphMode {
    /// Downgrades to the weakest level the active backends declare.
    ///
    /// With no active backend the graph mode is [`SupportLevel::Never`]: nothing has vouched for
    /// a captured routine, so every batch runs eagerly rather than replaying work no backend
    /// claims is valid.
    #[must_use]
    pub fn resolve(levels: impl IntoIterator<Item = SupportLevel>) -> Self {
        Self(levels.into_iter().min().unwrap_or(SupportLevel::Never))
    }

    /// The level captured routines are valid at.
    #[must_use]
    pub fn support_level(self) -> SupportLevel {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::{GraphMode, SupportLevel};

    #[test]
    fn support_levels_order_weakest_to_strongest() {
        assert!(SupportLevel::Never < SupportLevel::UniformSingleTokenDecode);
        assert!(SupportLevel::UniformSingleTokenDecode < SupportLevel::UniformBatch);
        assert!(SupportLevel::UniformBatch < SupportLevel::Always);
    }

    #[test]
    fn the_graph_mode_is_the_minimum_across_the_active_backends() {
        let graph_mode = GraphMode::resolve([
            SupportLevel::Always,
            SupportLevel::UniformSingleTokenDecode,
            SupportLevel::UniformBatch,
        ]);

        assert_eq!(
            graph_mode.support_level(),
            SupportLevel::UniformSingleTokenDecode
        );
    }

    #[test]
    fn one_backend_settles_the_graph_mode_at_its_own_level() {
        assert_eq!(
            GraphMode::resolve([SupportLevel::UniformBatch]).support_level(),
            SupportLevel::UniformBatch
        );
    }

    #[test]
    fn a_never_backend_downgrades_every_other() {
        let graph_mode = GraphMode::resolve([SupportLevel::Always, SupportLevel::Never]);

        assert_eq!(graph_mode.support_level(), SupportLevel::Never);
    }

    #[test]
    fn no_active_backend_resolves_to_never() {
        assert_eq!(
            GraphMode::resolve([]).support_level(),
            SupportLevel::Never,
            "nothing vouches for a captured routine, so nothing replays"
        );
    }
}
