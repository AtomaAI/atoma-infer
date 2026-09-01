//! What backends and the model declare, and what the engine settles from it at startup.

use tracing::info;

use crate::attention::break_point::{BreakPoint, BreakPoints, BreakSite, Declarer};
use crate::attention::graph_mode::{GraphMode, SupportLevel};

/// What one attention backend declares to the engine: the live batches its captured routine is
/// valid for, and the sites it cannot capture.
///
/// Device-free, so the startup resolution runs and is tested without a GPU. Break points are
/// added through this type, which stamps the backend's name into each, so a backend can only ever
/// state a capability — never a model's policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendDeclaration {
    name: String,
    support_level: SupportLevel,
    break_points: BreakPoints,
}

impl BackendDeclaration {
    /// Declares the level `name`'s captured routine is valid at, with nothing broken yet.
    #[must_use]
    pub fn new(name: impl Into<String>, support_level: SupportLevel) -> Self {
        Self {
            name: name.into(),
            support_level,
            break_points: BreakPoints::default(),
        }
    }

    /// Declares that this backend cannot capture the op at `site`.
    #[must_use]
    pub fn cannot_capture(mut self, site: BreakSite) -> Self {
        self.break_points.insert(BreakPoint::new(
            site,
            Declarer::BackendCannotCapture {
                backend: self.name.clone(),
            },
        ));
        self
    }

    /// Declares that this backend's work at `site` is rank-coupled.
    #[must_use]
    pub fn rank_coupled(mut self, site: BreakSite) -> Self {
        self.break_points.insert(BreakPoint::new(
            site,
            Declarer::BackendRankCoupled {
                backend: self.name.clone(),
            },
        ));
        self
    }

    /// The name this backend declared itself under.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The live batches its captured routine is valid for.
    #[must_use]
    pub fn support_level(&self) -> SupportLevel {
        self.support_level
    }

    /// The sites it cannot capture.
    #[must_use]
    pub fn break_points(&self) -> &BreakPoints {
        &self.break_points
    }
}

/// What a model definition declares: where its eager region lies.
///
/// A model states policy and nothing else: it declares no support level, because whether a
/// captured routine stays valid across batch shapes is the backend's statement, and it declares
/// no capability, because what can be captured is not the model's to say.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelDeclaration {
    name: String,
    break_points: BreakPoints,
}

impl ModelDeclaration {
    /// Declares `name` with no eager region.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            break_points: BreakPoints::default(),
        }
    }

    /// Declares that this model runs the op at `site` eagerly.
    #[must_use]
    pub fn eager_at(mut self, site: BreakSite) -> Self {
        self.break_points.insert(BreakPoint::new(
            site,
            Declarer::ModelPolicy {
                model: self.name.clone(),
            },
        ));
        self
    }

    /// The name this model declared itself under.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The sites its policy runs eagerly.
    #[must_use]
    pub fn break_points(&self) -> &BreakPoints {
        &self.break_points
    }
}

/// What the active backends and the model together settle at startup: the level every captured
/// routine is valid at, and every site the pass leaves the graph.
///
/// Built once, before any capture, and never rebuilt — the dispatcher is constructed from it, so
/// nothing at runtime can raise the graph mode or drop a break point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaptureContract {
    graph_mode: GraphMode,
    break_points: BreakPoints,
}

impl CaptureContract {
    /// Settles the graph mode at the minimum level across `backends`, and takes the union of
    /// every backend's capability break points with the model's policy break points.
    ///
    /// Logs the settled graph mode, the backend that pinned it, and every break point with the
    /// declarer behind it, so an engine replaying less than expected says why at startup rather
    /// than only through per-batch fallback counts.
    #[must_use]
    pub fn resolve(backends: &[BackendDeclaration], model: &ModelDeclaration) -> Self {
        let graph_mode = GraphMode::resolve(backends.iter().map(BackendDeclaration::support_level));
        let break_points = backends
            .iter()
            .fold(model.break_points().clone(), |union, backend| {
                union.union(backend.break_points())
            });
        let pinned_by = backends
            .iter()
            .find(|backend| backend.support_level() == graph_mode.support_level())
            .map_or("no active backend", BackendDeclaration::name);
        info!(
            support_level = ?graph_mode.support_level(),
            pinned_by,
            model = model.name(),
            break_points = %break_points,
            "graph mode settled"
        );
        Self {
            graph_mode,
            break_points,
        }
    }

    /// The level every captured routine is valid at.
    #[must_use]
    pub fn graph_mode(&self) -> GraphMode {
        self.graph_mode
    }

    /// Every site the pass leaves the graph, from both declarers.
    #[must_use]
    pub fn break_points(&self) -> &BreakPoints {
        &self.break_points
    }
}

#[cfg(test)]
mod tests {
    use super::{BackendDeclaration, CaptureContract, ModelDeclaration};
    use crate::attention::{DeclarerKind, SupportLevel};
    use crate::test_support::{captured_log, site};

    #[test]
    fn the_graph_mode_settles_at_the_weakest_backend() {
        let contract = CaptureContract::resolve(
            &[
                BackendDeclaration::new("full", SupportLevel::Always),
                BackendDeclaration::new("decode-only", SupportLevel::UniformSingleTokenDecode),
                BackendDeclaration::new("uniform", SupportLevel::UniformBatch),
            ],
            &ModelDeclaration::new("fake-model"),
        );

        assert_eq!(
            contract.graph_mode().support_level(),
            SupportLevel::UniformSingleTokenDecode
        );
        assert!(contract.break_points().is_empty());
    }

    #[test]
    fn the_contract_unions_capability_and_policy_break_points() {
        let attention = BackendDeclaration::new("fake-attention", SupportLevel::UniformBatch)
            .cannot_capture(site(0, 3));
        let collective = BackendDeclaration::new("fake-collective", SupportLevel::Always)
            .rank_coupled(site(4, 1));
        let model = ModelDeclaration::new("fake-model").eager_at(site(2, 7));

        let contract = CaptureContract::resolve(&[attention, collective], &model);

        assert_eq!(
            contract.break_points().sites(),
            [site(0, 3), site(2, 7), site(4, 1)]
        );
        assert_eq!(
            contract
                .break_points()
                .declarers_at(site(2, 7))
                .map(super::Declarer::kind)
                .collect::<Vec<_>>(),
            [DeclarerKind::Model]
        );
    }

    #[test]
    fn a_backend_and_a_model_may_name_the_same_site() {
        let site = site(1, 5);
        let contract = CaptureContract::resolve(
            &[BackendDeclaration::new("fake", SupportLevel::Always).cannot_capture(site)],
            &ModelDeclaration::new("fake-model").eager_at(site),
        );

        assert_eq!(contract.break_points().sites(), [site]);
        assert_eq!(contract.break_points().declarers_at(site).count(), 2);
    }

    #[test]
    fn a_declarer_stamps_its_own_name_into_every_point_it_declares() {
        let backend = BackendDeclaration::new("fake", SupportLevel::Always)
            .cannot_capture(site(0, 1))
            .rank_coupled(site(0, 2));

        assert!(backend
            .break_points()
            .iter()
            .all(|point| point.declarer().name() == "fake"
                && point.declarer().kind() == DeclarerKind::Backend));
    }

    #[test]
    fn no_active_backend_leaves_nothing_replayable() {
        let contract = CaptureContract::resolve(&[], &ModelDeclaration::new("fake-model"));

        assert_eq!(
            contract.graph_mode().support_level(),
            SupportLevel::Never,
            "no backend vouches for a captured routine"
        );
    }

    #[test]
    fn the_settled_graph_mode_is_logged_with_the_backend_that_pinned_it() {
        // The log capture is process-global, so this test's backend names appear in no other.
        let log = captured_log();
        let contract = CaptureContract::resolve(
            &[
                BackendDeclaration::new("settled-log-full", SupportLevel::Always),
                BackendDeclaration::new("settled-log-floor", SupportLevel::UniformBatch),
            ],
            &ModelDeclaration::new("settled-log-model").eager_at(site(0, 1)),
        );

        assert_eq!(
            contract.graph_mode().support_level(),
            SupportLevel::UniformBatch
        );
        let line = log
            .contents()
            .lines()
            .find(|line| line.contains("settled-log-floor"))
            .expect("the resolution logs once")
            .to_owned();
        assert!(line.contains("graph mode settled"), "got: {line}");
        assert!(line.contains("UniformBatch"), "got: {line}");
        assert!(
            line.contains("layer 0 op 1: model settled-log-model runs it eagerly"),
            "the log names every break point and its declarer: {line}"
        );
    }
}
