//! Break points: where a forward pass leaves the graph, and who says so.
//!
//! Two declarers state break points independently. A backend states a capability — an op it
//! cannot capture, or one that is rank-coupled. A model states a policy — where its eager region
//! lies. Neither can state the other's kind: capability break points are declared through
//! [`BackendDeclaration`](crate::attention::BackendDeclaration) and policy break points through
//! [`ModelDeclaration`](crate::attention::ModelDeclaration), and each stamps its own name into
//! every point it declares.
//!
//! A break point names a position in the pass's op order, never an op kind. The attention op has
//! no standing here: it is broken at only when a declarer names its site, exactly like any other
//! op.

use std::collections::BTreeSet;
use std::fmt;

/// Where in a forward pass a break point falls: one op of one layer.
///
/// `op` indexes the layer's own op order — the order a caller declares its ops in, and the one it
/// declares role lifetimes against — so a bridge buffer's lifetime and the break point it spans
/// are counted over the same ops. Ordering is pass order: by layer, then by op within the layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BreakSite {
    /// The zero-based layer the site falls in.
    pub layer: usize,
    /// The site's index in its layer's op order.
    pub op: usize,
}

impl fmt::Display for BreakSite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "layer {} op {}", self.layer, self.op)
    }
}

/// Who declared a break point, and on what grounds.
///
/// The grounds are part of the identity: a backend states what it cannot do, and a model states
/// what it has chosen. Two declarers naming the same site are two declarations of it, and the
/// site is broken at while either stands.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Declarer {
    /// The named backend cannot capture the op at this site.
    BackendCannotCapture {
        /// The backend that cannot capture it.
        backend: String,
    },
    /// The op at this site is rank-coupled: what the named backend does there depends on a value
    /// only a cross-rank exchange settles, so a capture would bake one exchange's answer into
    /// every replay.
    BackendRankCoupled {
        /// The backend whose work at the site is rank-coupled.
        backend: String,
    },
    /// The named model's policy places its eager region at this site.
    ModelPolicy {
        /// The model that declared it.
        model: String,
    },
}

impl Declarer {
    /// Whether a backend or a model declared it.
    #[must_use]
    pub fn kind(&self) -> DeclarerKind {
        match self {
            Declarer::BackendCannotCapture { backend: _ }
            | Declarer::BackendRankCoupled { backend: _ } => DeclarerKind::Backend,
            Declarer::ModelPolicy { model: _ } => DeclarerKind::Model,
        }
    }

    /// The name the declarer declared itself under.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Declarer::BackendCannotCapture { backend }
            | Declarer::BackendRankCoupled { backend } => backend,
            Declarer::ModelPolicy { model } => model,
        }
    }
}

impl fmt::Display for Declarer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Declarer::BackendCannotCapture { backend } => {
                write!(f, "backend {backend} cannot capture it")
            }
            Declarer::BackendRankCoupled { backend } => {
                write!(f, "backend {backend} is rank-coupled there")
            }
            Declarer::ModelPolicy { model } => write!(f, "model {model} runs it eagerly"),
        }
    }
}

/// Which of the two independent declarers stated a break point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DeclarerKind {
    /// A backend, stating a capability.
    Backend,
    /// A model definition, stating a policy.
    Model,
}

/// One declaration that a site is not captured.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BreakPoint {
    site: BreakSite,
    declarer: Declarer,
}

impl BreakPoint {
    /// Declares `site` broken on `declarer`'s grounds.
    ///
    /// Crate-internal: outside this crate a break point is declared through
    /// [`BackendDeclaration`](crate::attention::BackendDeclaration) or
    /// [`ModelDeclaration`](crate::attention::ModelDeclaration), so a backend cannot state a
    /// model's policy and a model cannot state a backend's capability.
    pub(super) fn new(site: BreakSite, declarer: Declarer) -> Self {
        Self { site, declarer }
    }

    /// Where the pass leaves the graph.
    #[must_use]
    pub fn site(&self) -> BreakSite {
        self.site
    }

    /// Who declared it, and on what grounds.
    #[must_use]
    pub fn declarer(&self) -> &Declarer {
        &self.declarer
    }
}

impl fmt::Display for BreakPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.site, self.declarer)
    }
}

/// The break points standing over a forward pass, in pass order.
///
/// A set, so declaring the same point twice declares it once, and two declarers naming one site
/// both stand against it.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct BreakPoints(BTreeSet<BreakPoint>);

impl BreakPoints {
    /// Declares `point`, returning whether it was not already declared.
    pub(super) fn insert(&mut self, point: BreakPoint) -> bool {
        self.0.insert(point)
    }

    /// Everything either set declares.
    #[must_use]
    pub fn union(&self, other: &Self) -> Self {
        Self(self.0.union(&other.0).cloned().collect())
    }

    /// Every declaration, in pass order.
    pub fn iter(&self) -> impl Iterator<Item = &BreakPoint> {
        self.0.iter()
    }

    /// The sites the pass leaves the graph at, once each, in pass order.
    ///
    /// This is what a segmentation is built from: the declarations behind a site explain it, but
    /// two declarers naming one site still break the pass in one place.
    #[must_use]
    pub fn sites(&self) -> Vec<BreakSite> {
        let mut sites: Vec<BreakSite> = self.0.iter().map(BreakPoint::site).collect();
        sites.dedup();
        sites
    }

    /// Who declared `site`, if anyone did.
    pub fn declarers_at(&self, site: BreakSite) -> impl Iterator<Item = &Declarer> {
        self.0
            .iter()
            .filter(move |point| point.site() == site)
            .map(BreakPoint::declarer)
    }

    /// Whether nothing breaks the pass.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl fmt::Display for BreakPoints {
    /// Every declaration in pass order, so a startup log says which sites break the pass and who
    /// declared each.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            return f.write_str("none");
        }
        for (position, point) in self.0.iter().enumerate() {
            if position > 0 {
                f.write_str("; ")?;
            }
            write!(f, "{point}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{BreakPoint, BreakPoints, BreakSite, Declarer, DeclarerKind};
    use crate::test_support::site;

    fn cannot_capture(backend: &str) -> Declarer {
        Declarer::BackendCannotCapture {
            backend: backend.to_owned(),
        }
    }

    fn policy(model: &str) -> Declarer {
        Declarer::ModelPolicy {
            model: model.to_owned(),
        }
    }

    fn points(declared: Vec<(BreakSite, Declarer)>) -> BreakPoints {
        let mut points = BreakPoints::default();
        for (site, declarer) in declared {
            points.insert(BreakPoint::new(site, declarer));
        }
        points
    }

    #[test]
    fn sites_order_by_layer_then_op() {
        let mut sites = [site(1, 0), site(0, 7), site(0, 2)];
        sites.sort_unstable();

        assert_eq!(sites, [site(0, 2), site(0, 7), site(1, 0)]);
    }

    #[test]
    fn a_declaration_names_its_kind_and_its_declarer() {
        assert_eq!(cannot_capture("flashinfer").kind(), DeclarerKind::Backend);
        assert_eq!(
            Declarer::BackendRankCoupled {
                backend: "flashinfer".to_owned()
            }
            .kind(),
            DeclarerKind::Backend
        );
        assert_eq!(policy("deepseek-v3").kind(), DeclarerKind::Model);
        assert_eq!(policy("deepseek-v3").name(), "deepseek-v3");
    }

    #[test]
    fn the_union_holds_every_declaration_in_pass_order() {
        let backend = points(vec![(site(2, 4), cannot_capture("fake"))]);
        let model = points(vec![(site(0, 1), policy("fake-model"))]);

        let union = backend.union(&model);

        assert_eq!(union.iter().count(), 2);
        assert_eq!(union.sites(), [site(0, 1), site(2, 4)]);
        assert_eq!(
            union.to_string(),
            "layer 0 op 1: model fake-model runs it eagerly; \
             layer 2 op 4: backend fake cannot capture it"
        );
    }

    #[test]
    fn one_site_declared_by_both_breaks_the_pass_once() {
        let shared = site(3, 6);
        let union = points(vec![(shared, cannot_capture("fake"))])
            .union(&points(vec![(shared, policy("fake-model"))]));

        assert_eq!(union.sites(), [shared]);
        assert_eq!(
            union.iter().count(),
            2,
            "both declarations stand behind the site"
        );
        assert_eq!(
            union
                .declarers_at(shared)
                .map(Declarer::kind)
                .collect::<Vec<_>>(),
            [DeclarerKind::Backend, DeclarerKind::Model]
        );
    }

    #[test]
    fn the_same_declaration_twice_is_one_declaration() {
        let declared_twice = site(1, 1);
        let mut declared = points(vec![(declared_twice, cannot_capture("fake"))]);

        assert!(!declared.insert(BreakPoint::new(declared_twice, cannot_capture("fake"))));
        assert_eq!(declared.iter().count(), 1);
    }

    #[test]
    fn a_site_nobody_declared_has_no_declarers() {
        let declared = points(vec![(site(1, 1), cannot_capture("fake"))]);

        assert_eq!(declared.declarers_at(site(1, 2)).count(), 0);
    }

    #[test]
    fn nothing_declared_breaks_nothing() {
        let empty = BreakPoints::default();

        assert!(empty.is_empty());
        assert_eq!(empty.sites().len(), 0);
        assert_eq!(empty.union(&BreakPoints::default()), empty);
        assert_eq!(empty.to_string(), "none");
    }

    #[test]
    fn a_declaration_renders_its_site_and_its_grounds() {
        let point = BreakPoint::new(site(2, 5), cannot_capture("fake"));

        assert_eq!(
            point.to_string(),
            "layer 2 op 5: backend fake cannot capture it"
        );
    }
}
