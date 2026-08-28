//! Layer-group accounting: what each group of layers caches, and what a block costs.
//!
//! A model no longer has one uniform cache: layers group by what they cache and how. Each group
//! declares its geometry, its cache kind and its fill rate, and a group can share another
//! group's cache instead of writing its own. Only the full-attention kind is implemented; the
//! shape is the deliverable, so a sliding-window or state-space group is an added variant, not a
//! redesign.

use std::num::NonZeroUsize;

use thiserror::Error;

use crate::types::{LayerGroupId, TokenCount};

/// What kind of cache a layer group writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheKind {
    /// Full attention: K and V for every token, kept for the whole sequence.
    Full,
}

impl CacheKind {
    /// The fraction of a block's declared capacity this kind fills over a long sequence.
    /// Full attention keeps every token, so it fills every slot it declares.
    #[must_use]
    pub fn fill_rate(self) -> f64 {
        match self {
            CacheKind::Full => 1.0,
        }
    }
}

/// Where a layer group's cache bytes live.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvSource {
    /// The group writes its own cache.
    Own,
    /// The group reads another group's cache and writes none — cross-layer KV sharing.
    SharedFrom(LayerGroupId),
}

/// The geometry of one group's cache blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockLayout {
    /// Tokens per block.
    pub block_size: TokenCount,
    /// KV heads per layer.
    pub kv_head_count: NonZeroUsize,
    /// Elements per KV head.
    pub head_width: NonZeroUsize,
    /// Bytes per element — the cache dtype's width.
    pub element_bytes: NonZeroUsize,
}

/// One group of layers with a common cache kind and geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerGroup {
    pub id: LayerGroupId,
    /// Layers in the group.
    pub layer_count: NonZeroUsize,
    pub kind: CacheKind,
    pub layout: BlockLayout,
    pub kv_source: KvSource,
}

impl LayerGroup {
    /// Bytes one block costs in this group: K and V per token across the group's layers, or
    /// nothing when the group shares another group's cache.
    #[must_use]
    pub fn block_bytes(&self) -> usize {
        match self.kv_source {
            KvSource::SharedFrom(_) => 0,
            KvSource::Own => {
                let per_token = 2
                    * self.layout.kv_head_count.get()
                    * self.layout.head_width.get()
                    * self.layout.element_bytes.get();
                self.layer_count.get() * self.layout.block_size.get() * per_token
            }
        }
    }
}

/// Every layer group of one model's cache, validated as a whole.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvCacheSpec {
    groups: Vec<LayerGroup>,
}

impl KvCacheSpec {
    /// Builds a spec from `groups`.
    ///
    /// # Errors
    ///
    /// Rejects an empty spec, a duplicated group id, and a shared source that is missing or
    /// itself shared — sharing never chains.
    pub fn new(groups: Vec<LayerGroup>) -> Result<Self, LayerGroupError> {
        if groups.is_empty() {
            return Err(LayerGroupError::EmptySpec);
        }
        for (position, group) in groups.iter().enumerate() {
            if groups[..position].iter().any(|other| other.id == group.id) {
                return Err(LayerGroupError::DuplicateId { id: group.id });
            }
            let KvSource::SharedFrom(source) = group.kv_source else {
                continue;
            };
            let Some(source_group) = groups.iter().find(|other| other.id == source) else {
                return Err(LayerGroupError::SharedSourceMissing {
                    id: group.id,
                    shared_from: source,
                });
            };
            if source_group.kv_source != KvSource::Own {
                return Err(LayerGroupError::SharedSourceNotOwn {
                    id: group.id,
                    shared_from: source,
                });
            }
        }
        Ok(Self { groups })
    }

    /// The groups in declaration order.
    #[must_use]
    pub fn groups(&self) -> &[LayerGroup] {
        &self.groups
    }

    /// Bytes one block id costs across the whole cache — the sum of every group's block cost.
    #[must_use]
    pub fn bytes_per_block(&self) -> usize {
        self.groups.iter().map(LayerGroup::block_bytes).sum()
    }
}

/// A layer-group configuration no model could have.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum LayerGroupError {
    /// A cache with no layer groups caches nothing.
    #[error("a cache spec needs at least one layer group")]
    EmptySpec,
    /// Two groups declared the same id.
    #[error("layer group id {id:?} is declared twice")]
    DuplicateId { id: LayerGroupId },
    /// A shared source names a group that does not exist.
    #[error("layer group {id:?} shares from {shared_from:?}, which is not declared")]
    SharedSourceMissing {
        id: LayerGroupId,
        shared_from: LayerGroupId,
    },
    /// A shared source must own its cache; sharing never chains.
    #[error("layer group {id:?} shares from {shared_from:?}, which does not own its cache")]
    SharedSourceNotOwn {
        id: LayerGroupId,
        shared_from: LayerGroupId,
    },
}

#[cfg(test)]
mod tests {
    use super::{CacheKind, KvCacheSpec, KvSource, LayerGroup, LayerGroupError};
    use crate::kv::test_support::full_attention_group;
    use crate::types::LayerGroupId;

    #[test]
    fn block_bytes_multiply_out_the_declared_geometry() {
        // 2 (K and V) x 32 layers x 16 tokens x 8 heads x 128 wide x 2 bytes = 2 MiB.
        assert_eq!(full_attention_group(0).block_bytes(), 2 * 1024 * 1024);
    }

    #[test]
    fn a_shared_group_costs_no_bytes() {
        let shared = LayerGroup {
            id: LayerGroupId::new(1),
            kv_source: KvSource::SharedFrom(LayerGroupId::new(0)),
            ..full_attention_group(1)
        };
        assert_eq!(shared.block_bytes(), 0);

        let spec = KvCacheSpec::new(vec![full_attention_group(0), shared]).unwrap();
        assert_eq!(
            spec.bytes_per_block(),
            2 * 1024 * 1024,
            "only the owner pays"
        );
        assert_eq!(
            spec.groups().len(),
            2,
            "the sharing group is still declared"
        );
    }

    #[test]
    fn full_attention_declares_a_full_fill_rate() {
        assert!((CacheKind::Full.fill_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn an_empty_spec_is_rejected() {
        assert_eq!(
            KvCacheSpec::new(Vec::new()),
            Err(LayerGroupError::EmptySpec)
        );
    }

    #[test]
    fn duplicate_group_ids_are_rejected() {
        assert_eq!(
            KvCacheSpec::new(vec![full_attention_group(0), full_attention_group(0)]),
            Err(LayerGroupError::DuplicateId {
                id: LayerGroupId::new(0)
            })
        );
    }

    #[test]
    fn a_missing_shared_source_is_rejected() {
        let dangling = LayerGroup {
            kv_source: KvSource::SharedFrom(LayerGroupId::new(7)),
            ..full_attention_group(1)
        };
        assert_eq!(
            KvCacheSpec::new(vec![full_attention_group(0), dangling]),
            Err(LayerGroupError::SharedSourceMissing {
                id: LayerGroupId::new(1),
                shared_from: LayerGroupId::new(7),
            })
        );
    }

    #[test]
    fn sharing_from_a_sharing_group_is_rejected() {
        let first_sharer = LayerGroup {
            kv_source: KvSource::SharedFrom(LayerGroupId::new(0)),
            ..full_attention_group(1)
        };
        let chained = LayerGroup {
            kv_source: KvSource::SharedFrom(LayerGroupId::new(1)),
            ..full_attention_group(2)
        };
        assert_eq!(
            KvCacheSpec::new(vec![full_attention_group(0), first_sharer, chained]),
            Err(LayerGroupError::SharedSourceNotOwn {
                id: LayerGroupId::new(2),
                shared_from: LayerGroupId::new(1),
            })
        );
    }
}
