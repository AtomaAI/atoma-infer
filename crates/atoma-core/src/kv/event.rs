//! The cache-event schema: the types describing changes to the cache's contents.
//!
//! This module defines only the event types; it contains no emitter, no ring and no sink, so
//! nothing here produces or delivers an event yet. The schema exists this early because adding
//! it later would break the wire format: every residence-bearing event carries its [`Tier`]
//! from day one, so introducing a second tier changes no consumer, and every batch names its
//! [`HashAlgorithm`], so a reader never mixes caches hashed under different versions.

use serde::{Deserialize, Serialize};

use crate::kv::HashAlgorithm;
use crate::types::BlockHash;

/// Where cached bytes reside. A tier is where bytes live, never a preemption mechanism.
///
/// `Host` is declared today though nothing emits it: adding a variant later would break every
/// written stream, while an unused variant costs nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    /// GPU memory — the only tier anything emits.
    Device,
    /// Host memory, post-launch.
    Host,
}

/// One observable change to the cache's contents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvEvent {
    /// `hash`'s bytes became resident in `tier`. `parent` lets a consumer rebuild the prefix
    /// index without ever seeing a token.
    BlockStored {
        hash: BlockHash,
        parent: Option<BlockHash>,
        tier: Tier,
    },
    /// `hash`'s bytes left `tier`.
    BlockRemoved { hash: BlockHash, tier: Tier },
    /// Every block left `tier` at once.
    TierCleared { tier: Tier },
}

/// A run of events under one hash algorithm.
///
/// The algorithm identifier is schema position, not payload: a consumer that reads a batch
/// minted under an algorithm it does not speak rejects the whole batch instead of indexing
/// hashes it can never reproduce.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvEventBatch {
    pub algorithm: HashAlgorithm,
    pub events: Vec<KvEvent>,
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::{KvEvent, KvEventBatch, Tier};
    use crate::kv::test_support::hash_of;
    use crate::kv::HashAlgorithm;

    fn every_event() -> Vec<KvEvent> {
        let hash = hash_of(&[1, 2, 3, 4]);
        vec![
            KvEvent::BlockStored {
                hash,
                parent: None,
                tier: Tier::Device,
            },
            KvEvent::BlockRemoved {
                hash,
                tier: Tier::Device,
            },
            KvEvent::TierCleared { tier: Tier::Device },
        ]
    }

    #[test]
    fn a_batch_round_trips_through_serde() {
        let batch = KvEventBatch {
            algorithm: HashAlgorithm::Sha256V1,
            events: every_event(),
        };
        let json = serde_json::to_string(&batch).unwrap();
        assert_eq!(serde_json::from_str::<KvEventBatch>(&json).unwrap(), batch);
    }

    #[test]
    fn every_event_carries_its_tier() {
        for event in every_event() {
            let json = serde_json::to_value(event).unwrap();
            let Value::Object(variant) = json else {
                panic!("events serialize as a tagged object");
            };
            let body = variant.values().next().expect("one variant body");
            assert_eq!(
                body.get("tier"),
                Some(&Value::String("device".to_owned())),
                "no residence-bearing event may omit its tier"
            );
        }
    }

    #[test]
    fn the_algorithm_is_schema_position_not_payload() {
        let batch = KvEventBatch {
            algorithm: HashAlgorithm::Sha256V1,
            events: Vec::new(),
        };
        let json = serde_json::to_value(&batch).unwrap();
        assert_eq!(
            json.get("algorithm"),
            Some(&Value::String("sha256_v1".to_owned()))
        );
    }

    #[test]
    fn the_host_tier_is_already_wire_representable() {
        assert_eq!(serde_json::to_string(&Tier::Host).unwrap(), "\"host\"");
        assert_eq!(
            serde_json::from_str::<Tier>("\"host\"").unwrap(),
            Tier::Host
        );
    }

    #[test]
    fn an_unknown_tier_is_rejected_not_guessed() {
        assert!(serde_json::from_str::<Tier>("\"disk\"").is_err());
    }
}
