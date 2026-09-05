//! The host's mirror of what each request slot holds on the device: whose record is written
//! there, and whether a step has sampled for it. It answers the two questions the executor asks
//! before a step: which records this step must write, because a slot changed hands, and which
//! rows can take their token from the device, because a step sampling for the request in that
//! slot has already been issued, so the token the slot holds is that request's last.
//!
//! Nothing the mirror knows comes back from the device: a slot is marked as sampled for when the
//! step that samples for it is staged, which is enough, since the stream runs that step's sample
//! before any later step's gather. Nothing tells the executor a request is over either; a slot is
//! released by the next request claiming it, which is when its record is rewritten and its
//! sampling forgotten.

use atoma_core::types::{RequestId, RequestSlot};
use thiserror::Error;

/// A slot the mirror refuses: one it does not cover, which is the engine handing out a slot past
/// the bound it declared, since the device arrays are sized to the mirror; or one that holds no
/// request, yet is said to sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum OwnersError {
    #[error("request slot {} is past the {slots} slots the sampler holds", slot.get())]
    SlotOutOfRange { slot: RequestSlot, slots: usize },
    #[error("request slot {} holds no request, yet a step samples for it", slot.get())]
    SlotUnclaimed { slot: RequestSlot },
}

/// What claiming a slot found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Claim {
    /// The slot already held the request; its record stands.
    Held,
    /// The slot changed hands, or was empty; its record must be written.
    Taken,
}

/// The request a slot holds and whether a step has sampled for it there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Owner {
    request: RequestId,
    sampled: bool,
}

/// One entry per request slot the sampler holds on the device.
#[derive(Debug)]
pub struct SlotOwners {
    slots: Vec<Option<Owner>>,
}

impl SlotOwners {
    /// A mirror over `slot_count` empty slots.
    #[must_use]
    pub fn new(slot_count: usize) -> Self {
        Self {
            slots: vec![None; slot_count],
        }
    }

    /// Claims `slot` for `request`: [`Claim::Held`] when the slot already holds it, and
    /// [`Claim::Taken`] when it did not, in which case whatever the slot held is forgotten.
    ///
    /// # Errors
    ///
    /// Returns [`OwnersError::SlotOutOfRange`] when the mirror does not cover `slot`.
    pub fn claim(&mut self, slot: RequestSlot, request: RequestId) -> Result<Claim, OwnersError> {
        let owner = self.owner_mut(slot)?;
        if owner.is_some_and(|held| held.request == request) {
            return Ok(Claim::Held);
        }
        *owner = Some(Owner {
            request,
            sampled: false,
        });
        Ok(Claim::Taken)
    }

    /// Whether a row for `request` in `slot` can take its token from the device: the slot holds
    /// the request, and a step sampling for it there has been issued, so the token the slot holds
    /// is the request's last.
    #[must_use]
    pub fn gathers(&self, slot: RequestSlot, request: RequestId) -> bool {
        self.slots
            .get(slot.index())
            .copied()
            .flatten()
            .is_some_and(|held| held.request == request && held.sampled)
    }

    /// Records that a step sampling for the request `slot` holds is issued: from the next step
    /// on, the token the slot holds is that request's last.
    ///
    /// # Errors
    ///
    /// Returns [`OwnersError`] when the mirror does not cover `slot` or the slot holds no
    /// request.
    pub fn samples(&mut self, slot: RequestSlot) -> Result<(), OwnersError> {
        let Some(owner) = self.owner_mut(slot)? else {
            return Err(OwnersError::SlotUnclaimed { slot });
        };
        owner.sampled = true;
        Ok(())
    }

    fn owner_mut(&mut self, slot: RequestSlot) -> Result<&mut Option<Owner>, OwnersError> {
        let slots = self.slots.len();
        self.slots
            .get_mut(slot.index())
            .ok_or(OwnersError::SlotOutOfRange { slot, slots })
    }
}

#[cfg(test)]
mod tests {
    use atoma_core::types::{RequestId, RequestSlot};

    use super::{Claim, OwnersError, SlotOwners};

    fn slot(slot: u32) -> RequestSlot {
        RequestSlot::new(slot)
    }

    fn request(request: u64) -> RequestId {
        RequestId::new(request)
    }

    #[test]
    fn a_slot_is_taken_once_and_held_until_another_request_claims_it() {
        let mut owners = SlotOwners::new(4);
        assert_eq!(owners.claim(slot(2), request(7)).unwrap(), Claim::Taken);
        assert_eq!(owners.claim(slot(2), request(7)).unwrap(), Claim::Held);
        assert_eq!(owners.claim(slot(2), request(8)).unwrap(), Claim::Taken);
        assert_eq!(owners.claim(slot(2), request(8)).unwrap(), Claim::Held);
        assert_eq!(
            owners.claim(slot(2), request(7)).unwrap(),
            Claim::Taken,
            "the earlier request coming back is a new claim"
        );
    }

    #[test]
    fn a_row_takes_its_token_from_the_device_once_a_step_has_sampled_for_its_slot() {
        let mut owners = SlotOwners::new(4);
        owners.claim(slot(1), request(7)).unwrap();
        assert!(!owners.gathers(slot(1), request(7)), "nothing sampled yet");
        owners.samples(slot(1)).unwrap();
        assert!(owners.gathers(slot(1), request(7)));
        assert!(!owners.gathers(slot(1), request(8)), "another request");
        assert!(!owners.gathers(slot(3), request(7)), "an empty slot");
        assert!(
            !owners.gathers(slot(9), request(7)),
            "a slot past the mirror holds nothing"
        );
    }

    #[test]
    fn a_slot_changing_hands_forgets_that_it_sampled_for_the_last_request() {
        let mut owners = SlotOwners::new(2);
        owners.claim(slot(0), request(1)).unwrap();
        owners.samples(slot(0)).unwrap();
        owners.claim(slot(0), request(2)).unwrap();
        assert!(!owners.gathers(slot(0), request(2)));
    }

    #[test]
    fn a_slot_past_the_mirror_or_holding_nothing_is_refused_by_name() {
        let mut owners = SlotOwners::new(2);
        assert_eq!(
            owners.claim(slot(2), request(1)).unwrap_err(),
            OwnersError::SlotOutOfRange {
                slot: slot(2),
                slots: 2
            }
        );
        assert_eq!(
            owners.samples(slot(2)).unwrap_err(),
            OwnersError::SlotOutOfRange {
                slot: slot(2),
                slots: 2
            }
        );
        assert_eq!(
            owners.samples(slot(1)).unwrap_err(),
            OwnersError::SlotUnclaimed { slot: slot(1) }
        );
        assert!(OwnersError::SlotOutOfRange {
            slot: slot(2),
            slots: 2
        }
        .to_string()
        .contains("past the 2 slots"));
    }
}
