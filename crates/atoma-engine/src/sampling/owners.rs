//! The host's mirror of what each request slot holds on the device: whose record is written
//! there, and the last token sampled for it. It answers the two questions the executor asks
//! before a step: which records this step must write, because a slot changed hands, and which
//! rows can take their token from the device, because the token the engine sent is the one the
//! device sampled for that slot last.
//!
//! Nothing tells the executor a request is over; a slot is released by the next request claiming
//! it, which is when its record is rewritten and its last token forgotten.

use atoma_core::types::{RequestId, RequestSlot};
use thiserror::Error;

/// A slot the mirror does not cover: the device arrays are sized to the mirror, so this is the
/// engine handing out a slot past the bound it declared.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum OwnersError {
    #[error("request slot {} is past the {slots} slots the sampler holds", slot.get())]
    SlotOutOfRange { slot: RequestSlot, slots: usize },
    #[error("request slot {} holds no request, yet a token was sampled for it", slot.get())]
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

/// The request a slot holds and what was last sampled for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Owner {
    request: RequestId,
    last_token: Option<u32>,
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

    #[must_use]
    pub fn slot_count(&self) -> usize {
        self.slots.len()
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
            last_token: None,
        });
        Ok(Claim::Taken)
    }

    /// Whether a row for `request` in `slot` whose input is `token` can take that token from the
    /// device: the slot holds the request, and `token` is what was last sampled for it there.
    #[must_use]
    pub fn holds_token(&self, slot: RequestSlot, request: RequestId, token: u32) -> bool {
        self.slots
            .get(index(slot))
            .copied()
            .flatten()
            .is_some_and(|held| held.request == request && held.last_token == Some(token))
    }

    /// Records that `token` was sampled for the request `slot` holds.
    ///
    /// # Errors
    ///
    /// Returns [`OwnersError`] when the mirror does not cover `slot` or the slot holds no
    /// request.
    pub fn sampled(&mut self, slot: RequestSlot, token: u32) -> Result<(), OwnersError> {
        let Some(owner) = self.owner_mut(slot)? else {
            return Err(OwnersError::SlotUnclaimed { slot });
        };
        owner.last_token = Some(token);
        Ok(())
    }

    fn owner_mut(&mut self, slot: RequestSlot) -> Result<&mut Option<Owner>, OwnersError> {
        let slots = self.slots.len();
        self.slots
            .get_mut(index(slot))
            .ok_or(OwnersError::SlotOutOfRange { slot, slots })
    }
}

fn index(slot: RequestSlot) -> usize {
    slot.get() as usize
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
        assert_eq!(owners.slot_count(), 4);
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
    fn a_row_takes_its_token_from_the_device_only_for_the_token_last_sampled_there() {
        let mut owners = SlotOwners::new(4);
        owners.claim(slot(1), request(7)).unwrap();
        assert!(
            !owners.holds_token(slot(1), request(7), 5),
            "nothing sampled yet"
        );
        owners.sampled(slot(1), 5).unwrap();
        assert!(owners.holds_token(slot(1), request(7), 5));
        assert!(!owners.holds_token(slot(1), request(7), 6), "another token");
        assert!(
            !owners.holds_token(slot(1), request(8), 5),
            "another request"
        );
        assert!(!owners.holds_token(slot(3), request(7), 5), "an empty slot");
        assert!(
            !owners.holds_token(slot(9), request(7), 5),
            "a slot past the mirror holds nothing"
        );
        owners.sampled(slot(1), 6).unwrap();
        assert!(!owners.holds_token(slot(1), request(7), 5));
        assert!(owners.holds_token(slot(1), request(7), 6));
    }

    #[test]
    fn a_slot_changing_hands_forgets_the_token_sampled_for_the_last_request() {
        let mut owners = SlotOwners::new(2);
        owners.claim(slot(0), request(1)).unwrap();
        owners.sampled(slot(0), 5).unwrap();
        owners.claim(slot(0), request(2)).unwrap();
        assert!(!owners.holds_token(slot(0), request(2), 5));
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
            owners.sampled(slot(2), 5).unwrap_err(),
            OwnersError::SlotOutOfRange {
                slot: slot(2),
                slots: 2
            }
        );
        assert_eq!(
            owners.sampled(slot(1), 5).unwrap_err(),
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
