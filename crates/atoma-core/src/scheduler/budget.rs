//! The token budget: the per-step cap on query tokens summed over entries, plus a request cap.

use crate::types::{RequestCount, TokenCount};

/// What one scheduling pass may spend: query tokens across entries, and entries themselves.
///
/// Spent by running requests first; the remainder is offered to admission. Reset at the start of
/// every pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenBudget {
    max_tokens: TokenCount,
    max_requests: RequestCount,
    tokens_spent: usize,
    requests_spent: usize,
}

impl TokenBudget {
    /// A fresh budget of `max_tokens` query tokens over at most `max_requests` entries.
    #[must_use]
    pub fn new(max_tokens: TokenCount, max_requests: RequestCount) -> Self {
        Self {
            max_tokens,
            max_requests,
            tokens_spent: 0,
            requests_spent: 0,
        }
    }

    #[must_use]
    pub fn max_requests(&self) -> RequestCount {
        self.max_requests
    }

    /// Query tokens still unspent this pass.
    #[must_use]
    pub fn tokens_remaining(&self) -> usize {
        self.max_tokens.get() - self.tokens_spent
    }

    /// Entries still unspent this pass.
    #[must_use]
    pub fn requests_remaining(&self) -> usize {
        self.max_requests.get() - self.requests_spent
    }

    /// Whether one more entry computing `tokens` fits.
    #[must_use]
    pub fn fits(&self, tokens: TokenCount) -> bool {
        self.requests_remaining() > 0 && tokens.get() <= self.tokens_remaining()
    }

    /// How many of `wanted` tokens one more entry may compute: all of them, or the remainder of
    /// the budget as a chunk. `None` when no entry fits at all — the request cap is spent, or no
    /// token is left.
    #[must_use]
    pub fn offer(&self, wanted: usize) -> Option<TokenCount> {
        if self.requests_remaining() == 0 {
            return None;
        }
        TokenCount::new(wanted.min(self.tokens_remaining()))
    }

    /// Spends one entry computing `tokens`.
    ///
    /// # Panics
    ///
    /// Panics when the entry does not fit: the caller asks first, through [`TokenBudget::fits`]
    /// or [`TokenBudget::offer`].
    pub fn spend(&mut self, tokens: TokenCount) {
        assert!(
            self.fits(tokens),
            "spend of {tokens} tokens over a budget with {} tokens and {} entries left",
            self.tokens_remaining(),
            self.requests_remaining()
        );
        self.tokens_spent += tokens.get();
        self.requests_spent += 1;
    }

    /// Starts a new pass with nothing spent.
    pub fn reset(&mut self) {
        self.tokens_spent = 0;
        self.requests_spent = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::TokenBudget;
    use crate::test_support::{requests, tokens};

    /// Ported from the vLLM-derived `test_scheduling_budget`. Divergence: the old budget tracked
    /// per-request-id additions and subtractions so a group could be counted once; here the
    /// budget is spent per entry and reset per pass, so there is nothing to subtract and no id
    /// to deduplicate on.
    #[test]
    fn a_budget_admits_exactly_what_it_claims_at_the_boundary() {
        let mut budget = TokenBudget::new(tokens(4), requests(4));
        assert!(budget.fits(tokens(1)));
        assert!(budget.fits(tokens(4)));
        assert!(!budget.fits(tokens(5)));
        assert_eq!(budget.tokens_remaining(), 4);
        assert_eq!(budget.requests_remaining(), 4);

        budget.spend(tokens(2));
        assert_eq!(budget.tokens_remaining(), 2);
        assert_eq!(budget.requests_remaining(), 3, "one entry spent");
        assert!(budget.fits(tokens(2)));
        assert!(!budget.fits(tokens(3)));

        budget.reset();
        assert_eq!(budget.tokens_remaining(), 4);
        assert_eq!(budget.requests_remaining(), 4);
        assert_eq!(budget, TokenBudget::new(tokens(4), requests(4)));
    }

    #[test]
    fn the_request_cap_is_spent_one_entry_at_a_time() {
        let mut budget = TokenBudget::new(tokens(100), requests(2));
        budget.spend(tokens(1));
        budget.spend(tokens(1));
        assert_eq!(budget.requests_remaining(), 0);
        assert!(!budget.fits(tokens(1)), "tokens left, entries spent");
        assert_eq!(budget.offer(10), None);
    }

    #[test]
    fn an_offer_is_the_whole_request_or_the_remainder_as_a_chunk() {
        let mut budget = TokenBudget::new(tokens(8), requests(4));
        assert_eq!(budget.offer(3), Some(tokens(3)), "fits whole");
        assert_eq!(budget.offer(8), Some(tokens(8)), "fits exactly");
        assert_eq!(
            budget.offer(20),
            Some(tokens(8)),
            "chunked to the remainder"
        );

        budget.spend(tokens(8));
        assert_eq!(budget.offer(1), None, "no token left");
        assert_eq!(budget.offer(0), None, "nothing wanted is nothing offered");
    }

    #[test]
    #[should_panic(expected = "spend of 5 tokens")]
    fn spending_past_the_budget_is_a_caller_bug() {
        let mut budget = TokenBudget::new(tokens(4), requests(4));
        budget.spend(tokens(5));
    }
}
