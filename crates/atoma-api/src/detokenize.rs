//! Detokenizing a request's generated tokens as they come, and matching its stop strings.
//!
//! A token is not a character. A byte-level tokenizer splits one character across tokens, and a
//! token decodes differently with and without the token before it, so decoding the new token alone
//! is wrong and decoding the whole sequence every time is quadratic. The detokenizer keeps a window
//! instead: the tokens it has already read out are the prefix the next decode starts from, what it
//! emits is exactly the text the window grew by, and a window ending in an incomplete character is
//! held back until the next token completes it.
//!
//! Stop strings are matched over the accumulated text, so a match split across tokens is found. A
//! tail that could begin a stop string is held back until later text rules it out, so what has been
//! streamed never holds part of a match; the tail is released when the request finishes.

use std::sync::Arc;

use tokenizers::Tokenizer;

/// What a decoder emits for an incomplete character.
const REPLACEMENT: char = '\u{FFFD}';

/// What feeding one token yielded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Emission {
    /// Text completed by the token: possibly none, when the token completes nothing yet.
    Text(String),
    /// A stop string was matched: the text before the match, and nothing after it is ever
    /// emitted.
    Stopped(String),
}

/// One request's detokenization state.
pub struct Detokenizer {
    tokenizer: Arc<Tokenizer>,
    tokens: Vec<u32>,
    /// Where the window's prefix starts: the tokens from here to `read_offset` were read out
    /// last, and give the next decode its context.
    prefix_offset: usize,
    /// Where the tokens not yet read out start.
    read_offset: usize,
    /// Everything decoded so far, truncated before a stop match.
    text: String,
    /// How much of `text` has been emitted.
    emitted: usize,
    stop: Vec<String>,
    /// The longest stop string's length in bytes; how far back a match can start into text
    /// already accumulated, and how much of the tail is held back.
    longest_stop: usize,
    stopped: bool,
}

impl Detokenizer {
    /// A detokenizer over `tokenizer` that stops on any of `stop`; an empty stop string is no
    /// stop string.
    #[must_use]
    pub fn new(tokenizer: Arc<Tokenizer>, stop: Vec<String>) -> Self {
        let stop: Vec<String> = stop.into_iter().filter(|s| !s.is_empty()).collect();
        let longest_stop = stop.iter().map(String::len).max().unwrap_or(0);
        Self {
            tokenizer,
            tokens: Vec::new(),
            prefix_offset: 0,
            read_offset: 0,
            text: String::new(),
            emitted: 0,
            stop,
            longest_stop,
            stopped: false,
        }
    }

    /// Feeds one generated token and returns whatever text it completed, or the stop it hit.
    /// After a stop, every further token emits nothing.
    ///
    /// # Errors
    ///
    /// Returns the tokenizer's error when the window cannot be decoded.
    pub fn feed(&mut self, token: u32) -> Result<Emission, tokenizers::Error> {
        if self.stopped {
            return Ok(Emission::Stopped(String::new()));
        }
        self.tokens.push(token);
        let Some(delta) = self.decode_window()? else {
            return Ok(Emission::Text(String::new()));
        };
        let search_from = self.char_boundary_at_or_before(
            self.text
                .len()
                .saturating_sub(self.longest_stop.saturating_sub(1)),
        );
        self.text.push_str(&delta);
        if let Some(at) = self.first_stop_match(search_from) {
            self.text.truncate(at);
            self.stopped = true;
            return Ok(Emission::Stopped(self.release(at)));
        }
        let releasable = self.char_boundary_at_or_before(
            self.text
                .len()
                .saturating_sub(self.longest_stop.saturating_sub(1)),
        );
        Ok(Emission::Text(self.release(releasable)))
    }

    /// Releases the held-back tail: what a request that finished without a stop still owes its
    /// client. Nothing is left to release after a stop, or a second time.
    #[must_use]
    pub fn finish(&mut self) -> String {
        if self.stopped {
            return String::new();
        }
        self.release(self.text.len())
    }

    /// Everything decoded so far, truncated before a stop match.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Decodes the window and returns the text it grew by, or nothing when it grew by nothing
    /// or ends in an incomplete character.
    fn decode_window(&mut self) -> Result<Option<String>, tokenizers::Error> {
        let prefix = self
            .tokenizer
            .decode(&self.tokens[self.prefix_offset..self.read_offset], true)?;
        let whole = self
            .tokenizer
            .decode(&self.tokens[self.prefix_offset..], true)?;
        if whole.len() <= prefix.len() || whole.ends_with(REPLACEMENT) {
            return Ok(None);
        }
        let delta = whole[prefix.len()..].to_owned();
        self.prefix_offset = self.read_offset;
        self.read_offset = self.tokens.len();
        Ok(Some(delta))
    }

    /// Where the earliest stop string matches in the text from `from` on, if one does.
    fn first_stop_match(&self, from: usize) -> Option<usize> {
        self.stop
            .iter()
            .filter_map(|stop| self.text[from..].find(stop.as_str()).map(|at| from + at))
            .min()
    }

    /// The text from what was last emitted up to `to`, marking it emitted.
    fn release(&mut self, to: usize) -> String {
        let to = to.max(self.emitted);
        let released = self.text[self.emitted..to].to_owned();
        self.emitted = to;
        released
    }

    fn char_boundary_at_or_before(&self, mut index: usize) -> usize {
        while !self.text.is_char_boundary(index) {
            index -= 1;
        }
        index
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tokenizers::Tokenizer;

    use super::{Detokenizer, Emission};
    use crate::test_support::tokenizer;

    fn ids(tokenizer: &Tokenizer, text: &str) -> Vec<u32> {
        tokenizer.encode(text, false).unwrap().get_ids().to_vec()
    }

    fn feed_all(detokenizer: &mut Detokenizer, tokens: &[u32]) -> Vec<Emission> {
        tokens
            .iter()
            .map(|&token| detokenizer.feed(token).unwrap())
            .collect()
    }

    fn texts(emissions: &[Emission]) -> String {
        emissions
            .iter()
            .map(|emission| match emission {
                Emission::Text(text) | Emission::Stopped(text) => text.as_str(),
            })
            .collect()
    }

    #[test]
    fn text_is_emitted_token_by_token_and_adds_up_to_the_whole() {
        let tokenizer = tokenizer();
        let tokens = ids(&tokenizer, "hello world");
        assert!(
            tokens.len() > 3,
            "the merges make multi-byte tokens: {tokens:?}"
        );
        let mut detokenizer = Detokenizer::new(Arc::clone(&tokenizer), Vec::new());
        let emissions = feed_all(&mut detokenizer, &tokens);
        assert_eq!(texts(&emissions), "hello world");
        assert!(
            emissions
                .iter()
                .all(|emission| matches!(emission, Emission::Text(text) if !text.is_empty())),
            "every token completes text: {emissions:?}"
        );
        assert_eq!(detokenizer.text(), "hello world");
        assert_eq!(detokenizer.finish(), "", "nothing was held back");
    }

    #[test]
    fn an_incomplete_character_is_held_back_until_its_bytes_complete() {
        let tokenizer = tokenizer();
        // "é" is two bytes, and each byte is its own token here.
        let tokens = ids(&tokenizer, "é");
        assert_eq!(tokens.len(), 2, "{tokens:?}");
        let mut detokenizer = Detokenizer::new(Arc::clone(&tokenizer), Vec::new());
        assert_eq!(
            detokenizer.feed(tokens[0]).unwrap(),
            Emission::Text(String::new()),
            "half a character is not text"
        );
        assert_eq!(
            detokenizer.feed(tokens[1]).unwrap(),
            Emission::Text("é".to_owned())
        );
    }

    #[test]
    fn a_stop_string_split_across_tokens_ends_the_text_before_the_match() {
        let tokenizer = tokenizer();
        let tokens = ids(&tokenizer, "hello world");
        let mut detokenizer = Detokenizer::new(Arc::clone(&tokenizer), vec!["lo w".to_owned()]);
        let emissions = feed_all(&mut detokenizer, &tokens);
        let stopped_at = emissions
            .iter()
            .position(|emission| matches!(emission, Emission::Stopped(_)))
            .expect("the stop string was matched");
        assert_eq!(texts(&emissions[..=stopped_at]), "hel");
        assert!(
            emissions[stopped_at + 1..]
                .iter()
                .all(|emission| *emission == Emission::Stopped(String::new())),
            "nothing after the stop is emitted: {emissions:?}"
        );
        assert_eq!(detokenizer.text(), "hel");
        assert_eq!(detokenizer.finish(), "");
    }

    #[test]
    fn a_tail_that_could_begin_a_stop_string_is_held_back_and_released_at_the_finish() {
        let tokenizer = tokenizer();
        let tokens = ids(&tokenizer, "hello");
        let mut detokenizer = Detokenizer::new(Arc::clone(&tokenizer), vec!["xyz".to_owned()]);
        let emissions = feed_all(&mut detokenizer, &tokens);
        assert_eq!(
            texts(&emissions),
            "hel",
            "two bytes could still start a match"
        );
        assert_eq!(detokenizer.text(), "hello");
        assert_eq!(detokenizer.finish(), "lo");
    }

    #[test]
    fn stop_strings_are_matched_over_multi_byte_text() {
        let tokenizer = tokenizer();
        let tokens = ids(&tokenizer, "héllo");
        let mut detokenizer = Detokenizer::new(Arc::clone(&tokenizer), vec!["éll".to_owned()]);
        let emissions = feed_all(&mut detokenizer, &tokens);
        assert_eq!(texts(&emissions), "h");
        assert!(emissions
            .iter()
            .any(|emission| matches!(emission, Emission::Stopped(_))));
    }

    #[test]
    fn empty_stop_strings_are_no_stop_strings() {
        let tokenizer = tokenizer();
        let tokens = ids(&tokenizer, "hi");
        let mut detokenizer = Detokenizer::new(Arc::clone(&tokenizer), vec![String::new()]);
        let emissions = feed_all(&mut detokenizer, &tokens);
        assert_eq!(texts(&emissions), "hi");
        assert_eq!(detokenizer.finish(), "");
    }
}
