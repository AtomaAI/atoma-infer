//! The logits one step selected, on the host: one row per selected entry, a vocabulary wide.

/// A view of `rows × vocab` f32 logits, row-major, on the host.
#[derive(Debug, Clone, Copy)]
pub struct Logits<'a> {
    data: &'a [f32],
    vocab: usize,
}

impl<'a> Logits<'a> {
    /// Views `data` as rows of `vocab` logits.
    ///
    /// # Panics
    ///
    /// Panics when `vocab` is zero or `data` is not a whole number of rows: the producer sized
    /// the copy, so either is its bug.
    #[must_use]
    pub fn new(data: &'a [f32], vocab: usize) -> Self {
        assert!(vocab > 0, "a vocabulary has at least one token");
        assert!(
            data.len().is_multiple_of(vocab),
            "{} logits are not a whole number of rows of {vocab}",
            data.len()
        );
        Self { data, vocab }
    }

    #[must_use]
    pub fn rows(&self) -> usize {
        self.data.len() / self.vocab
    }

    #[must_use]
    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// The logits of one row, or nothing when `row` is past the last.
    #[must_use]
    pub fn row(&self, row: usize) -> Option<&'a [f32]> {
        self.data.get(row * self.vocab..(row + 1) * self.vocab)
    }
}

#[cfg(test)]
mod tests {
    use super::Logits;

    #[test]
    fn rows_are_read_off_the_flat_buffer_and_the_last_is_the_last() {
        let data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let logits = Logits::new(&data, 3);
        assert_eq!(logits.rows(), 2);
        assert_eq!(logits.vocab(), 3);
        assert_eq!(logits.row(0), Some(&[0.0, 1.0, 2.0][..]));
        assert_eq!(logits.row(1), Some(&[3.0, 4.0, 5.0][..]));
        assert_eq!(logits.row(2), None);
    }

    #[test]
    #[should_panic(expected = "not a whole number of rows")]
    fn a_ragged_buffer_is_a_bug() {
        let data = [0.0, 1.0, 2.0, 3.0, 4.0];
        let _ = Logits::new(&data, 3);
    }
}
