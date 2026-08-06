//! The attention window the flash-attention kernels take, and what it means.
//!
//! The kernels have no separate "causal" flag: causality is read off the window bounds, so the
//! window a caller passes is the only thing standing between a prefill and full bidirectional
//! attention.

/// The window a causal call passes: no left bound, and no lookahead to the right.
///
/// Leaving both sides unset instead — the natural-looking `(None, None)` — asks for full
/// bidirectional attention, which on a prefill lets every token attend to its own future and
/// silently produces wrong results.
pub const CAUSAL_WINDOW: (Option<usize>, Option<usize>) = (None, Some(0));

/// Whether a window pair, normalised to the kernels' convention of `-1` for "unbounded", asks for
/// causal attention.
pub fn is_causal(window_size_left: i32, window_size_right: i32) -> bool {
    window_size_left < 0 && window_size_right == 0
}

/// Normalises an optional window bound to the kernels' convention: bounds wider than the key
/// sequence, and unset bounds, become `-1`.
pub fn normalize_bound(window_size: Option<usize>, max_seqlen_k: usize) -> i32 {
    match window_size {
        Some(window_size) if window_size <= max_seqlen_k => window_size as i32,
        Some(_) | None => -1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_window_is_causal() {
        const MAX_SEQLEN_K: usize = 128;

        let (left, right) = CAUSAL_WINDOW;
        assert!(is_causal(
            normalize_bound(left, MAX_SEQLEN_K),
            normalize_bound(right, MAX_SEQLEN_K)
        ));
    }

    /// The defect this rule exists to prevent: an unset window pair is bidirectional attention,
    /// not causal attention.
    #[test]
    fn test_unset_window_is_not_causal() {
        const MAX_SEQLEN_K: usize = 128;

        assert!(!is_causal(
            normalize_bound(None, MAX_SEQLEN_K),
            normalize_bound(None, MAX_SEQLEN_K)
        ));
    }

    #[test]
    fn test_a_left_bound_makes_the_window_local_rather_than_causal() {
        assert!(!is_causal(16, 0));
    }

    #[test]
    fn test_lookahead_is_not_causal() {
        assert!(!is_causal(-1, 4));
    }

    #[test]
    fn test_bounds_wider_than_the_keys_are_unbounded() {
        assert_eq!(normalize_bound(Some(129), 128), -1);
        assert_eq!(normalize_bound(Some(128), 128), 128);
        assert_eq!(normalize_bound(Some(0), 128), 0);
        assert_eq!(normalize_bound(None, 128), -1);
    }
}
