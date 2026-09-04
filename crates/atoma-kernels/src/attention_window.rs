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

/// A window as the kernel entry stores it: the causal flag and both bounds, `-1` for unbounded.
/// The launch template reads its causal and local switches off exactly these three values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelWindow {
    pub is_causal: bool,
    pub window_size_left: i32,
    pub window_size_right: i32,
}

/// Resolves a requested window for a call over `seqlen_k` keys and `seqlen_q` query rows, the
/// way the KV-cache path and upstream flash-attention do: a bound at or past the key length is
/// unbounded; a causal request over one query row is not causal, since every key precedes the
/// row, unless alibi slopes make the row's position matter; and a window bounded on one side
/// only is bounded by the key length on the other, so the kernel's local switch sees two
/// non-negative bounds.
pub fn resolve(
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
    seqlen_k: i32,
    seqlen_q: usize,
    has_alibi: bool,
) -> KernelWindow {
    let within_keys = |bound: Option<usize>| -> i32 {
        match bound.map(i32::try_from) {
            Some(Ok(bound)) if bound < seqlen_k => bound,
            Some(Ok(_) | Err(_)) | None => -1,
        }
    };
    let mut left = within_keys(window_size_left);
    let mut right = within_keys(window_size_right);
    let mut causal = is_causal(left, right);
    if seqlen_q == 1 && !has_alibi {
        causal = false;
    }
    if left < 0 && right >= 0 {
        left = seqlen_k;
    }
    if right < 0 && left >= 0 {
        right = seqlen_k;
    }
    KernelWindow {
        is_causal: causal,
        window_size_left: left,
        window_size_right: right,
    }
}

/// The window a decode step runs under: the causal window the models ask for, resolved for the
/// step's one query row over `seqlen_k` keys. The raw decode path passes exactly this, and the
/// KV-cache wrapper resolves the same request through [`resolve`], so both reach the same kernel
/// instantiation.
pub fn decode(seqlen_k: i32) -> KernelWindow {
    let (left, right) = CAUSAL_WINDOW;
    resolve(left, right, seqlen_k, 1, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEQLEN_K: i32 = 4096;

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

    #[test]
    fn a_causal_request_over_one_row_resolves_to_a_full_left_window_and_no_lookahead() {
        let (left, right) = CAUSAL_WINDOW;
        assert_eq!(
            resolve(left, right, SEQLEN_K, 1, false),
            KernelWindow {
                is_causal: false,
                window_size_left: SEQLEN_K,
                window_size_right: 0
            }
        );
    }

    #[test]
    fn a_decode_step_runs_under_the_causal_request_resolved_for_one_row() {
        let (left, right) = CAUSAL_WINDOW;
        assert_eq!(decode(SEQLEN_K), resolve(left, right, SEQLEN_K, 1, false));
        assert_eq!(
            decode(SEQLEN_K),
            KernelWindow {
                is_causal: false,
                window_size_left: SEQLEN_K,
                window_size_right: 0
            }
        );
    }

    #[test]
    fn a_causal_request_over_many_rows_stays_causal() {
        let (left, right) = CAUSAL_WINDOW;
        assert_eq!(
            resolve(left, right, SEQLEN_K, 8, false),
            KernelWindow {
                is_causal: true,
                window_size_left: SEQLEN_K,
                window_size_right: 0
            }
        );
    }

    #[test]
    fn alibi_keeps_one_row_causal() {
        let (left, right) = CAUSAL_WINDOW;
        assert!(resolve(left, right, SEQLEN_K, 1, true).is_causal);
    }

    #[test]
    fn an_unset_window_is_unbounded_on_both_sides() {
        assert_eq!(
            resolve(None, None, SEQLEN_K, 1, false),
            KernelWindow {
                is_causal: false,
                window_size_left: -1,
                window_size_right: -1
            }
        );
    }

    #[test]
    fn a_bound_at_or_past_the_key_length_is_unbounded_and_one_inside_it_is_local() {
        let at = resolve(Some(4096), Some(0), SEQLEN_K, 8, false);
        assert!(at.is_causal);
        assert_eq!(at.window_size_left, SEQLEN_K);
        assert_eq!(
            resolve(Some(4095), Some(0), SEQLEN_K, 8, false),
            KernelWindow {
                is_causal: false,
                window_size_left: 4095,
                window_size_right: 0
            }
        );
        let past = resolve(Some(usize::MAX), None, SEQLEN_K, 8, false);
        assert_eq!(past.window_size_left, -1);
    }

    #[test]
    fn a_window_bounded_on_one_side_is_bounded_by_the_keys_on_the_other() {
        assert_eq!(
            resolve(None, Some(4), SEQLEN_K, 8, false),
            KernelWindow {
                is_causal: false,
                window_size_left: SEQLEN_K,
                window_size_right: 4
            }
        );
        assert_eq!(
            resolve(Some(16), None, SEQLEN_K, 8, false),
            KernelWindow {
                is_causal: false,
                window_size_left: 16,
                window_size_right: SEQLEN_K
            }
        );
    }
}
