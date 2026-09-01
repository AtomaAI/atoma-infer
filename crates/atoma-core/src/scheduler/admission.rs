//! Admission policies: how one pass orders the window of waiting candidates.
//!
//! Every policy sees the same window and differs only in which request it picks from it; none
//! sorts or clones the queue, and preempted requests re-enter ahead of every policy, last in
//! first out.

use serde::{Deserialize, Serialize};

use crate::types::RequestCount;

/// How admission orders the bounded window of waiting requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdmissionPolicy {
    /// The front of the queue: earliest arrival first.
    Fcfs,
    /// The request with the most cached prefix blocks in the window; ties go to arrival.
    LongestPrefixMatch,
    /// The highest-priority request in the window; ties go to arrival. Traffic that asks for no
    /// priority shares the default and so admits first-come-first-served.
    Priority,
}

/// The bounded window admission examines in one pass: how many candidates, how the policy
/// orders them, and whether it is open to waiting requests at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AdmissionWindow {
    /// Candidates the pass examines.
    pub size: RequestCount,
    /// How the window is ordered.
    pub policy: AdmissionPolicy,
    /// Whether waiting requests may enter Running; preempted requests are offered either way,
    /// since they have already run.
    pub open: bool,
}

#[cfg(test)]
mod tests {
    use super::AdmissionPolicy;

    #[test]
    fn policies_serialize_as_snake_case_tags() {
        assert_eq!(
            serde_json::to_string(&AdmissionPolicy::LongestPrefixMatch).unwrap(),
            "\"longest_prefix_match\""
        );
        assert_eq!(
            serde_json::from_str::<AdmissionPolicy>("\"fcfs\"").unwrap(),
            AdmissionPolicy::Fcfs
        );
        assert_eq!(
            serde_json::to_string(&AdmissionPolicy::Priority).unwrap(),
            "\"priority\""
        );
    }
}
