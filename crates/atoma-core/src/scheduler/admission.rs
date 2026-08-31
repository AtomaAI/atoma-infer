//! Admission policies: how one pass orders the window of waiting candidates.
//!
//! Every policy sees the same window and differs only in which request it picks from it; none
//! sorts or clones the queue, and preempted requests re-enter ahead of every policy, last in
//! first out.

use serde::{Deserialize, Serialize};

/// How admission orders the bounded window of waiting requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdmissionPolicy {
    /// The front of the queue: earliest arrival first.
    Fcfs,
    /// The request with the most cached prefix blocks in the window; ties go to arrival.
    LongestPrefixMatch,
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
    }
}
