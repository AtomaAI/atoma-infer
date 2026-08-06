//! Translation from a request's validated sampling parameters to the decoding strategy the model
//! executor applies to the logits.

use candle_transformers::generation::{LogitsProcessor, Sampling};

use crate::validation::NextTokenChooserParameters;

/// Builds the `LogitsProcessor` a `SequenceGroup` decodes with.
pub fn logits_processor(parameters: &NextTokenChooserParameters) -> LogitsProcessor {
    LogitsProcessor::from_sampling(parameters.random_seed, sampling_strategy(parameters))
}

/// Selects the decoding strategy for a request.
///
/// Greedy decoding is chosen only when the request opts out of sampling. `temperature == 1.0` is
/// the OpenAI default and means "sample from the model's own distribution" — treating it as a
/// request for greedy decoding makes every default-parameter request deterministic.
pub fn sampling_strategy(parameters: &NextTokenChooserParameters) -> Sampling {
    if !parameters.do_sample {
        return Sampling::ArgMax;
    }

    let temperature = f64::from(parameters.temperature);
    // `top_k == 0` and `top_p == 1.0` are the "unset" values validation normalises to.
    match (parameters.top_k, parameters.top_p) {
        (0, top_p) if top_p >= 1.0 => Sampling::All { temperature },
        (0, top_p) => Sampling::TopP {
            p: f64::from(top_p),
            temperature,
        },
        (top_k, top_p) if top_p >= 1.0 => Sampling::TopK {
            k: top_k as usize,
            temperature,
        },
        (top_k, top_p) => Sampling::TopKThenTopP {
            k: top_k as usize,
            p: f64::from(top_p),
            temperature,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The parameters `Validation` produces for a request that sets nothing but `do_sample`, which
    /// is what an OpenAI request with default sampling arrives as.
    fn default_parameters() -> NextTokenChooserParameters {
        NextTokenChooserParameters {
            n: 1,
            best_of: 1,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            typical_p: 1.0,
            do_sample: true,
            random_seed: 42,
            repetition_penalty: 1.0,
            repeat_last_n: 0,
            frequency_penalty: 0.0,
        }
    }

    #[test]
    fn test_default_temperature_samples_instead_of_decoding_greedily() {
        assert_eq!(
            sampling_strategy(&default_parameters()),
            Sampling::All { temperature: 1.0 }
        );
    }

    #[test]
    fn test_sampling_disabled_decodes_greedily() {
        let parameters = NextTokenChooserParameters {
            do_sample: false,
            ..default_parameters()
        };
        assert_eq!(sampling_strategy(&parameters), Sampling::ArgMax);
    }

    #[test]
    fn test_sampling_disabled_wins_over_sampling_parameters() {
        let parameters = NextTokenChooserParameters {
            do_sample: false,
            temperature: 0.7,
            top_k: 40,
            top_p: 0.75,
            ..default_parameters()
        };
        assert_eq!(sampling_strategy(&parameters), Sampling::ArgMax);
    }

    #[test]
    fn test_temperature_is_carried_through() {
        let parameters = NextTokenChooserParameters {
            temperature: 0.25,
            ..default_parameters()
        };
        assert_eq!(
            sampling_strategy(&parameters),
            Sampling::All { temperature: 0.25 }
        );
    }

    #[test]
    fn test_top_p_alone_selects_nucleus_sampling() {
        let parameters = NextTokenChooserParameters {
            temperature: 0.5,
            top_p: 0.75,
            ..default_parameters()
        };
        assert_eq!(
            sampling_strategy(&parameters),
            Sampling::TopP {
                p: 0.75,
                temperature: 0.5
            }
        );
    }

    #[test]
    fn test_top_k_alone_selects_top_k_sampling() {
        let parameters = NextTokenChooserParameters {
            temperature: 0.5,
            top_k: 40,
            ..default_parameters()
        };
        assert_eq!(
            sampling_strategy(&parameters),
            Sampling::TopK {
                k: 40,
                temperature: 0.5
            }
        );
    }

    #[test]
    fn test_top_k_and_top_p_are_combined() {
        let parameters = NextTokenChooserParameters {
            temperature: 0.5,
            top_k: 40,
            top_p: 0.75,
            ..default_parameters()
        };
        assert_eq!(
            sampling_strategy(&parameters),
            Sampling::TopKThenTopP {
                k: 40,
                p: 0.75,
                temperature: 0.5
            }
        );
    }
}
