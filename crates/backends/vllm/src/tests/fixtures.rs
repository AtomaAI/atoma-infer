//! Request fixtures shared by the tests that start after validation.

use tokenizers::{Encoding, Token};

use crate::validation::{
    NextTokenChooserParameters, StoppingCriteriaParameters, ValidGenerateRequest,
};

/// A validated request whose prompt encodes to `prompt_len` distinct tokens, all of them derived
/// from `request_id` so that two fixtures never share a token id.
pub(crate) fn valid_request(
    request_id: &str,
    prompt_len: usize,
    max_new_tokens: u32,
) -> ValidGenerateRequest {
    let prompt_offset = prompt_token_offset(request_id);
    let tokens = (0..prompt_len)
        .map(|i| {
            let token_id = prompt_offset + i as u32;
            Token::new(token_id, format!("token-{token_id}"), (0, 0))
        })
        .collect::<Vec<_>>();
    let encoding = Encoding::from_tokens(tokens, 0);

    ValidGenerateRequest {
        request_id: request_id.to_string(),
        inputs: format!("prompt of {request_id}"),
        input_token_len: encoding.len(),
        encoding,
        truncate: 0,
        decoder_input_details: false,
        parameters: NextTokenChooserParameters {
            n: 1,
            best_of: 1,
            temperature: 1.0,
            top_p: 1.0,
            typical_p: 1.0,
            do_sample: true,
            ..Default::default()
        },
        stopping_parameters: StoppingCriteriaParameters {
            max_new_tokens,
            ..Default::default()
        },
        top_n_tokens: 0,
        return_full_text: false,
    }
}

/// First prompt token id of `request_id`'s fixture. Prompts are spaced far enough apart that a
/// request's token ids identify it.
pub(crate) fn prompt_token_offset(request_id: &str) -> u32 {
    const TOKENS_PER_REQUEST: u32 = 1_000;

    let request_number = request_id
        .rsplit('-')
        .next()
        .and_then(|suffix| suffix.parse::<u32>().ok())
        .unwrap_or(0);
    request_number * TOKENS_PER_REQUEST
}
