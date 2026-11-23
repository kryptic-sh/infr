//! Per-token log probabilities for the OpenAI `logprobs` reply fields.

/// One generated token's log probability and, when the request asked for them, its most likely
/// alternatives. `text`/`bytes` are filled in by the backend that owns the tokenizer; the decode
/// loop only knows the id and the probabilities.
#[derive(Clone, Debug, PartialEq)]
pub struct TokenLogprob {
    pub token: u32,
    /// The token decoded on its own. May be a replacement character for a byte-fallback token
    /// whose bytes only complete in context.
    pub text: String,
    /// The bytes of `text`, as OpenAI's `bytes` array.
    pub bytes: Vec<u8>,
    /// Natural log of the token's probability in the distribution it was sampled from.
    pub logprob: f32,
    /// The most likely tokens at this position, at most the request's `top_logprobs`, descending
    /// by probability. Does not necessarily contain `token`.
    pub top: Vec<TokenLogprobAlt>,
}

/// One alternative in [`TokenLogprob::top`].
#[derive(Clone, Debug, PartialEq)]
pub struct TokenLogprobAlt {
    pub token: u32,
    pub text: String,
    pub bytes: Vec<u8>,
    pub logprob: f32,
}

impl TokenLogprob {
    /// A token id and its probability, before the tokenizer-dependent fields are filled in.
    pub fn unresolved(token: u32, logprob: f32, top: Vec<TokenLogprobAlt>) -> Self {
        Self {
            token,
            text: String::new(),
            bytes: Vec::new(),
            logprob,
            top,
        }
    }
}
