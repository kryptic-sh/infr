//! The OpenAI `logprobs` reply fields: the wire shape of each generated token's log probability,
//! and [`LogprobStops`], which keeps each token's entry with the text it produced while stop
//! sequences are cut from that text.

use std::collections::VecDeque;

use infr_engine::TokenLogprob;
use serde::Serialize;

use crate::StopMatcher;

/// OpenAI's `choice.logprobs` / `choice.delta.logprobs`.
#[derive(Debug, Clone, Serialize)]
pub struct ChatLogprobs {
    pub content: Vec<ChatTokenLogprob>,
}

/// One token's entry in [`ChatLogprobs::content`]. `id` is a llama.cpp-style extension; OpenAI
/// clients ignore unknown fields.
#[derive(Debug, Clone, Serialize)]
pub struct ChatTokenLogprob {
    pub id: u32,
    pub token: String,
    pub bytes: Vec<u8>,
    pub logprob: f32,
    pub top_logprobs: Vec<ChatTokenLogprobAlt>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatTokenLogprobAlt {
    pub id: u32,
    pub token: String,
    pub bytes: Vec<u8>,
    pub logprob: f32,
}

impl ChatLogprobs {
    /// The wire shape for a run of generated tokens. Text and bytes come from the backend that
    /// filled the [`TokenLogprob`](infr_engine::TokenLogprob) entries in.
    pub fn from_tokens(tokens: &[infr_engine::TokenLogprob]) -> Self {
        Self {
            content: tokens
                .iter()
                .map(|t| ChatTokenLogprob {
                    id: t.token,
                    token: t.text.clone(),
                    bytes: t.bytes.clone(),
                    logprob: t.logprob,
                    top_logprobs: t
                        .top
                        .iter()
                        .map(|a| ChatTokenLogprobAlt {
                            id: a.token,
                            token: a.text.clone(),
                            bytes: a.bytes.clone(),
                            logprob: a.logprob,
                        })
                        .collect(),
                })
                .collect(),
        }
    }
}

/// A [`StopMatcher`] that carries each token's log probability along with the text it produced, so
/// the entries a reply reports are exactly those of the tokens whose text it returns. Shared by the
/// chat route and `/v1/completions`.
///
/// Text arrives as the decoder releases it: one piece and the entries of every token since the
/// previous piece (a character split across tokens, or a special token the stream skips, adds no
/// piece of its own). Such a group is the unit entries are attached to; this never assumes a
/// token's own bytes add up to the streamed text.
///
/// The rule: an entry is reported when its token's text is at least partly returned. A token whose
/// text a stop sequence cut in the middle keeps its entry, which still describes the whole sampled
/// token, as every entry does; the tokens of the stop sequence itself, and any after it, have none.
/// Entries pushed with no text cover nothing returned and are dropped.
#[derive(Debug, Default)]
pub struct LogprobStops {
    stops: StopMatcher,
    /// Text the matcher released that has not been handed out: the released head of a token group
    /// whose tail the matcher still holds.
    released: String,
    /// The groups covering `released` followed by what the matcher holds, oldest first.
    groups: VecDeque<Group>,
}

#[derive(Debug)]
struct Group {
    /// Bytes of this group's text not yet handed out.
    len: usize,
    logprobs: Vec<TokenLogprob>,
}

/// Text that is now safe to send, with the entries of the tokens it belongs to.
#[derive(Debug, Default, PartialEq)]
pub struct TokenText {
    pub text: String,
    pub logprobs: Vec<TokenLogprob>,
}

impl LogprobStops {
    pub fn new(stops: Vec<String>) -> Self {
        Self {
            stops: StopMatcher::new(stops),
            ..Self::default()
        }
    }

    /// A stop sequence has fired: generation must halt and no further text may be emitted.
    pub fn hit(&self) -> bool {
        self.stops.hit()
    }

    /// Feed one decoded piece and the entries of the tokens that produced it. Returns the text now
    /// safe to send, split where it changes group so a caller can keep a delta from splitting a
    /// token. A group with entries is handed out whole, so its text waits while the matcher holds
    /// any of it; text without entries is handed out as the matcher releases it, so a request that
    /// asked for no log probabilities streams exactly as through a bare [`StopMatcher`].
    pub fn push(&mut self, piece: &str, logprobs: Vec<TokenLogprob>) -> Vec<TokenText> {
        if self.stops.hit() {
            return Vec::new();
        }
        self.groups.push_back(Group {
            len: piece.len(),
            logprobs,
        });
        let safe = self.stops.push(piece);
        self.released.push_str(&safe);
        self.hand_out(self.stops.hit())
    }

    /// End of generation: what the matcher still holds was never a stop, so all of it goes out.
    pub fn flush(&mut self) -> Vec<TokenText> {
        let tail = self.stops.flush();
        self.released.push_str(&tail);
        self.hand_out(true)
    }

    /// Hand out the released text, group by group. While more text may follow (`all` false), a
    /// group with entries goes out only once all of its text has been released; at the end every
    /// released byte goes, and a group cut short by a stop takes its entries with what it returned.
    fn hand_out(&mut self, all: bool) -> Vec<TokenText> {
        let mut out: Vec<TokenText> = Vec::new();
        while let Some(g) = self.groups.front_mut() {
            let avail = self.released.len();
            let take = if g.len <= avail {
                g.len
            } else if all || g.logprobs.is_empty() {
                avail
            } else {
                break;
            };
            if take == 0 && g.len > 0 {
                break;
            }
            let text: String = self.released.drain(..take).collect();
            g.len -= take;
            let logprobs = if take > 0 {
                std::mem::take(&mut g.logprobs)
            } else {
                Vec::new()
            };
            if g.len == 0 {
                self.groups.pop_front();
            }
            match out.last_mut() {
                // Text without entries joins the run before it when that has none either, so it
                // goes out in one piece, as the matcher released it.
                Some(last) if last.logprobs.is_empty() && logprobs.is_empty() => {
                    last.text.push_str(&text)
                }
                _ if text.is_empty() => {}
                _ => out.push(TokenText { text, logprobs }),
            }
        }
        if all {
            // Nothing after a stop, or after the end, is returned.
            self.groups.clear();
            self.released.clear();
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A request without log probabilities streams exactly as through a bare [`StopMatcher`]: the
    /// same text in the same pieces, at a stop and at the end.
    #[test]
    fn without_entries_it_releases_what_the_stop_matcher_does() {
        let pieces = ["Hel", "lo E", "N", "d. ", "Then E", "ND", " more"];
        for stops in [vec![], vec!["END".to_owned()], vec!["\n\n".to_owned()]] {
            let mut bare = StopMatcher::new(stops.clone());
            let mut ours = LogprobStops::new(stops.clone());
            for p in pieces {
                let want = bare.push(p);
                let got = ours.push(p, Vec::new());
                let got: Vec<&str> = got.iter().map(|t| t.text.as_str()).collect();
                if want.is_empty() {
                    assert!(got.is_empty(), "{stops:?} {p:?}: {got:?}");
                } else {
                    assert_eq!(got, [want.as_str()], "{stops:?} {p:?}");
                }
            }
            let want = bare.flush();
            let got: String = ours.flush().into_iter().map(|t| t.text).collect();
            assert_eq!(got, want, "{stops:?}");
            assert_eq!(ours.hit(), bare.hit(), "{stops:?}");
        }
    }

    /// A clamped log probability goes out as a number, never as `null`.
    #[test]
    fn the_lowest_finite_logprob_serializes_as_a_number() {
        let lp = infr_engine::TokenLogprob::unresolved(1, f32::MIN, Vec::new());
        let v = serde_json::to_value(ChatLogprobs::from_tokens(&[lp])).unwrap();
        assert!(v["content"][0]["logprob"].is_number(), "{v}");
    }
}
