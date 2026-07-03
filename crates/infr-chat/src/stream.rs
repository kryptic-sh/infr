//! Streaming reasoning/content splitter — THE single place model output is separated into
//! reasoning vs answer (vs tool calls) as it streams. `infr run` (display styling), `infr serve`
//! (OpenAI deltas) and the history stripper all consume this module, so every thinking model is
//! exposed the same way on every surface; a new reasoning format (e.g. channel markers) lands
//! here once and everywhere at once.

/// Does this rendered prompt end with a `<think>` PREFILL (the template opened the reasoning
/// block itself — Qwen3.5/Qwen3-Next style — so the model's output starts MID-reasoning with only
/// the close marker)? Callers inject a synthetic `"<think>"` piece at the head of the stream when
/// this is true, so the streaming splitter, the display styler and the history stripper all see a
/// well-formed block through the one shared grammar.
pub fn prompt_prefills_think(prompt: &str) -> bool {
    prompt.trim_end().ends_with("<think>")
}

/// A streamed piece of a response.
#[derive(Clone, Debug)]
pub enum Delta {
    Reasoning(String),
    Content(String),
    ToolCall { name: String, arguments: String },
}

/// Incremental splitter for the streaming server path. Accumulates the raw decoded text and, on each
/// piece, emits the newly-stable Reasoning (`<think>…</think>`) and Content deltas — holding back a
/// marker-length tail so a partial `<think>`/`</think>`/`<tool_call>` marker is never emitted. Once a
/// `<tool_call>` opener appears (and tool calls are allowed) it stops streaming content; `finish()`
/// flushes the held-back tails and parses the buffered tool call(s) into ToolCall deltas.
pub struct ChatStream {
    raw: String,
    sent_r: usize, // reasoning bytes already emitted (offset within the reasoning region)
    sent_c: usize, // content bytes already emitted (offset within the content region)
    allow_tools: bool,
}

impl ChatStream {
    pub fn new(allow_tools: bool) -> Self {
        Self {
            raw: String::new(),
            sent_r: 0,
            sent_c: 0,
            allow_tools,
        }
    }

    pub fn push(&mut self, piece: &str, on_delta: &mut dyn FnMut(Delta)) {
        self.raw.push_str(piece);
        self.emit(false, on_delta);
    }

    pub fn finish(&mut self, on_delta: &mut dyn FnMut(Delta)) {
        self.emit(true, on_delta); // flush the held-back tails (no marker can still be forming)
        if self.allow_tools {
            let (_r, body) = crate::split_think(&self.raw);
            let (_content, calls) = crate::parse_hermes_tool_calls(&body);
            for call in calls {
                on_delta(Delta::ToolCall {
                    name: call.name,
                    arguments: serde_json::to_string(&call.arguments)
                        .unwrap_or_else(|_| "{}".to_string()),
                });
            }
        }
    }

    fn emit(&mut self, final_flush: bool, on_delta: &mut dyn FnMut(Delta)) {
        const TO: &str = "<think>";
        const TC: &str = "</think>";
        const TL: &str = "<tool_call>";
        let raw = &self.raw;
        let think_open = raw.find(TO);
        let think_close = raw.find(TC);
        let tool_open = if self.allow_tools { raw.find(TL) } else { None };
        // Reasoning region: between `<think>` and `</think>` (or end, while still thinking).
        if let Some(to) = think_open {
            let rs = to + TO.len();
            let (r_end, hold) = match think_close {
                Some(tc) => (tc, false),
                None => (raw.len(), !final_flush),
            };
            emit_region(raw, rs, r_end, hold, &mut self.sent_r, true, on_delta);
        }
        // Content region: after `</think>` (or from the start when there's no `<think>` at all), up to
        // a `<tool_call>` opener (whose block is buffered, not streamed) or the end.
        let c_start = match think_close {
            Some(tc) => Some(tc + TC.len()),
            None if think_open.is_none() => Some(0),
            None => None,
        };
        if let Some(cs) = c_start {
            let (c_end, hold) = match tool_open {
                Some(t) if t >= cs => (t, false),
                _ => (raw.len(), !final_flush),
            };
            emit_region(raw, cs, c_end, hold, &mut self.sent_c, false, on_delta);
        }
    }
}

/// Emit the not-yet-sent slice of `raw[region_start .. region_end]` (past `*sent` bytes), holding back
/// a marker-length tail when `hold` (so a partial marker isn't emitted mid-stream), clamped to a UTF-8
/// boundary. Advances `*sent`.
fn emit_region(
    raw: &str,
    region_start: usize,
    region_end: usize,
    hold: bool,
    sent: &mut usize,
    reasoning: bool,
    on_delta: &mut dyn FnMut(Delta),
) {
    const HOLD: usize = 12; // > the longest marker, so a partial one never streams
    let abs = region_start + *sent;
    if abs >= region_end {
        return;
    }
    let mut end = if hold {
        region_end.saturating_sub(HOLD).max(abs)
    } else {
        region_end
    };
    while end > abs && !raw.is_char_boundary(end) {
        end -= 1;
    }
    if end <= abs {
        return;
    }
    let text = &raw[abs..end];
    if text.is_empty() {
        return;
    }
    on_delta(if reasoning {
        Delta::Reasoning(text.to_owned())
    } else {
        Delta::Content(text.to_owned())
    });
    *sent += text.len();
}

#[cfg(test)]
mod chat_stream_tests {
    use super::*;

    /// Feed `pieces` through a `ChatStream` and collect every emitted delta (streaming + finish).
    fn run(pieces: &[&str], allow_tools: bool) -> (String, String, Vec<(String, String)>) {
        let mut out: Vec<Delta> = Vec::new();
        let mut s = ChatStream::new(allow_tools);
        {
            let mut od = |d: Delta| out.push(d);
            for p in pieces {
                s.push(p, &mut od);
            }
        }
        s.finish(&mut |d: Delta| out.push(d));
        let (mut r, mut c, mut t) = (String::new(), String::new(), Vec::new());
        for d in &out {
            match d {
                Delta::Reasoning(x) => r.push_str(x),
                Delta::Content(x) => c.push_str(x),
                Delta::ToolCall { name, arguments } => t.push((name.clone(), arguments.clone())),
            }
        }
        (r, c, t)
    }

    #[test]
    fn streams_think_then_content() {
        let (r, c, t) = run(
            &["<think>", "reason", "ing", "</think>", "the ", "answer"],
            true,
        );
        assert_eq!(r.trim(), "reasoning");
        assert_eq!(c.trim(), "the answer");
        assert!(t.is_empty());
    }

    #[test]
    fn plain_content_no_think() {
        let (r, c, _) = run(&["hello ", "world, this is a longer reply"], true);
        assert!(r.trim().is_empty());
        assert_eq!(c.trim(), "hello world, this is a longer reply");
    }

    #[test]
    fn tool_call_buffered_and_parsed() {
        let (r, _c, t) = run(
            &[
                "<think>plan</think>",
                "<tool_call>\n{\"name\": \"run_bash\", \"arguments\": {\"command\": \"ls\"}}\n</tool_call>",
            ],
            true,
        );
        assert_eq!(r.trim(), "plan");
        assert_eq!(t.len(), 1);
        assert_eq!(t[0].0, "run_bash");
        assert!(t[0].1.contains("ls"));
    }

    #[test]
    fn tool_choice_none_keeps_tool_text_as_content() {
        // allow_tools=false → a <tool_call> block is NOT extracted; it stays content.
        let (_r, c, t) = run(&["<tool_call>{\"name\":\"x\"}</tool_call>"], false);
        assert!(t.is_empty());
        assert!(c.contains("tool_call"));
    }
}
