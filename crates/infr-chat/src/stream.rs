//! Streaming reasoning/content splitter — THE single place model output is separated into
//! reasoning vs answer (vs tool calls) as it streams. `infr run` (display styling), `infr serve`
//! (OpenAI deltas) and the history stripper all consume this module, so every thinking model is
//! exposed the same way on every surface; a new reasoning format (e.g. channel markers) lands
//! here once and everywhere at once.

/// Does this rendered prompt end with a `<think>` PREFILL (the template opened the reasoning
/// block itself — Qwen3.5 style — so the model's output starts MID-reasoning with only
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
/// tool-call opener appears (and tool calls are allowed) it stops streaming content — matching every
/// dialect `finish()` parses (Hermes `<tool_call>`, the pipe form `<|tool_call>`, and a Llama-3
/// bare-JSON `{…}` body), so non-Hermes calls can't leak as content while also firing as a ToolCall.
/// `finish()` flushes the held-back tails and parses the buffered tool call(s) into ToolCall deltas.
pub struct ChatStream {
    raw: String,
    sent_r: usize, // reasoning bytes already emitted (offset within the reasoning region)
    sent_c: usize, // content bytes already emitted (offset within the content region)
    // Content before a `<think>` that does not open the reply, fixed once the opener is seen:
    // `(stop, end)`, where `stop` is a tool-call opener in it (or `end`), and how much of it has
    // been emitted.
    lead: Option<(usize, usize)>,
    sent_lead: usize,
    allow_tools: bool,
    // Per-marker cursors so `emit` doesn't re-`find` from offset 0 over the whole buffer on every
    // pushed piece (that was O(n²) across a response). Each caches the first hit once found and
    // otherwise resumes scanning from the previously-scanned tail.
    cur_think_open: Cursor,
    cur_think_close: Cursor,
    cur_tool: Cursor,  // Hermes/Qwen `<tool_call>`
    cur_pipe: Cursor,  // gemma4/E2B/DG `<|tool_call>`
    cur_final: Cursor, // channel-format `<channel|>`
    bare: BareJson,
}

/// Incremental scan for a bare-JSON tool call at the head of a content region: skips leading
/// whitespace, then tracks brace depth outside JSON strings until the first object closes. Each
/// call scans only the bytes appended since the last, so streaming stays linear in the reply.
#[derive(Default)]
struct BareJson {
    cs: Option<usize>, // the region start this state describes
    scanned: usize,
    head: BareHead,
    depth: usize,
    in_str: bool,
    esc: bool,
}

#[derive(Default, Clone, Copy, PartialEq)]
enum BareHead {
    /// Only whitespace so far.
    #[default]
    Pending,
    /// The region opens with `{` at this offset and the object has not closed yet.
    Open(usize),
    /// The first object closed and parses as a call; held until `finish` decides.
    Call(usize),
    /// Not a bare-JSON call: no leading `{`, or its first object is not a call.
    No,
}

impl BareJson {
    /// Where the region at `cs` opens a possible bare-JSON call, or `None` once it cannot be one.
    fn open(&mut self, raw: &str, cs: usize) -> Option<usize> {
        if self.cs != Some(cs) {
            *self = Self {
                cs: Some(cs),
                scanned: cs,
                ..Self::default()
            };
        }
        let bytes = raw.as_bytes();
        while self.scanned < bytes.len() {
            let i = self.scanned;
            let b = bytes[i];
            self.scanned += 1;
            match self.head {
                BareHead::Pending => match b {
                    b' ' | b'\n' | b'\t' | b'\r' => {}
                    b'{' => {
                        self.head = BareHead::Open(i);
                        self.depth = 1;
                    }
                    _ => self.head = BareHead::No,
                },
                BareHead::Open(start) => {
                    if self.in_str {
                        match b {
                            _ if self.esc => self.esc = false,
                            b'\\' => self.esc = true,
                            b'"' => self.in_str = false,
                            _ => {}
                        }
                    } else {
                        match b {
                            b'"' => self.in_str = true,
                            b'{' => self.depth += 1,
                            b'}' => {
                                self.depth -= 1;
                                if self.depth == 0 {
                                    // A call must be the whole region, so the first object
                                    // decides: one that is not a call means none is coming.
                                    let call = crate::tools::parse_bare_json_call(&raw[start..=i]);
                                    self.head = if call.is_some() {
                                        BareHead::Call(start)
                                    } else {
                                        BareHead::No
                                    };
                                }
                            }
                            _ => {}
                        }
                    }
                }
                BareHead::Call(_) | BareHead::No => {
                    self.scanned = bytes.len();
                }
            }
        }
        match self.head {
            BareHead::Open(i) | BareHead::Call(i) => Some(i),
            BareHead::Pending | BareHead::No => None,
        }
    }
}

/// A forward-only search for a fixed `needle` in an append-only buffer: once the needle is found its
/// byte offset is cached; until then, each call resumes scanning from just before the previously
/// scanned tail (so a needle straddling the append boundary is still caught).
#[derive(Default)]
struct Cursor {
    off: Option<usize>,
    scanned: usize,
}

impl Cursor {
    fn find(&mut self, raw: &str, needle: &str) -> Option<usize> {
        self.find_from(raw, needle, 0)
    }

    /// [`Cursor::find`] for hits at or after `from` only. A cached hit below `from` is dropped and
    /// the search resumes at `from`: latching it would answer "no hit" forever after, which is how a
    /// `<tool_call>` quoted in the reasoning hid the real call in the content region.
    fn find_from(&mut self, raw: &str, needle: &str, from: usize) -> Option<usize> {
        match self.off {
            Some(p) if p >= from => return Some(p),
            Some(_) => {
                self.off = None;
                self.scanned = from;
            }
            None => {}
        }
        let mut start = self
            .scanned
            .saturating_sub(needle.len().saturating_sub(1))
            .max(from);
        while start > from && !raw.is_char_boundary(start) {
            start -= 1;
        }
        if let Some(p) = raw[start..].find(needle) {
            self.off = Some(start + p);
        }
        self.scanned = raw.len();
        self.off
    }
}

impl ChatStream {
    pub fn new(allow_tools: bool) -> Self {
        Self {
            raw: String::new(),
            sent_r: 0,
            sent_c: 0,
            lead: None,
            sent_lead: 0,
            allow_tools,
            cur_think_open: Cursor::default(),
            cur_think_close: Cursor::default(),
            cur_tool: Cursor::default(),
            cur_pipe: Cursor::default(),
            cur_final: Cursor::default(),
            bare: BareJson::default(),
        }
    }

    pub fn push(&mut self, piece: &str, on_delta: &mut dyn FnMut(Delta)) {
        self.raw.push_str(piece);
        self.emit(false, on_delta);
    }

    /// Flush the held-back tails (no marker can still be forming) and settle every stretch that
    /// streaming held back from a tool-call opener: its calls fire, or it is emitted as content
    /// when it holds none.
    pub fn finish(&mut self, on_delta: &mut dyn FnMut(Delta)) {
        self.emit(true, on_delta);
    }

    /// The earliest tool-call opener at/after `cs`, across every dialect `finish()` parses — Hermes
    /// `<tool_call>`, the pipe form `<|tool_call>`, and (when the content region begins with `{`) a
    /// Llama-3 bare-JSON call. Returns `None` when tools aren't allowed or no opener is present, so
    /// the content region runs to the buffer end. Matching `finish()`'s dialect set here is what
    /// stops non-Hermes calls from leaking as `Delta::Content` while also firing as `Delta::ToolCall`.
    fn tool_open_at(&mut self, cs: usize, final_flush: bool) -> Option<usize> {
        if !self.allow_tools {
            return None;
        }
        const TL: &str = "<tool_call>";
        const PIPE: &str = "<|tool_call>";
        let hermes = self.cur_tool.find_from(&self.raw, TL, cs);
        let pipe = self.cur_pipe.find_from(&self.raw, PIPE, cs);
        let bare = self.bare_json_open(cs, final_flush);
        [hermes, pipe, bare].into_iter().flatten().min()
    }

    /// Position of a bare-JSON (`{…}`) tool-call opener in the content region starting at `cs`, or
    /// `None`. Scoped to the leading `{` of the region (skipping only whitespace) so ordinary prose
    /// that merely *contains* a brace is never suppressed. During streaming a leading `{` is held
    /// back until its first object closes ([`BareJson`]): one that is not a call streams as content
    /// from then on, instead of the whole reply arriving as one chunk at `finish`. On the final
    /// flush it's confirmed a real call via [`crate::tools::parse_bare_json_call`] so a leading
    /// brace that isn't a call still streams as content instead of being dropped.
    fn bare_json_open(&mut self, cs: usize, final_flush: bool) -> Option<usize> {
        let i = self.bare.open(&self.raw, cs)?;
        if final_flush && crate::tools::parse_bare_json_call(&self.raw[i..]).is_none() {
            return None; // leading `{` but not actually a call — let it stream as content
        }
        Some(i)
    }

    fn emit(&mut self, final_flush: bool, on_delta: &mut dyn FnMut(Delta)) {
        const TO: &str = "<think>";
        const TC: &str = "</think>";
        // Channel-format head (E2B/gpt-oss: `<|channel>thought…<channel|>answer`): reasoning runs
        // from after the thought marker to the final-answer marker, content after it. While the
        // head could still be a forming marker, hold everything back.
        const FINAL: &str = "<channel|>";
        if let Some(hm) = channel_thought_head(&self.raw) {
            let (r_end, hold, c_start) = match self.cur_final.find(&self.raw, FINAL) {
                Some(f) => (f, false, Some(f + FINAL.len())),
                None => (self.raw.len(), !final_flush, None),
            };
            emit_region(&self.raw, hm, r_end, hold, &mut self.sent_r, true, on_delta);
            if let Some(cs) = c_start {
                self.emit_content(cs, final_flush, on_delta);
            }
            return;
        }
        if !final_flush
            && CHANNEL_THOUGHT.iter().any(|m| {
                m.as_bytes()
                    .starts_with(&self.raw.as_bytes()[..self.raw.len().min(m.len())])
            })
            && self.raw.len() < CHANNEL_THOUGHT[0].len()
        {
            return; // head could still become a channel marker — hold
        }
        let think_open = self.cur_think_open.find(&self.raw, TO);
        let think_close = self.cur_think_close.find(&self.raw, TC);
        // Content before a `<think>` later in the reply. Until the opener arrives it streamed as
        // the content region at offset 0, so that region's progress carries over; its end is then
        // fixed, and streaming stops at a tool-call opener in it.
        let lead = think_open
            .filter(|&to| think_close.is_none_or(|tc| tc > to))
            .map(|to| match self.lead {
                Some(lead) => lead,
                None => {
                    self.sent_lead = std::mem::take(&mut self.sent_c);
                    // The streaming rule even on the final flush: a held opener is judged by
                    // `settle_held` on the lead-in alone, never with the `<think>` text after it.
                    let stop = match to {
                        0 => 0,
                        _ => self.tool_open_at(0, false).map_or(to, |t| t.min(to)),
                    };
                    self.lead = Some((stop, to));
                    (stop, to)
                }
            });
        if let Some((stop, end)) = lead {
            emit_region(
                &self.raw,
                0,
                stop,
                false,
                &mut self.sent_lead,
                false,
                on_delta,
            );
            if final_flush && stop < end {
                settle_held(&self.raw[stop..end], on_delta);
            }
        }
        // Reasoning region: between `<think>` and `</think>` (or end, while still thinking).
        if let Some(to) = think_open {
            let rs = to + TO.len();
            let (r_end, hold) = match think_close {
                Some(tc) => (tc, false),
                None => (self.raw.len(), !final_flush),
            };
            emit_region(&self.raw, rs, r_end, hold, &mut self.sent_r, true, on_delta);
        }
        // A lead-in held at a tool-call opener is settled only at `finish`; content after the
        // reasoning waits with it so the content arrives in order.
        if !final_flush && self.lead.is_some_and(|(stop, end)| stop < end) {
            return;
        }
        // Content region: after `</think>` (or from the start when there's no `<think>` at all), up to
        // a tool-call opener (whose block is buffered, not streamed) or the end.
        let c_start = match think_close {
            Some(tc) => Some(tc + TC.len()),
            None if think_open.is_none() => Some(0),
            None => None,
        };
        if let Some(cs) = c_start {
            self.emit_content(cs, final_flush, on_delta);
        }
    }

    /// Stream the content region starting at `cs` up to its first tool-call opener. On the final
    /// flush the stretch from that opener on is settled ([`settle_held`]).
    fn emit_content(&mut self, cs: usize, final_flush: bool, on_delta: &mut dyn FnMut(Delta)) {
        let tool_open = self.tool_open_at(cs, final_flush).filter(|&t| t >= cs);
        let (c_end, hold) = match tool_open {
            Some(t) => (t, false),
            None => (self.raw.len(), !final_flush),
        };
        emit_region(
            &self.raw,
            cs,
            c_end,
            hold,
            &mut self.sent_c,
            false,
            on_delta,
        );
        if let (true, Some(t)) = (final_flush, tool_open) {
            settle_held(&self.raw[t..], on_delta);
        }
    }
}

/// Settle a stretch streaming held back from a tool-call opener. A `<think>` span in it is
/// reasoning, not content, and goes out as such. Every call in the rest fires, and what is not a
/// call is emitted as content verbatim: prose around the calls with its own whitespace, and any
/// block that did not parse (see [`crate::parse_hermes_tool_calls`]) — nothing the model wrote is
/// dropped. The one exception is a remainder that is only whitespace once the calls are out,
/// which shows nothing.
fn settle_held(held: &str, on_delta: &mut dyn FnMut(Delta)) {
    let (body, reasoning) = split_reasoning_spans(held);
    if !reasoning.is_empty() {
        on_delta(Delta::Reasoning(reasoning));
    }
    // Dialect-aware: hermes JSON + qwen3.5/3.6 XML + gemma pipe-marker + llama3 bare JSON
    // (hermes-only here silently dropped every non-qwen3 model's calls over serve).
    let (content, calls) = crate::parse_any_tool_calls(&body);
    let content = if calls.is_empty() { body } else { content };
    if !content.trim().is_empty() {
        on_delta(Delta::Content(content));
    }
    for call in calls {
        on_delta(Delta::ToolCall {
            name: call.name,
            arguments: serde_json::to_string(&call.arguments).unwrap_or_else(|_| "{}".to_string()),
        });
    }
}

/// `text` without its `<think>…</think>` spans (an unclosed one runs to the end), and the
/// reasoning inside them.
fn split_reasoning_spans(text: &str) -> (String, String) {
    const TO: &str = "<think>";
    const TC: &str = "</think>";
    let (mut content, mut reasoning) = (String::new(), String::new());
    let mut rest = text;
    while let Some(o) = rest.find(TO) {
        content.push_str(&rest[..o]);
        let inner = &rest[o + TO.len()..];
        let Some(c) = inner.find(TC) else {
            reasoning.push_str(inner);
            return (content, reasoning);
        };
        reasoning.push_str(&inner[..c]);
        rest = &inner[c + TC.len()..];
    }
    content.push_str(rest);
    (content, reasoning)
}

/// The heads that put a reply in the channel format (E2B/gpt-oss): reasoning after the thought
/// marker, the answer after `<channel|>`.
const CHANNEL_THOUGHT: [&str; 2] = ["<|channel|>thought", "<|channel>thought"];

/// The length of the channel-format head `raw` starts with, or `None` for any other reply. THE test
/// for "this reply is in the channel format": streaming and [`ChatStream::finish`] both use it, so
/// they cannot disagree about where the content — and so a tool call — can be.
fn channel_thought_head(raw: &str) -> Option<usize> {
    CHANNEL_THOUGHT
        .iter()
        .find_map(|m| raw.starts_with(m).then_some(m.len()))
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
    fn streams_channel_thought_then_final() {
        // E2B/gpt-oss channel format: reasoning after the thought marker, content after the
        // final-answer marker — streamed as Reasoning/Content like the <think> form.
        let (r, c, t) = run(
            &[
                "<|channel>th",
                "ought\nreaso",
                "ning<chan",
                "nel|>the answer",
            ],
            false,
        );
        assert_eq!(r.trim(), "reasoning");
        assert_eq!(c.trim(), "the answer");
        assert!(t.is_empty());
    }

    #[test]
    fn prefilled_think_via_injected_opener() {
        // Template-prefilled thinking (Qwen3.5): the caller injects "<think>" before the model's
        // mid-reasoning output; the splitter then treats the head as Reasoning.
        let (r, c, t) = run(&["<think>", "reasoning here", "</think>", "answer"], false);
        assert_eq!(r.trim(), "reasoning here");
        assert_eq!(c.trim(), "answer");
        assert!(t.is_empty());
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

    #[test]
    fn pipe_form_tool_call_not_streamed_as_content() {
        // gemma4/E2B/DG pipe form has no `<tool_call>` substring — it must still gate content
        // and fire exactly once as a ToolCall (not leak as Content AND duplicate).
        let (_r, c, t) = run(
            &[
                "<|tool_",
                "call>call:get_weather{city:<|\"|>Par",
                "is<|\"|>}",
                "<tool_call|>",
            ],
            true,
        );
        assert!(
            !c.contains("tool_call") && !c.contains("get_weather"),
            "pipe markup leaked as content: {c:?}"
        );
        assert_eq!(t.len(), 1, "expected exactly one tool call, got {t:?}");
        assert_eq!(t[0].0, "get_weather");
    }

    #[test]
    fn bare_json_tool_call_not_streamed_as_content() {
        // Llama-3 bare-JSON form: the whole body is one JSON object — no markers at all.
        let (_r, c, t) = run(
            &[
                "{\"name\": \"get_weather\", ",
                "\"parameters\": {\"city\": \"Paris\"}}",
            ],
            true,
        );
        assert!(c.trim().is_empty(), "bare json leaked as content: {c:?}");
        assert_eq!(t.len(), 1, "expected exactly one tool call, got {t:?}");
        assert_eq!(t[0].0, "get_weather");
    }

    #[test]
    fn content_with_stray_brace_not_swallowed() {
        // Ordinary prose that merely contains a brace mid-text must NOT be treated as a call.
        let (_r, c, t) = run(
            &[
                "The config is {",
                "\"a\": 1} and that is the whole story here, nothing more.",
            ],
            true,
        );
        assert!(t.is_empty(), "stray brace misparsed as tool call: {t:?}");
        assert!(c.contains("The config is"), "content lost: {c:?}");
        assert!(c.contains("whole story"), "content truncated: {c:?}");
    }

    /// A `<tool_call>` mentioned inside the reasoning must not hide the real call that follows in
    /// the content: the real one is buffered (not streamed as raw markup) and fires exactly once.
    #[test]
    fn tool_marker_in_reasoning_does_not_hide_the_real_call() {
        let call = "<tool_call>\n{\"name\": \"run_bash\", \"arguments\": {\"command\": \"ls\"}}\n</tool_call>";
        for pipe in [false, true] {
            let (mention, real) = if pipe {
                (
                    "I could emit <|tool_call> here",
                    "<|tool_call>call:run_bash{command:<|\"|>ls<|\"|>}<tool_call|>",
                )
            } else {
                ("I could emit <tool_call> here", call)
            };
            let pieces = ["<think>", mention, "</think>", "Sure. ", real];
            let (_r, c, t) = run(&pieces, true);
            assert!(
                !c.contains("run_bash"),
                "the call streamed as content: {c:?}"
            );
            assert_eq!(t.len(), 1, "expected exactly one tool call, got {t:?}");
            assert_eq!(t[0].0, "run_bash");
        }
    }

    /// A call quoted inside a channel-format thought block is reasoning, not a call: the stream
    /// never treats it as one, and `finish` must not either.
    #[test]
    fn tool_call_in_channel_reasoning_is_not_a_call() {
        let (r, c, t) = run(
            &[
                "<|channel>thought\nmaybe <|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}",
                "<tool_call|> but no<channel|>It is sunny.",
            ],
            true,
        );
        assert!(r.contains("maybe"), "reasoning lost: {r:?}");
        assert_eq!(c.trim(), "It is sunny.");
        assert!(t.is_empty(), "a call quoted in the reasoning fired: {t:?}");
    }

    /// A `<think>`-format reply whose content merely mentions a channel marker is still a
    /// `<think>`-format reply: the call streaming held back must fire at `finish`, not vanish into
    /// an empty channel-format body.
    #[test]
    fn a_quoted_channel_marker_does_not_switch_formats() {
        let (_r, c, t) = run(
            &[
                "<think>plan</think>",
                "The <|channel> marker is how E2B splits. ",
                "<tool_call>\n{\"name\": \"run_bash\", \"arguments\": {\"command\": \"ls\"}}\n</tool_call>",
            ],
            true,
        );
        assert!(c.contains("<|channel> marker"), "content lost: {c:?}");
        assert_eq!(t.len(), 1, "expected exactly one tool call, got {t:?}");
        assert_eq!(t[0].0, "run_bash");
    }

    /// With tools on, a reply opening with a JSON object that is not a call streams once the
    /// object closes, instead of arriving as one chunk at `finish`. Braces and an escaped quote
    /// inside a string do not end the object early.
    #[test]
    fn a_leading_json_answer_streams_before_finish() {
        let pieces = [
            "{\"a\": [1, ",
            "2], \"b\": \"x} \\\"q\\\" {\"}",
            " and the prose after it keeps coming",
            " for a while longer.",
        ];
        let mut s = ChatStream::new(true);
        let mut streamed = String::new();
        for p in pieces {
            s.push(p, &mut |d| {
                if let Delta::Content(x) = d {
                    streamed.push_str(&x)
                }
            });
        }
        assert!(
            streamed.starts_with("{\"a\": [1, 2], \"b\": \"x} \\\"q\\\" {\"}"),
            "the object was not streamed before finish: {streamed:?}"
        );
        let (_r, c, t) = run(&pieces, true);
        assert!(t.is_empty(), "a plain object fired as a call: {t:?}");
        assert_eq!(c, pieces.concat());
    }

    /// Content before a `<think>` that does not open the reply streams, in order, and does not
    /// shift the content after `</think>`.
    #[test]
    fn content_before_a_late_think_is_streamed() {
        let pieces = [
            "Sure, let me check. ",
            "<think>",
            "plan",
            "</think>",
            "It is 4.",
        ];
        let mut out = Vec::new();
        let mut s = ChatStream::new(false);
        for p in pieces {
            s.push(p, &mut |d| out.push(d));
        }
        s.finish(&mut |d| out.push(d));
        let lead = out
            .iter()
            .position(|d| matches!(d, Delta::Content(x) if x.contains("check.")));
        let think = out.iter().position(|d| matches!(d, Delta::Reasoning(_)));
        assert!(
            lead < think && lead.is_some(),
            "lead missing or late: {out:?}"
        );
        let (r, c, _) = run(&pieces, false);
        assert_eq!(r, "plan");
        assert_eq!(c, "Sure, let me check. It is 4.");
    }

    /// A lead-in held at a tool-call opener before a `<think>` (a leading `{`, a `<tool_call>`)
    /// that turns out not to be a call is emitted at `finish`, ahead of the content after the
    /// reasoning, instead of vanishing.
    #[test]
    fn a_held_lead_in_that_is_no_call_is_emitted_in_order() {
        for (pieces, want) in [
            (
                &[
                    "{not a call} so ",
                    "<think>",
                    "plan",
                    "</think>",
                    "an answer long enough to stream",
                ][..],
                "{not a call} so an answer long enough to stream",
            ),
            (
                &[
                    "Hi <tool_call>oops ",
                    "<think>",
                    "plan",
                    "</think>",
                    "done, and more after that",
                ][..],
                "Hi <tool_call>oops done, and more after that",
            ),
        ] {
            let (r, c, t) = run(pieces, true);
            assert_eq!(r, "plan");
            assert!(t.is_empty(), "no call was made: {t:?}");
            assert_eq!(c, want);
        }
    }

    /// A call after the reasoning is judged on the content after `</think>` alone, the region
    /// streaming held: the lead-in before `<think>` does not stop it from firing.
    #[test]
    fn a_call_after_reasoning_fires_despite_a_lead_in() {
        let (_r, c, t) = run(
            &[
                "Sure. ",
                "<think>",
                "plan",
                "</think>",
                "{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\"}}",
            ],
            true,
        );
        assert_eq!(t.len(), 1, "expected exactly one tool call, got {t:?}");
        assert_eq!(t[0].0, "get_weather");
        assert_eq!(c, "Sure. ");
    }

    /// A held tool-call marker whose body is not a call is shown as content, not dropped.
    #[test]
    fn a_marker_that_holds_no_call_is_emitted_as_content() {
        let (_r, c, t) = run(&["Here: ", "<tool_call>not json</tool_call>"], true);
        assert!(t.is_empty(), "{t:?}");
        assert_eq!(c, "Here: <tool_call>not json</tool_call>");
    }

    /// A `<think>` span after a call in the held stretch is reasoning: it must not reach the
    /// content.
    #[test]
    fn reasoning_after_a_held_call_is_not_content() {
        let (r, c, t) = run(
            &[
                "<think>plan</think>",
                "<tool_call>{\"name\": \"run_bash\", \"arguments\": {\"command\": \"ls\"}}</tool_call>",
                "<think>second thoughts</think>",
                "Done after the call.",
            ],
            true,
        );
        assert_eq!(t.len(), 1, "{t:?}");
        assert!(!c.contains("think") && !c.contains("second"), "{c:?}");
        assert!(c.contains("Done after the call."), "{c:?}");
        assert!(r.contains("second thoughts"), "{r:?}");
    }

    /// A `<tool_call>` block that does not parse is shown as content whether or not a valid
    /// call sits beside it.
    #[test]
    fn an_unparseable_block_beside_a_call_is_shown() {
        let (_r, c, t) = run(
            &[
                "Here: ",
                "<tool_call>not json</tool_call> and ",
                "<tool_call>{\"name\": \"run_bash\", \"arguments\": {}}</tool_call>",
            ],
            true,
        );
        assert_eq!(t.len(), 1, "{t:?}");
        assert!(c.contains("<tool_call>not json</tool_call>"), "{c:?}");
    }

    /// Prose after a removed call keeps the whitespace that separated it from the prose before.
    #[test]
    fn prose_around_a_removed_call_is_not_glued() {
        let (_r, c, t) = run(
            &[
                "Sure.",
                "<tool_call>{\"name\": \"run_bash\", \"arguments\": {}}</tool_call>",
                " More text after it.",
            ],
            true,
        );
        assert_eq!(t.len(), 1, "{t:?}");
        assert_eq!(c, "Sure. More text after it.");
    }

    /// A Hermes call followed by a pipe-format call: both fire and neither is shown.
    #[test]
    fn mixed_dialect_calls_all_fire() {
        let (_r, c, t) = run(
            &[
                "<tool_call>{\"name\": \"run_bash\", \"arguments\": {}}</tool_call>",
                "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>",
            ],
            true,
        );
        let names: Vec<&str> = t.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["run_bash", "get_weather"]);
        assert!(!c.contains("tool_call"), "{c:?}");
    }

    #[test]
    fn leading_brace_non_call_survives_as_content() {
        // A leading `{` that is NOT a valid bare-JSON call must be flushed as content, not dropped.
        let (_r, c, t) = run(
            &["{not json here, just prose that opens with a brace}"],
            true,
        );
        assert!(t.is_empty(), "non-call misparsed: {t:?}");
        assert!(c.contains("not json here"), "content dropped: {c:?}");
    }
}
