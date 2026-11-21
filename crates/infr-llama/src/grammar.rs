//! Grammar-constrained decoding via [`llguidance`] — the reliability tier of tool calling (llama.cpp
//! parity). A [`Constraint`] wraps an llguidance `Matcher` over the model's tokenizer and, each decode
//! step, (a) masks the logits to the grammatically-allowed tokens and (b) consumes the sampled token
//! to advance the grammar (plus any deterministically-forced "fast-forward" tokens). When tools are in
//! play the grammar forces syntactically-valid, schema-conforming `<tool_call>` JSON, so even tiny
//! models can't emit malformed calls.
//!
//! The tokenizer bridge uses [`ByteTokenizer::from_json_bytes`] (the tokenizer's serialized JSON)
//! rather than `from_tokenizer`, so it's immune to the `tokenizers` crate-version skew between infr
//! (0.20) and toktrie (0.21).
//!
//! KNOWN ISSUE — the GGUF→`tokenizers`→serialized-JSON tokenizer bridge is NOT canonical in toktrie's
//! sense, so llguidance's token-level `compute_mask` returns a SUPERSET that the byte-forcing parser
//! then rejects ("forced bytes: got '{'"). Mitigations applied: (1) `NonCanonicalEnv` forces the
//! canonical flag false; (2) the decode loop validate-before-commits each pick ([`Constraint::try_accept`])
//! and re-picks on rejection instead of erroring. These stop the crash, but on the live model the mask
//! is still inconsistent enough that the forced JSON can mask out entirely (empty body) — so the server
//! treats an empty/unparseable constrained call as a miss and FALLS BACK to unconstrained generation.
//! Net today: ≥1.7B models get reliable tool calls via the (unconstrained) auto path; the constrained
//! reliability tier for tiny models needs the principled fix — driving the grammar at BYTE level
//! (`Matcher::compute_ff_bytes`, the non-canonical-safe path) instead of token level. TODO.

use anyhow::{anyhow, Result};

mod caps;
mod shape;
use llguidance::api::TopLevelGrammar;
use llguidance::toktrie::{TokEnv, TokTrie, TokenId, TokenizerEnv};
use llguidance::{Matcher, ParserFactory};
use serde_json::Value;
use std::sync::Arc;
use tokenizers::Tokenizer;
use toktrie_hf_tokenizers::{ByteTokenizer, ByteTokenizerEnv};

/// Wraps a [`ByteTokenizerEnv`] to report itself NON-canonical. `ByteTokenizerEnv` inherits the trait
/// default `tokenize_is_canonical() == true`, which makes llguidance enable token-healing: it computes
/// a byte-forced prefix and returns a HEALED mask that's only exact for a truly-canonical tokenizer.
/// The GGUF→`tokenizers`→serialized-JSON bridge is NOT canonical, so the healed mask is a SUPERSET that
/// `consume_token` then rejects (the live "forced bytes: got '{'; applying '!'" bug). Forcing the flag
/// to `false` makes llguidance use the conservative, non-canonical-safe masking path where
/// `compute_mask` and `consume_token` agree.
struct NonCanonicalEnv(ByteTokenizerEnv);

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl TokenizerEnv for NonCanonicalEnv {
    fn tok_trie(&self) -> &TokTrie {
        self.0.tok_trie()
    }
    fn tokenize_bytes(&self, s: &[u8]) -> Vec<TokenId> {
        self.0.tokenize_bytes(s)
    }
    fn tokenize_is_canonical(&self) -> bool {
        false
    }
}

/// Build an llguidance [`TokEnv`] from infr's in-memory tokenizer. Serializes the tokenizer to JSON
/// and reparses it on toktrie's side (decoupling the `tokenizers` versions); `eos_ids` mark stop
/// tokens; `vocab` is the model's logit width so the token trie matches the logits exactly. Wrapped in
/// `NonCanonicalEnv` so the mask is consistent with `consume` (see that type's docs).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn build_tok_env(tokenizer: &Tokenizer, vocab: usize, eos_ids: &[u32]) -> Result<TokEnv> {
    let json = tokenizer
        .to_string(false)
        .map_err(|e| anyhow!("serialize tokenizer: {e}"))?;
    let mut bt = ByteTokenizer::from_json_bytes(json.as_bytes())
        .map_err(|e| anyhow!("byte tokenizer: {e}"))?;
    if !eos_ids.is_empty() {
        bt.set_eos_tokens(eos_ids);
    }
    let env = ByteTokenizerEnv::new(bt, Some(vocab)).map_err(|e| anyhow!("tok env: {e}"))?;
    Ok(Arc::new(NonCanonicalEnv(env)))
}

/// Force every id at or beyond `vocab` to -inf. Those ids sit past the token trie, so they can
/// never be grammar-legal, yet a padded-vocab model's logits row is wider than `vocab`. Split out
/// so the tail-masking is unit-testable without a live tokenizer/matcher.
fn mask_padding_tail(logits: &mut [f32], vocab: usize) {
    for l in logits.iter_mut().skip(vocab) {
        *l = f32::NEG_INFINITY;
    }
}

/// The per-MODEL half of a [`Constraint`]: the llguidance parser factory over one tokenizer, with
/// its token trie. Building it serialises and re-parses the whole tokenizer, so it is built once
/// and shared; only the parser is per request ([`ConstraintEnv::constraint`]).
pub struct ConstraintEnv {
    factory: ParserFactory,
    vocab: usize,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl ConstraintEnv {
    /// Build the environment for `tokenizer` (see [`build_tok_env`] for `vocab` and `eos_ids`).
    pub fn new(tokenizer: &Tokenizer, vocab: usize, eos_ids: &[u32]) -> Result<Self> {
        Self::from_tok_env(&build_tok_env(tokenizer, vocab, eos_ids)?)
    }

    fn from_tok_env(tok_env: &TokEnv) -> Result<Self> {
        #[cfg(test)]
        tests::ENV_BUILDS.with(|n| n.set(n.get() + 1));
        let factory =
            ParserFactory::new_simple(tok_env).map_err(|e| anyhow!("parser factory: {e}"))?;
        Ok(Self {
            factory,
            vocab: tok_env.tok_trie().vocab_size(),
        })
    }

    /// `text` spelled as a lark grammar element that matches what this model emits for it: the
    /// special token of that name where the tokenizer has one (Qwen's `<tool_call>` is a single
    /// special token, which no byte literal can match), else the literal bytes.
    fn marker(&self, text: &str) -> String {
        if self
            .factory
            .tok_env()
            .tok_trie()
            .get_special_token(text)
            .is_some()
        {
            text.to_string()
        } else {
            format!("{text:?}")
        }
    }

    /// A fresh constraint for `grammar`. A grammar llguidance cannot compile — a JSON schema
    /// using a keyword it does not support, say — fails with [`GrammarRejected`], so a caller can
    /// tell the request's fault from the environment's.
    ///
    /// The compile runs on a thread of its own with a large stack (`COMPILE_STACK`): llguidance
    /// compiles a schema by recursion, one frame chain per level of nesting and per `$ref`
    /// followed, and a stack overflow aborts the whole process rather than failing the request.
    /// A caller's thread — a tokio blocking thread, 2 MiB — is far too small for that. This
    /// stack is the margin; the bound is the shape check this crate runs on the exact JSON it
    /// compiles (pinned dialects, plain local `#/…` references to walked subschemas only, two
    /// depth bounds, `$ref` and value counts), which keeps the compile inside it. A grammar built
    /// outside this crate gets the stack, not that check.
    pub fn constraint(&self, grammar: TopLevelGrammar) -> Result<Constraint> {
        let compiled = std::thread::scope(|scope| {
            std::thread::Builder::new()
                .name("grammar-compile".into())
                .stack_size(COMPILE_STACK)
                .spawn_scoped(scope, || self.factory.create_parser(grammar))
                .map(|t| t.join())
        });
        let parser = match compiled {
            Err(e) => return Err(anyhow!("start the grammar compile thread: {e}")),
            Ok(Err(_)) => return Err(anyhow!("the grammar compile thread panicked")),
            Ok(Ok(parsed)) => parsed.map_err(|e| GrammarRejected {
                message: e.to_string(),
                in_tools: false,
            })?,
        };
        Ok(Constraint {
            matcher: Matcher::new(Ok(parser)),
            vocab: self.vocab,
        })
    }
}

/// The stack the grammar compile runs on (see [`ConstraintEnv::constraint`]). Every schema this
/// crate compiles passes the shape check first, whose level bound is derived from this stack:
/// its arithmetic sits with that bound, in the `shape` module. It is address space, committed
/// only as deep as a compile actually goes.
const COMPILE_STACK: usize = 64 << 20;

/// llguidance refused to compile a grammar: the grammar (a request's JSON schema, a tool's
/// parameter schema) is at fault, not the tokenizer environment it was compiled against.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrammarRejected {
    /// llguidance's own explanation.
    pub message: String,
    /// Whether the request's `tools` are at fault — a malformed tool or a parameter schema the
    /// grammar engine refuses — rather than its `response_format` schema.
    pub in_tools: bool,
}

impl GrammarRejected {
    /// A rejection blamed on the request's `tools`.
    fn tools(message: impl Into<String>) -> anyhow::Error {
        GrammarRejected {
            message: message.into(),
            in_tools: true,
        }
        .into()
    }
}

/// `e` blamed on the request's `tools` when it is a [`GrammarRejected`]; any other error as is.
fn blame_tools(e: anyhow::Error) -> anyhow::Error {
    match e.downcast::<GrammarRejected>() {
        Ok(r) => GrammarRejected::tools(r.message),
        Err(e) => e,
    }
}

impl std::fmt::Display for GrammarRejected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for GrammarRejected {}

/// This backend cannot apply a grammar constraint — the default of
/// [`crate::chat::ChatModel::generate_constrained`]: the request needs something this model's
/// backend does not do, not a server fault.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstraintUnsupported;

impl std::fmt::Display for ConstraintUnsupported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("grammar-constrained generation is not supported by this backend")
    }
}

impl std::error::Error for ConstraintUnsupported {}

/// A live grammar constraint over a decode. Cheap-ish to construct (parser build); one per request.
pub struct Constraint {
    matcher: Matcher,
    vocab: usize,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl Constraint {
    /// Construct a constraint for `grammar` over `tok_env`.
    pub fn new(tok_env: TokEnv, grammar: TopLevelGrammar) -> Result<Self> {
        ConstraintEnv::from_tok_env(&tok_env)?.constraint(grammar)
    }

    /// Mask `logits` in place to the grammar's allowed tokens (disallowed → -inf). Then the caller
    /// samples as usual and feeds the chosen token to [`try_accept`](Self::try_accept).
    pub fn apply_mask(&mut self, logits: &mut [f32]) -> Result<()> {
        let mask = self.matcher.compute_mask().map_err(|e| anyhow!("{e}"))?;
        let n = logits.len().min(self.vocab);
        for (id, l) in logits.iter_mut().enumerate().take(n) {
            if !mask.is_allowed(id as u32) {
                *l = f32::NEG_INFINITY;
            }
        }
        // Padded-vocab GGUFs give a logits row WIDER than the token trie (`logits.len() > vocab`).
        // Those tail ids have no trie entry, so the mask can never allow them — but the loop above
        // stops at `vocab`, leaving them finite. Force them to -inf so a downstream full-slice
        // argmax/sample can't emit an out-of-grammar, out-of-trie token (fail CLOSED, not open).
        mask_padding_tail(logits, self.vocab);
        Ok(())
    }

    /// The grammar's deterministically-FORCED continuation at the current state (e.g. the literal
    /// `{`/`"`/`:` bytes a JSON object must emit). These must be consumed WITHOUT sampling — the right
    /// llguidance flow is to drain forced tokens first each step, then mask+sample only a free token.
    /// Returns empty when the next token is a real choice.
    pub fn forced(&mut self) -> Vec<u32> {
        self.matcher.compute_ff_tokens()
    }

    /// Try to advance the grammar by one freely-sampled `token`, validating BEFORE committing. Returns
    /// `true` if accepted, `false` if the token isn't actually grammar-legal (no state change). This is
    /// the guard against llguidance's token-healing returning a SUPERSET mask: `compute_mask` can allow
    /// a token that the parser then rejects (the GGUF tokenizer bridge isn't truly canonical), so the
    /// decode loop re-picks instead of failing the whole request.
    pub fn try_accept(&mut self, token: u32) -> Result<bool> {
        Ok(self
            .matcher
            .try_consume_tokens(&[token])
            .map_err(|e| anyhow!("{e}"))?
            == 1)
    }

    /// Whether the grammar is in an accepting state (a legal place to stop — EOS allowed here).
    pub fn accepting(&mut self) -> Result<bool> {
        self.matcher.is_accepting().map_err(|e| anyhow!("{e}"))
    }

    /// Advance the grammar by a run of forced tokens.
    pub fn consume(&mut self, tokens: &[u32]) -> Result<()> {
        self.matcher
            .consume_tokens(tokens)
            .map_err(|e| anyhow!("{e}"))
    }

    /// Whether the grammar has reached an accepting stop state (no further tokens required).
    pub fn stopped(&mut self) -> bool {
        self.matcher.is_stopped()
    }
}

/// One CONSTRAINED decode step over `logits`: drain llguidance's deterministically-forced tokens
/// first (no sampling — the intended flow, keeps compute_mask/consume consistent); otherwise mask
/// the logits and pick a grammar-legal token with validate-before-commit (the healed mask can be a
/// SUPERSET on the non-canonical GGUF tokenizer bridge — a rejected candidate is dropped and
/// re-picked, never failed). EOS terminates only in an accepting state.
///
/// The free token is drawn with the SAME [`crate::sampling::Sampler`]/`rng` the unconstrained decode path uses, so a
/// request that sets `temperature`/`top_p`/`seed` alongside `tool_choice` is honoured instead of
/// being silently forced to greedy. At `temp<=0` `sample_logits`
/// collapses to `argmax`, so the deterministic tool-call path is byte-for-byte unchanged; a masked
/// (-inf) token has probability 0 in the softmax, so sampling can never leave the grammar.
///
/// `budget` is how many tokens the request may still emit. A forced run can be many tokens long,
/// so only its first `budget` are consumed — the grammar stays where those leave it — and an
/// exhausted budget emits nothing.
///
/// `penalties`, when a request sets any, are applied to `logits` before the mask, as they are on
/// an unconstrained step, and every token this step emits is recorded in them, forced ones
/// included: a forced tool call honours `frequency_penalty`/`presence_penalty`/`repeat_penalty`
/// like any other generation (llama.cpp applies penalties and the grammar together). The
/// request's `logit_bias` travels in the same state and is applied with them. A penalty or bias
/// keeps a finite logit finite (see [`crate::sampling::Penalties::apply`]), so it can reorder the
/// grammar-legal tokens but never mask one out, and "mask exhausted" still means the grammar
/// itself allows nothing.
///
/// Returns `(tokens emitted this step, grammar finished)`; an EMPTY step means the constrained
/// span is done (accepting EOS, mask exhausted, or no budget left). ONE implementation shared by
/// the bespoke decode loop and both seam decode paths.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn constrained_step(
    c: &mut Constraint,
    logits: &mut [f32],
    eos_ids: &[u32],
    sampler: crate::sampling::Sampler,
    rng: &mut u64,
    budget: usize,
    mut penalties: Option<&mut crate::sampling::Penalties>,
) -> Result<(Vec<u32>, bool)> {
    if budget == 0 {
        return Ok((Vec::new(), false));
    }
    let mut forced = c.forced();
    if !forced.is_empty() {
        forced.truncate(budget);
        c.consume(&forced)?;
        if let Some(p) = penalties {
            for &t in &forced {
                p.observe(t);
            }
        }
        let done = c.stopped();
        return Ok((forced, done));
    }
    if let Some(p) = penalties.as_deref() {
        p.apply(logits);
    }
    c.apply_mask(logits)?;
    loop {
        let cand = crate::sampling::sample_logits(logits, sampler, rng);
        if !logits[cand as usize].is_finite() {
            return Ok((Vec::new(), true)); // mask exhausted — nothing grammar-legal left
        }
        if eos_ids.contains(&cand) {
            if c.accepting()? {
                return Ok((Vec::new(), true)); // legal end of the constrained span
            }
            logits[cand as usize] = f32::NEG_INFINITY; // EOS not allowed yet
            continue;
        }
        if c.try_accept(cand)? {
            if let Some(p) = penalties.as_deref_mut() {
                p.observe(cand);
            }
            let done = c.stopped();
            return Ok((vec![cand], done));
        }
        logits[cand as usize] = f32::NEG_INFINITY; // superset member — drop + retry
    }
}

/// Builds the grammar [`Constraint`] that FORCES a valid, schema-conforming tool call, for
/// `tool_choice` values that require one (`"required"`, or a named function). Tokenizer-based (no
/// `Llama` needed) so the seam backends share it. Built once per model: the [`ConstraintEnv`] is
/// made on the first request that forces a call and reused after that.
#[derive(Default)]
pub struct ToolConstrainer {
    env: std::sync::OnceLock<std::result::Result<ConstraintEnv, String>>,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl ToolConstrainer {
    /// The forced tool-call constraint for one request, or `None` for `"auto"`/`"none"`/absent
    /// `tool_choice` — and when `tools` is absent, without reading `tool_choice` at all.
    /// `tokenizer`, `vocab` and `eos_ids` must be the same on every call: they only shape the
    /// environment the first forced request builds.
    pub fn constraint(
        &self,
        tokenizer: &Tokenizer,
        vocab: usize,
        eos_ids: &[u32],
        tools: Option<&Value>,
        tool_choice: Option<&str>,
    ) -> Result<Option<Constraint>> {
        let Some(grammar) = forced_tool_grammar(tools, tool_choice)? else {
            return Ok(None);
        };
        self.env(tokenizer, vocab, eos_ids)?
            .constraint(grammar)
            .map(Some)
            .map_err(blame_tools)
    }

    /// The constraint for a `response_format` reply: a JSON value matching `schema`, or — when
    /// `tools` are offered and callable — one or more Hermes `<tool_call>` blocks instead, each
    /// body constrained like a forced call's. The JSON is the whole reply, so the prompt must be
    /// rendered with thinking off. Shares the forced calls' environment, under the same
    /// once-per-model contract for `tokenizer`, `vocab` and `eos_ids`. A schema llguidance
    /// cannot compile fails with [`GrammarRejected`], blaming the tools when one of their schemas
    /// is the one at fault.
    pub fn response_constraint(
        &self,
        tokenizer: &Tokenizer,
        vocab: usize,
        eos_ids: &[u32],
        schema: &Value,
        tools: Option<&Value>,
    ) -> Result<Constraint> {
        let env = self.env(tokenizer, vocab, eos_ids)?;
        let reply = with_json_options(&caps::cap_numbers(schema)?);
        // The shape check again, on exactly what llguidance will compile.
        shape::check_shape(&reply)?;
        let Some(tools) = tools else {
            return env.constraint(TopLevelGrammar::from_json_schema(reply));
        };
        let call = call_schema(tools)?;
        let grammar = response_format_grammar(
            &reply,
            &call,
            &env.marker("<tool_call>"),
            &env.marker("</tool_call>"),
        )?;
        env.constraint(grammar).map_err(|e| {
            // The combined grammar does not say which half failed: the call schema compiled on
            // its own does, and it is only compiled again on this failure path.
            match env.constraint(TopLevelGrammar::from_json_schema(call)) {
                Err(tools_err) => blame_tools(tools_err),
                Ok(_) => e,
            }
        })
    }

    /// The shared environment, built by the first request that needs one.
    fn env(&self, tokenizer: &Tokenizer, vocab: usize, eos_ids: &[u32]) -> Result<&ConstraintEnv> {
        self.env
            .get_or_init(|| {
                ConstraintEnv::new(tokenizer, vocab, eos_ids).map_err(|e| e.to_string())
            })
            .as_ref()
            .map_err(|e| anyhow!("{e}"))
    }
}

/// Whether a request's `tool_choice` forces a tool call over its `tools` — `"required"` or a
/// function name, with `tools` present — so its reply needs a grammar. The decision
/// [`ToolConstrainer::constraint`] makes, without building anything.
pub fn forces_tool_call(tools: Option<&Value>, tool_choice: Option<&str>) -> bool {
    tools.is_some() && !matches!(tool_choice, None | Some("auto") | Some("none"))
}

/// The grammar a `tool_choice` forces over `tools`: `"required"` (any tool) or a named function
/// (that one), `None` for `"auto"`/`"none"`/absent.
fn forced_tool_grammar(
    tools: Option<&Value>,
    tool_choice: Option<&str>,
) -> Result<Option<TopLevelGrammar>> {
    let Some(tools) = tools.filter(|_| forces_tool_call(tools, tool_choice)) else {
        return Ok(None);
    };
    let only = tool_choice
        .filter(|c| *c != "required")
        .map(|c| c.trim_matches('"'));
    let filtered;
    let tools = if let Some(name) = only {
        let arr = tools.as_array().cloned().unwrap_or_default();
        filtered = Value::Array(
            arr.into_iter()
                .filter(|t| {
                    t.get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(Value::as_str)
                        == Some(name)
                })
                .collect(),
        );
        &filtered
    } else {
        tools
    };
    forced_tool_call_grammar(tools).map(Some)
}

/// Build a JSON-schema grammar constraining the tool-call BODY — a single JSON object
/// `{"name": <one-of-the-tool-names>, "arguments": <that tool's parameter schema>}` (the union over
/// the request's tools). The caller prefills the `<tool_call>` opener and constrains only this body,
/// so the grammar stays pure JSON over normal byte tokens (no special-token / byte-grammar mismatch).
/// Used for `tool_choice: "required"` / a named tool — the model MUST emit one valid, schema-conforming
/// call. `tools` is the OpenAI `tools` array.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn forced_tool_call_grammar(tools: &Value) -> Result<TopLevelGrammar> {
    Ok(TopLevelGrammar::from_json_schema(call_schema(tools)?))
}

/// The call-body schema over `tools` as llguidance compiles it — [`tool_call_schema`] with the
/// JSON options — after the shape check has passed on exactly that JSON (a refusal blames the
/// tools).
fn call_schema(tools: &Value) -> Result<Value> {
    let call = with_json_options(&tool_call_schema(tools)?);
    shape::check_shape(&call).map_err(|r| GrammarRejected::tools(r.message))?;
    Ok(call)
}

/// The JSON schema of one tool-call body over `tools`: `{"name": .., "arguments": ..}`, one
/// `anyOf` alternative per tool.
fn tool_call_schema(tools: &Value) -> Result<Value> {
    let arr = tools
        .as_array()
        .ok_or_else(|| GrammarRejected::tools("`tools` is not an array"))?;
    // One JSON-schema alternative per tool: { name: const, arguments: <params> }.
    let mut alts: Vec<Value> = Vec::new();
    for (i, t) in arr.iter().enumerate() {
        let f = t
            .get("function")
            .ok_or_else(|| GrammarRejected::tools("a tool has no `function`"))?;
        let name = f
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| GrammarRejected::tools("a tool has no string `function.name`"))?;
        let mut params = f
            .get("parameters")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({"type": "object"}));
        // Refused here, for the forced path as for `response_format`: a schema llguidance would
        // miscompile or not survive compiling (see `caps::check_schema`). Checked as the client
        // wrote it, its references relative to the parameters themselves…
        caps::check_schema(&params).map_err(|r| GrammarRejected::tools(r.message))?;
        // …which is how a tool's `#/$defs/…` means them, so they are moved to where the
        // parameters sit in this schema.
        shape::relocate_refs(&mut params, &format!("/anyOf/{i}/properties/arguments"));
        // A `$schema` there passed the check, so it names a dialect the shape check accepts, its
        // rules applied; below the root of this schema it would be refused. It is dropped, and
        // the tool compiled as 2020-12 means what it meant: see `shape::drop_dialect`.
        shape::drop_dialect(&mut params);
        alts.push(serde_json::json!({
            "type": "object",
            "properties": {
                "name": { "const": name },
                "arguments": params,
            },
            "required": ["name", "arguments"],
            "additionalProperties": false,
        }));
    }
    Ok(serde_json::json!({ "anyOf": alts }))
}

/// The whitespace a JSON value under one of these grammars may put between two of its tokens: a
/// run of at most this many characters, as llama.cpp's JSON grammars bound it. llguidance's default
/// allows any run, and a small model under the grammar will drift into one — tabs until the budget
/// runs out.
const JSON_WHITESPACE: &str = r"[\x20\x0A\x0D\x09]{1,20}";

/// `schema` with the llguidance JSON options every grammar here compiles with: whitespace bounded
/// to [`JSON_WHITESPACE`], and `oneOf` read as `anyOf`, as llama.cpp's `json-schema-to-grammar`
/// reads it — exclusivity is a property a generating grammar cannot check, and llguidance refuses
/// a `oneOf` whose branches it cannot prove disjoint. A schema that sets its own options
/// (`x-guidance`) keeps them.
fn with_json_options(schema: &Value) -> Value {
    let mut schema = schema.clone();
    if let Some(obj) = schema.as_object_mut() {
        obj.entry("x-guidance").or_insert_with(
            || serde_json::json!({ "whitespace_pattern": JSON_WHITESPACE, "coerce_one_of": true }),
        );
    }
    schema
}

/// The grammar for a `response_format` reply that may also be tool calls: one JSON value matching
/// `reply`, or — `tools` offered, and callable (`tool_choice` `"auto"` or absent), as OpenAI
/// allows — one or more Hermes `<tool_call>` blocks, the dialect the forced tool path primes, each
/// body matching `call`. The choice is the model's; the grammar only guarantees that whichever it
/// makes is well formed. `open` and `close` are the block markers already spelled for lark (see
/// [`ConstraintEnv::marker`]).
///
/// The JSON is the WHOLE reply — no reasoning block, no code fence — so the caller must render
/// the prompt with thinking off: a thinking model's `<think>` markers are its own special tokens,
/// and a grammar that let the model reason first would have to know each template's markers.
fn response_format_grammar(
    reply: &Value,
    call: &Value,
    open: &str,
    close: &str,
) -> Result<TopLevelGrammar> {
    // `%json` takes the schema as JSON text, so serialising it is the whole embedding. The
    // whitespace between call blocks, and after the last one, is bounded like the JSON's own.
    let lark = format!(
        "start: reply | calls\n\
         reply: %json {reply}\n\
         calls: call (WS? call)* WS?\n\
         call: {open} WS? %json {call} WS? {close}\n\
         WS: /{ws}/\n",
        ws = JSON_WHITESPACE,
        reply = serde_json::to_string(reply)?,
        call = serde_json::to_string(call)?,
    );
    Ok(TopLevelGrammar::from_lark(lark))
}

/// Fixtures the crate's tests share.
#[cfg(test)]
pub(crate) mod tests_support {
    use tokenizers::Tokenizer;

    /// A byte-level tokenizer of one token per byte: enough for llguidance to build a trie and a
    /// JSON grammar over, with no model file.
    pub(crate) fn byte_tokenizer() -> Tokenizer {
        use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
        use tokenizers::models::bpe::BPE;
        use tokenizers::pre_tokenizers::byte_level::ByteLevel;
        let mut chars: Vec<char> = ByteLevel::alphabet().into_iter().collect();
        chars.sort_unstable();
        let vocab = chars
            .iter()
            .enumerate()
            .map(|(i, c)| (c.to_string(), i as u32))
            .collect();
        let mut tok = Tokenizer::new(
            BPE::builder()
                .vocab_and_merges(vocab, vec![])
                .build()
                .unwrap(),
        );
        // No prefix space: the tests' texts are tokenized as a model would emit them.
        tok.with_pre_tokenizer(Some(ByteLevel::default().add_prefix_space(false)));
        tok.with_decoder(Some(ByteLevelDecoder::default()));
        tok
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    thread_local! {
        /// [`ConstraintEnv`]s built on this thread — the per-model cost a forced request must not
        /// pay again.
        pub(super) static ENV_BUILDS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    use tests_support::byte_tokenizer;

    /// Forced tool calls build the tokenizer environment once per model, not once per request:
    /// three `tool_choice: "required"` requests on one [`ToolConstrainer`] build it once, and an
    /// `auto` request builds nothing. Each request still gets a working constraint of its own.
    #[test]
    fn forced_tool_requests_share_one_constraint_env() {
        let tok = byte_tokenizer();
        let vocab = tok.get_vocab_size(true);
        let tools = serde_json::json!([{
            "type": "function",
            "function": {"name": "get_weather", "parameters": {
                "type": "object", "properties": {"city": {"type": "string"}}
            }}
        }]);
        let constrainer = ToolConstrainer::default();
        let before = ENV_BUILDS.with(|n| n.get());
        for _ in 0..3 {
            let mut c = constrainer
                .constraint(&tok, vocab, &[0], Some(&tools), Some("required"))
                .expect("constraint")
                .expect("`required` forces a call");
            let mut logits = vec![0.0f32; vocab];
            c.apply_mask(&mut logits).expect("mask");
            assert!(
                logits.iter().any(|l| l.is_finite()),
                "the grammar allows nothing"
            );
        }
        let auto = constrainer
            .constraint(&tok, vocab, &[0], Some(&tools), Some("auto"))
            .expect("auto");
        assert!(auto.is_none());
        assert_eq!(ENV_BUILDS.with(|n| n.get()) - before, 1);
    }

    /// A request's frequency penalty, as `infr serve` builds it.
    fn frequency_penalty(f: f32) -> crate::sampling::RequestCtx {
        crate::sampling::RequestCtx::new(crate::sampling::RequestSampling {
            frequency_penalty: f,
            ..Default::default()
        })
    }

    /// How far `penalties` push each of `ids` down from a zero logit.
    fn pushed_down(penalties: &crate::sampling::Penalties, ids: &[u32], vocab: usize) -> Vec<f32> {
        let mut row = vec![0.0f32; vocab];
        penalties.apply(&mut row);
        ids.iter().map(|&i| row[i as usize]).collect()
    }

    /// **Penalties reach a constrained generation.** A grammar of four letters, each `a` or `b`,
    /// over logits that prefer `a` slightly: greedy picks `aaaa` with no penalty, and a frequency
    /// penalty must push the repeated `a` below `b`, so the run alternates. The penalty state must
    /// also have seen every constrained token. Before the fix the constrained step neither applied
    /// nor recorded penalties, so both runs came out `aaaa` and the state stayed empty.
    #[test]
    fn a_constrained_step_applies_and_records_penalties() {
        let tok = byte_tokenizer();
        let vocab = tok.get_vocab_size(true);
        let id = |s: &str| tok.token_to_id(s).expect("single-byte token");
        let (a, b, eos) = (id("a"), id("b"), id("z"));
        let greedy = crate::sampling::Sampler {
            temp: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
        };
        let run = |mut penalties: Option<&mut crate::sampling::Penalties>| -> Vec<u32> {
            let env = build_tok_env(&tok, vocab, &[eos]).expect("tok env");
            let grammar = TopLevelGrammar::from_lark(
                "start: c c c c
c: \"a\" | \"b\""
                    .to_string(),
            );
            let mut c = Constraint::new(env, grammar).expect("constraint");
            let mut out = Vec::new();
            for _ in 0..8 {
                let mut logits = vec![0.0f32; vocab];
                logits[a as usize] = 1.0;
                logits[b as usize] = 0.9;
                let (step, done) = constrained_step(
                    &mut c,
                    &mut logits,
                    &[eos],
                    greedy,
                    &mut 1,
                    usize::MAX,
                    penalties.as_deref_mut(),
                )
                .expect("constrained step");
                out.extend(step);
                if done || out.len() >= 4 {
                    break;
                }
            }
            out
        };

        assert_eq!(
            run(None),
            [a, a, a, a],
            "without a penalty greedy repeats `a`"
        );
        let ctx = frequency_penalty(1.0);
        let mut p = crate::sampling::Penalties::resolve(Some(&ctx)).expect("penalty active");
        assert_eq!(
            run(Some(&mut p)),
            [a, b, a, b],
            "a frequency penalty must steer the constrained run off the repeated token"
        );
        assert_eq!(
            pushed_down(&p, &[a, b], vocab),
            [-2.0, -2.0],
            "the penalty state did not see the constrained tokens"
        );
    }

    /// **`logit_bias` reaches a constrained generation.** The same four-letter `a`/`b` grammar
    /// over logits that prefer `a`: banning `a` must make every position `b`, greedy and
    /// sampled.
    #[test]
    fn a_constrained_step_applies_logit_bias() {
        let tok = byte_tokenizer();
        let vocab = tok.get_vocab_size(true);
        let id = |s: &str| tok.token_to_id(s).expect("single-byte token");
        let (a, b, eos) = (id("a"), id("b"), id("z"));
        let run = |bias: Vec<(u32, f32)>, sampler: crate::sampling::Sampler, seed: u64| {
            let ctx = crate::sampling::RequestCtx::new(crate::sampling::RequestSampling {
                logit_bias: bias,
                ..Default::default()
            });
            let mut p = crate::sampling::Penalties::resolve(Some(&ctx)).expect("bias active");
            let env = build_tok_env(&tok, vocab, &[eos]).expect("tok env");
            let grammar = TopLevelGrammar::from_lark(
                "start: c c c c
c: \"a\" | \"b\""
                    .to_string(),
            );
            let mut c = Constraint::new(env, grammar).expect("constraint");
            let mut rng = seed;
            let mut out = Vec::new();
            for _ in 0..8 {
                let mut logits = vec![0.0f32; vocab];
                logits[a as usize] = 1.0;
                logits[b as usize] = 0.9;
                let (step, done) = constrained_step(
                    &mut c,
                    &mut logits,
                    &[eos],
                    sampler,
                    &mut rng,
                    usize::MAX,
                    Some(&mut p),
                )
                .expect("constrained step");
                out.extend(step);
                if done || out.len() >= 4 {
                    break;
                }
            }
            out
        };
        let greedy = crate::sampling::Sampler {
            temp: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
        };
        let sampled = crate::sampling::Sampler {
            temp: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
        };
        assert_eq!(
            run(vec![(a, -100.0)], greedy, 1),
            [b, b, b, b],
            "a banned token was drawn under a grammar at temperature 0"
        );
        for seed in 1..=64 {
            assert_eq!(
                run(vec![(a, -100.0)], sampled, seed),
                [b, b, b, b],
                "seed {seed}: a banned token was drawn under a grammar"
            );
        }
    }

    /// Tokens the grammar FORCES are emitted without a draw, and they must reach the penalty state
    /// too. The env is canonical and the grammar a fixed literal, which is what makes llguidance
    /// fast-forward (see [`a_forced_run_is_cut_to_the_budget`]).
    #[test]
    fn forced_tokens_reach_the_penalty_state() {
        // No prefix space: llguidance tokenizes the literal to fast-forward it, and a leading
        // space token would not be the grammar's first byte.
        let mut tok = byte_tokenizer();
        tok.with_pre_tokenizer(Some(
            tokenizers::pre_tokenizers::byte_level::ByteLevel::default().add_prefix_space(false),
        ));
        let vocab = tok.get_vocab_size(true);
        let eos = tok.token_to_id("z").expect("single-byte token");
        let json = tok.to_string(false).expect("serialize tokenizer");
        let mut bt = ByteTokenizer::from_json_bytes(json.as_bytes()).expect("byte tokenizer");
        bt.set_eos_tokens(&[eos]);
        let env: TokEnv =
            Arc::new(ByteTokenizerEnv::new(bt, Some(vocab)).expect("canonical tok env"));
        let grammar = TopLevelGrammar::from_lark("start: \"abab\"".to_string());
        let mut c = Constraint::new(env, grammar).expect("constraint");
        let ctx = frequency_penalty(1.0);
        let mut p = crate::sampling::Penalties::resolve(Some(&ctx)).expect("penalty active");
        let greedy = crate::sampling::Sampler {
            temp: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
        };
        let mut logits = vec![0.0f32; vocab];
        let (step, _) = constrained_step(
            &mut c,
            &mut logits,
            &[eos],
            greedy,
            &mut 1,
            usize::MAX,
            Some(&mut p),
        )
        .expect("constrained step");
        assert!(step.len() >= 2, "the literal must force a run: {step:?}");
        let counts: Vec<f32> = step
            .iter()
            .map(|t| -(step.iter().filter(|&u| u == t).count() as f32))
            .collect();
        assert_eq!(
            pushed_down(&p, &step, vocab),
            counts,
            "a forced token was not recorded"
        );
    }

    fn qwen3_06b() -> Option<PathBuf> {
        crate::test_hub_gguf("unsloth/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q4_K_M.gguf")
    }

    /// **Finding 2 — `apply_mask` must fail CLOSED past `vocab`.** On a padded-vocab model the
    /// logits row is wider than the token trie; ids `>= vocab` have no trie entry and must be forced
    /// to -inf so a downstream full-slice argmax can't emit an out-of-grammar token. Tests the tail
    /// masker directly (no live matcher needed).
    #[test]
    fn padding_tail_is_masked_to_neg_inf() {
        let mut logits = vec![1.0f32; 10];
        mask_padding_tail(&mut logits, 6);
        for (i, &l) in logits.iter().enumerate() {
            if i < 6 {
                assert_eq!(l, 1.0, "id {i} within vocab must be untouched");
            } else {
                assert!(
                    l.is_infinite() && l.is_sign_negative(),
                    "id {i} beyond vocab must be -inf"
                );
            }
        }
        // vocab >= len is a no-op (no padding tail).
        let mut exact = vec![2.0f32; 4];
        mask_padding_tail(&mut exact, 4);
        assert!(exact.iter().all(|&l| l == 2.0));
    }

    /// **Finding 3 — constrained decoding honours the `Sampler`.** At `temp==0` `constrained_step`
    /// stays deterministic (greedy within the mask); at `temp>0` with a fixed seed it draws — and a
    /// masked token has probability 0, so every emitted token is still grammar-legal. Model-gated
    /// (needs the real GGUF tokenizer); self-skips when the model isn't cached.
    #[test]
    fn constrained_step_honours_sampler_and_stays_in_grammar() {
        let Some(path) = qwen3_06b() else {
            eprintln!("skip: Qwen3-0.6B not cached");
            return;
        };
        let g = infr_gguf::Gguf::open(&path).expect("open gguf");
        let cfg = crate::Config::from_gguf(&g).expect("config");
        let tok = crate::build_tokenizer(&g).expect("tokenizer");
        let tools = serde_json::json!([{
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": { "city": { "type": "string" } },
                    "required": ["city"],
                },
            },
        }]);

        // Deterministic (temp==0) run — collapses to greedy within the mask.
        let run = |sampler: crate::sampling::Sampler, seed: u64| -> Vec<u32> {
            let env = build_tok_env(&tok, cfg.vocab, &[cfg.eos]).expect("tok env");
            let grammar = forced_tool_call_grammar(&tools).expect("grammar");
            let mut c = Constraint::new(env, grammar).expect("constraint");
            let mut rng = seed;
            let mut out = Vec::new();
            for _ in 0..40 {
                if c.stopped() {
                    break;
                }
                let mut logits: Vec<f32> =
                    (0..cfg.vocab).map(|i| (i as f32 * 0.01).sin()).collect();
                let (step, done) = constrained_step(
                    &mut c,
                    &mut logits,
                    &[cfg.eos],
                    sampler,
                    &mut rng,
                    usize::MAX,
                    None,
                )
                .expect("constrained step");
                if step.is_empty() {
                    break;
                }
                out.extend(step);
                if done {
                    break;
                }
            }
            out
        };

        let greedy = crate::sampling::Sampler {
            temp: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
        };
        // temp==0 is deterministic regardless of seed.
        assert_eq!(
            run(greedy, 1),
            run(greedy, 999),
            "temp==0 must stay greedy/deterministic"
        );

        // temp>0, seeded: reproducible, AND every emitted token is grammar-legal (a masked token
        // could never be drawn). We re-validate the run against a fresh grammar to prove the tokens
        // stayed inside the constraint.
        let sampled = crate::sampling::Sampler {
            temp: 1.2,
            top_k: 0,
            top_p: 0.95,
            min_p: 0.0,
        };
        let a = run(sampled, 12345);
        let b = run(sampled, 12345);
        assert_eq!(a, b, "a seeded temp>0 constrained run must be reproducible");
        let env = build_tok_env(&tok, cfg.vocab, &[cfg.eos]).expect("tok env");
        let grammar = forced_tool_call_grammar(&tools).expect("grammar");
        let mut check = Constraint::new(env, grammar).expect("constraint");
        for &t in &a {
            if t == cfg.eos {
                continue;
            }
            assert!(
                check.try_accept(t).expect("try_accept"),
                "sampled token {t} escaped the grammar mask"
            );
        }
    }

    /// A forced run longer than the request's remaining budget is cut to that budget, and the
    /// grammar is left where the consumed tokens put it, so `max_tokens` holds even when one step
    /// would otherwise fast-forward several tokens. Model-gated like the test above.
    ///
    /// Two things make the forced branch reachable here, and neither holds in production today.
    /// The env is CANONICAL, because llguidance only fast-forwards when `tokenize_is_canonical()`
    /// is true, so under [`build_tok_env`]'s `NonCanonicalEnv` `forced()` is always empty. And the
    /// grammar is a fixed literal, because the tool-call schema allows flexible JSON whitespace
    /// and so forces almost nothing.
    #[test]
    fn a_forced_run_is_cut_to_the_budget() {
        let Some(path) = qwen3_06b() else {
            eprintln!("skip: Qwen3-0.6B not cached");
            return;
        };
        let g = infr_gguf::Gguf::open(&path).expect("open gguf");
        let cfg = crate::Config::from_gguf(&g).expect("config");
        let tok = crate::build_tokenizer(&g).expect("tokenizer");
        let json = tok.to_string(false).expect("serialize tokenizer");
        let mut bt = ByteTokenizer::from_json_bytes(json.as_bytes()).expect("byte tokenizer");
        bt.set_eos_tokens(&[cfg.eos]);
        let env: TokEnv =
            Arc::new(ByteTokenizerEnv::new(bt, Some(cfg.vocab)).expect("canonical tok env"));
        let new_constraint = || {
            let grammar = TopLevelGrammar::from_lark(
                r#"start: "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Oslo\"}}""#
                    .to_string(),
            );
            Constraint::new(env.clone(), grammar).expect("constraint")
        };
        let greedy = crate::sampling::Sampler {
            temp: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
        };
        let step = |c: &mut Constraint, budget: usize| {
            let mut logits = vec![0.0f32; cfg.vocab];
            constrained_step(c, &mut logits, &[cfg.eos], greedy, &mut 1, budget, None)
                .expect("constrained step")
        };

        let mut c = new_constraint();
        let forced = c.forced();
        assert!(
            forced.len() >= 2,
            "the literal must force a multi-token run: {forced:?}"
        );

        assert!(
            step(&mut c, 0).0.is_empty(),
            "a spent budget must emit nothing"
        );
        assert_eq!(
            c.forced(),
            forced,
            "a spent budget must not move the grammar"
        );

        let (cut, done) = step(&mut c, 1);
        assert_eq!(
            cut,
            forced[..1],
            "budget 1 must emit exactly the first forced token"
        );
        assert!(!done);

        // The grammar stands exactly where the one emitted token put it: a fresh constraint fed
        // that token forces the same continuation, and the rest of the run is still there.
        let mut check = new_constraint();
        check.consume(&cut).expect("the cut token left the grammar");
        let rest = c.forced();
        assert_eq!(check.forced(), rest, "the cut left the grammar out of step");
        assert!(!rest.is_empty(), "the rest of the forced run was dropped");

        // An unlimited budget still drains the whole run.
        let (tail, done) = step(&mut c, usize::MAX);
        assert_eq!(tail, rest);
        assert!(done, "the literal must be complete after its forced run");
    }

    /// End-to-end: build the TokEnv from a real GGUF tokenizer, build the forced tool-call grammar,
    /// and verify the constraint actually restricts the first token to a strict subset of the vocab
    /// (the grammar must START a `<tool_call>` — not "anything goes"). Self-skips without the model.
    #[test]
    fn forced_grammar_constrains_first_token() {
        let Some(path) = qwen3_06b() else {
            eprintln!("skip: Qwen3-0.6B not cached");
            return;
        };
        let g = infr_gguf::Gguf::open(&path).expect("open gguf");
        let cfg = crate::Config::from_gguf(&g).expect("config");
        let tok = crate::build_tokenizer(&g).expect("tokenizer");
        let env = build_tok_env(&tok, cfg.vocab, &[cfg.eos]).expect("tok env");

        let tools = serde_json::json!([{
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": { "city": { "type": "string" } },
                    "required": ["city"],
                },
            },
        }]);
        let grammar = forced_tool_call_grammar(&tools).expect("grammar");
        let mut c = Constraint::new(env, grammar).expect("constraint");

        // Drive the constraint, greedily picking the first allowed token each step. (This naive picker
        // wanders inside JSON strings and may never choose a closing quote, so we don't require full
        // termination — only that every step stays constrained and `accept` never rejects, i.e. the
        // mask and the parser agree. Full termination is covered by the live-model server test.)
        // Mirror the real decode loop: drain forced tokens first, else mask + argmax-pick a free token.
        let mut out: Vec<u32> = Vec::new();
        let mut drops = 0usize;
        for step in 0..60 {
            if c.stopped() {
                break;
            }
            let forced = c.forced();
            if !forced.is_empty() {
                c.consume(&forced).expect("consume forced");
                out.extend(forced);
                continue;
            }
            // ADVERSARIAL logits: bias toward high token ids (and away from the genuinely-valid low-id
            // delimiters). The healed mask can be a SUPERSET, so a naive argmax here would pick a
            // superset member that `consume_token` rejects — exactly the live-server failure. The
            // retry-on-reject loop (mirroring `decode_loop`) must recover and still produce valid JSON.
            let mut logits: Vec<f32> = (0..cfg.vocab).map(|i| i as f32).collect();
            c.apply_mask(&mut logits).expect("mask");
            let allowed = logits.iter().filter(|l| l.is_finite()).count();
            assert!(allowed > 0, "step {step}: grammar masked out EVERY token");
            assert!(
                allowed < cfg.vocab,
                "step {step}: grammar allowed everything"
            );
            let picked = loop {
                let cand = crate::sampling::argmax(&logits) as u32;
                assert!(
                    logits[cand as usize].is_finite(),
                    "step {step}: mask exhausted — every allowed token was rejected"
                );
                if c.try_accept(cand).expect("try_accept") {
                    break cand;
                }
                drops += 1;
                logits[cand as usize] = f32::NEG_INFINITY; // superset member — drop + retry
            };
            out.push(picked);
        }
        let text = tok.decode(&out, false).expect("decode");
        // `drops` counts superset-member rejections recovered via retry. With the canonical GGUF-built
        // tokenizer the mask is usually exact (drops==0); the SUPERSET only appears with the live model
        // tokenizer (the server test exercises that), so we don't assert drops>0 here — we assert the
        // retry path is sound: never deadlocks (handled in-loop) and keeps the output valid JSON.
        eprintln!("constrained prefix ({drops} superset drops): {text:?}");
        assert!(
            text.trim_start().starts_with('{'),
            "constrained output must stay valid JSON: {text:?}"
        );
    }

    // --- response_format (JSON mode) -----------------------------------------------------------

    fn weather_tools() -> Value {
        serde_json::json!([{
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": { "city": { "type": "string" } },
                    "required": ["city"],
                },
            },
        }])
    }

    fn person_schema() -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "age": { "type": "integer" },
            },
            "required": ["name", "age"],
            "additionalProperties": false,
        })
    }

    /// Whether `v` matches [`person_schema`] — checked by hand, the workspace has no JSON-schema
    /// validator: an object with exactly a string `name` and an integer `age`.
    fn is_person(v: &Value) -> bool {
        v.as_object().is_some_and(|o| {
            o.len() == 2
                && o.get("name").is_some_and(Value::is_string)
                && o.get("age").is_some_and(|a| a.is_i64() || a.is_u64())
        })
    }

    /// What a `response_format` grammar makes of a whole text on `tok`: `Some(true)` when it
    /// admits the text and may end there, `Some(false)` when it admits the text as a prefix
    /// only, `None` when some token of it is refused.
    fn judge(
        tok: &Tokenizer,
        vocab: usize,
        eos: u32,
        schema: &Value,
        tools: Option<&Value>,
        text: &str,
    ) -> Option<bool> {
        let mut c = ToolConstrainer::default()
            .response_constraint(tok, vocab, &[eos], schema, tools)
            .expect("constraint");
        let ids = tok.encode(text, false).expect("encode").get_ids().to_vec();
        for t in ids {
            if !c.try_accept(t).expect("try_accept") {
                return None;
            }
        }
        Some(c.accepting().expect("accepting"))
    }

    /// The token sequences a `response_format` grammar admits, on a synthetic byte vocabulary (no
    /// model): with tools offered, well-formed calls — any number, whitespace between and after
    /// them — or a schema answer, and nothing else; numbers bounded as llama.cpp bounds them;
    /// `oneOf` read as `anyOf`.
    #[test]
    fn the_response_grammar_admits_answers_and_calls_and_nothing_else() {
        let tok = byte_tokenizer();
        let vocab = tok.get_vocab_size(true);
        let eos = 0;
        let tools = weather_tools();
        let person = person_schema();
        let j = |schema: &Value, tools: Option<&Value>, text: &str| {
            judge(&tok, vocab, eos, schema, tools, text)
        };
        let call =
            "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Oslo\"}}\n\
                    </tool_call>";
        let answer = r#"{"name": "Ada", "age": 36}"#;

        // Both branches; trailing whitespace after the last call may end the reply.
        assert_eq!(j(&person, Some(&tools), call), Some(true), "a call");
        assert_eq!(j(&person, Some(&tools), &format!("{call}\n")), Some(true));
        assert_eq!(
            j(&person, Some(&tools), &format!("{call}\n{call}")),
            Some(true)
        );
        assert_eq!(j(&person, Some(&tools), answer), Some(true), "an answer");
        assert_eq!(j(&person, None, answer), Some(true));
        assert_eq!(j(&person, None, call), None, "no tool is callable");
        assert_eq!(
            j(&person, None, r#"{"name": "Ada""#),
            Some(false),
            "a prefix"
        );
        assert_eq!(
            j(&person, None, r#"{"name": "Ada"}"#),
            None,
            "a required key missing"
        );
        assert_eq!(
            j(&person, None, r#"{"name": "Ada", "age": 36, "x": 1}"#),
            None
        );
        assert_eq!(
            j(&person, Some(&tools), "Sure! Here it is."),
            None,
            "free text"
        );
        let other = "<tool_call>\n{\"name\": \"get_time\", \"arguments\": {}}\n</tool_call>";
        assert_eq!(j(&person, Some(&tools), other), None, "a tool not offered");

        // Whitespace between tokens is bounded.
        let padded = |n: usize| format!(r#"{{"name": "Ada", "age": 36{}}}"#, " ".repeat(n));
        assert_eq!(j(&person, None, &padded(8)), Some(true));
        assert_eq!(
            j(&person, None, &padded(40)),
            None,
            "an unbounded whitespace run"
        );

        // Integers: llama.cpp's 16 digits; numbers: that integral part, eight fraction digits.
        let int = |n: usize| format!(r#"{{"name": "Ada", "age": {}}}"#, "9".repeat(n));
        assert_eq!(j(&person, None, &int(16)), Some(true));
        assert_eq!(j(&person, None, &int(18)), None, "an 18-digit integer");
        let num = serde_json::json!({"type": "number"});
        assert_eq!(j(&num, None, "-2.25"), Some(true));
        assert_eq!(j(&num, None, &format!("0.{}", "3".repeat(8))), Some(true));
        assert_eq!(
            j(&num, None, &format!("0.{}", "3".repeat(9))),
            None,
            "9 decimals"
        );
        assert_eq!(j(&num, None, &"9".repeat(18)), None);
        // The numbers in `json_object` and in every value a schema leaves open are bounded too.
        let object = serde_json::json!({"type": "object"});
        assert_eq!(
            j(&object, None, r#"{"a": [1, {"b": 2.5}], "c": null}"#),
            Some(true)
        );
        let deep = format!(r#"{{"a": [{{"b": {}}}]}}"#, "9".repeat(18));
        assert_eq!(
            j(&object, None, &deep),
            None,
            "an open value's 18-digit integer"
        );
        assert_eq!(
            j(&serde_json::json!({}), None, "[1, true, \"x\"]"),
            Some(true)
        );
        // A schema's own bounds are kept.
        let small = serde_json::json!({"type": "integer", "maximum": 5});
        assert_eq!(j(&small, None, "5"), Some(true));
        assert_eq!(j(&small, None, "6"), None);

        // `oneOf` compiles as `anyOf`, in a reply and in a tool's parameters.
        // Two object branches: llguidance cannot prove these exclusive, and refuses a `oneOf`
        // it cannot unless told to read it as `anyOf`.
        let one_of = serde_json::json!({"oneOf": [
            {"type": "object", "properties": {"a": {"type": "string"}}},
            {"type": "object", "properties": {"b": {"type": "integer"}}},
        ]});
        assert_eq!(j(&one_of, None, r#"{"b": 3}"#), Some(true));
        let tools_one_of = serde_json::json!([{"type": "function", "function": {
            "name": "f", "parameters": {"type": "object", "properties": {"v": one_of}}
        }}]);
        let f_call =
            "<tool_call>{\"name\": \"f\", \"arguments\": {\"v\": {\"a\": \"x\"}}}</tool_call>";
        assert_eq!(j(&person, Some(&tools_one_of), f_call), Some(true));
        assert!(
            forced_tool_call_grammar(&tools_one_of).is_ok()
                && ToolConstrainer::default()
                    .constraint(&tok, vocab, &[eos], Some(&tools_one_of), Some("required"))
                    .is_ok(),
            "a forced call over a `oneOf` parameter"
        );
    }

    /// The number bounds never contradict the schema's own: a range beyond the default one keeps
    /// its values, `const`/`enum` are left alone, a range too narrow for the fraction step gets
    /// none. Values the schema leaves open through annotation-only nodes or an untyped node's
    /// keywords are bounded like any other. A `multipleOf` llguidance would miscompute is refused
    /// up front, on the `response_format` path and the forced-tool path, and the fraction step is
    /// never combined with a `multipleOf` it would overflow against.
    #[test]
    fn number_bounds_respect_the_schema_and_refuse_what_cannot_be_enforced() {
        let tok = byte_tokenizer();
        let vocab = tok.get_vocab_size(true);
        let j = |schema: Value, text: &str| judge(&tok, vocab, 0, &schema, None, text);

        assert_eq!(
            j(
                serde_json::json!({"type": "integer", "minimum": 1e17}),
                "100000000000000000"
            ),
            Some(true)
        );
        assert_eq!(
            j(
                serde_json::json!({"type": "integer", "exclusiveMinimum": 1e17}),
                "100000000000000001"
            ),
            Some(true)
        );
        assert_eq!(
            j(
                serde_json::json!({"type": "number", "maximum": -1e17}),
                "-100000000000000000"
            ),
            Some(true)
        );
        let big = serde_json::json!(1e20);
        assert_eq!(
            j(
                serde_json::json!({"type": "number", "const": big}),
                "100000000000000000000"
            ),
            Some(true)
        );
        let tiny = serde_json::json!(1e-9);
        assert_eq!(
            j(
                serde_json::json!({"type": "number", "enum": [tiny]}),
                "0.000000001"
            ),
            Some(true)
        );
        let narrow =
            serde_json::json!({"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1e-9});
        assert_eq!(j(narrow, "0.0000000005"), Some(true));

        // Open values: an annotation-only node, and an untyped node with `items`.
        let annotated = serde_json::json!({"type": "object",
            "properties": {"a": {"description": "d", "readOnly": true}}});
        assert_eq!(j(annotated.clone(), r#"{"a": 5}"#), Some(true));
        assert_eq!(
            j(annotated, &format!(r#"{{"a": {}}}"#, "9".repeat(20))),
            None
        );
        let untyped = serde_json::json!({"items": {"type": "integer"}});
        assert_eq!(j(untyped.clone(), "[1, 2]"), Some(true));
        assert_eq!(j(untyped.clone(), "\"x\""), Some(true));
        assert_eq!(j(untyped.clone(), &"9".repeat(20)), None);
        assert_eq!(j(untyped, &format!(r#"{{"a": {}}}"#, "9".repeat(20))), None);

        // Subschemas llguidance intersects get no recursive reference — two of them meeting
        // would expand without end — yet every such schema compiles.
        for shared in [
            serde_json::json!({"allOf": [{"properties": {"a": {"type": "object"}}},
                {"properties": {"a": {"type": "object"}}}]}),
            serde_json::json!({"properties": {"a": {"type": "object"}},
                "anyOf": [{"properties": {"a": {"type": "object"}}}]}),
            serde_json::json!({"$defs": {"A": {"type": "object"}, "B": {"type": "object"}},
                "allOf": [{"$ref": "#/$defs/A"}, {"$ref": "#/$defs/B"}]}),
        ] {
            assert_eq!(j(shared.clone(), r#"{"a": {}}"#), Some(true), "{shared}");
        }
        // A reference nothing intersects keeps its definition's open values bounded.
        let referenced = serde_json::json!({"$defs": {"A": {"type": "object"}},
            "type": "object", "properties": {"x": {"$ref": "#/$defs/A"}}});
        assert_eq!(j(referenced.clone(), r#"{"x": {"n": 5}}"#), Some(true));
        assert_eq!(
            j(
                referenced,
                &format!(r#"{{"x": {{"n": {}}}}}"#, "9".repeat(20))
            ),
            None
        );

        // A `multipleOf` the step would overflow against gets no step, and compiles.
        let multiples = serde_json::json!({"allOf": [{"multipleOf": 43}, {"type": "number"}]});
        assert_eq!(j(multiples.clone(), "86"), Some(true));
        assert_eq!(j(multiples, "87"), Some(false), "87 is no multiple of 43");

        // A keyword llguidance knows but refuses is not an open node: the schema is refused
        // rather than the constraint dropped.
        let not_null = serde_json::json!({"type": "object",
            "properties": {"x": {"not": {"type": "null"}}}});
        let e = ToolConstrainer::default()
            .response_constraint(&tok, vocab, &[0], &not_null, None)
            .err()
            .expect("`not` must be refused");
        assert!(e.to_string().contains("not"), "{e}");
        // A named property meets a matching pattern property: both open, and it compiles.
        let both = serde_json::json!({"type": "object",
            "properties": {"a": {}}, "patternProperties": {"^a$": {}}});
        assert_eq!(j(both, r#"{"a": 1}"#), Some(true));
        // A step past derivre's boundary is refused before compiling (the boundary itself is
        // `caps`' unit test: compiling a divisor that large takes llguidance minutes in a debug
        // build before its own expression budget gives up).
        let step = |m: u64| serde_json::json!({"type": "integer", "multipleOf": m});
        let e = ToolConstrainer::default()
            .response_constraint(&tok, vocab, &[0], &step(429_496_729), None)
            .err()
            .expect("refused");
        assert!(e.to_string().contains("multipleOf"), "{e}");
        // A step whose coefficient llguidance cannot match cheaply is refused up front, on both
        // paths and where two steps meet; the largest allowed one compiles and works.
        assert_eq!(j(step(1000), "3000"), Some(true));
        assert_eq!(
            j(step(1000), "3500"),
            Some(false),
            "3500 is no multiple of 1000"
        );
        assert_eq!(
            j(
                serde_json::json!({"type": "number", "multipleOf": 0.001}),
                "1.234"
            ),
            Some(true)
        );
        let refused = |schema: Value, tools: bool| {
            let tc = ToolConstrainer::default();
            let r = if tools {
                let t = serde_json::json!([{"type": "function", "function": {"name": "f",
                    "parameters": {"type": "object", "properties": {"v": schema}}}}]);
                tc.constraint(&tok, vocab, &[0], Some(&t), Some("f"))
                    .map(drop)
            } else {
                tc.response_constraint(&tok, vocab, &[0], &schema, None)
                    .map(drop)
            };
            let e = r.expect_err("refused");
            let g = e.downcast_ref::<GrammarRejected>().expect("typed").clone();
            assert_eq!(g.in_tools, tools, "{g:?}");
            g.message
        };
        for tools in [false, true] {
            let m = refused(step(1001), tools);
            assert!(m.contains("too fine or too large"), "{m}");
        }
        let meet = serde_json::json!({"allOf": [{"multipleOf": 997}, {"multipleOf": 991}]});
        assert!(refused(meet, false).contains("combine"));

        // Steps are checked in pairs wherever they sit: meeting in two `allOf` branches on one
        // property, in a property and a matching pattern property, through a `$ref` beside a
        // literal step.
        for meeting in [
            serde_json::json!({"allOf": [{"properties": {"a": {"multipleOf": 997}}},
                {"properties": {"a": {"multipleOf": 991}}}]}),
            serde_json::json!({"properties": {"a": {"multipleOf": 997}},
                "patternProperties": {"^a$": {"multipleOf": 991}}}),
            serde_json::json!({"$defs": {"s": {"multipleOf": 997}},
                "allOf": [{"$ref": "#/$defs/s"}, {"multipleOf": 991}]}),
        ] {
            assert!(
                refused(meeting.clone(), false).contains("could combine"),
                "{meeting}"
            );
        }
        // A node that may hold an integer keeps added bounds in `i64`, typed with others or
        // untyped beside an `integer` branch; its own bounds past that are refused.
        assert_eq!(
            j(
                serde_json::json!({"type": ["integer", "null"], "maximum": -9.215e18}),
                "null"
            ),
            Some(true)
        );
        // A bound on one side of an intersection is not contradicted by one added to the other.
        assert_eq!(
            j(
                serde_json::json!({"allOf": [{"type": "integer"}, {"minimum": 1e17}]}),
                "100000000000000000"
            ),
            Some(true)
        );
        // A schema naming a resource another way is refused, even with plain references.
        assert!(
            refused(serde_json::json!({"$id": "urn:x", "type": "string"}), false)
                .contains("`$id` is not supported")
        );
        // Under an intersection nothing is added; llguidance's own integer range refuses a bound
        // this close to `i64`'s edge cleanly — no panic, nothing past the limit inserted.
        let edge = serde_json::json!({"allOf": [{"maximum": -9.215e18}, {"type": "integer"}]});
        let e = ToolConstrainer::default()
            .response_constraint(&tok, vocab, &[0], &edge, None)
            .map(drop)
            .expect_err("llguidance's integer range refuses it");
        assert!(!e.to_string().contains("panic"), "{e}");
        assert!(refused(
            serde_json::json!({"allOf": [{"maximum": -1e19}, {"type": "integer"}]}),
            false
        )
        .contains("beyond"));

        // Integer bounds stay inside llguidance's `i64` arithmetic: an added bound is clamped,
        // a schema's own bound past that range is refused.
        assert_eq!(
            j(
                serde_json::json!({"type": "integer", "maximum": -9.22e18}),
                "-9220000000000000000"
            ),
            Some(true)
        );
        for bound in ["minimum", "maximum"] {
            let e = ToolConstrainer::default()
                .response_constraint(
                    &tok,
                    vocab,
                    &[0],
                    &serde_json::json!({"type": "integer", bound: -1e19}),
                    None,
                )
                .err()
                .expect("refused");
            let r = e.downcast_ref::<GrammarRejected>().expect("typed");
            assert!(r.message.contains("beyond"), "{r:?}");
        }

        // A `multipleOf` finer than llguidance can match is refused, blaming the right field.
        let tc = ToolConstrainer::default();
        let fine = serde_json::json!({"type": "number", "multipleOf": 3e-9});
        let e = tc
            .response_constraint(&tok, vocab, &[0], &fine, None)
            .err()
            .expect("refused");
        let r = e.downcast_ref::<GrammarRejected>().expect("typed");
        assert!(!r.in_tools && r.message.contains("multipleOf"), "{r:?}");
        let tools = serde_json::json!([{"type": "function", "function": {"name": "f",
            "parameters": {"type": "object", "properties": {"v": fine}}}}]);
        let e = tc
            .constraint(&tok, vocab, &[0], Some(&tools), Some("required"))
            .err()
            .expect("refused on the forced path too");
        let r = e.downcast_ref::<GrammarRejected>().expect("typed");
        assert!(r.in_tools && r.message.contains("multipleOf"), "{r:?}");
    }

    /// The commonest real tool schema — Pydantic's, with `#/$defs/…` references relative to the
    /// parameters — compiles once the parameters sit inside the call schema, on the forced path
    /// and beside a `response_format`, and constrains the referenced definition.
    /// Each accepted dialect compiles a schema with `definitions` and a `$ref` and enforces it;
    /// draft-04, whose plain `id` moves the base references resolve against, is refused.
    #[test]
    fn accepted_dialects_resolve_definitions_and_draft_04_is_refused() {
        let tok = byte_tokenizer();
        let vocab = tok.get_vocab_size(true);
        let eos = vocab as u32 - 1;
        let schema = |dialect: &str| {
            serde_json::json!({
                "$schema": dialect,
                "type": "object",
                "properties": {"pet": {"$ref": "#/definitions/Pet"}},
                "required": ["pet"], "additionalProperties": false,
                "definitions": {"Pet": {"type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"], "additionalProperties": false}}
            })
        };
        for dialect in [
            "https://json-schema.org/draft/2020-12/schema",
            "https://json-schema.org/draft/2019-09/schema#",
            "http://json-schema.org/draft-07/schema#",
            "http://json-schema.org/draft-06/schema",
        ] {
            let s = schema(dialect);
            let good = r#"{"pet": {"name": "Rex"}}"#;
            let bad = r#"{"pet": {"age": 3}}"#;
            assert_eq!(
                judge(&tok, vocab, eos, &s, None, good),
                Some(true),
                "{dialect}"
            );
            assert_eq!(judge(&tok, vocab, eos, &s, None, bad), None, "{dialect}");
        }
        let err = ToolConstrainer::default()
            .response_constraint(
                &tok,
                vocab,
                &[eos],
                &schema("http://json-schema.org/draft-04/schema#"),
                None,
            )
            .map(drop)
            .expect_err("draft-04 is refused");
        assert!(err.to_string().contains("`$schema`"), "{err}");
    }

    /// Keys and data that look like keywords compile as what they are: a property named `$ref`
    /// is a property, a `const` holding `{"$ref": "#"}` is that constant — also once relocated
    /// into the call schema, where the constant must not be rewritten.
    #[test]
    fn keyword_lookalikes_in_names_and_data_compile_as_written() {
        let tok = byte_tokenizer();
        let vocab = tok.get_vocab_size(true);
        let params = serde_json::json!({"type": "object",
            "properties": {
                "$ref": {"type": "string"},
                "$id": {"type": "string"},
                "k": {"const": {"$ref": "#"}}
            },
            "required": ["$ref", "$id", "k"], "additionalProperties": false});
        let tools = serde_json::json!([{"type": "function",
            "function": {"name": "f", "parameters": params}}]);
        let good = r##"{"name": "f", "arguments": {"$ref": "a", "$id": "b", "k": {"$ref": "#"}}}"##;
        let moved = r##"{"name": "f", "arguments": {"$ref": "a", "$id": "b",
            "k": {"$ref": "#/anyOf/0/properties/arguments"}}}"##;
        let tc = ToolConstrainer::default();
        let feed = |text: &str| {
            let mut c = tc
                .constraint(&tok, vocab, &[0], Some(&tools), Some("f"))
                .expect("compiles")
                .expect("forced");
            tok.encode(text, false)
                .expect("encode")
                .get_ids()
                .iter()
                .all(|&t| c.try_accept(t).expect("try_accept"))
                && c.accepting().expect("accepting")
        };
        assert!(feed(good));
        assert!(!feed(moved));
        let reply = tc.response_constraint(&tok, vocab, &[0], &params, None);
        assert!(reply.is_ok(), "{:?}", reply.err().map(|e| e.to_string()));
    }

    #[test]
    fn a_tool_with_local_defs_compiles_in_the_call_schema() {
        let tok = byte_tokenizer();
        let vocab = tok.get_vocab_size(true);
        let tools = serde_json::json!([
            {"type": "function", "function": {"name": "other", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "adopt", "parameters": {
                "type": "object",
                "properties": {"pet": {"$ref": "#/$defs/Pet"}},
                "required": ["pet"],
                "$defs": {"Pet": {"type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"], "additionalProperties": false}}}}}
        ]);
        let good = r#"{"name": "adopt", "arguments": {"pet": {"name": "Rex"}}}"#;
        let bad = r#"{"name": "adopt", "arguments": {"pet": {"age": 3}}}"#;
        let tc = ToolConstrainer::default();
        for choice in ["required", "adopt"] {
            let feed = |text: &str| {
                let mut c = tc
                    .constraint(&tok, vocab, &[0], Some(&tools), Some(choice))
                    .expect("compiles")
                    .expect("forced");
                tok.encode(text, false)
                    .expect("encode")
                    .get_ids()
                    .iter()
                    .all(|&t| c.try_accept(t).expect("try_accept"))
                    && c.accepting().expect("accepting")
            };
            assert!(feed(good), "{choice}");
            assert!(!feed(bad), "{choice}");
        }
        let call = format!("<tool_call>{good}</tool_call>");
        assert_eq!(
            judge(&tok, vocab, 0, &person_schema(), Some(&tools), &call),
            Some(true)
        );
    }

    /// Tools declaring their own dialect — zod-to-json-schema's default draft-07, a 2019-09 tuple,
    /// one with none — compile together in the call schema, on the forced and the mixed paths,
    /// and each means there what it means on its own.
    #[test]
    fn tools_in_their_own_dialects_compile_together_and_mean_the_same() {
        let tok = byte_tokenizer();
        let vocab = tok.get_vocab_size(true);
        let adopt = serde_json::json!({
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {"pet": {"$ref": "#/definitions/Pet"}},
            "required": ["pet"], "additionalProperties": false,
            "definitions": {"Pet": {"type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"], "additionalProperties": false}}
        });
        let pair = serde_json::json!({
            "$schema": "https://json-schema.org/draft/2019-09/schema",
            "type": "object",
            "properties": {"xy": {"type": "array",
                "items": [{"type": "integer"}, {"type": "string"}], "additionalItems": false}},
            "required": ["xy"], "additionalProperties": false
        });
        let tools = serde_json::json!([
            {"type": "function", "function": {"name": "adopt", "parameters": adopt}},
            {"type": "function", "function": {"name": "pair", "parameters": pair}},
            {"type": "function", "function": {"name": "plain", "parameters": {"type": "object"}}}
        ]);
        let tc = ToolConstrainer::default();
        let forced = |choice: &str, text: &str| {
            let mut c = tc
                .constraint(&tok, vocab, &[0], Some(&tools), Some(choice))
                .expect("compiles")
                .expect("forced");
            tok.encode(text, false)
                .expect("encode")
                .get_ids()
                .iter()
                .all(|&t| c.try_accept(t).expect("try_accept"))
                && c.accepting().expect("accepting")
        };
        let call = |name: &str, args: &str| format!(r#"{{"name": "{name}", "arguments": {args}}}"#);
        for choice in ["required", "adopt"] {
            assert!(forced(
                choice,
                &call("adopt", r#"{"pet": {"name": "Rex"}}"#)
            ));
            assert!(!forced(choice, &call("adopt", r#"{"pet": {"age": 3}}"#)));
        }
        // The tuple means the same embedded as it does compiled on its own, in its dialect.
        for xy in [r#"[1, "a"]"#, r#"[1, "a", 2]"#, r#"["a", 1]"#, "[1]"] {
            let args = format!(r#"{{"xy": {xy}}}"#);
            let alone = judge(&tok, vocab, 0, &pair, None, &args) == Some(true);
            assert_eq!(forced("pair", &call("pair", &args)), alone, "{xy}");
            assert_eq!(forced("required", &call("pair", &args)), alone, "{xy}");
        }
        assert!(forced("pair", &call("pair", r#"{"xy": [1, "a"]}"#)));
        let mixed = |text: &str| {
            let text = format!("<tool_call>{text}</tool_call>");
            judge(&tok, vocab, 0, &person_schema(), Some(&tools), &text)
        };
        assert_eq!(
            mixed(&call("adopt", r#"{"pet": {"name": "Rex"}}"#)),
            Some(true)
        );
        assert_eq!(mixed(&call("adopt", r#"{"pet": {"age": 3}}"#)), None);
        assert_eq!(mixed(&call("pair", r#"{"xy": [1, "a"]}"#)), Some(true));
        assert_eq!(mixed(&call("pair", r#"{"xy": [1, "a", 2]}"#)), None);
        // A keyword the declared dialect does not have is refused by name, before the dialect is
        // dropped: embedded in 2020-12 it would turn from annotation into constraint.
        let newer = serde_json::json!([{"type": "function", "function": {"name": "t",
            "parameters": {"$schema": "http://json-schema.org/draft-07/schema#",
                "type": "array", "prefixItems": [{"type": "integer"}]}}}]);
        let err = tc
            .constraint(&tok, vocab, &[0], Some(&newer), Some("t"))
            .map(drop)
            .expect_err("refused");
        let err = err.downcast_ref::<GrammarRejected>().expect("typed");
        assert!(
            err.in_tools && err.message.contains("`prefixItems`"),
            "{}",
            err.message
        );
    }

    /// A schema llguidance cannot compile is the request's fault, typed so serve answers `400`,
    /// and blames the field at fault: `if` is a keyword it does not implement. A malformed tool is
    /// the tools' fault, under `response_format` as under a forced call. No model needed.
    #[test]
    fn a_rejected_schema_is_typed_and_blames_its_field() {
        let tok = byte_tokenizer();
        let vocab = tok.get_vocab_size(true);
        let tc = ToolConstrainer::default();
        let rejected = |r: Result<Constraint>| -> GrammarRejected {
            let e = r.err().expect("must be refused");
            e.downcast_ref::<GrammarRejected>()
                .unwrap_or_else(|| panic!("untyped error: {e:#}"))
                .clone()
        };
        let if_schema = serde_json::json!({"if": {}});
        let r = rejected(tc.response_constraint(&tok, vocab, &[0], &if_schema, None));
        assert!(r.message.contains("if") && !r.in_tools, "{r:?}");
        let tools = weather_tools();
        let r = rejected(tc.response_constraint(&tok, vocab, &[0], &if_schema, Some(&tools)));
        assert!(!r.in_tools, "{r:?}");

        let bad_params = serde_json::json!([{"type": "function", "function": {
            "name": "f", "parameters": {"if": {}}
        }}]);
        let r = rejected(tc.response_constraint(
            &tok,
            vocab,
            &[0],
            &person_schema(),
            Some(&bad_params),
        ));
        assert!(r.in_tools, "{r:?}");
        let r = rejected(
            tc.constraint(&tok, vocab, &[0], Some(&bad_params), Some("required"))
                .map(|c| c.expect("forced")),
        );
        assert!(r.in_tools, "{r:?}");

        let nameless = serde_json::json!([{"type": "function", "function": {}}]);
        let r =
            rejected(tc.response_constraint(&tok, vocab, &[0], &person_schema(), Some(&nameless)));
        assert!(r.in_tools && r.message.contains("function.name"), "{r:?}");
    }

    /// On Qwen3's tokenizer `<tool_call>` is one special token, which no byte literal matches: the
    /// call branch must spell it as that token, and EOS must be legal after the last call and the
    /// whitespace that may follow it. Also the one-step check behind the digit cap: right after an
    /// integer's first digit, `}` (and `,` where the schema allows another key) are in the mask
    /// AND accepted — so a run of digits is the model's choice, which the cap bounds.
    #[test]
    fn qwen_markers_end_states_and_the_mask_after_a_digit() {
        let Some(path) = qwen3_06b() else {
            eprintln!("skip: Qwen3-0.6B not cached");
            return;
        };
        let g = infr_gguf::Gguf::open(&path).expect("open gguf");
        let cfg = crate::Config::from_gguf(&g).expect("config");
        let tok = crate::build_tokenizer(&g).expect("tokenizer");
        let (vocab, eos) = (cfg.vocab, cfg.eos);
        let tools = weather_tools();
        let call =
            "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Oslo\"}}\n\
                    </tool_call>";
        let open = tok.encode("<tool_call>", false).expect("encode");
        assert_eq!(open.get_ids().len(), 1, "`<tool_call>` is one token here");
        for text in [call.to_string(), format!("{call}\n")] {
            assert_eq!(
                judge(&tok, vocab, eos, &person_schema(), Some(&tools), &text),
                Some(true),
                "{text:?}"
            );
        }

        let id = |s: &str| {
            let e = tok.encode(s, false).expect("encode");
            assert_eq!(e.get_ids().len(), 1, "{s:?}");
            e.get_ids()[0]
        };
        let open_age =
            serde_json::json!({"type": "object", "properties": {"age": {"type": "integer"}}});
        let closed_age = serde_json::json!({"type": "object",
            "properties": {"age": {"type": "integer"}},
            "required": ["age"], "additionalProperties": false});
        for (schema, comma) in [(open_age, true), (closed_age, false)] {
            let fresh = || {
                let mut c = ToolConstrainer::default()
                    .response_constraint(&tok, vocab, &[eos], &schema, None)
                    .expect("constraint");
                for t in tok.encode(r#"{"age": 3"#, false).expect("encode").get_ids() {
                    assert!(c.try_accept(*t).expect("try_accept"));
                }
                c
            };
            let mut logits = vec![0.0f32; vocab];
            fresh().apply_mask(&mut logits).expect("mask");
            for (s, want) in [("}", true), (",", comma), ("4", true)] {
                let t = id(s);
                let in_mask = logits[t as usize].is_finite();
                let accepted = fresh().try_accept(t).expect("try_accept");
                eprintln!("after {{\"age\": 3, {s:?}: in mask {in_mask}, accepted {accepted}");
                assert_eq!((in_mask, accepted), (want, want), "{s:?} in {schema}");
            }
        }
    }

    /// One `response_format` reply from Qwen3-0.6B on the CPU, greedy, generated the way serve
    /// generates it: rendered with thinking off, decoded under the response constraint. Returns
    /// the text with the stats, and the model's tokenizer for checking it.
    fn qwen3_json_reply(
        user: &str,
        schema: &Value,
        tools: Option<&Value>,
        max_new: usize,
    ) -> Option<(String, crate::GenStats, Tokenizer, usize, u32)> {
        let path = qwen3_06b()?;
        let mut ecfg = crate::EngineConfig::default();
        ecfg.sampling.temp = 0.0;
        let ecfg = Arc::new(ecfg);
        let model = crate::SeamModel::load_with(&path, None, ecfg.clone()).expect("model load");
        let renderer = crate::chat::OaiRenderer::open(&path, ecfg).expect("renderer");
        let messages = [infr_chat::ChatMessage {
            role: "user".into(),
            content: user.into(),
            ..Default::default()
        }];
        let prompt = renderer.render_no_think(&messages, tools).expect("render");
        assert!(
            prompt.ends_with("<think>\n\n</think>\n\n"),
            "Qwen3 must be told not to think: {prompt:?}"
        );
        let mut c = renderer
            .response_constraint(schema, tools)
            .expect("constraint");
        let mut out = String::new();
        let stats = model
            .generate_cpu_constrained(&prompt, max_new, Some(&mut c), None, |p| out.push_str(p))
            .expect("generate");
        eprintln!("response_format reply ({} tokens): {out:?}", stats.n_gen);
        let g = infr_gguf::Gguf::open(&path).expect("open gguf");
        let cfg = crate::Config::from_gguf(&g).expect("config");
        let tok = crate::build_tokenizer(&g).expect("tokenizer");
        Some((out, stats, tok, cfg.vocab, cfg.eos))
    }

    /// A real model's `response_format` replies, judged by what the grammar guarantees rather
    /// than by what the model happens to choose: the text is grammar-valid; a reply the grammar
    /// ended (`natural_stop`) is complete and matches its schema (or is calls to an offered tool);
    /// a reply the budget ended used the whole budget and is a valid prefix.
    #[test]
    fn real_replies_are_grammar_valid_and_complete_when_they_stop() {
        let tools = weather_tools();
        let object = serde_json::json!({"type": "object"});
        let cases: [(&str, &Value, Option<&Value>, usize); 4] = [
            ("Describe a famous scientist.", &person_schema(), None, 96),
            ("List three primary colours.", &object, None, 96),
            (
                "What is the weather in Oslo? Use the get_weather tool.",
                &person_schema(),
                Some(&tools),
                96,
            ),
            ("Describe a famous scientist.", &person_schema(), None, 4),
        ];
        for (user, schema, tools, max_new) in cases {
            let Some((out, stats, tok, vocab, eos)) =
                qwen3_json_reply(user, schema, tools, max_new)
            else {
                eprintln!("skip: Qwen3-0.6B not cached");
                return;
            };
            let verdict = judge(&tok, vocab, eos, schema, tools, &out);
            assert!(verdict.is_some(), "grammar-invalid output: {out:?}");
            if stats.natural_stop {
                assert_eq!(verdict, Some(true), "stopped incomplete: {out:?}");
                let (_, calls) = infr_chat::parse_hermes_tool_calls(&out);
                if calls.is_empty() {
                    let v: Value = serde_json::from_str(&out)
                        .unwrap_or_else(|e| panic!("not JSON ({e}): {out:?}"));
                    if schema == &object {
                        assert!(v.is_object(), "{out:?}");
                    } else {
                        assert!(is_person(&v), "off-schema: {out:?}");
                    }
                } else {
                    assert!(tools.is_some(), "a call with no tool offered: {out:?}");
                    assert!(
                        calls
                            .iter()
                            .all(|c| c.name == "get_weather" && c.arguments["city"].is_string()),
                        "{out:?}"
                    );
                }
            } else {
                assert_eq!(stats.n_gen, max_new, "{out:?}");
            }
        }
    }

    // --- schemas deep enough to overflow a compile stack ----------------------------------------

    /// The environment variable naming the case a re-executed test binary runs as a child.
    const DEPTH_CHILD: &str = "INFR_TEST_GRAMMAR_DEPTH_CHILD";

    /// A chain of `n` `$ref`s, each definition pointing at the next, ending in a string.
    fn ref_chain(n: usize) -> Value {
        let mut defs = serde_json::Map::new();
        for i in 0..n {
            defs.insert(
                format!("d{i}"),
                serde_json::json!({"$ref": format!("#/$defs/d{}", i + 1)}),
            );
        }
        defs.insert(format!("d{n}"), serde_json::json!({"type": "string"}));
        serde_json::json!({"$ref": "#/$defs/d0", "$defs": defs})
    }

    /// A chain of `n` links the way each guard bypass spells its references: `kind` is `pct`
    /// (percent-encoded pointers), `anchor` (`$anchor` fragments), `id` (`$id` URIs) or
    /// `unknown` (definitions under a key no keyword names). The root starts the chain.
    fn bypass_chain(kind: &str, n: usize) -> Value {
        let target = |i: usize| match kind {
            "pct" => format!("#/%24defs/d{i}"),
            "anchor" => format!("#a{i}"),
            "id" => format!("urn:d{i}"),
            "unknown" => format!("#/stash/d{i}"),
            "pctstash" => format!("#/st%61sh/d{i}"),
            "anchorstash" => format!("#a{i}"),
            other => panic!("unknown chain {other}"),
        };
        let mut defs = serde_json::Map::new();
        for i in 0..=n {
            let mut d = if i == n {
                serde_json::json!({"type": "string"})
            } else {
                serde_json::json!({"$ref": target(i + 1)})
            };
            match kind {
                "anchor" | "anchorstash" => d["$anchor"] = format!("a{i}").into(),
                "id" => d["$id"] = format!("urn:d{i}").into(),
                _ => {}
            }
            defs.insert(format!("d{i}"), d);
        }
        let home = if kind.ends_with("stash") || kind == "unknown" {
            "stash"
        } else {
            "$defs"
        };
        serde_json::json!({"$ref": target(0), home: defs})
    }

    /// `levels` of nested `properties` around `inner`.
    fn nested_props(levels: usize, inner: Value) -> Value {
        (0..levels).fold(
            inner,
            |v, _| serde_json::json!({"type": "object", "properties": {"p": v}}),
        )
    }

    /// Definitions `A1…An`, each `k` levels of nested `properties` ending in a reference to the
    /// next, `An` ending in a string; the root reaches `A1` through a property. With `memo_decoy`
    /// the root also lists the definitions in reverse under `anyOf`, so a walk that takes the
    /// applicator first measures each definition's height bottom-up before the property chain.
    fn stacked_defs(n: usize, k: usize, memo_decoy: bool) -> Value {
        let mut defs = serde_json::Map::new();
        for i in 1..=n {
            let inner = if i == n {
                serde_json::json!({"type": "string"})
            } else {
                serde_json::json!({"$ref": format!("#/$defs/A{}", i + 1)})
            };
            defs.insert(format!("A{i}"), nested_props(k, inner));
        }
        let mut root = serde_json::json!({"type": "object",
            "properties": {"p": {"$ref": "#/$defs/A1"}}, "$defs": defs});
        if memo_decoy {
            let any: Vec<Value> = (2..=n)
                .rev()
                .map(|i| serde_json::json!({"$ref": format!("#/$defs/A{i}")}))
                .collect();
            root["anyOf"] = Value::Array(any);
        }
        root
    }

    /// Definitions `A0…An-1` chained by `$ref`, each `k` nested applicators of `kind`.
    fn intersecting_defs(kind: &str, n: usize, k: usize) -> serde_json::Map<String, Value> {
        let wrap = |v: Value| match kind {
            "anyof" => serde_json::json!({"anyOf": [v, {"type": "null"}]}),
            "allof" => serde_json::json!({"allOf": [v, {"type": ["object", "string"]}]}),
            _ => serde_json::json!({"allOf": [
                {"type": "object", "properties": {"p": v}},
                {"required": ["p"]}
            ]}),
        };
        (0..n)
            .map(|i| {
                let inner = if i + 1 == n {
                    serde_json::json!({"type": "object"})
                } else {
                    serde_json::json!({"$ref": format!("#/$defs/A{}", i + 1)})
                };
                (format!("A{i}"), (0..k).fold(inner, |v, _| wrap(v)))
            })
            .collect()
    }

    /// A chain of `n` targets, each `k` levels deep, held in a malformed container under the root
    /// — `items` as an object with digit keys, `anyOf` as an object, or `properties` as an array —
    /// which a pointer reaches by a spelling that looks like a schema position. The root's own
    /// `$ref` comes first, so llguidance compiles the target before it meets the container; each
    /// target's reference to the next is percent-encoded, which `referencing` decodes.
    fn malformed_chain(shape: &str, n: usize, k: usize) -> Value {
        let keyword = match shape {
            "items" | "anyof" => {
                if shape == "items" {
                    "items"
                } else {
                    "anyOf"
                }
            }
            _ => "properties",
        };
        let targets: Vec<Value> = (0..n)
            .map(|i| {
                let inner = if i + 1 == n {
                    serde_json::json!({"type": "string"})
                } else {
                    // `%3X` is the digit X, percent-encoded.
                    let next: String = (i + 1)
                        .to_string()
                        .chars()
                        .map(|c| format!("%3{c}"))
                        .collect();
                    serde_json::json!({"$ref": format!("#/{keyword}/{next}")})
                };
                nested_props(k, inner)
            })
            .collect();
        let container = if shape == "props" {
            Value::Array(targets)
        } else {
            Value::Object(
                targets
                    .into_iter()
                    .enumerate()
                    .map(|(i, t)| (i.to_string(), t))
                    .collect(),
            )
        };
        let mut root = serde_json::Map::new();
        root.insert("$ref".into(), serde_json::json!(format!("#/{keyword}/0")));
        root.insert(keyword.into(), container);
        Value::Object(root)
    }

    /// Draft-04 `id` moving the base `#` resolves against: a subschema with an `id` holds the real
    /// chain of `n` definitions, each `k` levels deep, while the root's `definitions` hold decoys
    /// of the same names that the root-relative reading resolves to.
    fn draft4_id_decoy(n: usize, k: usize) -> Value {
        let mut real = serde_json::Map::new();
        let mut decoy = serde_json::Map::new();
        for i in 0..n {
            let inner = if i + 1 == n {
                serde_json::json!({"type": "string"})
            } else {
                serde_json::json!({"$ref": format!("#/definitions/d{}", i + 1)})
            };
            real.insert(format!("d{i}"), nested_props(k, inner));
            decoy.insert(format!("d{i}"), serde_json::json!({"type": "string"}));
        }
        serde_json::json!({
            "$schema": "http://json-schema.org/draft-04/schema#",
            "type": "object",
            "properties": {"x": {"id": "urn:inner", "definitions": real,
                "$ref": "#/definitions/d0"}},
            "definitions": decoy
        })
    }

    /// A tool whose parameters reach a chain of `n` definitions through references that only
    /// resolve against the assembled call schema (`#` is the call root there).
    fn tool_chain(n: usize) -> Value {
        let at = |i: usize| format!("#/anyOf/0/properties/arguments/$defs/d{i}");
        let mut defs = serde_json::Map::new();
        for i in 0..n {
            defs.insert(format!("d{i}"), serde_json::json!({"$ref": at(i + 1)}));
        }
        defs.insert(format!("d{n}"), serde_json::json!({"type": "string"}));
        serde_json::json!([{"type": "function", "function": {"name": "f", "parameters": {
            "type": "object", "properties": {"x": {"$ref": at(0)}}, "$defs": defs}}}])
    }

    /// `n` levels of `anyOf` around a string.
    fn any_of_nest(n: usize) -> Value {
        (0..n).fold(
            serde_json::json!({"type": "string"}),
            |v, _| serde_json::json!({ "anyOf": [v] }),
        )
    }

    /// One child case: what it compiles, through which path, on a thread with the 2 MiB stack a
    /// tokio blocking thread has — the server's. Prints `OUTCOME <compiled|refused[ tools]|
    /// error>` for the parent to read; a stack overflow kills the child instead.
    fn run_depth_case(case: &str) {
        let tok = byte_tokenizer();
        let vocab = tok.get_vocab_size(true);
        let case = case.to_string();
        let outcome = std::thread::Builder::new()
            .stack_size(2 << 20)
            .spawn(move || {
                let tc = ToolConstrainer::default();
                let tools = |params: Value| {
                    serde_json::json!([{"type": "function",
                        "function": {"name": "f", "parameters": params}}])
                };
                let r: Result<()> = match case.as_str() {
                    "chain_2000" => tc
                        .response_constraint(&tok, vocab, &[0], &ref_chain(2000), None)
                        .map(drop),
                    "chain_2000_tool" => tc
                        .constraint(&tok, vocab, &[0], Some(&tools(ref_chain(2000))), Some("f"))
                        .map(drop),
                    "chain_at_limit" => tc
                        .response_constraint(
                            &tok,
                            vocab,
                            &[0],
                            &ref_chain(shape::MAX_SCHEMA_DEPTH - 8),
                            None,
                        )
                        .map(drop),
                    "any_of_60" => tc
                        .response_constraint(&tok, vocab, &[0], &any_of_nest(60), None)
                        .map(drop),
                    "self_recursive" => tc
                        .response_constraint(
                            &tok,
                            vocab,
                            &[0],
                            &serde_json::json!({"$ref": "#/$defs/n", "$defs": {"n": {
                                "type": "object",
                                "properties": {"next": {"$ref": "#/$defs/n"}}}}}),
                            None,
                        )
                        .map(drop),
                    "too_many_nodes" => {
                        let props: serde_json::Map<String, Value> = (0..=shape::MAX_JSON_VALUES
                            / 2)
                            .map(|i| (format!("p{i}"), serde_json::json!({"type": "string"})))
                            .collect();
                        tc.response_constraint(
                            &tok,
                            vocab,
                            &[0],
                            &serde_json::json!({"type": "object", "properties": props}),
                            None,
                        )
                        .map(drop)
                    }
                    // No guard: llguidance alone, on the compile thread, four times the limit.
                    "margin_unguarded" => ConstraintEnv::new(&tok, vocab, &[0])
                        .and_then(|env| {
                            env.constraint(TopLevelGrammar::from_json_schema(ref_chain(
                                4 * shape::MAX_SCHEMA_DEPTH,
                            )))
                        })
                        .map(drop),
                    // No rewrite: two recursive references meeting in an intersection.
                    "refs_meet_unguarded" => ConstraintEnv::new(&tok, vocab, &[0])
                        .and_then(|env| {
                            let any = serde_json::json!({"$ref": "#/$defs/a"});
                            env.constraint(TopLevelGrammar::from_json_schema(serde_json::json!({
                                "$defs": {"a": {"anyOf": [{"type": "string"},
                                    {"type": "object", "additionalProperties": any}]}},
                                "allOf": [{"type": "object", "additionalProperties": any},
                                          {"type": "object", "additionalProperties": any}]
                            })))
                        })
                        .map(drop),
                    c if c.starts_with("stacked_") => {
                        // `stacked_<n>_<k>[_memo|_raw]`
                        let parts: Vec<&str> = c.split('_').collect();
                        let (n, k) = (parts[1].parse().unwrap(), parts[2].parse().unwrap());
                        let schema = stacked_defs(n, k, parts.get(3) == Some(&"memo"));
                        if parts.get(3) == Some(&"raw") {
                            ConstraintEnv::new(&tok, vocab, &[0])
                                .and_then(|env| {
                                    env.constraint(TopLevelGrammar::from_json_schema(schema))
                                })
                                .map(drop)
                        } else {
                            tc.response_constraint(&tok, vocab, &[0], &schema, None)
                                .map(drop)
                        }
                    }
                    c if c.starts_with("wide_") => {
                        // `wide_<n>_<k>`: `n` independent definitions `k` levels deep, each
                        // reached straight from the root.
                        let parts: Vec<&str> = c.split('_').collect();
                        let (n, k): (usize, usize) =
                            (parts[1].parse().unwrap(), parts[2].parse().unwrap());
                        let defs: serde_json::Map<String, Value> = (0..n)
                            .map(|i| (format!("D{i}"), nested_props(k, serde_json::json!({}))))
                            .collect();
                        let any: Vec<Value> = (0..n)
                            .map(|i| serde_json::json!({"$ref": format!("#/$defs/D{i}")}))
                            .collect();
                        let schema = serde_json::json!({"anyOf": any, "$defs": defs});
                        tc.response_constraint(&tok, vocab, &[0], &schema, None)
                            .map(drop)
                    }
                    c if c.starts_with("inter_") => {
                        // `inter_<anyof|allof|props>_<n>_<k>`: definitions `A0…An-1` chained by
                        // `$ref`, each `k` nested applicators — `anyOf` with a null branch,
                        // `allOf` with a second schema, or `allOf` of a `properties` object and a
                        // `required` — the shapes that make llguidance intersect as it descends.
                        let parts: Vec<&str> = c.split('_').collect();
                        let (n, k) = (parts[2].parse().unwrap(), parts[3].parse().unwrap());
                        let defs = intersecting_defs(parts[1], n, k);
                        // Unguarded: the path bound would refuse it long before this one.
                        let schema = serde_json::json!({"$ref": "#/$defs/A0", "$defs": defs});
                        ConstraintEnv::new(&tok, vocab, &[0])
                            .and_then(|env| {
                                env.constraint(TopLevelGrammar::from_json_schema(schema))
                            })
                            .map(drop)
                    }
                    c if c.starts_with("malformed_") => {
                        // `malformed_<items|anyof|props>_<n>_<k>`
                        let parts: Vec<&str> = c.split('_').collect();
                        let (n, k) = (parts[2].parse().unwrap(), parts[3].parse().unwrap());
                        let schema = malformed_chain(parts[1], n, k);
                        tc.response_constraint(&tok, vocab, &[0], &schema, None)
                            .map(drop)
                    }
                    c if c.starts_with("draft4_") => {
                        let parts: Vec<&str> = c.split('_').collect();
                        let (n, k) = (parts[1].parse().unwrap(), parts[2].parse().unwrap());
                        tc.response_constraint(&tok, vocab, &[0], &draft4_id_decoy(n, k), None)
                            .map(drop)
                    }
                    c if c.starts_with("bypass_") => {
                        // `bypass_<kind>_<n>`
                        let mut parts = c.trim_start_matches("bypass_").rsplitn(2, '_');
                        let n: usize = parts.next().unwrap().parse().unwrap();
                        let kind = parts.next().unwrap();
                        if kind == "tool" {
                            tc.constraint(&tok, vocab, &[0], Some(&tool_chain(n)), Some("f"))
                                .map(drop)
                        } else {
                            tc.response_constraint(&tok, vocab, &[0], &bypass_chain(kind, n), None)
                                .map(drop)
                        }
                    }
                    other => panic!("unknown case {other}"),
                };
                match r {
                    Ok(()) => "compiled".to_string(),
                    Err(e) => match e.downcast_ref::<GrammarRejected>() {
                        Some(g) if g.in_tools => format!("refused tools: {}", g.message),
                        Some(g) => format!("refused: {}", g.message),
                        None => format!("error: {e}"),
                    },
                }
            })
            .expect("spawn")
            .join()
            .expect("join");
        println!("OUTCOME {outcome}");
    }

    /// A schema deep enough to overflow llguidance's recursive compile must never abort the
    /// process: a long flat `$ref` chain (reachable in a small request body, on the
    /// `response_format` path and through a tool's parameters) is refused before compiling, a
    /// schema within the limit compiles from a caller with the server's 2 MiB blocking-thread
    /// stack, and so do the serde-reachable nesting and a self-recursive schema. Each case runs
    /// in a re-executed child so an overflow fails the case instead of killing the suite. The
    /// unguarded case shows the compile thread's margin: llguidance alone survives four times the
    /// depth limit there. Two recursive references meeting in an intersection end in llguidance's
    /// own refusal (its intersection depth limit) on the compile thread, not an overflow.
    #[test]
    fn deep_schemas_never_abort_the_process() {
        if let Ok(case) = std::env::var(DEPTH_CHILD) {
            run_depth_case(&case);
            return;
        }
        // serde's own depth limit bounds what a request body can nest.
        let deep_json = format!("{}{}", "[".repeat(200), "]".repeat(200));
        let e = serde_json::from_str::<Value>(&deep_json).unwrap_err();
        assert!(e.to_string().contains("recursion limit"), "{e}");

        let exe = std::env::current_exe().expect("test binary");
        for (case, want) in [
            (
                "chain_2000",
                "refused: the schema has more than 1000 `$ref`s",
            ),
            (
                "chain_2000_tool",
                "refused tools: the schema has more than 1000 `$ref`s",
            ),
            ("chain_at_limit", "compiled"),
            ("any_of_60", "compiled"),
            ("self_recursive", "compiled"),
            ("too_many_nodes", "refused: the schema has more than"),
            // References the shape check could not follow, which llguidance resolves through
            // its `referencing` crate: refused now. Before, each passed the check; the
            // percent-encoded chain under an unknown key aborted the process at 9000 links in a
            // debug build and 25000 in a release one, on the 64 MiB compile stack.
            ("bypass_pct_2000", "refused: `$ref`"),
            ("bypass_pct_300", "refused: `$ref`"),
            // Refused for the reference or the `$anchor`/`$id` naming its target, whichever the
            // check meets first.
            ("bypass_anchor_300", "refused: `$"),
            ("bypass_id_300", "refused: `$"),
            ("bypass_tool_300", "refused tools: `$ref`"),
            // A pointer into a key no keyword names addresses data, not a subschema.
            ("bypass_unknown_2000", "refused: `$ref`"),
            ("bypass_unknown_300", "refused: `$ref`"),
            ("bypass_pctstash_25000", "refused: `$ref`"),
            ("bypass_pctstash_300", "refused: `$ref`"),
            // Reused definitions reached deeper than a memoised walk measured: definitions
            // A1…A200, each 100 levels deep and referencing the next, reached through a property
            // and listed again in reverse under `anyOf`. Before, the guard passed it and the
            // compile overflowed its 64 MiB stack in debug and release alike.
            ("stacked_200_100_memo", "refused: the schema"),
            // The longest-path bound on its own, under the same memo decoy: 5 × 60 levels.
            (
                "stacked_5_60_memo",
                "refused: the schema nests deeper than 256",
            ),
            ("stacked_8_20", "compiled"),
            // The order-independent backstop on its own: 30 definitions 40 levels deep, none
            // deeper than the path limit, together past the level limit.
            (
                "wide_30_40",
                "refused: the schema's definitions together nest deeper",
            ),
            ("wide_10_40", "compiled"),
            // Draft-04 `id` moving the base: before, the guard measured the root's shallow decoys
            // and passed 960 real levels; the dialect is now pinned.
            ("draft4_24_40", "refused: `$schema`"),
            // Targets under an unknown key of `items`, or in a malformed container, reached by a
            // pointer spelled like a schema position, their own references percent-encoded.
            // Before, the guard passed all three and the `items` and `anyOf` chains aborted the
            // process in debug and release.
            ("malformed_items_300_100", "refused: `$ref`"),
            ("malformed_anyof_300_100", "refused: `anyOf` holds"),
            ("malformed_props_300_100", "refused: `properties` holds"),
            // Intersection-heavy definitions as deep as the level bound admits — root 1 plus
            // 30 × (33 + 1) — compile on the compile stack in debug, with nothing guarding them.
            ("inter_anyof_30_32", "compiled"),
            ("margin_unguarded", "compiled"),
            (
                "refs_meet_unguarded",
                "refused: Schema intersection stack level exceeded",
            ),
        ] {
            let out = std::process::Command::new(&exe)
                .args([
                    "--exact",
                    "grammar::tests::deep_schemas_never_abort_the_process",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(DEPTH_CHILD, case)
                .output()
                .expect("re-run the test binary");
            let stdout = String::from_utf8_lossy(&out.stdout);
            let outcome = stdout
                .lines()
                .find_map(|l| l.split_once("OUTCOME ").map(|(_, o)| o))
                .unwrap_or_else(|| {
                    panic!(
                        "{case}: the child died ({:?}): {}",
                        out.status,
                        String::from_utf8_lossy(&out.stderr)
                            .lines()
                            .filter(|l| l.contains("overflow") || l.contains("panicked"))
                            .collect::<Vec<_>>()
                            .join(" | ")
                    )
                });
            eprintln!("{case}: {outcome}");
            assert!(outcome.starts_with(want), "{case}: {outcome}");
        }
    }
}
