//! Regression test: the LFM3 chat template (extracted verbatim from LiquidAI/LFM3-1.7B-GGUF
//! `tokenizer.chat_template`) renders byte for byte what llama.cpp renders for the same
//! conversation. The template uses the `{% generation %}` tag around assistant turns, opens with
//! the BOS, empties a past assistant turn's thinking to `<think></think>`, and ends the generation
//! prompt in `<think>` or, with thinking off, `<think></think>`.

use infr_chat::render_template;
use serde_json::{json, Value};

const LFM3_TEMPLATE: &str = include_str!("fixtures/lfm3_chat_template.jinja");

/// A system turn, a user/assistant exchange, and a second user turn with CJK text, with thinking
/// on (the template's default) and off. The expected strings are llama-server's `/apply-template`
/// output (`--jinja`) for the same messages.
#[test]
fn lfm3_render_matches_the_reference() {
    let msgs = || {
        vec![
            json!({"role": "system", "content": "You are a helpful assistant."}),
            json!({"role": "user", "content": "Hi, who are you?"}),
            json!({"role": "assistant", "content": "I am LFM."}),
            json!({"role": "user", "content": "北京在哪里？"}),
        ]
    };
    let conversation =
        "<|startoftext|><|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
                        <|im_start|>user\nHi, who are you?<|im_end|>\n\
                        <|im_start|>assistant\n<think></think>I am LFM.<|im_end|>\n\
                        <|im_start|>user\n北京在哪里？<|im_end|>\n\
                        <|im_start|>assistant\n";
    for (thinking, tail) in [(true, "<think>"), (false, "<think></think>")] {
        let out = render_template(
            LFM3_TEMPLATE,
            msgs(),
            Value::Null,
            "<|startoftext|>",
            "<|im_end|>",
            true,
            thinking,
        )
        .unwrap();
        assert_eq!(out, format!("{conversation}{tail}"), "thinking={thinking}");
    }
}
