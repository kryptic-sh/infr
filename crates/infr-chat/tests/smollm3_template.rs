//! Regression test: the SmolLM3 chat template (extracted verbatim from
//! unsloth/SmolLM3-3B-GGUF `tokenizer.chat_template`) renders byte for byte what llama.cpp renders
//! for the same input. Its leading comment lines end in plain `#}`, so the render only starts at
//! `<|im_start|>` when the newline after each comment is trimmed, as the reference does.

use infr_chat::render_template;
use serde_json::{json, Value};

const SMOLLM3_TEMPLATE: &str = include_str!("fixtures/smollm3_chat_template.jinja");

/// The conversation `cpu_golden_smollm3` uses: `/system_override` replaces the dated default
/// system block, so the render is the same on every day.
#[test]
fn smollm3_render_matches_the_reference() {
    let msgs = vec![
        json!({
            "role": "system",
            "content": "You are a helpful AI assistant named SmolLM, trained by Hugging Face.                  /system_override /no_think",
        }),
        json!({"role": "user", "content": "The capital of France is"}),
    ];
    let out = render_template(
        SMOLLM3_TEMPLATE,
        msgs,
        Value::Null,
        "",
        "<|im_end|>",
        true,
        true,
    )
    .unwrap();
    // llama.cpp's `test-chat-template --no-common` output for the same template and variables.
    assert_eq!(
        out,
        "<|im_start|>system\n\
         You are a helpful AI assistant named SmolLM, trained by Hugging Face.<|im_end|>\n\
         <|im_start|>user\n\
         The capital of France is<|im_end|>\n\
         <|im_start|>assistant\n\
         <think>\n\n</think>\n"
    );
}
