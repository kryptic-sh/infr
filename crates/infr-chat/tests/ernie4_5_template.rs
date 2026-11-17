//! Regression test: the ERNIE 4.5 chat template (extracted verbatim from
//! unsloth/ERNIE-4.5-0.3B-PT-GGUF `tokenizer.chat_template`) renders byte for byte what llama.cpp
//! renders for the same conversation. The template opens with its own `cls_token`
//! (`<|begin_of_sentence|>`), not the GGUF's BOS, and closes only assistant turns, with
//! `<|end_of_sentence|>`.

use infr_chat::render_template;
use serde_json::{json, Value};

const ERNIE4_5_TEMPLATE: &str = include_str!("fixtures/ernie4_5_chat_template.jinja");

/// A system turn, a user/assistant exchange, and a second user turn with CJK text.
#[test]
fn ernie4_5_render_matches_the_reference() {
    let msgs = vec![
        json!({"role": "system", "content": "You are a helpful assistant."}),
        json!({"role": "user", "content": "Hi, who are you?"}),
        json!({"role": "assistant", "content": "I am ERNIE."}),
        json!({"role": "user", "content": "北京在哪里？"}),
    ];
    let out = render_template(
        ERNIE4_5_TEMPLATE,
        msgs,
        Value::Null,
        "<s>",
        "</s>",
        true,
        true,
    )
    .unwrap();
    // llama.cpp's `test-chat-template --no-common` output for the same template and variables
    // (the default common path renders the same bytes).
    assert_eq!(
        out,
        "<|begin_of_sentence|>You are a helpful assistant.\n\
         User: Hi, who are you?\n\
         Assistant: I am ERNIE.<|end_of_sentence|>\
         User: 北京在哪里？\n\
         Assistant: "
    );
}
