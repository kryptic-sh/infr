//! The request's JSON (messages, tool definitions, tool-call arguments) as template values whose
//! lists and dicts print as Python's `repr` everywhere minijinja prints them itself.
//!
//! minijinja's `~` operator and string-argument conversions go through its own `Display`, which
//! has no hook (`ops::string_concat` in its `value/ops.rs`). A list or dict it builds there is
//! written JSON-style. So every list and dict that comes from the request is wrapped in
//! [`PyContainer`], which hands every lookup, iteration, length and comparison to the native value
//! it wraps and only replaces how it renders: `' Default=' ~ default` then prints Python's
//! `['a', 'b']` for a list default, as Hugging Face's renderer does. Floats are not reachable this
//! way (they must stay numbers for arithmetic), and a container the template builds itself is
//! minijinja's own; both stay the gaps `tests/jinja2_text.rs` lists.

use std::fmt;
use std::sync::Arc;

use minijinja::value::{DynObject, Enumerator, Object, ObjectRepr};
use minijinja::Value;
use serde_json::Value as Json;

use crate::pystr::py_str;

/// `json` as a template value, every list and dict in it a [`PyContainer`].
pub(crate) fn from_json(json: &Json) -> Value {
    match json {
        Json::Null => Value::from(()),
        Json::Bool(b) => Value::from(*b),
        Json::Number(n) => match (n.as_u64(), n.as_i64()) {
            (Some(u), _) => Value::from(u),
            (None, Some(i)) => Value::from(i),
            _ => Value::from(
                n.as_f64()
                    .expect("without arbitrary_precision a number is an f64"),
            ),
        },
        Json::String(s) => Value::from(s.as_str()),
        Json::Array(items) => wrap(Value::from(items.iter().map(from_json).collect::<Vec<_>>())),
        Json::Object(map) => wrap(Value::from_iter(
            map.iter()
                .map(|(k, v)| (Value::from(k.as_str()), from_json(v))),
        )),
    }
}

fn wrap(native: Value) -> Value {
    let inner = native
        .as_object()
        .expect("a list or a dict is an object")
        .clone();
    Value::from_object(PyContainer { inner })
}

/// A list or dict from the request: minijinja's own value, rendered as Python's `repr`.
#[derive(Debug)]
pub(crate) struct PyContainer {
    inner: DynObject,
}

impl Object for PyContainer {
    fn repr(self: &Arc<Self>) -> ObjectRepr {
        self.inner.repr()
    }

    fn get_value(self: &Arc<Self>, key: &Value) -> Option<Value> {
        self.inner.get_value(key)
    }

    fn get_value_by_str(self: &Arc<Self>, key: &str) -> Option<Value> {
        self.inner.get_value_by_str(key)
    }

    fn enumerate(self: &Arc<Self>) -> Enumerator {
        self.inner.enumerate()
    }

    fn enumerator_len(self: &Arc<Self>) -> Option<usize> {
        self.inner.enumerator_len()
    }

    fn is_true(self: &Arc<Self>) -> bool {
        self.inner.is_true()
    }

    /// Python's `repr`; past [`crate::tojson::NESTING_MAX`] (where [`py_str`] refuses), the native
    /// rendering, since a `Display` that fails would abort the render with a panic.
    fn render(self: &Arc<Self>, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let me = Value::from_dyn_object(self.inner.clone());
        match py_str(&me) {
            Ok(text) => f.write_str(&text),
            Err(_) => self.inner.render(f),
        }
    }
}
