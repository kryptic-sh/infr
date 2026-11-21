//! Bounding the numbers a `response_format` grammar admits, and refusing `multipleOf` values
//! llguidance's decimal arithmetic cannot enforce.
//!
//! llguidance's JSON numbers are unbounded: a small model that does not want to close a value
//! keeps choosing digits the grammar still allows, out to its token budget. [`cap_numbers`]
//! rewrites a schema so every number it admits is bounded, the way llama.cpp's
//! `json-schema-to-grammar` bounds its `integral-part` and `decimal-part`, without refusing any
//! value the schema's own keywords single out.

use serde_json::{Map, Value};

use super::GrammarRejected;

/// The largest magnitude a number the schema leaves unbounded may have: an integral part of at
/// most llama.cpp's 16 digits, give or take the one 17-digit value this bound itself spells, which
/// is exact in an `f64` where `10^16 - 1` is not.
pub(super) const NUMBER_LIMIT: f64 = 1e16;

/// The granularity of a non-integer the schema leaves unbounded: a multiple of this has at most
/// as many fraction digits as its exponent says. llama.cpp allows 16. llguidance 1.9 can say no
/// more than this: its `multipleOf` matcher (derivre's `RemainderIs`) computes
/// `digit * 10^scale` in a `u32`, which overflows one power of ten finer — a panic in a debug
/// build, a silently wrong grammar in a release one.
pub(super) const NUMBER_STEP: f64 = 1e-8;

/// The largest magnitude an `integer` bound may have. llguidance converts integer bounds to `i64`
/// (`normalize_integer_bounds`), and one past that range saturates and is then negated — a panic
/// in a debug build. This stays inside `i64` with room for that arithmetic; a schema bound past it
/// is refused.
const INTEGER_BOUND_LIMIT: f64 = 9.22e18;

/// The most distinct `multipleOf` steps a schema may use: every pair is checked (see
/// [`check_schema`]), so this bounds that check's work.
const MAX_DISTINCT_STEPS: usize = 64;

/// The largest `multipleOf` coefficient (llguidance's `coef` in `coef * 10^-exp`: the step's
/// significant digits, or the whole integer for an integer step) a grammar may use, the user's own
/// or one formed where two steps meet. derivre matches a step with one state per remainder, so
/// its cost grows with the coefficient: measured in a release build on Qwen3's vocabulary under
/// the number bounds the rewrite adds, a coefficient of 10007 compiles in 6 ms with first masks of
/// 2 ms, 20011 compiles and then fails its first mask with "too many expressions constructed" —
/// a mid-reply `500` — and without those bounds the compile alone takes 2.2 s at 1000003 and
/// 60 s at 10000019, which llguidance's `ParserLimits` fuel does not meter. This sits twenty times
/// below the first failure, and above the steps schemas use (`0.01`, `0.25`, `5`, `100`, `1000`).
const MAX_STEP_COEFFICIENT: u64 = 1000;

/// Where the rewrite keeps the bounded stand-in for "any JSON value".
const ANY_DEF: &str = "infr_any_json";

/// The reference the rewrite inserts to [`ANY_DEF`]; the shape check does not count these toward
/// its reference limit, since all of them reach the one small definition.
pub(super) const ANY_REF: &str = "#/$defs/infr_any_json";

/// The keywords llguidance 1.9 implements (`json::schema::IMPLEMENTED`, which is crate-private
/// there). A test compiles each alone against the installed llguidance.
const LLGUIDANCE_KEYWORDS: [&str; 27] = [
    "anyOf",
    "oneOf",
    "allOf",
    "$ref",
    "const",
    "enum",
    "type",
    "items",
    "additionalItems",
    "prefixItems",
    "minItems",
    "maxItems",
    "properties",
    "additionalProperties",
    "patternProperties",
    "required",
    "minProperties",
    "maxProperties",
    "minLength",
    "maxLength",
    "pattern",
    "format",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
];

/// The keywords llguidance 1.9's default draft (2020-12) knows but does not implement, beside the
/// annotations it ignores (`json::context_simple::Draft::is_known_keyword` less `IMPLEMENTED` and
/// `META_AND_ANNOTATIONS`). It refuses a schema that uses one ("Unimplemented keys"); a test
/// checks each is refused, so an upgrade that implements one fails it.
const REFUSED_KEYWORDS: [&str; 18] = [
    "not",
    "uniqueItems",
    "dependencies",
    "contains",
    "propertyNames",
    "if",
    "then",
    "else",
    "$recursiveAnchor",
    "$recursiveRef",
    "dependentRequired",
    "dependentSchemas",
    "maxContains",
    "minContains",
    "unevaluatedItems",
    "unevaluatedProperties",
    "$dynamicAnchor",
    "$dynamicRef",
];

/// The keywords holding subschemas that llguidance compiles: lists of them, a single one (or a
/// tuple list, for `items`), and maps of them. Every one is walked.
const SUBSCHEMA_LISTS: [&str; 4] = ["anyOf", "oneOf", "allOf", "prefixItems"];
const SUBSCHEMA_ONES: [&str; 3] = ["items", "additionalItems", "additionalProperties"];
const SUBSCHEMA_MAPS: [&str; 4] = ["properties", "patternProperties", "$defs", "definitions"];

/// Visit every subschema directly below `obj`.
fn each_subschema(obj: &Map<String, Value>, f: &mut dyn FnMut(&Value)) {
    for key in SUBSCHEMA_LISTS {
        if let Some(Value::Array(subs)) = obj.get(key) {
            subs.iter().for_each(&mut *f);
        }
    }
    for key in SUBSCHEMA_ONES {
        match obj.get(key) {
            Some(Value::Array(subs)) => subs.iter().for_each(&mut *f),
            Some(sub) => f(sub),
            None => {}
        }
    }
    for key in SUBSCHEMA_MAPS {
        if let Some(Value::Object(subs)) = obj.get(key) {
            subs.values().for_each(&mut *f);
        }
    }
}

/// [`each_subschema`], mutably, with the keyword each subschema sits under.
fn each_subschema_mut(obj: &mut Map<String, Value>, f: &mut dyn FnMut(&str, &mut Value)) {
    for key in SUBSCHEMA_LISTS {
        if let Some(Value::Array(subs)) = obj.get_mut(key) {
            subs.iter_mut().for_each(|s| f(key, s));
        }
    }
    for key in SUBSCHEMA_ONES {
        match obj.get_mut(key) {
            Some(Value::Array(subs)) => subs.iter_mut().for_each(|s| f(key, s)),
            Some(sub) => f(key, sub),
            None => {}
        }
    }
    for key in SUBSCHEMA_MAPS {
        if let Some(Value::Object(subs)) = obj.get_mut(key) {
            subs.values_mut().for_each(|s| f(key, s));
        }
    }
}

/// llguidance's decimal for a `multipleOf` value — `coef * 10^-exp`, reduced — computed as its
/// `Decimal::try_from` and `Decimal::new` compute it. `None` where llguidance refuses the value
/// itself (too many digits for its `u32` coefficient), which it reports as a clean error.
fn decimal(value: f64) -> Option<(u64, u32)> {
    let mut v = value.abs();
    let mut exp = 0u32;
    while v.fract() != 0.0 {
        v *= 10.0;
        exp += 1;
        if v > f64::from(u32::MAX) {
            return None;
        }
    }
    if v > f64::from(u32::MAX) {
        return None;
    }
    let mut coef = v as u64;
    if coef == 0 {
        return Some((0, 0));
    }
    while exp > 0 && coef.is_multiple_of(10) {
        coef /= 10;
        exp -= 1;
    }
    Some((coef, exp))
}

/// Whether derivre can match multiples of `d` without overflowing: its `RemainderIs` step
/// computes `remainder * 10 + digit * 10^exp` in a `u32`, and starts with the remainder equal to
/// the divisor `coef`.
fn matchable((coef, exp): (u64, u32)) -> bool {
    let worst = 10u64
        .checked_pow(exp)
        .and_then(|p| p.checked_mul(9))
        .and_then(|d| coef.checked_mul(10)?.checked_add(d));
    worst.is_some_and(|w| w <= u64::from(u32::MAX))
}

/// Whether a step is one llguidance computes correctly ([`matchable`]) and cheaply enough
/// ([`MAX_STEP_COEFFICIENT`]).
fn enforceable(d: (u64, u32)) -> bool {
    matchable(d) && d.0 <= MAX_STEP_COEFFICIENT
}

/// llguidance's `Decimal::lcm` of two steps, which it takes where two `multipleOf` meet in one
/// value (`allOf`, or a keyword beside an applicator); `None` where its `u32` arithmetic
/// overflows.
fn lcm(a: (u64, u32), b: (u64, u32)) -> Option<(u64, u32)> {
    if a.0 == 0 || b.0 == 0 {
        return Some((0, 0));
    }
    let fits = |v: u64| (v <= u64::from(u32::MAX)).then_some(v);
    let x = fits(a.0.checked_mul(10u64.checked_pow(b.1.saturating_sub(a.1))?)?)?;
    let y = fits(b.0.checked_mul(10u64.checked_pow(a.1.saturating_sub(b.1))?)?)?;
    let product = fits(x * y)?;
    let (mut g, mut h) = (x, y);
    while h != 0 {
        (g, h) = (h, g % h);
    }
    let (mut coef, mut exp) = (product / g, a.1.max(b.1));
    while exp > 0 && coef.is_multiple_of(10) {
        coef /= 10;
        exp -= 1;
    }
    Some((coef, exp))
}

/// Refuse a schema llguidance would compile wrongly, slowly, or not survive compiling: one
/// `shape::check_shape` refuses, a bound past [`INTEGER_BOUND_LIMIT`] on a node that may hold an
/// integer, a `multipleOf` finer than derivre can match or with a coefficient past
/// [`MAX_STEP_COEFFICIENT`], more than [`MAX_DISTINCT_STEPS`] distinct steps, or two steps anywhere
/// in the schema whose combined step would be unenforceable — checked for every pair, whether or
/// not they meet, so a schema whose two such steps never meet is refused too. Returns whether
/// every `multipleOf` in the schema also combines safely with [`NUMBER_STEP`] — when one does not,
/// no step is inserted anywhere, and that schema's free fractions stay unbounded.
///
/// Applies to every schema a grammar here compiles — tool parameters on the forced path as well as
/// a `response_format` — because the miscomputation is llguidance's, not the rewrite's.
pub(super) fn check_schema(schema: &Value) -> Result<bool, GrammarRejected> {
    super::shape::check_shape(schema)?;
    let refuse = |message: String| GrammarRejected {
        message,
        in_tools: false,
    };
    let mut err = None;
    let mut steps: Vec<(f64, (u64, u32))> = Vec::new();
    super::shape::for_each_schema_object(schema, &mut |obj| {
        if can_be_integer(obj) {
            for key in ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"] {
                if let Some(v) = obj.get(key).and_then(Value::as_f64) {
                    if v.abs() > INTEGER_BOUND_LIMIT {
                        err.get_or_insert(format!(
                            "integer bound {key} {v} is beyond what the grammar engine can hold"
                        ));
                    }
                }
            }
        }
        let Some(m) = obj.get("multipleOf").and_then(Value::as_f64) else {
            return;
        };
        // A value llguidance cannot read as a decimal is its own clean error.
        let Some(d) = decimal(m) else { return };
        if !matchable(d) {
            err.get_or_insert(format!(
                "multipleOf {m} is finer than the grammar engine can enforce"
            ));
        } else if d.0 > MAX_STEP_COEFFICIENT {
            err.get_or_insert(format!(
                "multipleOf {m} is too fine or too large a step for the grammar engine to enforce"
            ));
        } else if !steps.iter().any(|&(_, e)| e == d) {
            steps.push((m, d));
        }
    });
    if let Some(message) = err {
        return Err(refuse(message));
    }
    if steps.len() > MAX_DISTINCT_STEPS {
        return Err(refuse(format!(
            "the schema uses more than {MAX_DISTINCT_STEPS} distinct multipleOf steps"
        )));
    }
    // Every pair, wherever the two sit: whether two steps meet in one value depends on
    // intersections, `$ref`s and pattern matches that are not worth predicting, so a pair that
    // would combine unenforceably is refused even if it never actually meets.
    for (i, &(m, d)) in steps.iter().enumerate() {
        for &(n, e) in &steps[i + 1..] {
            if !lcm(d, e).is_some_and(enforceable) {
                return Err(refuse(format!(
                    "multipleOf {m} and {n} could combine into a step the grammar engine cannot \
                     enforce (every pair of steps in a schema is checked, whether or not they meet)"
                )));
            }
        }
    }
    let step = decimal(NUMBER_STEP).expect("the step is a decimal llguidance reads");
    Ok(steps
        .iter()
        .all(|&(_, d)| lcm(d, step).is_some_and(enforceable)))
}

/// Whether a numeric node may hold an integer: typed `integer`, or untyped (an untyped bound can
/// meet an `integer` branch under an intersection).
fn can_be_integer(obj: &Map<String, Value>) -> bool {
    let types = schema_types(obj);
    types.is_empty() || types.iter().any(|t| t == "integer")
}

/// `schema` with every number it admits bounded, so a reply cannot run a digit string out to its
/// token budget, after [`check_schema`] has refused what llguidance cannot enforce.
///
/// A number the schema leaves unbounded gets a range of [`NUMBER_LIMIT`] — measured from the
/// schema's own bound where it has one on the other side, so the range is never empty — and a
/// non-integer gets [`NUMBER_STEP`] unless the schema gives its own `multipleOf` (or the step
/// would not combine safely with one, or would leave the range empty); a bounded llguidance number
/// has no exponent form. A node with `const`, `enum` or `$ref` is left as it is. A value the
/// schema leaves open — a node llguidance reads as any value (`{}`, `true`, annotations and
/// unknown keywords only; see [`is_open`]), an object's unlisted properties, an array's unlisted
/// items, `json_object` — becomes a reference to a bounded "any" definition added at the root.
/// JSON Schema applies the numeric keywords to numbers only, and the object and array ones to
/// those types only, so adding them beside other types (or to an untyped node) changes nothing
/// else.
pub(super) fn cap_numbers(schema: &Value) -> Result<Value, GrammarRejected> {
    let step_ok = check_schema(schema)?;
    if schema.pointer(&format!("/$defs/{ANY_DEF}")).is_some() {
        return Err(GrammarRejected {
            message: format!("`$defs/{ANY_DEF}` is a name this server reserves"),
            in_tools: false,
        });
    }
    let any_ref = serde_json::json!({ "$ref": ANY_REF });
    let mut root = schema.clone();
    match &mut root {
        Value::Bool(true) => root = any_ref.clone(),
        // An open root keeps its own keys (its `$defs` among them) and gains the reference.
        Value::Object(obj) if is_open(obj) => {
            obj.insert("$ref".into(), any_ref["$ref"].clone());
        }
        _ => {
            let ctx = CapCtx {
                any_ref: any_ref.clone(),
                step_ok,
                defs_intersected: refs_meet(&root, false),
            };
            cap_node(&mut root, &ctx, false)
        }
    }
    let mut number = serde_json::json!({ "type": "number" });
    if let Some(obj) = number.as_object_mut() {
        cap_number(obj, step_ok);
    }
    let any = serde_json::json!({ "anyOf": [
        { "type": "null" },
        { "type": "boolean" },
        number,
        { "type": "string" },
        { "type": "array", "items": any_ref.clone() },
        { "type": "object", "additionalProperties": any_ref },
    ]});
    if let Some(obj) = root.as_object_mut() {
        let defs = obj
            .entry("$defs")
            .or_insert_with(|| Value::Object(Default::default()));
        if let Some(defs) = defs.as_object_mut() {
            defs.insert(ANY_DEF.to_string(), any);
        }
    }
    Ok(root)
}

/// Whether llguidance reads `obj` as "any value": every key is an annotation or a keyword its
/// draft does not know. A node with a keyword it knows but refuses is not open — llguidance
/// refuses the schema, and so must the rewrite, rather than drop that constraint.
fn is_open(obj: &Map<String, Value>) -> bool {
    !obj.keys().any(|k| {
        LLGUIDANCE_KEYWORDS.contains(&k.as_str()) || REFUSED_KEYWORDS.contains(&k.as_str())
    })
}

/// The `type` names of a schema object (`"type": "x"` or `"type": [..]`); empty when untyped.
fn schema_types(obj: &Map<String, Value>) -> Vec<String> {
    match obj.get("type") {
        Some(Value::String(t)) => vec![t.clone()],
        Some(Value::Array(ts)) => ts
            .iter()
            .filter_map(Value::as_str)
            .map(Into::into)
            .collect(),
        _ => Vec::new(),
    }
}

/// What [`cap_node`] carries down the schema.
struct CapCtx {
    /// The reference to the bounded "any" definition.
    any_ref: Value,
    /// Whether the fraction step may be added (see [`check_schema`]).
    step_ok: bool,
    /// Whether a reference into the schema's definitions sits where llguidance intersects it,
    /// which makes every definition intersected too ([`refs_meet`]).
    defs_intersected: bool,
}

/// Whether llguidance intersects what lies below `obj` with another subschema: an `allOf`'s
/// members meet each other, and an applicator's branches (or a `$ref`'s target) meet the node's
/// own subschema keywords.
fn shares(obj: &Map<String, Value>) -> bool {
    let applied = ["anyOf", "oneOf", "allOf", "$ref"]
        .iter()
        .any(|k| obj.contains_key(*k));
    obj.contains_key("allOf")
        // llguidance intersects a named property with each pattern that matches its name.
        || (obj.contains_key("properties") && obj.contains_key("patternProperties"))
        || (applied
            && SUBSCHEMA_ONES
                .iter()
                .chain(&["properties", "patternProperties", "prefixItems"])
                .any(|k| obj.contains_key(*k)))
}

/// Whether any `$ref` below `schema` sits where llguidance intersects its target with another
/// subschema — then the definitions are treated as intersected throughout.
fn refs_meet(schema: &Value, intersected: bool) -> bool {
    let Some(obj) = schema.as_object() else {
        return false;
    };
    let below = intersected || shares(obj);
    if obj.contains_key("$ref") && below {
        return true;
    }
    let mut met = false;
    each_subschema(obj, &mut |s| met |= refs_meet(s, below));
    met
}

/// [`cap_numbers`] over one subschema and everything below it.
///
/// `intersected` marks a subschema llguidance may intersect with another — everything below an
/// `allOf`, below a node whose applicator branches (or `$ref` target) meet its own subschema
/// keywords, below an object whose named properties meet its pattern properties, and the
/// definitions when a reference to them is intersected. There the rewrite adds no reference to
/// the "any" definition and leaves an open node open: llguidance intersects two such recursive
/// references by expanding them until its intersection depth limit refuses the schema (on the
/// compile thread; on a 2 MiB stack a debug build overflows first), so the rewrite would turn a
/// valid schema into a `400`. Nor does it add number bounds or a step there, or to a node whose
/// own keywords meet its applicators: a bound added to one side of an intersection can contradict
/// the other side's (`allOf: [{"type": "integer"}, {"minimum": 1e17}]`) and refuse a valid schema.
/// The numbers in intersected subschemas therefore stay as the schema bounds them.
fn cap_node(node: &mut Value, ctx: &CapCtx, intersected: bool) {
    let (any_ref, step_ok) = (&ctx.any_ref, ctx.step_ok);
    if matches!(node, Value::Bool(true)) {
        if !intersected {
            *node = any_ref.clone();
        }
        return;
    }
    let Some(obj) = node.as_object_mut() else {
        return;
    };
    if is_open(obj) {
        if !intersected {
            // Replaced whole: a nested `$id` would move where the reference resolves.
            *node = any_ref.clone();
        }
        return;
    }
    let applied = ["anyOf", "oneOf", "allOf"]
        .iter()
        .any(|k| obj.contains_key(*k));
    let below = intersected || shares(obj);
    each_subschema_mut(obj, &mut |key, s| {
        let defs = ctx.defs_intersected && matches!(key, "$defs" | "definitions");
        cap_node(s, ctx, below || defs)
    });
    if ["const", "enum", "$ref"]
        .iter()
        .any(|k| obj.contains_key(*k))
    {
        return;
    }
    let types = schema_types(obj);
    let can_be = |t: &str| types.is_empty() || types.iter().any(|x| x == t);
    // A node with applicators is intersected with their branches, which are bounded themselves.
    if !intersected && !applied {
        if can_be("object")
            && matches!(
                obj.get("additionalProperties"),
                None | Some(Value::Bool(true))
            )
        {
            obj.insert("additionalProperties".into(), any_ref.clone());
        }
        if can_be("array") {
            let rest = if obj.get("items").is_some_and(Value::is_array) {
                "additionalItems"
            } else {
                "items"
            };
            if matches!(obj.get(rest), None | Some(Value::Bool(true))) {
                obj.insert(rest.into(), any_ref.clone());
            }
        }
    }
    if !intersected && !applied && (can_be("integer") || can_be("number")) {
        cap_number(obj, step_ok);
    }
}

/// The schema's own bound on one side: the tighter of the inclusive and exclusive keywords.
fn own_bound(
    obj: &Map<String, Value>,
    inclusive: &str,
    exclusive: &str,
    lower: bool,
) -> Option<f64> {
    let a = obj.get(inclusive).and_then(Value::as_f64);
    let b = obj.get(exclusive).and_then(Value::as_f64);
    match (a, b) {
        (Some(a), Some(b)) => Some(if lower { a.max(b) } else { a.min(b) }),
        (a, b) => a.or(b),
    }
}

/// Bound a numeric schema object as [`cap_numbers`] describes.
fn cap_number(obj: &mut Map<String, Value>, step_ok: bool) {
    let lo = own_bound(obj, "minimum", "exclusiveMinimum", true);
    let hi = own_bound(obj, "maximum", "exclusiveMaximum", false);
    let types = schema_types(obj);
    let integer_only = !types.is_empty() && types.iter().all(|t| t == "integer");
    // An integer's bounds go through llguidance's `i64` arithmetic, so wherever the node may hold
    // one: a bound added here stays in the range `check_schema` admits for the schema's own, so
    // the range is still non-empty.
    let integer = can_be_integer(obj);
    let clamp = |v: f64| {
        if integer {
            v.clamp(-INTEGER_BOUND_LIMIT, INTEGER_BOUND_LIMIT)
        } else {
            v
        }
    };
    // The side the schema leaves open is bounded relative to the side it sets: a range beyond
    // the default one keeps its own digits instead of becoming empty.
    let lo = lo.unwrap_or_else(|| {
        let lo = clamp(hi.map_or(-NUMBER_LIMIT, |hi| (-NUMBER_LIMIT).min(hi - NUMBER_LIMIT)));
        obj.insert("minimum".into(), serde_json::json!(lo));
        lo
    });
    let hi = hi.unwrap_or_else(|| {
        let hi = clamp(NUMBER_LIMIT.max(lo + NUMBER_LIMIT));
        obj.insert("maximum".into(), serde_json::json!(hi));
        hi
    });
    // A range narrower than two steps may hold no multiple of the step at all.
    if !integer_only && step_ok && !obj.contains_key("multipleOf") && hi - lo >= 2.0 * NUMBER_STEP {
        obj.insert("multipleOf".into(), serde_json::json!(NUMBER_STEP));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The mirrors of llguidance's decimal arithmetic agree with what it computes, checked by
    /// hand against its `Decimal::try_from`, `Decimal::new`, `Decimal::lcm` and derivre's
    /// `RemainderIs` step.
    #[test]
    fn the_decimal_mirrors_match_llguidance() {
        assert_eq!(decimal(0.25), Some((25, 2)));
        assert_eq!(decimal(7.0), Some((7, 0)));
        assert_eq!(decimal(NUMBER_STEP), Some((1, 8)));
        assert_eq!(
            decimal(1e-16),
            None,
            "too many digits for llguidance itself"
        );
        assert!(matchable((1, 8)));
        assert!(!matchable((1, 9)), "9 * 10^9 overflows a u32");
        assert!(
            !matchable((500_000_000, 0)),
            "remainder * 10 overflows a u32"
        );
        assert_eq!(lcm((3, 1), (1, 8)), Some((3, 1)), "lcm(0.3, 1e-8) = 0.3");
        assert_eq!(lcm((7, 0), (1, 8)), Some((7, 0)));
        // lcm(7, 0.00000013): 7e8 * 13 overflows llguidance's u32 product.
        assert_eq!(lcm((7, 0), (13, 8)), None);
        // lcm(43, 1e-8): 43 * 10^8 overflows llguidance's u32 before any product.
        assert_eq!(lcm((43, 0), (1, 8)), None);
        assert_eq!(lcm((70_000, 0), (70_000, 0)), None, "x * y overflows a u32");
    }

    /// An open node is one llguidance reads as any value: annotation and unknown keywords only.
    #[test]
    fn a_node_is_open_when_llguidance_would_read_it_as_any_value() {
        let open = |v: Value| is_open(v.as_object().unwrap());
        assert!(open(serde_json::json!({})));
        assert!(open(
            serde_json::json!({"description": "d", "readOnly": true, "$id": "x", "x-a": 1})
        ));
        assert!(!open(serde_json::json!({"items": {}})));
        assert!(!open(serde_json::json!({"minimum": 1})));
    }

    /// The keyword knowledge copied from llguidance holds for the installed version: each
    /// keyword treated as implemented compiles on its own, each known-but-unimplemented one is
    /// refused, and an unknown one is read as any value. An upgrade that changes either list
    /// fails here.
    #[test]
    fn the_keyword_lists_match_the_installed_llguidance() {
        let tok = crate::grammar::tests_support::byte_tokenizer();
        let vocab = tok.get_vocab_size(true);
        let env = crate::grammar::ConstraintEnv::new(&tok, vocab, &[0]).expect("env");
        let compiles = |schema: Value| {
            env.constraint(llguidance::api::TopLevelGrammar::from_json_schema(
                schema.clone(),
            ))
            .map(drop)
            .map_err(|e| format!("{schema}: {e}"))
        };
        let sample = |k: &str| -> Value {
            match k {
                "anyOf" | "oneOf" | "allOf" => serde_json::json!([{"type": "string"}]),
                "const" => serde_json::json!(1),
                "enum" | "required" | "prefixItems" => serde_json::json!(["a"]),
                "type" => serde_json::json!("string"),
                "properties" | "patternProperties" | "dependentSchemas" => {
                    serde_json::json!({"a": {}})
                }
                "pattern" | "$dynamicAnchor" => serde_json::json!("a"),
                "format" => serde_json::json!("date"),
                "multipleOf" => serde_json::json!(2),
                "dependencies" | "dependentRequired" => serde_json::json!({"a": ["b"]}),
                "uniqueItems" | "$recursiveAnchor" => serde_json::json!(true),
                "$recursiveRef" | "$dynamicRef" => serde_json::json!("#"),
                "unevaluatedItems" | "unevaluatedProperties" => serde_json::json!(false),
                k if k.starts_with("min") || k.starts_with("max") || k.starts_with("exclusive") => {
                    serde_json::json!(1)
                }
                _ => serde_json::json!({}),
            }
        };
        for k in LLGUIDANCE_KEYWORDS {
            let schema = match k {
                "$ref" => serde_json::json!({"$ref": "#/$defs/a", "$defs": {"a": {}}}),
                "prefixItems" => serde_json::json!({"prefixItems": [{}]}),
                k => serde_json::json!({ k: sample(k) }),
            };
            compiles(schema).unwrap_or_else(|e| panic!("{k} must compile alone: {e}"));
        }
        for k in REFUSED_KEYWORDS {
            let e = compiles(serde_json::json!({ k: sample(k) }))
                .expect_err(&format!("{k} must be refused"));
            assert!(e.contains(k), "{k}: {e}");
        }
        // The rest of the JSON Schema vocabulary, drafts 4 to 2020-12: llguidance must read each
        // as an annotation or an unknown keyword (any value). Its draft keyword table is private
        // to it (and the `referencing` crate behind it is not a dependency here), so this is the
        // check within reach: an upgrade that starts refusing one of these, or implements one of
        // the refused list above, fails; one that starts implementing an annotation is not seen.
        let vocabulary_rest: [(&str, Value); 18] = [
            (
                "$schema",
                serde_json::json!("https://json-schema.org/draft/2020-12/schema"),
            ),
            ("$id", serde_json::json!("urn:x")),
            ("id", serde_json::json!("urn:x")),
            ("$anchor", serde_json::json!("a")),
            ("$defs", serde_json::json!({})),
            ("definitions", serde_json::json!({})),
            ("$comment", serde_json::json!("c")),
            ("$vocabulary", serde_json::json!({})),
            ("contentEncoding", serde_json::json!("base64")),
            ("contentMediaType", serde_json::json!("text/plain")),
            ("contentSchema", serde_json::json!({})),
            ("title", serde_json::json!("t")),
            ("description", serde_json::json!("d")),
            ("default", serde_json::json!(1)),
            ("deprecated", serde_json::json!(true)),
            ("readOnly", serde_json::json!(true)),
            ("writeOnly", serde_json::json!(true)),
            ("examples", serde_json::json!([1])),
        ];
        for (k, v) in vocabulary_rest {
            assert!(
                !LLGUIDANCE_KEYWORDS.contains(&k) && !REFUSED_KEYWORDS.contains(&k),
                "{k} is listed twice"
            );
            compiles(serde_json::json!({ k: v }))
                .unwrap_or_else(|e| panic!("{k} must read as any value: {e}"));
            assert!(
                is_open(serde_json::json!({ k: v }).as_object().unwrap()),
                "{k}"
            );
        }
        compiles(serde_json::json!({"x-unknown": 1})).expect("an unknown keyword is any value");
        assert!(is_open(
            serde_json::json!({"x-unknown": 1}).as_object().unwrap()
        ));
        assert!(!is_open(
            serde_json::json!({"not": {}}).as_object().unwrap()
        ));
    }

    /// derivre starts its remainder at the divisor: the largest step it can match at exponent 0
    /// passes, the next does not.
    #[test]
    fn the_multiple_of_boundary_is_derivres() {
        let largest = (u64::from(u32::MAX) - 9) / 10;
        assert!(matchable((largest, 0)), "{largest}");
        assert!(!matchable((largest + 1, 0)), "{}", largest + 1);
        assert_eq!(largest + 1, 429_496_729);
    }
}
