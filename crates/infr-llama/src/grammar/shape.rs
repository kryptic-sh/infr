//! What a schema must look like before llguidance compiles it: references it can be shown to
//! follow, one dialect, and a size and depth its compile survives.
//!
//! llguidance builds a schema by recursion — one chain of stack frames per level of nesting and
//! per `$ref` followed — and a stack overflow aborts the whole process, not the request. Every
//! schema this crate compiles passes [`check_shape`] first, on the exact JSON compiled.

use std::collections::{HashMap, HashSet};

use serde_json::{Map, Value};

use super::GrammarRejected;

/// The longest path a schema may make llguidance's compile descend: levels of nesting plus each
/// `$ref` followed, along the longest path through the reference graph (see [`check_shape`]).
/// Nothing a person writes by hand nests this deep; it is the first of two bounds, the second
/// being [`MAX_COMPILE_LEVELS`].
pub(super) const MAX_SCHEMA_DEPTH: usize = 256;

/// The backstop bound: the root's own nesting plus, for every distinct `$ref` target, that
/// target's own nesting and one hop. llguidance compiles a target at most once along any chain,
/// so this sum bounds how deep its compile can go however the references are arranged or walked.
///
/// The arithmetic, from this llguidance measured in a debug build — the costlier per level — on
/// the compile thread's 64 MiB stack, unguarded, with levels counted as this bound counts them:
///
/// | definitions chained by `$ref`, each made of | compiled | overflowed |
/// |---|---|---|
/// | nested `properties` | 2048 | 4096 |
/// | nested `anyOf` with a null branch | 2640 | 3168 |
/// | nested `allOf` of two schemas | 3168 | 4224 |
/// | nested `allOf` of `properties` and `required` | 3120 | 4160 |
///
/// The costliest, `anyOf`, overflowed below 3168 levels but held 2640, so a level there costs at
/// most 64 MiB / 2640 ≈ 25 KiB. This bound at 25 KiB a level is 25 MiB, under two fifths of the
/// stack. A release build held 4096 levels of nested `properties` and overflowed at 16384.
pub(super) const MAX_COMPILE_LEVELS: usize = 1024;

/// The most JSON values (every object, array, string, number… anywhere in it, data and unknown
/// keys included) a schema handed to llguidance may hold: a bound on its compile's work.
pub(super) const MAX_JSON_VALUES: usize = 100_000;

/// The most `$ref`s a schema may contain at schema positions, not counting the rewrite's own
/// references to its bounded "any" definition.
pub(super) const MAX_REFS: usize = 1_000;

/// The dialects accepted, by the `$schema` URI that names them (with or without a trailing `#`,
/// as `referencing` reads it), each with the keywords refused under it: those 2020-12 knows —
/// whether llguidance compiles them or refuses them — that the dialect does not, or reads
/// differently.
///
/// Accepted are the dialects in which, once [`IDENTITY_KEYWORDS`] are refused, nothing names a
/// resource or moves the base a reference resolves against, so every `#/…` resolves against the
/// document root: 2020-12 (also llguidance's default when `$schema` is absent) and 2019-09 name
/// resources by `$id` and `$anchor`, draft-06 and draft-07 by `$id` alone. Draft-04 is left out:
/// its plain `id` — a word also used as a property name — moves the base.
///
/// llguidance reads a keyword its dialect does not know as an annotation, so in an older dialect
/// a node holding only such keywords would compile as "any value" while the rewrite here, and a
/// tool's schema once embedded in 2020-12 (see [`drop_dialect`]), would read them as keywords.
/// The listed ones are refused instead; the rest of the difference is annotations to llguidance
/// in every dialect, or identity keywords, refused everywhere.
const DIALECTS: [(&str, &[&str]); 4] = [
    ("https://json-schema.org/draft/2020-12/schema", &[]),
    (
        "https://json-schema.org/draft/2019-09/schema",
        // Known to 2019-09, but llguidance reads `items` and `additionalItems` there instead.
        &["prefixItems"],
    ),
    ("http://json-schema.org/draft-07/schema", &NEWER_THAN_07),
    ("http://json-schema.org/draft-06/schema", &NEWER_THAN_06),
];
/// Keywords 2020-12 knows and draft-07 does not, besides annotations and identity keywords.
const NEWER_THAN_07: [&str; 7] = [
    "dependentRequired",
    "dependentSchemas",
    "maxContains",
    "minContains",
    "prefixItems",
    "unevaluatedItems",
    "unevaluatedProperties",
];
/// Keywords 2020-12 knows and draft-06 does not: draft-07's, and the conditionals.
const NEWER_THAN_06: [&str; 10] = [
    "dependentRequired",
    "dependentSchemas",
    "maxContains",
    "minContains",
    "prefixItems",
    "unevaluatedItems",
    "unevaluatedProperties",
    "if",
    "then",
    "else",
];

/// Keywords that name a resource, switch the base a reference resolves against, or change the
/// dialect: refused at every schema position (`$schema` everywhere but the root, where one of
/// [`DIALECTS`] is accepted).
const IDENTITY_KEYWORDS: [&str; 8] = [
    "$id",
    "$anchor",
    "$dynamicAnchor",
    "$dynamicRef",
    "$recursiveRef",
    "$recursiveAnchor",
    "$vocabulary",
    "$schema",
];

/// Keywords whose value is one subschema (`items` also a list of them, in older drafts): the union
/// of what llguidance compiles and what `referencing`'s 2020-12 walk visits.
const SCHEMA_ONE: [&str; 12] = [
    "additionalProperties",
    "additionalItems",
    "contains",
    "contentSchema",
    "else",
    "if",
    "items",
    "not",
    "propertyNames",
    "then",
    "unevaluatedItems",
    "unevaluatedProperties",
];
/// Keywords whose value is a list of subschemas.
const SCHEMA_LIST: [&str; 4] = ["allOf", "anyOf", "oneOf", "prefixItems"];
/// Keywords whose value maps names to subschemas — the keys are names, not keywords.
const SCHEMA_MAP: [&str; 6] = [
    "$defs",
    "definitions",
    "dependentSchemas",
    "dependencies",
    "patternProperties",
    "properties",
];
/// Map keywords llguidance does not compile in place: definitions are reached only by `$ref`.
const DEFINITIONS: [&str; 2] = ["$defs", "definitions"];

/// The subschemas directly below `obj`: every value at a schema position. Everything else in a
/// schema object — `const`, `enum`, `default`, `examples`, unknown keys — is data. A container of
/// the wrong kind contributes nothing here; [`malformed`] refuses it before this is relied on.
fn subschemas(obj: &Map<String, Value>, with_definitions: bool) -> Vec<&Value> {
    let mut out = Vec::new();
    for (k, v) in obj {
        let k = k.as_str();
        if SCHEMA_ONE.contains(&k) {
            match v {
                Value::Array(items) if k == "items" => out.extend(items),
                v => out.push(v),
            }
        } else if SCHEMA_LIST.contains(&k) {
            if let Value::Array(items) = v {
                out.extend(items);
            }
        } else if SCHEMA_MAP.contains(&k) && (with_definitions || !DEFINITIONS.contains(&k)) {
            if let Value::Object(m) = v {
                out.extend(m.values());
            }
        }
    }
    out
}

/// The first schema keyword of `obj` whose value is no container of its kind: a list keyword
/// holding no array, a map keyword no object, a one-schema keyword neither a schema (object or
/// boolean) nor — `items` only — an array. Its insides would be neither walked nor data.
fn malformed(obj: &Map<String, Value>) -> Option<&str> {
    obj.iter().find_map(|(k, v)| {
        let k = k.as_str();
        let ok = if SCHEMA_LIST.contains(&k) {
            v.is_array()
        } else if SCHEMA_MAP.contains(&k) {
            v.is_object()
        } else if SCHEMA_ONE.contains(&k) {
            v.is_object() || v.is_boolean() || (k == "items" && v.is_array())
        } else {
            true
        };
        (!ok).then_some(k)
    })
}

/// Call `f` on every schema object reachable from `root` through schema positions — the root
/// included, with whether it is the root — without recursion.
fn each_schema_object(root: &Value, f: &mut dyn FnMut(&Map<String, Value>, bool)) {
    let mut stack = vec![(root, true)];
    while let Some((v, is_root)) = stack.pop() {
        if let Value::Object(obj) = v {
            f(obj, is_root);
            stack.extend(subschemas(obj, true).into_iter().map(|s| (s, false)));
        }
    }
}

/// [`each_schema_object`] over `root`'s schema objects, for the checks outside this module.
pub(super) fn for_each_schema_object(root: &Value, f: &mut dyn FnMut(&Map<String, Value>)) {
    each_schema_object(root, &mut |obj, _| f(obj));
}

/// Every value at a schema position of `root`, the root included, by address: the nodes
/// [`each_schema_object`] walks, and the boolean schemas among them.
fn schema_positions(root: &Value) -> HashSet<*const Value> {
    let mut seen = HashSet::new();
    let mut stack = vec![root];
    while let Some(v) = stack.pop() {
        seen.insert(v as *const Value);
        if let Value::Object(obj) = v {
            stack.extend(subschemas(obj, true));
        }
    }
    seen
}

/// Refuse a schema llguidance's compile could not be shown to survive, before anything recurses
/// over it — this crate's own passes or the compile, whose stack overflow would abort the server.
/// Run on the very JSON handed to llguidance as well as on the request's own schema.
///
/// Sound by construction rather than by listing what to watch:
///
/// * **Schema positions, walked.** The check walks every value at a schema position: the values
///   of the subschema keywords this module lists — the union of what llguidance compiles and
///   what `referencing` visits in every accepted dialect — and their lists and maps. A keyword
///   holding the wrong kind of container is refused ([`malformed`]), so nothing at a schema
///   position escapes the walk.
/// * **Root-relative references.** `$schema` may only name one of [`DIALECTS`], and only at the
///   root; the keywords listed there against that dialect are refused; and no walked node may
///   hold a keyword that names a resource, moves a reference's base or switches dialect
///   ([`IDENTITY_KEYWORDS`]). So every reference resolves against the document root.
/// * **References into the walk.** Every `$ref` on a walked node must be `#` or `#/` and a plain
///   JSON pointer (no `%`, which `referencing` would decode and this check would not) that
///   resolves to a node the walk visited. So every target is itself walked: its own references
///   met the same rules, and the depth check follows only checked references.
/// * **Everything counted.** Every JSON value anywhere, data included, up to [`MAX_JSON_VALUES`]
///   and nested at most [`MAX_COMPILE_LEVELS`] deep; the walked references up to [`MAX_REFS`].
/// * **Depth, twice.** The longest path through the reference graph — each target's own nesting
///   plus the references inside it, strongly connected targets bounded by the sum of their
///   nesting (a path enters each at most once) — up to [`MAX_SCHEMA_DEPTH`]; and the
///   order-independent sum over distinct targets up to [`MAX_COMPILE_LEVELS`].
///
/// Data is data: names under `properties` and `$defs`, and the values of `const`, `enum`,
/// `default`, `examples` and unknown keys, may hold `$ref` or `$id` freely. llguidance compiles
/// none of them, and no reference may point into them.
pub(super) fn check_shape(schema: &Value) -> Result<(), GrammarRejected> {
    let refuse = |message: String| GrammarRejected {
        message,
        in_tools: false,
    };
    let mut values = 0usize;
    let mut stack = vec![(schema, 1usize)];
    while let Some((v, level)) = stack.pop() {
        values += 1;
        if values > MAX_JSON_VALUES {
            return Err(refuse(format!(
                "the schema has more than {MAX_JSON_VALUES} JSON values"
            )));
        }
        // This crate's own passes recurse over every subschema, referenced or not.
        if level > MAX_COMPILE_LEVELS {
            return Err(refuse(too_deep()));
        }
        match v {
            Value::Object(obj) => stack.extend(obj.values().map(|c| (c, level + 1))),
            Value::Array(items) => stack.extend(items.iter().map(|c| (c, level + 1))),
            _ => {}
        }
    }

    let any_ref = super::caps::ANY_REF;
    let positions = schema_positions(schema);
    // A malformed container first: the reference into it is only its symptom.
    let mut err = None;
    each_schema_object(schema, &mut |obj, _| {
        if let Some(k) = malformed(obj) {
            err.get_or_insert(format!("`{k}` holds the wrong kind of value"));
        }
    });
    if let Some(message) = err {
        return Err(refuse(message));
    }
    let dialect = match schema.get("$schema") {
        None => Some(&DIALECTS[0]),
        Some(declared) => declared.as_str().and_then(|d| {
            DIALECTS
                .iter()
                .find(|(uri, _)| d.trim_end_matches('#') == *uri)
        }),
    };
    let mut refs = 0usize;
    each_schema_object(schema, &mut |obj, is_root| {
        if let Some((uri, unknown)) = dialect {
            if let Some(k) = unknown.iter().find(|k| obj.contains_key(**k)) {
                err.get_or_insert(format!("`{k}` is not part of the schema's dialect, {uri}"));
            }
        }
        for k in IDENTITY_KEYWORDS {
            let Some(v) = obj.get(k) else { continue };
            let pinned = k == "$schema" && is_root && dialect.is_some();
            if !pinned {
                err.get_or_insert(if k == "$schema" {
                    format!(
                        "`$schema` {v} is not supported: only draft-06, draft-07, 2019-09 or \
                         2020-12, at the root"
                    )
                } else {
                    format!("`{k}` is not supported: only local `#/…` references are")
                });
            }
        }
        if let Some(target) = obj.get("$ref") {
            if target.as_str() != Some(any_ref) {
                refs += 1;
            }
            // The target must be a node this walk visits, so the rules here held inside it too.
            let resolved = target
                .as_str()
                .and_then(|t| t.strip_prefix('#'))
                .filter(|p| (p.is_empty() || p.starts_with('/')) && !p.contains('%'))
                .and_then(|p| schema.pointer(p))
                .is_some_and(|t| positions.contains(&(t as *const Value)));
            if !resolved {
                err.get_or_insert(format!(
                    "`$ref` {target} is not supported: only local `#/…` references to a \
                     subschema in the same schema are"
                ));
            }
        }
    });
    if let Some(message) = err {
        return Err(refuse(message));
    }
    if refs > MAX_REFS {
        return Err(refuse(format!(
            "the schema has more than {MAX_REFS} `$ref`s"
        )));
    }
    check_depth(schema).map_err(refuse)
}

/// One vertex of the reference graph — the root or a `$ref` target: its own nesting (levels of
/// compiled subschemas below and including it, references not followed) and the references inside
/// it, each with the level it sits at.
struct Vertex {
    own: usize,
    refs: Vec<(usize, usize)>,
}

/// The two depth bounds of [`check_shape`], on a schema whose references all resolve.
fn check_depth(schema: &Value) -> Result<(), String> {
    // Vertices by node identity: one per distinct target, however many pointers spell it.
    let mut index: HashMap<*const Value, usize> = HashMap::new();
    let mut nodes: Vec<&Value> = vec![schema];
    index.insert(schema as *const Value, 0);
    let mut vertices: Vec<Vertex> = Vec::new();
    let mut next = 0;
    while next < nodes.len() {
        let root = nodes[next];
        let mut own = 0;
        let mut refs = Vec::new();
        let mut stack = vec![(root, 1usize)];
        while let Some((v, level)) = stack.pop() {
            let Value::Object(obj) = v else { continue };
            own = own.max(level);
            if own > MAX_COMPILE_LEVELS {
                return Err(too_deep());
            }
            if let Some(p) = obj
                .get("$ref")
                .and_then(Value::as_str)
                .and_then(|t| t.strip_prefix('#'))
            {
                if let Some(target) = schema.pointer(p) {
                    let id = *index.entry(target as *const Value).or_insert_with(|| {
                        nodes.push(target);
                        nodes.len() - 1
                    });
                    refs.push((level, id));
                }
            }
            stack.extend(subschemas(obj, false).into_iter().map(|s| (s, level + 1)));
        }
        vertices.push(Vertex { own, refs });
        next += 1;
    }

    // The backstop: no graph reasoning, just the sum.
    let backstop: usize = vertices[0].own + vertices[1..].iter().map(|v| v.own + 1).sum::<usize>();
    if backstop > MAX_COMPILE_LEVELS {
        return Err(too_deep());
    }

    // The longest path: strongly connected components (iterative Tarjan), whose emission order
    // is reverse topological, so every successor's height is known when a component is done.
    let n = vertices.len();
    let mut height = vec![0usize; n];
    let mut order = vec![usize::MAX; n];
    let mut low = vec![0usize; n];
    let mut on_stack = vec![false; n];
    let mut scc_stack: Vec<usize> = Vec::new();
    let mut counter = 0;
    for start in 0..n {
        if order[start] != usize::MAX {
            continue;
        }
        let mut call: Vec<(usize, usize)> = vec![(start, 0)];
        order[start] = counter;
        low[start] = counter;
        counter += 1;
        scc_stack.push(start);
        on_stack[start] = true;
        while let Some(&mut (v, ref mut edge)) = call.last_mut() {
            if let Some(&(_, w)) = vertices[v].refs.get(*edge) {
                *edge += 1;
                if order[w] == usize::MAX {
                    order[w] = counter;
                    low[w] = counter;
                    counter += 1;
                    scc_stack.push(w);
                    on_stack[w] = true;
                    call.push((w, 0));
                } else if on_stack[w] {
                    low[v] = low[v].min(order[w]);
                }
                continue;
            }
            call.pop();
            if let Some(&(parent, _)) = call.last() {
                low[parent] = low[parent].min(low[v]);
            }
            if low[v] != order[v] {
                continue;
            }
            let mut members = Vec::new();
            loop {
                let w = scc_stack.pop().expect("v is on the stack");
                on_stack[w] = false;
                members.push(w);
                if w == v {
                    break;
                }
            }
            let inside = |w: usize| members.contains(&w);
            let h = if members.len() == 1 {
                // A reference back to itself is a cycle llguidance stops; the rest are exact.
                let m = members[0];
                vertices[m]
                    .refs
                    .iter()
                    .filter(|&&(_, t)| t != m)
                    .map(|&(level, t)| level + height[t])
                    .fold(vertices[m].own, usize::max)
            } else {
                // A path enters each member at most once: their nesting summed, then the deepest
                // way out.
                let within: usize = members.iter().map(|&m| vertices[m].own).sum();
                let out = members
                    .iter()
                    .flat_map(|&m| vertices[m].refs.iter())
                    .filter(|&&(_, t)| !inside(t))
                    .map(|&(level, t)| level + height[t])
                    .max()
                    .unwrap_or(0);
                within + out
            };
            for &m in &members {
                height[m] = h;
            }
        }
    }
    if height[0] > MAX_SCHEMA_DEPTH {
        return Err(format!(
            "the schema nests deeper than {MAX_SCHEMA_DEPTH} levels, counting `$ref`s followed"
        ));
    }
    Ok(())
}

fn too_deep() -> String {
    format!(
        "the schema's definitions together nest deeper than {MAX_COMPILE_LEVELS} levels, \
         counting `$ref`s followed"
    )
}

/// Remove the root `$schema` of a schema that passed [`check_shape`], so it can sit below the
/// root of another document compiled as 2020-12.
///
/// What llguidance compiles reads the same in every accepted dialect once [`check_shape`] has
/// passed under the declared one:
///
/// * `items` as a list and `additionalItems`: llguidance reads them the older way in 2020-12 too
///   whenever either form appears (a list `items`, or any `additionalItems`), so a draft-07 tuple
///   is the same tuple embedded.
/// * `prefixItems` is the one array keyword read differently, and it is refused under every
///   dialect before 2020-12.
/// * `definitions` and `$defs` are annotations to llguidance in every dialect: they hold targets
///   for references, which resolve by plain pointer whatever the dialect.
/// * `dependencies`, `contains`, `if`/`then`/`else`, `propertyNames`: keywords every dialect that
///   has them shares with 2020-12, and llguidance refuses each under both, so nothing changes.
/// * The keywords only 2020-12 or 2019-09 know (`dependentSchemas`, `unevaluated*`,
///   `min`/`maxContains`…) would turn from annotation into keyword: refused under the older
///   dialects, they never reach this.
pub(super) fn drop_dialect(schema: &mut Value) {
    if let Value::Object(obj) = schema {
        obj.remove("$schema");
    }
}

/// Move every local `$ref` at a schema position of `schema` from its own root to `at`, the JSON
/// pointer where `schema` sits inside the document llguidance compiles — so a tool's `#/$defs/…`
/// still names the tool's own definitions once its parameters are placed inside the call schema.
/// A `$ref` in data (a `const`, a `default`…) is data, and kept as it is.
pub(super) fn relocate_refs(schema: &mut Value, at: &str) {
    let mut stack = vec![schema];
    while let Some(v) = stack.pop() {
        let Value::Object(obj) = v else { continue };
        if let Some(Value::String(r)) = obj.get_mut("$ref") {
            if let Some(rest) = r.strip_prefix('#') {
                *r = format!("#{at}{rest}");
            }
        }
        for (k, child) in obj.iter_mut() {
            let k = k.as_str();
            let one = SCHEMA_ONE.contains(&k);
            if (one && !child.is_array()) || (one && k != "items") {
                stack.push(child);
            } else if one || SCHEMA_LIST.contains(&k) {
                if let Value::Array(items) = child {
                    stack.extend(items.iter_mut());
                }
            } else if SCHEMA_MAP.contains(&k) {
                if let Value::Object(m) = child {
                    stack.extend(m.values_mut());
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(schema: Value) -> Result<(), String> {
        check_shape(&schema).map_err(|e| e.message)
    }

    /// `$ref` and `$id` in data and in names are not keywords: accepted, and still counted.
    #[test]
    fn data_and_names_are_not_schema() {
        for ok in [
            serde_json::json!({"properties": {"$id": {"type": "string"}}}),
            serde_json::json!({"properties": {"id": {"type": "string"}}}),
            serde_json::json!({"properties": {"$ref": {"type": "string"}}}),
            serde_json::json!({"properties": {"$schema": {"type": "string"}}}),
            serde_json::json!({"default": {"$ref": "x"}}),
            serde_json::json!({"const": {"$ref": "#", "$id": "urn:x"}}),
            serde_json::json!({"enum": [{"$ref": "#/nowhere"}, {"$anchor": "a"}]}),
            serde_json::json!({"examples": [{"$id": "urn:y"}]}),
            serde_json::json!({"$schema": "https://json-schema.org/draft/2020-12/schema#"}),
            serde_json::json!({"$schema": "https://json-schema.org/draft/2020-12/schema"}),
            serde_json::json!({"$schema": "http://json-schema.org/draft-07/schema#",
                "if": {}, "then": {}, "definitions": {"a": {}}}),
            serde_json::json!({"prefixItems": [{}], "unevaluatedProperties": false}),
        ] {
            assert_eq!(shape(ok.clone()), Ok(()), "{ok}");
        }
        for (bad, why) in [
            (
                serde_json::json!({"$schema": "http://json-schema.org/draft-04/schema#"}),
                "`$schema`",
            ),
            (
                serde_json::json!({"$schema": "https://json-schema.org/draft/2021-01/schema"}),
                "`$schema`",
            ),
            (serde_json::json!({"$schema": 7}), "`$schema`"),
            (
                serde_json::json!({"$schema": "http://json-schema.org/draft-07/schema#",
                    "properties": {"a": {"prefixItems": [{}]}}}),
                "`prefixItems` is not part",
            ),
            (
                serde_json::json!({"$schema": "http://json-schema.org/draft-06/schema",
                    "if": {}, "then": {}}),
                "is not part",
            ),
            (
                serde_json::json!({"properties": {"a": {
                    "$schema": "https://json-schema.org/draft/2020-12/schema"}}}),
                "`$schema`",
            ),
            (
                serde_json::json!({"properties": {"a": {"$vocabulary": {}}}}),
                "`$vocabulary`",
            ),
            (
                serde_json::json!({"properties": {"a": {"$ref": "#/default"}},
                    "default": {"type": "string"}}),
                "`$ref`",
            ),
            (
                serde_json::json!({"properties": {"a": {"$ref": "#/$defs/none"}}}),
                "`$ref`",
            ),
        ] {
            let got = shape(bad.clone()).expect_err(&bad.to_string());
            assert!(got.contains(why), "{bad}: {got}");
        }
        // Data still counts toward the value limit.
        let big: Vec<Value> = (0..=MAX_JSON_VALUES).map(Value::from).collect();
        let got = shape(serde_json::json!({"const": big})).unwrap_err();
        assert!(got.contains("JSON values"), "{got}");
    }

    /// References must land on a walked subschema: into data, under an unknown key, or into a
    /// container of the wrong kind, they are refused; a malformed container is refused itself.
    #[test]
    fn a_reference_must_land_on_a_walked_subschema() {
        for ok in [
            serde_json::json!({"$ref": "#"}),
            serde_json::json!({"$ref": "#/$defs/A", "$defs": {"A": {"type": "string"}}}),
            serde_json::json!({"$ref": "#/$defs/a~1b", "$defs": {"a/b": {}}}),
            serde_json::json!({"properties": {"a": {"items": {}}},
                "$ref": "#/properties/a/items"}),
            serde_json::json!({"items": [{}, true], "$ref": "#/items/1"}),
            serde_json::json!({"anyOf": [{}, {"properties": {"b": {}}}],
                "$ref": "#/anyOf/1/properties/b"}),
        ] {
            assert_eq!(shape(ok.clone()), Ok(()), "{ok}");
        }
        for (bad, why) in [
            (
                serde_json::json!({"properties": {}, "$ref": "#/properties"}),
                "`$ref`",
            ),
            (
                serde_json::json!({"default": {}, "$ref": "#/default"}),
                "`$ref`",
            ),
            (
                serde_json::json!({"x-unknown": {"A": {}}, "$ref": "#/x-unknown/A"}),
                "`$ref`",
            ),
            (
                serde_json::json!({"const": {"a": {}}, "$ref": "#/const/a"}),
                "`$ref`",
            ),
            // `items` as an object is one schema; its key `0` is unknown, so data.
            (
                serde_json::json!({"$ref": "#/items/0", "items": {"0": {}}}),
                "`$ref`",
            ),
            // Malformed containers: the shapes whose insides the walk would skip.
            (
                serde_json::json!({"$ref": "#/anyOf/0", "anyOf": {"0": {}}}),
                "`anyOf`",
            ),
            (
                serde_json::json!({"$ref": "#/properties/0", "properties": [{}]}),
                "`properties`",
            ),
            (serde_json::json!({"items": "string"}), "`items`"),
            (serde_json::json!({"not": [{}]}), "`not`"),
        ] {
            let got = shape(bad.clone()).expect_err(&bad.to_string());
            assert!(got.contains(why), "{bad}: {got}");
        }
    }

    /// A Pydantic-style schema — dozens of models, each a handful of fields, referencing each
    /// other — passes both depth bounds.
    #[test]
    fn a_realistic_model_schema_passes() {
        const MODELS: usize = 40;
        let mut defs = Map::new();
        for i in 0..MODELS {
            let mut props = serde_json::json!({
                "id": {"title": "Id", "type": "integer"},
                "name": {"title": "Name", "type": "string"},
                "tags": {"title": "Tags", "type": "array", "items": {"type": "string"}},
                "note": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": null},
            });
            if i + 1 < MODELS {
                props["child"] = serde_json::json!({"$ref": format!("#/$defs/M{}", i + 1)});
                props["children"] = serde_json::json!({"type": "array",
                    "items": {"$ref": format!("#/$defs/M{}", (i + 2).min(MODELS - 1))}});
            }
            defs.insert(
                format!("M{i}"),
                serde_json::json!({"title": format!("M{i}"), "type": "object",
                    "properties": props, "required": ["id", "name"]}),
            );
        }
        let schema = serde_json::json!({"$defs": defs, "title": "Root", "type": "object",
            "properties": {"root": {"$ref": "#/$defs/M0"}}, "required": ["root"]});
        assert_eq!(shape(schema), Ok(()));
    }

    /// The relocation rewrites references at schema positions only: a `$ref` in a constant keeps
    /// its value.
    #[test]
    fn relocation_leaves_data_alone() {
        let mut s = serde_json::json!({"properties": {
            "a": {"$ref": "#/$defs/A"},
            "c": {"const": {"$ref": "#"}},
            "d": {"default": {"$ref": "#"}}
        }, "$defs": {"A": {"items": {"$ref": "#"}}}});
        relocate_refs(&mut s, "/x");
        assert_eq!(s["properties"]["a"]["$ref"], "#/x/$defs/A");
        assert_eq!(s["$defs"]["A"]["items"]["$ref"], "#/x");
        assert_eq!(s["properties"]["c"]["const"]["$ref"], "#");
        assert_eq!(s["properties"]["d"]["default"]["$ref"], "#");
    }
}
