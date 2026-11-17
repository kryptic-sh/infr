"""Generate `jinja2_text.json`: how Python Jinja2 turns values into text in a chat template.

Each row is a template, the variables it is rendered with, and what Python's Jinja2 printed (or
the error it raised). `tests/jinja2_text.rs` renders every row through infr and checks it, so the
test needs no Python. Regenerate after changing the cases below:

    python -m pip install --user "jinja2>=3.1.6"
    python crates/infr-chat/tests/fixtures/jinja2_text.py

It reads the whole templates it renders (`TEMPLATE_RENDERS`) from its own directory, so it must
stay beside those sibling `.jinja` files.

The environment is the one Hugging Face transformers renders chat templates with
(`_cached_compile_jinja_template` in `transformers/utils/chat_template_utils.py`, read in
transformers 5.22.2): an `ImmutableSandboxedEnvironment` with `trim_blocks` and `lstrip_blocks`,
the `loopcontrols` extension, its own `tojson` (`json.dumps` with `ensure_ascii=False`),
`raise_exception` and `strftime_now`. Its `generation` tag (`AssistantTracker`) is kept as a
tag that renders its body, which is what it does when no assistant mask is asked for; the
offsets it records then are not needed here.

Every row is rendered with the variables infr's `render_template` passes. The row's own variables
travel as `tools` and the template reads them as `v` (`{% set v = tools %}`).
"""

import json
import pathlib
import sys
import unicodedata
from datetime import datetime

import jinja2
import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment

OUT = pathlib.Path(__file__).with_name("jinja2_text.json")


class Generation(jinja2.ext.Extension):
    """transformers' `AssistantTracker` with no tracker active: `{% generation %}` renders its
    body unchanged."""

    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
        return jinja2.nodes.CallBlock(self.call_method("_body"), [], [], body).set_lineno(lineno)

    def _body(self, caller):
        return caller()


def make_env():
    """The environment `apply_chat_template` builds."""

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(
            x,
            ensure_ascii=ensure_ascii,
            indent=indent,
            separators=separators,
            sort_keys=sort_keys,
        )

    def strftime_now(format):
        return datetime.now().strftime(format)

    env = ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=[Generation, jinja2.ext.loopcontrols],
    )
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = strftime_now
    return env


def render(env, template, variables):
    """Render with the variables `infr_chat::render_template` passes."""
    return env.from_string(template).render(
        messages=[],
        tools=variables,
        add_generation_prompt=False,
        bos_token="",
        eos_token="",
        enable_thinking=True,
    )


# Values reaching a template from the request (JSON), read as `v.<name>`.
VARS = {
    "t": True,
    "f": False,
    "n": None,
    "i": 7,
    "neg": -42,
    "big": 18446744073709551615,
    "min": -9223372036854775808,
    "half": 2.5,
    "two": 2.0,
    "nz": -0.0,
    "tiny": 1e-07,
    "small": 0.0001,
    "smaller": 1e-05,
    "e15": 1e15,
    "e16": 1e16,
    "below16": 9999999999999998.0,
    "e22": 1e22,
    "third": 0.3333333333333333,
    # Two shortest digit strings are equally close: Python picks the even one.
    "tie1": 1125899906842624.25,
    "tie2": 1324350514534.90625,
    "tie3": 73637923426838.62,
    "s": "it's",
    "dq": 'say "hi"',
    "both": "both ' and \"",
    "esc": "back\\slash\nnew\ttab\r\x01\x7f",
    "uni": "café 日本 \U0001f600",
    # Python escapes these in repr (not printable): NBSP, ideographic space, soft hyphen,
    # zero-width space, BOM, line and paragraph separators, a private-use and a C1 control.
    "unprintable": " 　­​﻿  \x85",
    "empty_list": [],
    "empty_obj": {},
    "list": [True, None, 1e-07, 'q"uote', 2.0, [1, "x"]],
    "obj": {"b": {"c": [1, "x", True, None, 2.5]}, "a": "it's", "z": {}},
    "ptype": ["string", "null"],
    "schema": {"type": "array", "items": {"type": "string"}, "keys": 1, "get": 2},
    "args": {
        "city": "Paris",
        "unit": None,
        "days": 3,
        "precise": True,
        "ratio": 0.5,
        "tiny": 1e-07,
        "huge": 1e16,
        "filters": {"min": 1.0, "tags": ["a", "b"]},
    },
}



def nonprinting():
    """Every assigned character `repr` escapes (private use by its range ends), each followed by
    its assigned, printable neighbours, so a rule that escapes too much fails as well."""
    escaped = {"Cc", "Cf", "Zs", "Zl", "Zp"}
    private_use_ends = [0xE000, 0xF8FF, 0xF0000, 0xFFFFD, 0x100000, 0x10FFFD]
    out = []
    for c in range(0x110000):
        cat = unicodedata.category(chr(c))
        if cat in escaped or c in private_use_ends:
            out.append(chr(c))
            for n in (c - 1, c + 1):
                if not 0 <= n < 0x110000:
                    continue
                ch = chr(n)
                if unicodedata.category(ch) != "Cn" and ch.isprintable():
                    out.append(ch)
    return "".join(out)


# Read only inside a container (`repr`), not through every construct.
REPR_VARS = {
    "nonprinting": nonprinting(),
    # Unassigned in Unicode 17.0 (category Cn).
    "unassigned": "͸",
    # Everything Python's `str.isspace` is true for (`str.strip()` strips these).
    "spaces": "".join(chr(c) for c in range(0x110000) if chr(c).isspace()),
}

# Every way a template turns a value into text, applied to each value.
CONSTRUCTS = [
    "{{ X }}",
    "{{ X | string }}",
    "{{ 'p' ~ X }}",
    "{{ X ~ 'p' }}",
    "{{ [X] | join(',') }}",
    "{{ '{}'.format(X) }}",
    "{{ '%s' | format(X) }}",
    "{{ [X] }}",
    "{{ {'k': X} }}",
    "{{ X | string | length }}",
    "{{ X | tojson }}",
]

# Values written in the template itself, and from the request.
VALUES = [
    "true",
    "false",
    "none",
    "0",
    "-7",
    "2.0",
    "2.5",
    "-0.0",
    "1e16",
    "1.5e-7",
    "(0.1 + 0.2)",
    "(7 / 2)",
    "'it\\'s'",
    "[1, 'a', true, none]",
    "{'a': 1, 'b': [none]}",
    "[]",
    "{}",
    "undefined_name",
] + ["v." + name for name in VARS]

# Constructs that are not a value run through every construct.
EXTRA = [
    # Comparisons and arithmetic, printed.
    "{{ 1 == 1 }}",
    "{{ 1 > 2 }}",
    "{{ 'a' in 'abc' }}",
    "{{ none is none }}",
    "{{ 7 / 2 }}",
    "{{ 6 / 3 }}",
    "{{ 7 // 2 }}",
    "{{ -7 // 2 }}",
    "{{ 7.5 // 2 }}",
    "{{ 7 % 3 }}",
    "{{ -7 % 3 }}",
    "{{ 7 % 2.5 }}",
    "{{ 2 ** 0.5 }}",
    "{{ 2 ** 10 }}",
    "{{ 2 ** -1 }}",
    "{{ 2 ** 64 }}",
    "{{ 3 * 1.0 }}",
    "{{ 1 / 3 }}",
    "{{ 1e16 + 1 }}",
    "{{ 1e300 * 1e10 }}",
    "{{ -1e300 * 1e10 }}",
    "{{ 0.1 * 3 }}",
    "{{ 123456789.123456789 }}",
    "{{ 5e-324 }}",
    "{{ 1.7976931348623157e308 }}",
    "{{ 1e-4 }}",
    "{{ 9.999999999999999e-05 }}",
    "{{ 1e100 }}",
    "{{ 1e-100 }}",
    "{{ -1.5e-7 }}",
    "{{ 1000000.0 }}",
    "{{ 12345678901234567890 }}",
    "{{ true + 1 }}",
    "{{ -v.e16 }}",
    "{{ v.half * 2 }}",
    "{{ v.tiny * 10 }}",
    # Number filters.
    "{{ 3 | float }}",
    "{{ '3.5' | float }}",
    "{{ 'abc' | float }}",
    "{{ 'nan' | float }}",
    "{{ 'inf' | float }}",
    "{{ '-inf' | float }}",
    "{{ ['nan' | float] }}",
    "{{ 'p' ~ ('nan' | float) }}",
    "{{ 2.7 | int }}",
    "{{ '12' | int + 1 }}",
    "{{ 'abc' | int }}",
    "{{ true | int }}",
    "{{ 1e16 | int }}",
    "{{ 2.5 | round }}",
    "{{ 3.5 | round }}",
    "{{ 2.567 | round(2) }}",
    "{{ 3 | round }}",
    "{{ 2.5 | round(0, 'floor') }}",
    "{{ -0.0 | abs }}",
    "{{ -3 | abs }}",
    "{{ [0.1, 0.2] | sum }}",
    "{{ [1, 2] | sum }}",
    "{{ [1, 2.5] | max }}",
    # String filters on non-strings.
    "{{ true | upper }}",
    "{{ none | upper }}",
    "{{ 5 | trim }}",
    "{{ 2.5 | trim }}",
    "{{ [1] | trim }}",
    "{{ true | lower }}",
    "{{ none | capitalize }}",
    "{{ true | replace('T', 'X') }}",
    "{{ 1e-7 | replace('e', 'E') }}",
    "{{ none | length }}",
    "{{ 'abc' | length }}",
    "{{ true | center(8) }}",
    "{{ none | wordcount }}",
    "{{ true | indent(2) }}",
    # default.
    "{{ undefined_name | default('d') }}",
    "{{ none | default('d') }}",
    "{{ none | default('d', true) }}",
    "{{ false | default('d', true) }}",
    "{{ v.n | default('d') }}",
    "{{ v.missing | default(none) }}",
    "{{ v.missing }}",
    "{{ v.missing | string }}",
    "{{ 'p' ~ v.missing }}",
    "{{ [v.missing] | join(',') }}",
    # Containers and their filters.
    "{{ [1, 2, 3][1:] }}",
    "{{ 'abc'[1:] }}",
    "{{ [1, 2, 3][::-1] }}",
    "{{ 'ab' | list }}",
    "{{ v.obj | list }}",
    "{{ v.obj | first }}",
    "{{ v.args.filters | items | list }}",
    "{{ v.obj | dictsort }}",
    "{{ {'b': 1, 'a': 2} | dictsort }}",
    "{{ [3, 1, 2] | sort }}",
    "{{ [1, 1, 2] | unique | list }}",
    "{{ [3, 1, 2] | reverse | list }}",
    "{{ ['a', 'b'] | map('upper') | list }}",
    "{{ [1, 2, 3] | batch(2) | list }}",
    "{{ range(3) | list }}",
    "{{ (1, 2) }}",
    "{{ (1,) }}",
    "{{ ('a', none) | join('-') }}",
    "{{ {1: 'a', 2.5: none} }}",
    "{{ {'k': v.obj} }}",
    "{{ [[]] }}",
    "{{ [{}] }}",
    "{{ 'a,b'.split(',') }}",
    "{{ [v.s, v.dq, v.both] }}",
    "{{ [v.esc] }}",
    "{{ [v.uni] }}",
    "{{ [v.unprintable] }}",
    "{{ [v.nonprinting] }}",
    "{{ [v.unassigned] }}",
    "{{ {v.s: v.dq} }}",
    "{{ [1, 2] | join }}",
    "{{ [true, none, 2.5] | join(',') }}",
    "{{ [1.0, 1e-7, 1e16] | join(' ') }}",
    "{{ [[1, 'a'], {'b': true}] | join(';') }}",
    "{{ v.list | join('|') }}",
    "{{ 'abc' | join('-') }}",
    # format, both styles.
    "{{ '%s %s' | format(true, none) }}",
    "{{ '{} {}'.format(true, none) }}",
    "{{ '%d' | format(true) }}",
    "{{ '%.2f' | format(1) }}",
    "{{ '%s' | format(0.0) }}",
    "{{ '{}'.format(0.0) }}",
    "{{ '%s|%s' | format(1e-7, 1e16) }}",
    "{{ '{}|{}'.format(1e-7, 1e16) }}",
    "{{ '%s' | format([1, 'a']) }}",
    "{{ '{}'.format({'a': none}) }}",
    "{{ '%r' | format('x') }}",
    "{{ '{!r}'.format('x') }}",
    "{{ '{!s}'.format(none) }}",
    "{{ '{:>6}'.format(true) }}",
    "{{ '{:.3}'.format(1e-7) }}",
    "{{ '{:>12}'.format(1e-7) }}",
    "{{ '{:>20}'.format(0.1 + 0.2) }}",
    "{{ '%5s' | format(true) }}",
    "{{ '%-8s|' | format(none) }}",
    "{{ '{0}-{0}'.format(2.0) }}",
    "{{ '{a}'.format(a=none) }}",
    "{{ '{a}'.format(a=[true]) }}",
    "{{ '%(a)s' | format(a=none) }}",
    "{{ '%s' | format(v.args) }}",
    "{{ '{}'.format(v.args) }}",
    # Loops.
    "{% for x in [true, none] %}{{ loop.index }}:{{ x }};{% endfor %}",
    "{% for k, val in v.args | items %}{{ k }}={{ val | string }};{% endfor %}",
    "{% for k, val in v.args.items() %}{{ k }}={{ val }};{% endfor %}",
    "{% for x in [1, 2, 3] %}{% if x == 2 %}{% break %}{% endif %}{{ x }}{% endfor %}",
    # Whole request values.
    "{{ v.args }}",
    "{{ v.args | string }}",
    "{{ 'args: ' ~ v.args }}",
    "{{ v.obj }}",
    "{{ v.list }}",
    "{{ v.args | tojson }}",
    "{{ v.list | tojson }}",
    "{{ v.obj | tojson(indent=2) }}",
    # Fields of `str.format` that look up an attribute or an index first.
    "{{ '{a.b}'.format(a={'b': true}) }}",
    "{{ '{0.b}'.format({'b': 1}) }}",
    "{{ '{0[k]}'.format({'k': none}) }}",
    "{{ '{a[0]}'.format(a=[2.5e-7]) }}",
    "{{ '{0.b}-{1}'.format({'b': 1e16}, none) }}",
    "{{ '{0[1]}'.format(v.list) }}",
    # Tie floats written in the template.
    "{{ 1125899906842624.25 }}",
    "{{ [1324350514534.90625, 73637923426838.62] }}",
    "{{ 1125899906842624.25 | tojson }}",
    # Loop controls (Hugging Face loads `jinja2.ext.loopcontrols`).
    "{% for x in [1, 2, 3] %}{% if x == 2 %}{% continue %}{% endif %}{{ x }}{% endfor %}",
    "{% for x in range(5) %}{% if x > 2 %}{% break %}{% endif %}{{ loop.index }}{% endfor %}",
    "{% for a in [1, 2] %}{% for b in [1, 2, 3] %}{% if b == 2 %}{% break %}{% endif %}{{ a }}{{ b }};{% endfor %}{% endfor %}",
    # String filters coerce a non-string with `str()` (`soft_str`).
    "{{ v.ptype | upper }}",
    "{{ v.ptype | lower }}",
    "{{ v.ptype | capitalize }}",
    "{{ v.ptype | title }}",
    "{{ v.ptype | trim }}",
    "{{ v.ptype | replace('string', 'str') }}",
    "{{ v.ptype | trim('[]') }}",
    "{{ '\\x1c x \\x1f' | trim }}",
    "{{ (v.spaces ~ 'x y' ~ v.spaces) | trim | length }}",
    "{{ ('\\u200b x \\u180e') | trim | length }}",
    "{{ \"it's o'neil's\" | title }}",
    "{{ 'well-known co-op' | title }}",
    "{{ '3rd 1st x2y 42abc' | title }}",
    "{{ 'a\\tb\\nc\\u3000d\\x1ce' | title }}",
    "{{ 'e.g. mr.smith a.b.c' | title }}",
    "{{ 'ΟΔΟΣ ΣΑΣ' | title }}",
    "{{ 'straße ǆungla' | title }}",
    "{{ 'ß' | capitalize }}",
    "{{ 'ǆ' | capitalize }}",
    "{{ 'aΣ.' | capitalize }}",
    "{{ 'ΟΔΟΣ' | capitalize }}",
    "{{ 'İx' | capitalize }}",
    "{{ \"it's a-b (c) [d] <e> {f} x_y o'neil\tTAB\" | title }}",
    "{{ 'ÉCOLE straße' | title }}",
    "{{ 1e-7 | upper }}",
    "{{ v.tiny | title }}",
    "{{ {'a': true} | capitalize }}",
    "{{ none | lower }}",
    "{{ [none] | trim }}",
    "{{ 'aTrue' | replace(true, 'x') }}",
    "{{ 'a1.0' | replace(1.0, 2) }}",
    "{{ v.args | replace('None', 'null') }}",
    "{% for k, p in v.args.items() %}{{ k }}:{{ p | upper }};{% endfor %}",
    # printf and str.format with specs, too few and too many arguments.
    "{{ '%s %d' | format([1, 'a'], 1) }}",
    "{{ '%.3s' | format(true) }}",
    "{{ '%5s|%-6s|%.2s' | format(none, [1], 'abc') }}",
    "{{ '%s and %d and %.1f' | format({'k': 1e-7}, 3, 2.25) }}",
    "{{ '{:>8}'.format(none) }}",
    "{{ '{:5}'.format(true) }}",
    "{{ '%s' | format(1, 2) }}",
    "{{ '%s %s' | format(1) }}",
    "{{ '{} {}'.format(1) }}",
    "{{ '{}'.format(1, 2) }}",
    "{{ '{0}/{0}'.format([1.5e-7]) }}",
    # What minijinja 2.22-2.24 changed.
    "{% for x in [1, 2] %}{% if x == 1 %}{% set y = 'one' %}{% endif %}{{ y }};{% endfor %}",
    "{% for k, val in {'items': 1, 'b': 2}.items() %}{{ k }}={{ val }};{% endfor %}",
    "{% for k, val in v.schema.items() %}{{ k }};{% endfor %}",
    "{{ v.schema.get('type') }}|{{ v.schema.keys() | list }}",
    "{{ {'get': 1}.get('get') }}",
    "{{ {'a': 1} | tojson(indent=2 if true else none) }}",
    "{{ 'a b'.split() }}",
    "{{ 'a,b'.split(',') | length }}",
    "{{ ('a b c'.split())[1:] }}",
    "{{ ['a', 'b', 'c'][::-1] }}",
    "{{ (['a', 1e-7] + [none])[1:] }}",
    "{{ range(3) }}",
    "{{ ('ab' * 3) }}",
    "{{ ('x' * 100000000) | length }}",
    "{{ ('x' * 100000001) | length }}",
    "{{ ('nan' | float) == ('nan' | float) }}",
    "{{ ('nan' | float) < 1 }}",
    "{{ 1 < ('nan' | float) }}",
    # Tests on none (Nemotron 3 Nano and Qwen3-Coder ask `tools is iterable and tools | length`).
    "{{ none is iterable }}|{{ v.n is iterable }}|{{ undefined_name is iterable }}",
    "{{ 'ab' is iterable }}|{{ 1 is iterable }}|{{ true is iterable }}|{{ v.args is iterable }}",
    "{% if v.n is iterable and v.n | length > 0 %}x{% else %}y{% endif %}",
    "{{ none is sequence }}|{{ 'a' is sequence }}|{{ v.args is sequence }}|{{ v.list is sequence }}",
    "{{ none is mapping }}|{{ v.args is mapping }}|{{ v.list is mapping }}",
    "{{ (v.list | reverse) is sequence }}|{{ (v.list | map('string')) is sequence }}",
    "{{ (v.list | reverse | list) is sequence }}|{{ (v.list | map('string') | list) is sequence }}",
    "{{ undefined_name is sequence }}|{{ ('a b'.split())[1:] is sequence }}|{{ range(2) is iterable }}",
    # Containers the template builds, under `~`.
    "{{ (v.ptype | list) ~ '' }}",
    "{{ v.ptype[0:1] ~ '' }}",
    "{{ (v.ptype + ['x']) ~ '' }}",
    "{{ dict(a=none) ~ '' }}",
    # printf with keyword arguments: Jinja2 formats `str(fmt) % kwargs`.
    "{{ '%s' | format(a=0) }}",
    "{{ '%s' | format(1, a=none) }}",
    "{{ 'x' | format(a=0) }}",
    "{{ '%s %s' | format(a=0) }}",
    # Specs Python applies differently or refuses.
    "{{ '{:,}'.format(false) }}",
    "{{ '{:.3}'.format(5) }}",
    # `{% generation %}`: transformers' tag, its body in a scope of its own.
    "{% set x = 1 %}{% generation %}{% set x = 2 %}in={{ x }}{% endgeneration %} out={{ x }}",
    # Deeper than infr's nesting limit.
    "{% set ns = namespace(x=[]) %}{% for i in range(200) %}{% set ns.x = [ns.x] %}{% endfor %}{{ ns.x | string | length }}",
    # Errors.
    "{{ 'a' + 1 }}",
    "{{ 1 / 0 }}",
    "{{ v.missing.x }}",
]

ALL_VARS = {**VARS, **REPR_VARS}

# Whole templates already in this directory, rendered with a replayed tool call whose arguments
# hold every kind of value the template prints with `| string` (Qwen3.5's `args_value | string`).
TEMPLATE_RENDERS = [
    {
        "template": "qwen35_chat_template.jinja",
        "messages": [
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {
                                "city": "Paris",
                                "unit": None,
                                "precise": True,
                                "late": False,
                                "days": 3,
                                "ratio": 0.5,
                                "tiny": 1e-07,
                                "huge": 1e16,
                                "tie": 1125899906842624.25,
                                "two": 2.0,
                                "tags": ["a", "it's"],
                                "filters": {"min": 1.0, "on": None},
                            },
                        },
                        "id": "call00001",
                    }
                ],
            },
            {"role": "tool", "content": "sunny", "tool_call_id": "call00001", "name": "get_weather"},
            {"role": "user", "content": "Thanks"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "unit": {"type": ["string", "null"], "default": None},
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
    }
]


def cases():
    for value in VALUES:
        for construct in CONSTRUCTS:
            yield construct.replace("X", value)
    yield from EXTRA


def main():
    env = make_env()
    rows = []
    seen = set()
    for expr in cases():
        if expr in seen:
            continue
        seen.add(expr)
        template = "{% set v = tools %}" + expr if "v." in expr else expr
        row = {"template": template}
        try:
            row["python"] = render(env, template, ALL_VARS)
        except Exception as e:  # every error is a row: infr must refuse it too
            row["python_error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
    renders = []
    for case in TEMPLATE_RENDERS:
        source = OUT.with_name(case["template"]).read_text(encoding="utf-8")
        template = env.from_string(source)
        renders.append(
            {
                **case,
                "python": template.render(
                    messages=case["messages"],
                    tools=case["tools"],
                    add_generation_prompt=True,
                    bos_token="<s>",
                    eos_token="</s>",
                    enable_thinking=True,
                ),
            }
        )
    doc = {
        "generated_by": "crates/infr-chat/tests/fixtures/jinja2_text.py",
        "jinja2": jinja2.__version__,
        "python": sys.version.split()[0],
        "vars": ALL_VARS,
        "rows": rows,
        "renders": renders,
    }
    OUT.write_text(
        json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"{len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
