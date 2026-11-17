//! The arguments of Jinja2's printf-style `| format` and of `str.format`, converted one by one the
//! way Python converts them before minijinja's formatter sees them.
//!
//! minijinja's formatter (`format_utils.rs`) writes an argument that is not a string with its own
//! rules: a float in a field without a type as `%g` does (`0.3` for `0.1 + 0.2`), a container
//! JSON-style. Python's `%s` and a `{}` field without a spec are `str(arg)`. So each argument that
//! only ever meets such a field is replaced by its [`py_str`] first; one that meets a field with a
//! spec, an attribute or an index lookup, or a conversion is left to minijinja, as is everything
//! when the format string is not one these parsers understand.

use minijinja::value::{Kwargs, ValueKind};
use minijinja::{Error, ErrorKind, Value};

use crate::pystr::py_str;

/// Which argument a field reads.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Arg {
    Pos(usize),
    Name(String),
}

/// How a `str.format` field uses its argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FieldUse {
    /// `{}`, `{0}`, `{name}`, `{:}`: `format(arg, '')`, which is `str(arg)`.
    Str,
    /// A non-empty spec (`{:>8}`): `format(arg, spec)`.
    Spec,
    /// An attribute or index lookup (`{0.b}`, `{0[k]}`) or a conversion (`{!r}`).
    Other,
}

/// Each field of a `str.format` string: the argument it reads and how.
type Fields = Vec<(Arg, FieldUse)>;

/// Each conversion of a printf-style string: the argument it reads and its type character.
type Conversions = Vec<(Arg, char)>;

/// The arguments of `fmt.format(*args, **kwargs)` with each argument that only meets `str()`
/// fields replaced by its [`py_str`]. Refuses, as Python does, a non-empty spec on none, a list
/// or a dict (`format(None, '>8')` raises: those types have no `__format__` of their own).
pub(crate) fn str_format_args(fmt: &str, args: &[Value]) -> Result<Vec<Value>, Error> {
    let Some(fields) = str_format_fields(fmt) else {
        return Ok(args.to_vec());
    };
    for (arg, used) in &fields {
        if *used != FieldUse::Spec {
            continue;
        }
        if let Some(value) = lookup(args, arg)? {
            if matches!(
                value.kind(),
                ValueKind::None | ValueKind::Seq | ValueKind::Map
            ) {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    format!("str.format: a {} has no format spec", value.kind()),
                ));
            }
        }
    }
    convert(args, |arg| {
        let mut uses = fields.iter().filter(|(a, _)| a == arg).peekable();
        uses.peek().is_some() && uses.all(|(_, u)| *u == FieldUse::Str)
    })
}

/// The arguments of `fmt | format(*args, **kwargs)` (Jinja2: `str(fmt) % (kwargs or args)`)
/// with each argument whose every conversion is `s` replaced by its [`py_str`]: Python's `%s`
/// is `str(arg)`, padded or cut to the width and precision afterwards. With positional
/// arguments only, a count that does not match the conversions is refused, as Python refuses it;
/// positional and keyword arguments together are refused, as Jinja2 refuses them; and a lone `%s`
/// with keyword arguments formats the whole mapping (`'%s' | format(a=0)` is `{'a': 0}`).
pub(crate) fn printf_args(fmt: &str, args: &[Value]) -> Result<Vec<Value>, Error> {
    let Some(convs) = printf_conversions(fmt) else {
        return Ok(args.to_vec());
    };
    let has_kwargs = args.last().is_some_and(Value::is_kwargs);
    let keyed = convs.iter().any(|(a, _)| matches!(a, Arg::Name(_)));
    if has_kwargs && args.len() > 1 {
        // Jinja2's `do_format` raises "can't handle positional and keyword arguments".
        return Err(Error::new(
            ErrorKind::InvalidOperation,
            "format: positional and keyword arguments at once",
        ));
    }
    if has_kwargs && !keyed && convs.len() == 1 && convs[0].1 == 's' {
        // `'%s' % kwargs`: the one conversion formats the whole mapping.
        return Ok(vec![Value::from(py_str(&args[0])?)]);
    }
    if has_kwargs != keyed || (keyed && convs.iter().any(|(a, _)| matches!(a, Arg::Pos(_)))) {
        // A mapping read by position, or keys with no mapping: Python's answer depends on the
        // argument's own type, so it is left to minijinja.
        return Ok(args.to_vec());
    }
    if !keyed && convs.len() != args.len() {
        return Err(Error::new(
            ErrorKind::InvalidOperation,
            format!(
                "format: {} conversions for {} arguments",
                convs.len(),
                args.len()
            ),
        ));
    }
    convert(args, |arg| {
        let mut uses = convs.iter().filter(|(a, _)| a == arg).peekable();
        uses.peek().is_some() && uses.all(|(_, ty)| *ty == 's')
    })
}

/// The argument `arg` names, if the call has it.
fn lookup(args: &[Value], arg: &Arg) -> Result<Option<Value>, Error> {
    match arg {
        Arg::Pos(i) => Ok(args.get(*i).filter(|v| !v.is_kwargs()).cloned()),
        Arg::Name(name) => match args.last().filter(|v| v.is_kwargs()) {
            Some(kwargs) => Ok(Kwargs::try_from(kwargs.clone())?.peek::<Option<Value>>(name)?),
            None => Ok(None),
        },
    }
}

/// `args` with every argument `to_str` picks replaced by its [`py_str`] (a string stays as it is).
fn convert(args: &[Value], to_str: impl Fn(&Arg) -> bool) -> Result<Vec<Value>, Error> {
    let str_of = |value: &Value| -> Result<Value, Error> {
        if value.kind() == ValueKind::String {
            Ok(value.clone())
        } else {
            py_str(value).map(Value::from)
        }
    };
    args.iter()
        .enumerate()
        .map(|(i, value)| {
            if !value.is_kwargs() {
                return if to_str(&Arg::Pos(i)) {
                    str_of(value)
                } else {
                    Ok(value.clone())
                };
            }
            let kwargs = Kwargs::try_from(value.clone())?;
            let converted = kwargs
                .args()
                .map(|k| {
                    let v = kwargs.peek::<Value>(k)?;
                    let v = if to_str(&Arg::Name(k.to_owned())) {
                        str_of(&v)?
                    } else {
                        v
                    };
                    Ok((k.to_owned(), v))
                })
                .collect::<Result<Kwargs, Error>>()?;
            Ok(Value::from(converted))
        })
        .collect()
}

/// Every replacement field of a `str.format` string, as the argument it reads and how; `None`
/// for a string this does not parse the way Python does (a nested field in a spec, a stray `}`,
/// automatic and manual numbering mixed), which then gets no conversion at all. `{{` and `}}`
/// are literal braces.
fn str_format_fields(fmt: &str) -> Option<Fields> {
    let mut fields = Vec::new();
    let (mut next_auto, mut auto, mut manual) = (0, false, false);
    let mut chars = fmt.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '{' | '}' if chars.peek() == Some(&c) => {
                chars.next();
            }
            '}' => return None,
            '{' => {
                let mut field = String::new();
                loop {
                    match chars.next()? {
                        '}' => break,
                        '{' => return None,
                        c => field.push(c),
                    }
                }
                let name_len = field.find(['.', '[', '!', ':']).unwrap_or(field.len());
                let (name, rest) = field.split_at(name_len);
                let arg = if name.is_empty() {
                    auto = true;
                    next_auto += 1;
                    Arg::Pos(next_auto - 1)
                } else if name.bytes().all(|b| b.is_ascii_digit()) {
                    manual = true;
                    Arg::Pos(name.parse().ok()?)
                } else {
                    Arg::Name(name.to_owned())
                };
                let used = match rest {
                    "" | ":" => FieldUse::Str,
                    _ if rest.starts_with(':') => FieldUse::Spec,
                    _ => FieldUse::Other,
                };
                fields.push((arg, used));
            }
            _ => {}
        }
    }
    (!(auto && manual)).then_some(fields)
}

/// Every conversion of a printf-style string, as the argument it reads and its type character;
/// `None` for one this does not parse (a `*` width or precision, an unclosed or nested key, a
/// string ending inside a conversion). `%%` is a literal `%`.
fn printf_conversions(fmt: &str) -> Option<Conversions> {
    let mut convs = Vec::new();
    let mut next_pos = 0;
    let mut chars = fmt.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '%' {
            continue;
        }
        let mut key = None;
        if chars.next_if_eq(&'(').is_some() {
            let mut name = String::new();
            loop {
                match chars.next()? {
                    ')' => break,
                    '(' => return None,
                    c => name.push(c),
                }
            }
            key = Some(name);
        }
        while chars.next_if(|c| "#0- +".contains(*c)).is_some() {}
        while chars.next_if(char::is_ascii_digit).is_some() {}
        if chars.next_if_eq(&'.').is_some() {
            while chars.next_if(char::is_ascii_digit).is_some() {}
        }
        while chars.next_if(|c| "hlL".contains(*c)).is_some() {}
        match chars.next()? {
            '*' => return None,
            '%' if key.is_none() => {}
            ty => {
                let arg = match key {
                    Some(name) => Arg::Name(name),
                    None => {
                        next_pos += 1;
                        Arg::Pos(next_pos - 1)
                    }
                };
                convs.push((arg, ty));
            }
        }
    }
    Some(convs)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// How each field reads its argument. A field that looks one up (`{a.b}`, `{0[k]}`) must not
    /// be counted as `str()`: its argument has to reach the lookup unconverted.
    #[test]
    fn str_format_fields_are_read_as_python_reads_them() {
        use FieldUse::*;
        let pos = |i| Arg::Pos(i);
        let name = |n: &str| Arg::Name(n.to_owned());
        let cases: &[(&str, Option<Fields>)] = &[
            ("x", Some(vec![])),
            ("{{}} {}", Some(vec![(pos(0), Str)])),
            ("{} {}", Some(vec![(pos(0), Str), (pos(1), Str)])),
            ("{0}-{0:}", Some(vec![(pos(0), Str), (pos(0), Str)])),
            ("{a}", Some(vec![(name("a"), Str)])),
            ("{a.b}", Some(vec![(name("a"), Other)])),
            ("{0.b}", Some(vec![(pos(0), Other)])),
            ("{0[k]}", Some(vec![(pos(0), Other)])),
            ("{!r}", Some(vec![(pos(0), Other)])),
            ("{:>8}", Some(vec![(pos(0), Spec)])),
            ("{0:{1}}", None),
            ("{} {0}", None),
            ("a}b", None),
            ("{", None),
        ];
        for (fmt, want) in cases {
            assert_eq!(&str_format_fields(fmt), want, "{fmt:?}");
        }
    }

    #[test]
    fn printf_conversions_are_read_as_python_reads_them() {
        let pos = |i| Arg::Pos(i);
        let cases: &[(&str, Option<Conversions>)] = &[
            ("100%% %s", Some(vec![(pos(0), 's')])),
            ("%s %d", Some(vec![(pos(0), 's'), (pos(1), 'd')])),
            ("%-5.2s|%05.1f", Some(vec![(pos(0), 's'), (pos(1), 'f')])),
            (
                "%(a)s %(b)r",
                Some(vec![
                    (Arg::Name("a".into()), 's'),
                    (Arg::Name("b".into()), 'r'),
                ]),
            ),
            ("%*d", None),
            ("%(a", None),
            ("%", None),
        ];
        for (fmt, want) in cases {
            assert_eq!(&printf_conversions(fmt), want, "{fmt:?}");
        }
    }
}
