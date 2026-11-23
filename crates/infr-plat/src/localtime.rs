//! `strftime` over the machine's LOCAL time. The chat templates' `strftime_now` is the caller.
//!
//! The clock is `std`'s; turning an instant into local calendar fields and formatting them is the
//! C library's (`localtime_r` / `localtime_s`, then `strftime`), which is what llama.cpp's template
//! engine does for the same function. Rust has no local-time support of its own.
//!
//! The format string comes from a chat template, which ships inside a downloaded model, so it is
//! untrusted. It never reaches the C library: [`strftime_local`] walks it here, copies literal text
//! itself, and hands `strftime` one conversion at a time, each drawn from a fixed set. That is
//! what keeps a hostile format from being a problem on any platform —
//!
//! - Microsoft's C runtime treats a conversion it does not know as an invalid parameter, whose
//!   default handler terminates the process; glibc and Apple's libc each accept extensions the
//!   others do not. Only the conversions every one of them implements are let through, and anything
//!   else is an [`StrftimeError::Conversion`].
//! - Microsoft's `strftime` converts the format to wide characters and its output back through
//!   the time locale's code page, which the startup "C" locale sets to the ANSI code page
//!   (`CP_ACP`, in `ucrt/locale/nlsdata.cpp`). UTF-8 literal text would not survive that on a
//!   machine whose ANSI code page is not UTF-8; literal text never goes through it.
//! - A NUL in the format is literal text like any other, so no C string can be cut short by one.
//!
//! Day and month names are English as long as the process stays in the C library's startup "C"
//! locale, where `%c`, `%x`, `%X` and `%r` also have one fixed spelling. No Rust source in this
//! workspace calls `setlocale`; whether a C library loaded into the process does (the Vulkan
//! loader, a GPU driver) was not checked.
//!
//! **`%Z` on Windows can fail.** There the zone's name is the system's localized one, written in
//! that same code page. From the UCRT's source, not from a run: a name that is not ASCII comes
//! out as bytes that are not UTF-8 unless the ANSI code page is UTF-8, which is a
//! [`StrftimeError::NotUtf8`]; and if the conversion back from wide characters fails, `strftime`
//! returns zero, which is indistinguishable from a full buffer and so a
//! [`StrftimeError::FieldTooLong`]. Either way a format containing `%Z` is an error on such a
//! machine, and one without it is unaffected.
//!
//! Platforms: the Linux arm was built and run locally (x86-64, glibc). The macOS and Windows arms
//! were only type-checked locally (`cargo clippy --target`) and are run by CI's test matrix. The
//! Windows arm was written against the UCRT's own source (`ucrt/time/wcsftime.cpp`,
//! `localtime.cpp` and `tzset.cpp` in the Windows SDK), not against a run.

use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

/// The conversion characters accepted after `%` (besides `%%`): those C99 defines that glibc,
/// Apple's libc and Microsoft's UCRT all implement.
const CONVERSIONS: &[u8] = b"aAbBcCdDeFgGhHIjmMnprRStTuUVwWxXyYzZ";

/// The conversions that write one number, and so the ones `%-` (no padding) applies to.
const NUMERIC: &[u8] = b"CdegGHIjmMSuUVwWyY";

/// Room for one conversion's output. The widest fixed one is `%c`; `%Z` is as long as the zone's
/// name, which the UCRT caps well below this and which a POSIX `TZ` value can make arbitrarily
/// long — past this it is a [`StrftimeError::FieldTooLong`], not a truncated date.
const FIELD_CAPACITY: usize = 256;

/// Appended to every one-conversion format, so that a successful `strftime` never returns zero:
/// `%p` and `%Z` can legitimately produce nothing, and zero is also how `strftime` reports that
/// the buffer was too small.
const SENTINEL: u8 = b'.';

/// Why [`strftime_local`] produced no string.
#[derive(Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum StrftimeError {
    /// The format holds something other than literal text, `%%`, or `%` (optionally `%-` on a
    /// numeric field) followed by one of the accepted conversions. Carries the offending
    /// text.
    Conversion(String),
    /// The instant is outside what the C library's local-time conversion handles.
    Time,
    /// One conversion's output did not fit the buffer it is formatted into, or the C library
    /// failed to produce it. Carries the conversion.
    FieldTooLong(char),
    /// One conversion's output is not UTF-8. Carries the conversion.
    NotUtf8(char),
}

impl fmt::Display for StrftimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Conversion(text) => write!(f, "unsupported conversion {text:?}"),
            Self::Time => write!(f, "the time is outside the C library's local-time range"),
            Self::FieldTooLong(conv) => write!(f, "the output of %{conv} is too long"),
            Self::NotUtf8(conv) => write!(f, "the output of %{conv} is not UTF-8"),
        }
    }
}

impl std::error::Error for StrftimeError {}

/// A conversion character that is a member of [`CONVERSIONS`]. Only [`Conversion::parse`] makes
/// one, so [`LocalTm::field`] cannot be handed a character the C library might reject.
#[derive(Clone, Copy)]
struct Conversion(u8);

impl Conversion {
    fn parse(c: char) -> Option<Self> {
        let byte = u8::try_from(c).ok()?;
        CONVERSIONS.contains(&byte).then_some(Self(byte))
    }

    fn is_numeric(self) -> bool {
        NUMERIC.contains(&self.0)
    }

    fn char(self) -> char {
        char::from(self.0)
    }
}

#[cfg(unix)]
use libc::strftime;

// The `libc` crate declares no `strftime` for Windows. The UCRT exports this one, and `tm` there
// is the same nine `int`s whichever width `time_t` has.
#[cfg(all(windows, target_env = "msvc"))]
unsafe extern "C" {
    fn strftime(
        s: *mut libc::c_char,
        max: libc::size_t,
        format: *const libc::c_char,
        tm: *const libc::tm,
    ) -> libc::size_t;
}

// [`CONVERSIONS`] is the set the UCRT accepts; a MinGW build may link the older `msvcrt.dll`, which
// accepts fewer, so that target is refused rather than given a format it could abort on.
#[cfg(not(any(unix, all(windows, target_env = "msvc"))))]
compile_error!("infr-plat's strftime_local is implemented for Unix and for Windows with the UCRT");

/// Local calendar fields, as the C library's local-time conversion filled them. Only
/// [`LocalTm::at`] makes one, so the `tm` that reaches `strftime` is never caller-built: its fields
/// are in range, and its `tm_zone` pointer, where the platform has one, is the C library's own.
struct LocalTm(libc::tm);

impl LocalTm {
    fn at(now: SystemTime) -> Result<Self, StrftimeError> {
        // Whole seconds since the epoch, rounded toward the past on both sides of it.
        let secs = match now.duration_since(UNIX_EPOCH) {
            Ok(since) => i64::try_from(since.as_secs()).ok(),
            Err(before) => {
                let before = before.duration();
                i64::try_from(before.as_secs())
                    .ok()
                    .map(|secs| -secs - i64::from(before.subsec_nanos() > 0))
            }
        };
        // The C library's `time_t`, by inference from the call below: `i64` on the 64-bit targets
        // and narrower on some 32-bit ones, where this conversion is the range check.
        #[allow(clippy::useless_conversion)]
        let time = secs
            .and_then(|secs| secs.try_into().ok())
            .ok_or(StrftimeError::Time)?;
        // SAFETY: every field of `tm` is an integer or, on the platforms that carry `tm_zone`, a
        // raw pointer, so all-zero bytes are a valid value of it.
        let mut tm: libc::tm = unsafe { std::mem::zeroed() };
        // SAFETY: both pointers are to locals that are live and exclusively ours for the call,
        // which reads one `time_t` and writes one `tm`. The call may also read the `TZ`
        // environment variable (when is per C library, see `strftime_local`), and does so without
        // `std`'s environment lock; nothing here can rule out a write racing that read, which is
        // the limitation `strftime_local` documents, along with why it holds in this workspace.
        #[cfg(unix)]
        let converted = unsafe { !libc::localtime_r(&time, &mut tm).is_null() };
        // SAFETY: as above, with the UCRT's argument order (destination first). It reports a
        // time it cannot convert through its return value.
        #[cfg(windows)]
        let converted = unsafe { libc::localtime_s(&mut tm, &time) == 0 };
        if converted {
            Ok(Self(tm))
        } else {
            Err(StrftimeError::Time)
        }
    }

    /// What `strftime` writes for `%<conv>`.
    fn field(&self, conv: Conversion) -> Result<String, StrftimeError> {
        let format = [b'%', conv.0, SENTINEL, 0];
        let mut buf = [0u8; FIELD_CAPACITY];
        // SAFETY: `buf` is writable for the `buf.len()` bytes `strftime` is told it may use, NUL
        // included. `format` is NUL-terminated by its last element and holds no earlier NUL
        // (`conv.0` is one of `CONVERSIONS`). `self.0` came from the C library's own local-time
        // conversion (see `LocalTm`), and `conv` is a conversion every supported C library
        // implements, so the call cannot reach the UCRT's invalid-parameter handler.
        let written = unsafe {
            strftime(
                buf.as_mut_ptr().cast(),
                buf.len(),
                format.as_ptr().cast(),
                &self.0,
            )
        };
        field_text(conv, buf.get(..written).unwrap_or_default())
    }
}

/// The text of one conversion's output, given what `strftime` wrote for it: `written` is empty
/// when the output did not fit, and otherwise ends with the sentinel.
fn field_text(conv: Conversion, written: &[u8]) -> Result<String, StrftimeError> {
    let field = written
        .strip_suffix(&[SENTINEL])
        .ok_or(StrftimeError::FieldTooLong(conv.char()))?;
    // A zone name is the one field that can be non-ASCII (the UCRT writes it in the ANSI code
    // page). It becomes prompt text, so bytes that are not UTF-8 are refused rather than replaced:
    // a `U+FFFD` in a prompt is a silent change to what the model reads.
    String::from_utf8(field.to_vec()).map_err(|_| StrftimeError::NotUtf8(conv.char()))
}

/// A numeric field without its padding, which is what `%-` asks for: `05` is `5`, ` 5` is `5`,
/// `00` is `0`.
fn unpadded(field: &str) -> &str {
    let digits = field.trim_start_matches(['0', ' ']);
    if digits.is_empty() && !field.is_empty() {
        &field[field.len() - 1..]
    } else {
        digits
    }
}

/// Formats `now`, in the machine's local time zone, by the `strftime` format `fmt`.
///
/// Accepted: literal text, `%%`, `%` followed by one of the C99 conversions that glibc, Apple's
/// libc and Microsoft's UCRT all implement (`aAbBcCdDeFgGhHIjmMnprRStTuUVwWxXyYzZ`; a test holds
/// this list to the set the function accepts), and `%-` (no padding) before one that writes a
/// single number. Anything else is an error, never a panic and never an abort. The work is
/// proportional to `fmt`'s length — one C library call per conversion — so a caller formatting
/// untrusted input bounds that length.
///
/// **Not the same string everywhere**: `%Z` is the zone's name as the C library has it — an
/// abbreviation such as `PST` on Unix, the system's localized long name on Windows, where it can
/// also make the call fail (see the module documentation). Every other conversion writes the same
/// text on every supported platform.
///
/// # Limitation: the environment must not be written while the zone is read
///
/// The C library takes the zone from the `TZ` environment variable, reading the process
/// environment directly and not under `std`'s environment lock. A `std::env::set_var` or
/// `remove_var` on another thread during that read is a data race, and this function cannot
/// prevent it: edition 2021 still lets safe code call `set_var`, and `std`'s lock is private. It
/// is the condition `set_var` itself documents — no other thread may be reading the environment
/// — seen from the reader's side.
///
/// When the read happens, and so how wide the window is and whether a later change to `TZ` is
/// seen, depends on the C library:
///
/// - **glibc** (run here, 2.44): the zone is read at the process's first local-time conversion
///   and is then fixed. A `setenv("TZ")` after it changes nothing for `localtime_r` or
///   `strftime`; only a `tzset` or `localtime` call, which nothing in this crate makes, takes
///   the new value. The window is that first read.
/// - **musl** (run here, the static libc of Rust's `x86_64-unknown-linux-musl` target; not a
///   target the project ships): a later `setenv("TZ")` is seen by the next `localtime_r`, so
///   every call reads the variable and every call is a window.
/// - **Apple's libc**: not checked. Assume the wider, musl-like window.
/// - **UCRT** (from its source, `ucrt/time/tzset.cpp`, not run): the zone is initialised once,
///   under the runtime's own lock, reading `TZ` through `_wgetenv_s`; later calls return early.
///   How that read interacts with a concurrent `std::env::set_var` was not checked.
///
/// It holds in this workspace because the only `set_var` outside tests, `publish_thread_count`
/// in infr-cli, runs in `main` before any other thread exists. A new environment write that can
/// run after threads start breaks it.
pub fn strftime_local(fmt: &str, now: SystemTime) -> Result<String, StrftimeError> {
    let tm = LocalTm::at(now)?;
    let mut out = String::with_capacity(fmt.len());
    let mut chars = fmt.chars();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        let first = chars.next();
        let (no_pad, second) = match first {
            Some('-') => (true, chars.next()),
            _ => (false, first),
        };
        if second == Some('%') && !no_pad {
            out.push('%');
            continue;
        }
        match second.and_then(Conversion::parse) {
            Some(conv) if !no_pad => out.push_str(&tm.field(conv)?),
            Some(conv) if conv.is_numeric() => out.push_str(unpadded(&tm.field(conv)?)),
            _ => {
                let mut text = String::from("%");
                text.extend(no_pad.then_some('-'));
                text.extend(second);
                return Err(StrftimeError::Conversion(text));
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;
    use std::time::Duration;

    /// 2024-07-05 12:00:00 UTC, a Friday: a single-digit day, for the padded fields.
    const NOON_UTC: Duration = Duration::from_secs(1_720_180_800);

    fn at_noon(fmt: &str) -> Result<String, StrftimeError> {
        strftime_local(fmt, UNIX_EPOCH + NOON_UTC)
    }

    /// Set in the child of [`fixed_zones_format_exactly`] to the zone it was started in.
    const ZONE_VAR: &str = "INFR_PLAT_LOCALTIME_TEST_ZONE";

    /// What every accepted conversion writes at [`NOON_UTC`] in a zone five and a half hours east
    /// of UTC, where it is 17:30 on the same Friday. `TZ` here is the POSIX form, whose sign is
    /// west-positive, with the three-letter name the UCRT's parser requires.
    const EAST_TZ: &str = "XXX-5:30";
    const EAST: &[(&str, &str)] = &[
        ("%a", "Fri"),
        ("%A", "Friday"),
        ("%b", "Jul"),
        ("%B", "July"),
        ("%c", "Fri Jul  5 17:30:00 2024"),
        ("%C", "20"),
        ("%d", "05"),
        ("%D", "07/05/24"),
        ("%e", " 5"),
        ("%F", "2024-07-05"),
        ("%g", "24"),
        ("%G", "2024"),
        ("%h", "Jul"),
        ("%H", "17"),
        ("%I", "05"),
        ("%j", "187"),
        ("%m", "07"),
        ("%M", "30"),
        ("%n", "\n"),
        ("%p", "PM"),
        ("%r", "05:30:00 PM"),
        ("%R", "17:30"),
        ("%S", "00"),
        ("%t", "\t"),
        ("%T", "17:30:00"),
        ("%u", "5"),
        ("%U", "26"),
        ("%V", "27"),
        ("%w", "5"),
        ("%W", "27"),
        ("%x", "07/05/24"),
        ("%X", "17:30:00"),
        ("%y", "24"),
        ("%Y", "2024"),
        ("%z", "+0530"),
        ("%Z", "XXX"),
        // The formats chat templates use, and the no-padding flag.
        ("%d %b %Y", "05 Jul 2024"),
        ("%B %d, %Y", "July 05, 2024"),
        ("%-d %B %Y, %-I:%M %p", "5 July 2024, 5:30 PM"),
        ("%-m/%-e/%-y %-H h, day %-j", "7/5/24 17 h, day 187"),
    ];

    /// Zones on the far side of midnight from UTC at [`NOON_UTC`]: the date is the LOCAL one.
    const FAR_EAST_TZ: &str = "YYY-13";
    const FAR_EAST: &[(&str, &str)] = &[("%F %T %z %Z", "2024-07-06 01:00:00 +1300 YYY")];
    const FAR_WEST_TZ: &str = "ZZZ12:30";
    const FAR_WEST: &[(&str, &str)] = &[("%F %T %z %Z", "2024-07-04 23:30:00 -1230 ZZZ")];

    const ZONES: &[(&str, &[(&str, &str)])] = &[
        (EAST_TZ, EAST),
        (FAR_EAST_TZ, FAR_EAST),
        (FAR_WEST_TZ, FAR_WEST),
    ];

    /// Runs the test named `name` in this module again, in a child process whose `TZ` is `tz`: the
    /// zone is process state that cannot be changed soundly once threads exist, so it is set
    /// before the child starts. The child finds `tz` in [`ZONE_VAR`] and is the half that compares.
    fn rerun_in_zone(name: &str, tz: &str) {
        let (_, tests) = module_path!()
            .split_once("::")
            .expect("a module path starts with the crate's name");
        let child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", &format!("{tests}::{name}"), "--nocapture"])
            .env("TZ", tz)
            .env(ZONE_VAR, tz)
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&child.stdout);
        let stderr = String::from_utf8_lossy(&child.stderr);
        // Success alone would also be what a child that matched no test reports.
        assert!(
            child.status.success() && stdout.contains("1 passed"),
            "TZ={tz}: {}\n{stdout}\n{stderr}",
            child.status
        );
    }

    /// Exact output in known time zones.
    #[test]
    fn fixed_zones_format_exactly() {
        let Ok(zone) = std::env::var(ZONE_VAR) else {
            for (tz, _) in ZONES {
                rerun_in_zone("fixed_zones_format_exactly", tz);
            }
            return;
        };
        let (_, want) = ZONES
            .iter()
            .find(|(tz, _)| *tz == zone)
            .expect("the parent passes one of ZONES");
        let mut wrong: Vec<String> = want
            .iter()
            .filter_map(|&(fmt, want)| {
                let got = at_noon(fmt);
                (got.as_deref() != Ok(want)).then(|| format!("{fmt:?}: got {got:?}, want {want:?}"))
            })
            .collect();
        if zone == EAST_TZ {
            // The table has a row for every accepted conversion.
            for &conv in CONVERSIONS {
                let spec = format!("%{}", char::from(conv));
                if !want.iter().any(|(fmt, _)| *fmt == spec) {
                    wrong.push(format!("no row for {spec}"));
                }
            }
            // An instant before the epoch rounds toward the past, like one after it. Unix only:
            // the UCRT converts nothing before the epoch.
            #[cfg(unix)]
            {
                let got = strftime_local("%F %T", UNIX_EPOCH - Duration::from_millis(500));
                if got.as_deref() != Ok("1970-01-01 05:29:59") {
                    wrong.push(format!("half a second before the epoch: {got:?}"));
                }
            }
        }
        assert!(wrong.is_empty(), "TZ={zone}\n{}", wrong.join("\n"));
    }

    /// A field that does not fit [`FIELD_CAPACITY`] is an error, not a truncated or empty string.
    /// Only glibc is known to take a zone name of any length from `TZ`, so only it runs this.
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    #[test]
    fn an_overlong_zone_name_is_an_error() {
        if std::env::var(ZONE_VAR).is_err() {
            let tz = format!("<{}>0", "A".repeat(FIELD_CAPACITY));
            return rerun_in_zone("an_overlong_zone_name_is_an_error", &tz);
        }
        assert_eq!(at_noon("%Y %Z"), Err(StrftimeError::FieldTooLong('Z')));
        // The zone itself was understood: the other fields still format.
        assert_eq!(
            at_noon("%F %T %z").as_deref(),
            Ok("2024-07-05 12:00:00 +0000")
        );
    }

    /// In whatever zone this machine is in, noon UTC falls on that date or a neighbouring one, and
    /// the fields agree with the offset `%z` reports.
    #[test]
    fn the_local_date_is_within_a_day_of_utc() {
        let date = at_noon("%Y-%m-%d").unwrap();
        assert!(
            ["2024-07-04", "2024-07-05", "2024-07-06"].contains(&date.as_str()),
            "{date}"
        );
        let stamp = at_noon("%d %H %M %z").unwrap();
        let fields: Vec<&str> = stamp.split(' ').collect();
        let [day, hour, minute, offset] = fields[..] else {
            panic!("{stamp}");
        };
        let num = |s: &str| s.parse::<i64>().unwrap_or_else(|_| panic!("{stamp}"));
        let sign = if offset.starts_with('-') { -1 } else { 1 };
        let offset_minutes = sign * (num(&offset[1..3]) * 60 + num(&offset[3..5]));
        // Minutes since 2024-07-05 00:00 UTC, counted both ways.
        assert_eq!(
            (num(day) - 5) * 24 * 60 + num(hour) * 60 + num(minute),
            12 * 60 + offset_minutes,
            "{stamp}"
        );
    }

    #[test]
    fn literal_text_is_copied_untouched() {
        for (fmt, want) in [
            ("", ""),
            ("no conversions", "no conversions"),
            ("100%%", "100%"),
            ("%%Y", "%Y"),
            ("日付 \u{1F4C5} é", "日付 \u{1F4C5} é"),
            ("a\0b", "a\0b"),
        ] {
            assert_eq!(at_noon(fmt).as_deref(), Ok(want), "{fmt:?}");
        }
        // …around a conversion too.
        assert_eq!(at_noon("年\0%Y年").as_deref(), Ok("年\x002024年"));
    }

    /// Everything outside the accepted set is an error naming the offending text — including the
    /// glibc and chrono extensions, the modifiers, and a `%` with nothing after it.
    #[test]
    fn other_conversions_are_errors() {
        for (fmt, bad) in [
            ("%Q", "%Q"),
            ("%Y-%m-%q", "%q"),
            ("%", "%"),
            ("%Y%", "%"),
            ("%-", "%-"),
            ("%-%", "%-%"),
            ("%-b", "%-b"),
            ("%-Z", "%-Z"),
            ("%_d", "%_"),
            ("%0d", "%0"),
            ("%#d", "%#"),
            ("%Ey", "%E"),
            ("%Od", "%O"),
            ("%s", "%s"),
            ("%k", "%k"),
            ("%l", "%l"),
            ("%P", "%P"),
            ("%f", "%f"),
            ("%.3f", "%."),
            ("%v", "%v"),
            ("%+", "%+"),
            ("%:z", "%:"),
            ("%\0", "%\0"),
            ("%é", "%é"),
            ("%10Y", "%1"),
        ] {
            assert_eq!(
                at_noon(fmt),
                Err(StrftimeError::Conversion(bad.to_string())),
                "{fmt:?}"
            );
        }
    }

    /// The list the public documentation spells out is the accepted set.
    #[test]
    fn the_documented_conversions_are_the_accepted_ones() {
        assert_eq!(
            std::str::from_utf8(CONVERSIONS),
            Ok("aAbBcCdDeFgGhHIjmMnprRStTuUVwWxXyYzZ")
        );
        assert!(NUMERIC.iter().all(|conv| CONVERSIONS.contains(conv)));
    }

    #[test]
    fn unpadded_keeps_the_last_digit() {
        for (field, want) in [
            ("05", "5"),
            (" 5", "5"),
            ("15", "15"),
            ("007", "7"),
            ("00", "0"),
            ("0", "0"),
            ("2024", "2024"),
            ("", ""),
        ] {
            assert_eq!(unpadded(field), want, "{field:?}");
        }
    }

    /// What `strftime` wrote becomes text only if it ends with the sentinel and is UTF-8; the
    /// second is not reachable through the C library on this machine, so it is fed bytes here.
    #[test]
    fn a_field_is_its_bytes_before_the_sentinel() {
        let zone = Conversion::parse('Z').unwrap();
        assert_eq!(field_text(zone, b"CET.").as_deref(), Ok("CET"));
        assert_eq!(field_text(zone, b".").as_deref(), Ok(""));
        assert_eq!(
            field_text(zone, "Mitteleuropäische Zeit.".as_bytes()).as_deref(),
            Ok("Mitteleuropäische Zeit")
        );
        // The same name in Windows-1252, where `ä` is the single byte 0xE4.
        assert_eq!(
            field_text(zone, b"Mitteleurop\xE4ische Zeit."),
            Err(StrftimeError::NotUtf8('Z'))
        );
        assert_eq!(field_text(zone, b""), Err(StrftimeError::FieldTooLong('Z')));
        assert_eq!(
            field_text(zone, b"CET"),
            Err(StrftimeError::FieldTooLong('Z'))
        );
    }

    /// An instant the C library cannot convert is an error. Which instants those are differs: the
    /// UCRT refuses everything before the epoch, the Unix libraries a year that overflows `int`.
    #[test]
    fn an_unconvertible_time_is_an_error() {
        #[cfg(windows)]
        let unconvertible = UNIX_EPOCH - Duration::from_secs(86_400);
        #[cfg(unix)]
        let unconvertible = UNIX_EPOCH + Duration::from_secs(1 << 62);
        assert_eq!(
            strftime_local("%Y", unconvertible),
            Err(StrftimeError::Time)
        );
        assert_eq!(
            strftime_local("literal", unconvertible),
            Err(StrftimeError::Time)
        );
    }
}
