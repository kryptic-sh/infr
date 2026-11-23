//! `infr tune check`: the GPU tolerance gate's evaluation on one model and the selected Vulkan
//! device, from the shipped binary (`docs/gpu-tolerance.md`, "The command").

use super::{resolve, selected_backend, tune_backend, Backend};
use anyhow::anyhow;
use infr_core::config::Config;
use infr_llama::tolerance::gate::{self, Event, FileIdentity, Overridden};
use infr_llama::tolerance::{Check, Limit, PathClass};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// `infr tune check`'s exit status when the device is outside the model's tolerance rows. A
/// usage error has clap's status and any other failure (no row for the model, no device, a
/// failed load) the general one (`exit_status` in `main.rs`), so a script can tell "ran and is
/// out of tolerance" from "did not run".
pub(crate) const EXIT_OUT_OF_TOLERANCE: u8 = 3;

/// The error [`run`] returns for a check that ran and found the device outside the model's
/// rows: `main` turns it into [`EXIT_OUT_OF_TOLERANCE`] once everything the command owned is
/// dropped. The report is already printed.
#[derive(Debug)]
pub(crate) struct OutOfTolerance;

impl std::fmt::Display for OutOfTolerance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("outside the GPU tolerance of the CPU reference")
    }
}

impl std::error::Error for OutOfTolerance {}

/// One configuration's part of the report: the limits it was held to, each with what was
/// measured, and what it ran without.
struct CheckSection {
    config: String,
    class: String,
    /// The settings of this invocation that the configuration's own edit wrote over.
    overridden: Vec<Overridden>,
    /// `(phase, check)`. The phase is where the rows came from (`first`, `prefill`, `decode`),
    /// or what else was compared (`routing`, `tokens`).
    checks: Vec<(String, Check)>,
}

/// What `infr tune check` reports: the same content for the text table and the JSON document.
struct CheckReport {
    /// The name the rows were found by: the file's name.
    row: String,
    /// The file that was scored, and what it says of itself.
    file: PathBuf,
    identity: FileIdentity,
    device: String,
    tune_hash: String,
    runs: Vec<CheckSection>,
    greedy: Vec<CheckSection>,
    failures: Vec<String>,
    /// What a pass does not cover on this device and model.
    not_exercised: Vec<String>,
}

/// What the report says, wherever it shows the row, of how the row was found.
const ROW_MATCH: &str = "file name";

impl CheckReport {
    /// Within tolerance: the evaluation crossed no limit. `failures` is the evaluation's own
    /// list, the one the `gpu_seam_tolerance_*` tests assert empty; the per-check verdicts are
    /// the same comparisons shown one by one.
    fn pass(&self) -> bool {
        self.failures.is_empty()
    }

    /// The file's own account of itself, in words.
    fn identity_text(&self) -> String {
        let id = &self.identity;
        format!(
            "architecture {}, general.name {}, {} weights, {} tensors, {} bytes of them",
            id.architecture.as_deref().unwrap_or("(none)"),
            id.name
                .as_deref()
                .map_or("(none)".to_string(), |n| format!("{n:?}")),
            if id.quantized { "quantized" } else { "float" },
            id.tensors,
            id.tensor_bytes
        )
    }

    /// `infr tune check`'s text.
    fn text(&self) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        // Writing to a String cannot fail.
        let mut line = |l: std::fmt::Arguments| writeln!(out, "{l}").expect("write to a String");
        line(format_args!(
            "row:      {} (matched by {ROW_MATCH} only)",
            self.row
        ));
        line(format_args!("file:     {}", self.file.display()));
        line(format_args!("          {}", self.identity_text()));
        line(format_args!(
            "device:   {} (tune:{})",
            self.device, self.tune_hash
        ));
        line(format_args!("against:  cpu-reference"));
        let mut section = |title: &str, sections: &[CheckSection]| {
            for s in sections {
                line(format_args!(""));
                line(format_args!("{title}{} ({})", s.config, s.class));
                for o in &s.overridden {
                    line(format_args!(
                        "  ran with {} = {}, not the {} that was set: this configuration sets it",
                        o.path, o.ran, o.set
                    ));
                }
                line(format_args!(
                    "  {:<8} {:<14} {:>12}  {:<15} verdict",
                    "phase", "metric", "measured", "bound"
                ));
                for (phase, c) in &s.checks {
                    let (op, bound) = match c.limit {
                        Limit::AtMost(b) => ("<=", b),
                        Limit::AtLeast(b) => (">=", b),
                    };
                    line(format_args!(
                        "  {:<8} {:<14} {:>12}  {op} {:<12} {}",
                        phase,
                        c.metric,
                        check_number(c.measured),
                        check_number(bound),
                        if c.pass { "ok" } else { "FAIL" }
                    ));
                }
            }
        };
        section("", &self.runs);
        section("greedy decode, ", &self.greedy);
        if !self.failures.is_empty() {
            line(format_args!(""));
            line(format_args!("outside tolerance:"));
            for f in &self.failures {
                line(format_args!("  {f}"));
            }
        }
        line(format_args!(""));
        if !self.not_exercised.is_empty() {
            line(format_args!(
                "not exercised (the verdict says nothing of these):"
            ));
            for n in &self.not_exercised {
                line(format_args!("  {n}"));
            }
        }
        line(format_args!(
            "verdict:  {}",
            if self.pass() { "PASS" } else { "FAIL" }
        ));
        out
    }

    /// `infr tune check --json`'s document.
    fn json(&self) -> serde_json::Value {
        let sections = |sections: &[CheckSection]| {
            sections
                .iter()
                .map(|s| {
                    serde_json::json!({
                        "config": s.config,
                        "class": s.class,
                        "overridden": s.overridden.iter().map(|o| serde_json::json!({
                            "path": o.path,
                            "set": o.set,
                            "ran": o.ran,
                        })).collect::<Vec<_>>(),
                        "checks": s.checks.iter().map(|(phase, c)| {
                            let (limit, bound) = match c.limit {
                                Limit::AtMost(b) => ("at_most", b),
                                Limit::AtLeast(b) => ("at_least", b),
                            };
                            serde_json::json!({
                                "phase": phase,
                                "metric": c.metric,
                                "measured": c.measured,
                                "limit": limit,
                                "bound": bound,
                                "pass": c.pass,
                            })
                        }).collect::<Vec<_>>(),
                    })
                })
                .collect::<Vec<_>>()
        };
        let id = &self.identity;
        serde_json::json!({
            "against": "cpu-reference",
            "model": self.row,
            "row": { "model": self.row, "matched_by": ROW_MATCH },
            "file": {
                "path": self.file,
                "architecture": id.architecture,
                "name": id.name,
                "quantized": id.quantized,
                "tensors": id.tensors,
                "tensor_bytes": id.tensor_bytes,
            },
            "device": self.device,
            "tune_hash": self.tune_hash,
            "pass": self.pass(),
            "runs": sections(&self.runs),
            "greedy": sections(&self.greedy),
            "failures": self.failures,
            "not_exercised": self.not_exercised,
        })
    }
}

/// A measured value or a bound as the check table prints it: a whole number as one, a value of
/// ordinary size in fixed notation, a very small or very large one in scientific notation (the
/// branches below hold the cut-offs and the digits).
fn check_number(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e9 {
        format!("{v:.0}")
    } else if (1e-3..1e3).contains(&v.abs()) {
        format!("{v:.6}")
    } else {
        format!("{v:.3e}")
    }
}

/// The table's name for a model file: its file name, which is what the tolerance rows are keyed
/// by.
fn table_name(gguf: &Path) -> anyhow::Result<&str> {
    gguf.file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| anyhow!("`{}` has no file name", gguf.display()))
}

/// What `infr tune check` says of a model with no row: which name was looked up, that nothing
/// was scored, the models that do have rows, and how a row is made.
fn no_row_error(table: &str) -> anyhow::Error {
    let mut models: Vec<&str> = infr_llama::tolerance::TABLE
        .iter()
        .map(|e| e.model)
        .filter(|m| m.ends_with(".gguf"))
        .collect();
    models.dedup();
    anyhow!(
        "the GPU tolerance table has no row for `{table}` (rows are keyed by the GGUF file \
         name), so nothing was scored and this is not a pass. Models with rows: {}. A row is \
         measured, not guessed: docs/gpu-tolerance.md (\"The table\", \"Running it\") has the \
         procedure (`measure_gpu_tolerance` on each device, then a row and its planted-bug runs).",
        models.join(", ")
    )
}

/// The file name a model argument resolves to, where that is known without asking the hub: a
/// path's own, a cached model's (the hub store's own lookup, the one `infr_hub::ensure` tries
/// before it pulls), or an explicit `.gguf` selector's. `None` for a quant selector of a
/// repository not in the cache, whose file name only the hub's listing gives, and for an
/// argument that is no model reference (resolving it says why).
fn name_without_fetching(model: &str) -> Option<String> {
    let file_name = |p: &Path| Some(p.file_name()?.to_str()?.to_string());
    let r = infr_hub::ModelRef::parse(model).ok()?;
    if let infr_hub::ModelRef::Path(p) = &r {
        return file_name(p);
    }
    let cached = infr_hub::Store::discover()
        .ok()
        .and_then(|store| store.resolve(&r).ok().flatten());
    if let Some(p) = cached {
        return file_name(&p);
    }
    match r {
        infr_hub::ModelRef::Repo { sel: Some(sel), .. }
            if sel.to_ascii_lowercase().ends_with(".gguf") =>
        {
            file_name(Path::new(&sel))
        }
        _ => None,
    }
}

/// The model file to score and its gate runs. A model with no row is refused before `fetch`
/// (which may download it) wherever its file name is known without fetching
/// ([`name_without_fetching`]), and after it otherwise.
fn target(
    model: &str,
    fetch: impl FnOnce(&str) -> anyhow::Result<PathBuf>,
) -> anyhow::Result<(PathBuf, Vec<(&'static str, PathClass)>)> {
    if let Some(name) = name_without_fetching(model) {
        if gate::runs_for(&name).is_none() {
            return Err(no_row_error(&name));
        }
    }
    let gguf = fetch(model)?;
    let table = table_name(&gguf)?;
    let runs = gate::runs_for(table).ok_or_else(|| no_row_error(table))?;
    Ok((gguf, runs))
}

/// `infr tune check --against cpu-reference`: the tolerance gate's evaluation
/// (`infr_llama::tolerance::gate::check`, the function the `gpu_seam_tolerance_*` tests call) on
/// one model and the selected Vulkan device, under the configuration this invocation resolved,
/// so a tuning override given by `--set`, `INFR_TUNE` or a tuning file is in effect in every run.
pub(crate) fn run(model: &str, json: bool, cfg: &Arc<Config>) -> anyhow::Result<()> {
    let Backend::Vulkan(_) = selected_backend(cfg)? else {
        anyhow::bail!(
            "`infr tune check` scores a Vulkan device, and the selected device is not one \
             (`--dev` / INFR_DEV / `[device] dev`)"
        );
    };
    let (gguf, runs) = target(model, |m| Ok(resolve(m, cfg)?.0))?;
    let table = table_name(&gguf)?;
    let (device, tune_hash) = {
        let be = tune_backend(cfg)?;
        let report = be.tuning_report();
        let index = infr_vulkan::VulkanBackend::enumerate_devices(cfg)
            .map_err(|e| anyhow!("{e}"))?
            .into_iter()
            .find(|d| d.is_default_pick)
            .map(|d| format!("Vulkan{} ", d.index))
            .unwrap_or_default();
        (format!("{index}{}", report.device.name), report.hash())
    };
    // The GGUF's own vocabulary, never a `tokenizer.json` beside it: the rows were measured on
    // the prompts as the embedded tokenizer splits them. The reference is loaded under the
    // shipped defaults, so nothing this invocation sets (a KV format, say) moves it; the device
    // runs under everything it sets.
    let load = |ecfg: Arc<Config>| -> anyhow::Result<gate::Case> {
        gate::Case::of_model(infr_llama::SeamModel::load_with(&gguf, None, ecfg)?)
    };
    let reference = load(Arc::new(Config::default()))?;
    let case = load(Arc::clone(cfg))?;
    let identity = FileIdentity::of(&case.model);
    gate::check_file(table, &identity).map_err(|e| {
        anyhow!("{e}. Rows are found by file name only; nothing was scored and this is not a pass")
    })?;
    let mut report = CheckReport {
        row: table.to_string(),
        file: gguf.clone(),
        identity,
        device,
        tune_hash,
        runs: Vec::new(),
        greedy: Vec::new(),
        failures: Vec::new(),
        not_exercised: Vec::new(),
    };
    tracing::info!(
        "checking {} on {} (tune:{}) against the CPU reference: {} configurations on {} \
         prompts, then the greedy check",
        gguf.display(),
        report.device,
        report.tune_hash,
        runs.len(),
        case.prompts.len()
    );
    tracing::info!(
        "the rows of `{table}` were found by {ROW_MATCH} only, and no check ties them to this \
         file's contents beyond its kind of weights; the file says: {}",
        report.identity_text()
    );
    let verdict = gate::check(table, table, &reference, &case, &runs, &mut |e| match e {
        Event::Time { .. } => tracing::info!("{e}"),
        // Every prompt's values: diagnostics, off by default.
        Event::Values(_) => tracing::debug!("{e}"),
    })?;
    let class = |c: PathClass| format!("{c:?}");
    report.runs = verdict
        .forced
        .runs
        .iter()
        .map(|r| CheckSection {
            config: r.name.clone(),
            class: class(r.class),
            overridden: r.overridden.clone(),
            checks: r
                .checks()
                .into_iter()
                .map(|(phase, c)| (phase.to_string(), c))
                .collect(),
        })
        .collect();
    report.greedy = verdict
        .greedy
        .runs
        .iter()
        .map(|r| CheckSection {
            config: r.name.clone(),
            class: class(r.class),
            // The greedy check runs a configuration the forced runs ran too.
            overridden: verdict
                .forced
                .runs
                .iter()
                .find(|f| f.name == r.name)
                .map(|f| f.overridden.clone())
                .unwrap_or_default(),
            checks: r
                .checks()
                .into_iter()
                .map(|c| ("tokens".to_string(), c))
                .collect(),
        })
        .collect();
    report.failures = verdict.failures();
    report.not_exercised = verdict.not_exercised();
    if json {
        // print-ok: program OUTPUT — the `infr tune check --json` document.
        println!("{}", serde_json::to_string_pretty(&report.json())?);
    } else {
        // print-ok: program OUTPUT — the `infr tune check` table.
        print!("{}", report.text());
    }
    if report.pass() {
        Ok(())
    } else {
        Err(OutOfTolerance.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{cli_flag_layer, exit_status, Against, Cli, Cmd, TuneCmd};
    use clap::Parser;

    fn tune_check(args: &[&str]) -> Result<TuneCmd, clap::Error> {
        let argv = ["infr", "tune", "check"].iter().chain(args);
        match Cli::try_parse_from(argv)?.cmd {
            Some(Cmd::Tune { action }) => Ok(action),
            _ => panic!("`infr tune check` parsed as another command"),
        }
    }

    /// `infr tune check` needs a model, takes `cpu-reference` as its only reference (the
    /// default), and anything else is a usage error rather than a check of something else.
    #[test]
    fn tune_check_takes_a_model_and_the_cpu_reference() {
        use clap::error::ErrorKind;
        let missing = tune_check(&[]).err().expect("no model is an error");
        assert_eq!(missing.kind(), ErrorKind::MissingRequiredArgument);
        assert_eq!(missing.exit_code(), 2);
        for args in [
            &["m.gguf"][..],
            &["m.gguf", "--against", "cpu-reference"],
            &[
                "--against=cpu-reference",
                "m.gguf",
                "--dev",
                "Vulkan1",
                "--json",
            ],
        ] {
            let Ok(TuneCmd::Check { model, against, .. }) = tune_check(args) else {
                panic!("{args:?} did not parse");
            };
            assert_eq!((model.as_str(), against), ("m.gguf", Against::CpuReference));
        }
        let unknown = tune_check(&["m.gguf", "--against", "cpu"])
            .err()
            .expect("an unknown reference is an error");
        assert_eq!(unknown.kind(), ErrorKind::InvalidValue);
        assert_eq!(unknown.exit_code(), 2);
        // The help states the exit statuses the command uses.
        let help = tune_check(&["--help"]).err().expect("help").to_string();
        for status in [
            format!("{EXIT_OUT_OF_TOLERANCE} outside it"),
            format!(
                "{} when the check could not run",
                exit_status(&anyhow!("x"))
            ),
            format!("{} on a usage error", unknown.exit_code()),
        ] {
            assert!(help.contains(&status), "{status:?} missing from: {help}");
        }
    }

    /// The exit status `main` gives each way the command can end badly: outside tolerance has
    /// its own, a model with no row has the general failure's, and neither is clap's usage
    /// status.
    #[test]
    fn tune_check_exit_statuses() {
        let out = exit_status(&anyhow::Error::from(OutOfTolerance));
        let no_row = exit_status(&no_row_error("Some-Model-Q4_K_M.gguf"));
        let usage = tune_check(&["m.gguf", "--against", "cpu"])
            .err()
            .expect("a usage error")
            .exit_code();
        assert_eq!(out, EXIT_OUT_OF_TOLERANCE);
        assert_eq!((out, no_row, usage), (3, 1, 2));
        // An error that wraps the verdict in context is still the verdict.
        let wrapped = anyhow::Error::from(OutOfTolerance).context("while checking");
        assert_eq!(exit_status(&wrapped), EXIT_OUT_OF_TOLERANCE);
    }

    /// `infr tune check --dev` reaches the device selection like `show --dev`, and only a Vulkan
    /// device is accepted.
    #[test]
    fn tune_check_dev_selects_a_vulkan_device() {
        let layer = |dev: &str| {
            let action = tune_check(&["m.gguf", "--dev", dev]).expect("parses");
            cli_flag_layer(&Cmd::Tune { action })
        };
        assert_eq!(
            layer("Vulkan1").expect("a Vulkan device").device.dev,
            Some(Some("Vulkan1".to_string()))
        );
        assert!(layer("cpu").is_err());
    }

    /// A model with no row is an error that names the file it looked up and says nothing was
    /// scored; a model with rows gets its gate runs.
    #[test]
    fn tune_check_refuses_a_model_with_no_row() {
        let table = table_name(Path::new("/models/Some-Model-Q4_K_M.gguf")).expect("a name");
        assert_eq!(table, "Some-Model-Q4_K_M.gguf");
        assert!(gate::runs_for(table).is_none());
        let msg = no_row_error(table).to_string();
        for part in [
            "no row for `Some-Model-Q4_K_M.gguf`",
            "not a pass",
            "Qwen3-0.6B-Q4_K_M.gguf",
            "docs/gpu-tolerance.md",
        ] {
            assert!(msg.contains(part), "{part:?} missing from: {msg}");
        }
        assert!(!msg.contains("synthetic"), "{msg}");
        assert!(gate::runs_for("Qwen3-0.6B-Q4_K_M.gguf").is_some());
    }

    /// A repository no cache holds, for the references below.
    const NO_SUCH_REPO: &str = "infr-test-no-such-org/no-such-repo";

    /// A model whose file name the reference itself gives is refused for having no row before
    /// anything is fetched: an explicit `.gguf` selector of a repository that is not cached, and
    /// a path. A file name with rows goes on to the fetch, and so does a quant selector of an
    /// uncached repository, whose file name is not known until the hub is asked.
    #[test]
    fn tune_check_refuses_before_fetching() {
        let fetched = std::cell::Cell::new(false);
        let fetch = |_: &str| -> anyhow::Result<PathBuf> {
            fetched.set(true);
            Err(anyhow!("the test fetches nothing"))
        };
        for model in [
            format!("{NO_SUCH_REPO}:Some-Model-Q4_K_M.gguf"),
            format!("{NO_SUCH_REPO}:quants/Some-Model-Q4_K_M.gguf"),
            "/no/such/dir/Some-Model-Q4_K_M.gguf".to_string(),
        ] {
            let e = target(&model, fetch).expect_err("no row");
            assert!(
                e.to_string()
                    .contains("no row for `Some-Model-Q4_K_M.gguf`"),
                "{model}: {e}"
            );
            assert!(!fetched.get(), "{model} was fetched before it was refused");
        }
        for model in [
            format!("{NO_SUCH_REPO}:Qwen3-0.6B-Q4_K_M.gguf"),
            format!("{NO_SUCH_REPO}:Q4_K_M"),
        ] {
            fetched.set(false);
            let e = target(&model, fetch).expect_err("the fetch fails");
            assert!(e.to_string().contains("fetches nothing"), "{model}: {e}");
            assert!(fetched.get(), "{model} was refused without a fetch");
        }
        // After the fetch the name is the file's, and a file with no row is refused then.
        let e = target(&format!("{NO_SUCH_REPO}:Q4_K_M"), |_| {
            Ok(PathBuf::from("/cache/Some-Model-Q4_K_M.gguf"))
        })
        .expect_err("no row");
        assert!(e.to_string().contains("no row for"), "{e}");
    }

    fn report(measured: f64, failures: Vec<String>) -> CheckReport {
        CheckReport {
            row: "M.gguf".into(),
            file: "/models/M.gguf".into(),
            identity: FileIdentity {
                architecture: Some("qwen3".into()),
                name: Some("A Model".into()),
                quantized: true,
                tensors: 7,
                tensor_bytes: 4096,
            },
            device: "Vulkan0 Test GPU".into(),
            tune_hash: "abcd1234".into(),
            runs: vec![CheckSection {
                config: "gpu-float".into(),
                class: "Float".into(),
                overridden: Vec::new(),
                checks: vec![
                    ("decode".into(), Check::at_most("kl_max", measured, 1e-4)),
                    ("decode".into(), Check::at_least("top1_scored", 90.0, 16.0)),
                ],
            }],
            greedy: vec![CheckSection {
                config: "gpu".into(),
                class: "Int8".into(),
                overridden: Vec::new(),
                checks: vec![("tokens".into(), Check::at_most("token_misses", 0.0, 0.0))],
            }],
            failures,
            not_exercised: Vec::new(),
        }
    }

    /// The report's verdict is the evaluation's failure list, and the text and the JSON carry
    /// every check with its measured value, its bound and its own verdict.
    #[test]
    fn tune_check_report_says_pass_and_fail() {
        let green = report(2.5e-5, Vec::new());
        assert!(green.pass());
        let text = green.text();
        assert!(text.contains("gpu-float (Float)"), "{text}");
        assert!(text.contains("greedy decode, gpu (Int8)"), "{text}");
        assert!(
            text.contains("2.500e-5") && text.contains("<= 1.000e-4"),
            "{text}"
        );
        assert!(text.contains(">= 16"), "{text}");
        assert!(text.ends_with("verdict:  PASS\n"), "{text}");
        assert!(!text.contains("FAIL"), "{text}");
        assert!(!text.contains("not exercised"), "{text}");
        assert!(!text.contains("ran with"), "{text}");
        let j = green.json();
        assert_eq!(j["pass"], true);
        assert_eq!(j["against"], "cpu-reference");
        assert_eq!(j["tune_hash"], "abcd1234");
        let c = &j["runs"][0]["checks"][0];
        assert_eq!(
            (&c["phase"], &c["metric"], &c["limit"], &c["pass"]),
            (
                &serde_json::json!("decode"),
                &serde_json::json!("kl_max"),
                &serde_json::json!("at_most"),
                &serde_json::json!(true)
            )
        );
        assert_eq!(c["measured"], 2.5e-5);
        assert_eq!(c["bound"], 1e-4);
        assert_eq!(j["runs"][0]["checks"][1]["limit"], "at_least");
        assert_eq!(j["greedy"][0]["config"], "gpu");
        assert_eq!(j["not_exercised"], serde_json::json!([]));
        assert_eq!(j["runs"][0]["overridden"], serde_json::json!([]));

        let red = report(
            3.0e-3,
            vec!["gpu-float prompt 1 decode: KL 3.0000e-3 at position 40 over 1.0000e-4".into()],
        );
        assert!(!red.pass());
        let text = red.text();
        assert!(text.ends_with("verdict:  FAIL\n"), "{text}");
        assert!(text.contains("outside tolerance:"), "{text}");
        assert!(text.contains("at position 40"), "{text}");
        let line = text
            .lines()
            .find(|l| l.contains("kl_max"))
            .expect("the KL line");
        assert!(line.ends_with("FAIL"), "{line}");
        let j = red.json();
        assert_eq!(j["pass"], false);
        assert_eq!(j["runs"][0]["checks"][0]["pass"], false);
        assert_eq!(j["failures"].as_array().map(Vec::len), Some(1));
    }

    /// The report says how the row was found and what the file says of itself, which settings a
    /// configuration ran without, and what the verdict does not cover, in the text (the last
    /// beside the verdict) and in the JSON.
    #[test]
    fn tune_check_report_says_what_was_scored_and_what_was_not() {
        let mut r = report(2.5e-5, Vec::new());
        r.runs[0].overridden = vec![Overridden {
            path: "device.ubatch".into(),
            set: "77".into(),
            ran: "32".into(),
        }];
        r.not_exercised = vec!["M: no chained run is compared".into()];
        let text = r.text();
        assert!(
            text.starts_with("row:      M.gguf (matched by file name only)\n"),
            "{text}"
        );
        assert!(text.contains("file:     /models/M.gguf\n"), "{text}");
        for part in ["architecture qwen3", "\"A Model\"", "quantized weights"] {
            assert!(text.contains(part), "{part:?} missing from: {text}");
        }
        let lines: Vec<&str> = text.lines().collect();
        let at = |what: &str| {
            lines
                .iter()
                .position(|l| l.contains(what))
                .unwrap_or_else(|| panic!("{what:?} missing from: {text}"))
        };
        // The setting a configuration ran without is under that configuration's title.
        assert_eq!(
            at("ran with device.ubatch = 32, not the 77"),
            at("gpu-float (Float)") + 1
        );
        // What was not exercised is the block right above the verdict.
        assert_eq!(at("no chained run is compared") + 1, at("verdict:  PASS"));
        assert_eq!(at("not exercised") + 1, at("no chained run is compared"));
        let j = r.json();
        assert_eq!(
            j["row"],
            serde_json::json!({ "model": "M.gguf", "matched_by": "file name" })
        );
        assert_eq!(j["file"]["path"], "/models/M.gguf");
        assert_eq!(j["file"]["architecture"], "qwen3");
        assert_eq!(j["file"]["quantized"], true);
        assert_eq!(
            j["runs"][0]["overridden"],
            serde_json::json!([{ "path": "device.ubatch", "set": "77", "ran": "32" }])
        );
        assert_eq!(
            j["not_exercised"],
            serde_json::json!(["M: no chained run is compared"])
        );
        assert_eq!(j["pass"], true);
    }
}
