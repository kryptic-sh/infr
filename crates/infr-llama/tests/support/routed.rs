//! The routed-expert (MoE) half of the teacher-forced comparison (`forced.rs`, its parent): the
//! synthetic routed-expert model, a pair of runs' routing comparison, and the driver's self-check
//! of reading a routing back and running on a supplied one.

use super::{assert_same_bits, forced, forced_along, forced_reference, open_case, synthetic_llama};
use infr_llama::seam::{ForcedLogits, ForcedOn, PrefillHead};
use infr_llama::tolerance::RouteScore;
use infr_testkit::gguf::{Fill, Meta, SyntheticModel, TensorSpec};

/// The synthetic routed-expert model's key, and its name in the tolerance table.
pub const SYNTHETIC_MOE: &str = "synthetic-moe";

/// Experts of the synthetic MoE, how many each token uses, and their width.
const MOE_EXPERTS: usize = 8;
const MOE_USED: usize = 2;
const MOE_FF: usize = 128;
/// The router bias's amplitude: on the order of the gaps between the router's expert
/// probabilities, so it changes which experts some tokens pick.
const MOE_BIAS_AMP: f32 = 0.2;

/// [`synthetic_llama`] as an `ernie4_5-moe`: every layer's FFN replaced by a softmax router over
/// [`MOE_EXPERTS`] f32 experts with a selection bias (`exp_probs_b.bias`) and an ungated shared
/// expert, the layout of ERNIE-4.5-21B-A3B (B174's model) at toy size.
pub fn synthetic_moe() -> SyntheticModel {
    const ARCH: &str = "ernie4_5-moe";
    let llama = synthetic_llama(Fill::Rand(0.0));
    let ne = llama
        .tensors
        .iter()
        .find(|t| t.name == "token_embd.weight")
        .expect("token_embd")
        .shape[0];
    let mut meta: Vec<(String, Meta)> = llama
        .meta
        .into_iter()
        .map(|(k, v)| match k.strip_prefix("llama.") {
            Some(rest) => (format!("{ARCH}.{rest}"), v),
            None if k == "general.architecture" => (k, Meta::Str(ARCH.to_string())),
            None => (k, v),
        })
        .collect();
    let n_layer = meta
        .iter()
        .find_map(|(k, v)| match v {
            Meta::U32(n) if k == &format!("{ARCH}.block_count") => Some(*n as usize),
            _ => None,
        })
        .expect("block_count");
    for (k, v) in [
        ("expert_count", MOE_EXPERTS),
        ("expert_used_count", MOE_USED),
        ("expert_feed_forward_length", MOE_FF),
        ("expert_shared_feed_forward_length", MOE_FF),
        ("leading_dense_block_count", 0),
        ("interleave_moe_layer_step", 1),
    ] {
        meta.push((format!("{ARCH}.{k}"), Meta::U32(v as u32)));
    }
    let mut tensors: Vec<TensorSpec> = llama
        .tensors
        .into_iter()
        .filter(|t| {
            !["ffn_gate.", "ffn_up.", "ffn_down."]
                .iter()
                .any(|f| t.name.contains(f))
        })
        .collect();
    for l in 0..n_layer {
        let p = |s: &str| format!("blk.{l}.{s}");
        let r = |name: String, shape: Vec<usize>, amp: f32| {
            TensorSpec::new(name, shape, Fill::Rand(amp))
        };
        tensors.extend([
            r(p("ffn_gate_inp.weight"), vec![ne, MOE_EXPERTS], 0.3),
            r(
                p("ffn_gate_exps.weight"),
                vec![ne, MOE_FF, MOE_EXPERTS],
                0.2,
            ),
            r(p("ffn_up_exps.weight"), vec![ne, MOE_FF, MOE_EXPERTS], 0.2),
            r(
                p("ffn_down_exps.weight"),
                vec![MOE_FF, ne, MOE_EXPERTS],
                0.1,
            ),
            r(p("ffn_gate_shexp.weight"), vec![ne, MOE_FF], 0.2),
            r(p("ffn_up_shexp.weight"), vec![ne, MOE_FF], 0.2),
            r(p("ffn_down_shexp.weight"), vec![MOE_FF, ne], 0.1),
            r(p("exp_probs_b.bias"), vec![MOE_EXPERTS], MOE_BIAS_AMP),
        ]);
    }
    SyntheticModel { meta, tensors }
}

/// `test`'s own expert selections against `reference`'s at the near tie `near_tie`
/// (`gate::score_routing`); `None` for a model with no router-selected layer. Panics when only
/// one of the two runs carries a routing, or the two cannot be compared.
pub fn score_routing(
    reference: &ForcedLogits,
    test: &ForcedLogits,
    near_tie: f64,
) -> Option<RouteScore> {
    infr_llama::tolerance::gate::score_routing(reference, test, near_tie)
        .unwrap_or_else(|e| panic!("{e}"))
}

/// One machine-readable line for the measurement log with a pair's routing comparison, every
/// differing selection listed as `position/layer:margin`; empty for a model with no
/// router-selected layer.
pub fn route_line(tag: &str, reference: &ForcedLogits, test: &ForcedLogits) -> Option<String> {
    let s = score_routing(reference, test, 0.0)?;
    let flips: Vec<String> = s
        .flips
        .iter()
        .map(|f| format!("{}/{}:{:.3e}", f.pos, f.layer, f.margin))
        .collect();
    Some(format!(
        "ROUTE {tag} calls={} flips={} widest_flip={:.4e} narrowest={:.4e} at=[{}]",
        s.calls,
        s.flips.len(),
        s.widest_flip(),
        s.narrowest,
        flips.join(" ")
    ))
}

/// The routing half of the driver, on the CPU reference and the synthetic routed-expert model
/// (no GPU or cached model): the selections a run reads back are the ones its expert layers ran
/// on, and a supplied selection is the one they run on instead.
pub fn self_check_on_a_synthetic_moe() {
    let case = open_case(SYNTHETIC_MOE).expect("synthetic model");
    let (model, prompt, n) = (&case.model, &case.prompts[1], 10);
    let r = forced_reference(model, prompt, n);
    let own = r
        .routing
        .as_ref()
        .expect("a routed model reads its routing back");
    assert_eq!(
        own.positions(),
        r.seq.len() - 1,
        "one selection per fed token"
    );
    assert_eq!(own.layers, [0, 1]);
    assert_eq!(
        own.scores.len(),
        own.positions() * own.layers.len() * own.n_expert,
        "the CPU gives the selection scores"
    );

    // Forced along its own selections the run is the same run: the readback is what the expert
    // layers selected, and a supplied selection is computed on exactly like a router's own.
    let again = forced(model, ForcedOn::CpuReference, |_| {}, &r);
    assert_same_bits(
        &again.logits,
        &r.logits,
        r.vocab,
        "forced along its own routing",
    );
    assert_eq!(again.routing, r.routing, "its own selections moved");

    // One selection replaced, at a decode position: the last selected expert for the best one
    // the router rejected. Every row before it is untouched, its own row is not, and the router
    // there still reports the selection it would have made.
    let (at, layer) = (r.positions[r.prefill_rows] + 2, own.layers[0]);
    let scores = own.scores_at(at, layer).expect("scores");
    let want = own.ids_at(at, layer).expect("ids").to_vec();
    let rejected = (0..own.n_expert as u32)
        .filter(|e| !want.contains(e))
        .max_by(|a, b| scores[*a as usize].total_cmp(&scores[*b as usize]))
        .expect("more experts than a token uses");
    let mut other = own.clone();
    let slot = (at * own.layers.len()) * own.n_used + own.n_used - 1;
    assert_eq!(other.ids[slot], want[own.n_used - 1]);
    other.ids[slot] = rejected;
    let moved = forced_along(
        model,
        ForcedOn::CpuReference,
        |_| {},
        PrefillHead::All,
        &r,
        Some(&other),
    );
    let row = |g: &ForcedLogits, pos: usize| {
        let k = g.positions.iter().position(|&p| p == pos).expect("a row");
        g.logits[k * g.vocab..(k + 1) * g.vocab]
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>()
    };
    for pos in r.positions.iter().copied().filter(|&p| p < at) {
        assert_eq!(row(&moved, pos), row(&r, pos), "position {pos} moved");
    }
    assert_ne!(
        row(&moved, at),
        row(&r, at),
        "a supplied selection was not run"
    );
    let theirs = moved.routing.as_ref().expect("routing");
    assert_eq!(
        theirs.ids_at(at, layer),
        own.ids_at(at, layer),
        "the router's own selection follows the supplied one"
    );

    // The comparison sees that one replaced selection, at the margin between the two experts.
    let mut s = RouteScore::new(0.0);
    s.add(own, &other).expect("comparable");
    let flips: Vec<(usize, usize)> = s.flips.iter().map(|f| (f.pos, f.layer)).collect();
    assert_eq!(flips, [(at, layer)]);
    let margin = (scores[want[own.n_used - 1] as usize] - scores[rejected as usize]) as f64;
    assert_eq!(s.flips[0].margin, margin);
    assert!(
        s.violations()
            .iter()
            .any(|v| v.contains(&format!("{at}/{layer}"))),
        "{:?}",
        s.violations()
    );
}
