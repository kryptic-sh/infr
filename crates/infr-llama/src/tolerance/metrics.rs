//! Distances between two next-token logit rows, and their aggregate over a run.
//!
//! Every function here is pure and computes in f64 from the two f32 rows, so a score depends on
//! nothing but the logits it was given.

use infr_core::math;

/// The share of the reference's probability mass [`RowScore::topk_cos`] covers: the smallest set
/// of the reference's most probable tokens whose probabilities sum to at least this.
pub const TOPK_MASS: f64 = 0.99;

/// The most tokens [`RowScore::topk_cos`] covers, whatever mass they hold: a near-flat row (a
/// synthetic model's, or a position the model has no idea about) would otherwise take most of the
/// vocabulary, which is the whole-vocabulary cosine B60 showed cannot separate a match from a
/// mismatch.
pub const TOPK_MAX: usize = 64;

/// One position's distances between a reference row and a row under test.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RowScore {
    /// KL(reference ‖ test) over the full vocabulary, in nats: the expected extra log-loss of
    /// sampling from the test row where the reference was meant. Zero for identical rows and for
    /// rows that differ by a constant (softmax is shift-invariant).
    pub kl: f64,
    /// The cosine between the two rows' PROBABILITIES over the reference's top tokens (see
    /// [`TOPK_MASS`] and [`TOPK_MAX`]): 1 when the test row puts its mass where the reference does,
    /// in the same proportions.
    pub topk_cos: f64,
    /// The largest absolute difference between the two rows after each is centred on its own
    /// mean, divided by the reference row's range (max − min). Centring makes it shift-invariant;
    /// unlike log-probabilities (centring on the log-sum-exp), the mean does not move with the top
    /// logits, so one wrong top logit is counted once rather than as a shift of every token.
    pub logit_dev: f64,
    /// The reference's top-1 logit minus its top-2 logit, in nats: how far from a tie the
    /// reference's choice is. [`RunScore`] counts top-1 agreement only above a threshold.
    pub margin: f64,
    /// Whether the test row's argmax is the reference's (lowest index on an exact tie, both rows).
    pub top1_agrees: bool,
    /// How many tokens the test row ranks strictly above the reference's top-1 token: 0 when it is
    /// the test row's argmax too.
    pub ref_top1_rank: usize,
}

/// Why two rows could not be scored.
#[derive(Clone, Debug, PartialEq)]
pub enum RowError {
    /// The rows have different lengths.
    Shape { reference: usize, test: usize },
    /// The row is empty.
    Empty,
    /// A logit is NaN or infinite: `which` row, at vocabulary index `index`.
    NonFinite {
        which: &'static str,
        index: usize,
        value: f32,
    },
}

impl std::fmt::Display for RowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RowError::Shape { reference, test } => {
                write!(f, "row lengths differ: reference {reference}, test {test}")
            }
            RowError::Empty => write!(f, "empty logit row"),
            RowError::NonFinite {
                which,
                index,
                value,
            } => write!(f, "{which} logit {index} is {value}"),
        }
    }
}

impl std::error::Error for RowError {}

fn check_finite(row: &[f32], which: &'static str) -> Result<(), RowError> {
    match row.iter().position(|v| !v.is_finite()) {
        Some(index) => Err(RowError::NonFinite {
            which,
            index,
            value: row[index],
        }),
        None => Ok(()),
    }
}

/// The row's probabilities, in f64. Each exponential is `infr_core::math::exp`'s (the workspace
/// takes no math from the platform library, `docs/plan/11-cpu-bit-identical.md`), of the f32
/// nearest `x - max`: a relative error near 2^-23 per probability, far below any distance a
/// tolerance is set at, and normalized away from the sum.
fn softmax(row: &[f32]) -> Vec<f64> {
    let max = row.iter().fold(f64::NEG_INFINITY, |a, &v| a.max(v as f64));
    let w: Vec<f64> = row
        .iter()
        .map(|&v| math::exp((v as f64 - max) as f32) as f64)
        .collect();
    let s: f64 = w.iter().sum();
    w.iter().map(|x| x / s).collect()
}

/// Below this magnitude [`expm1`] sums its Taylor series.
const EXPM1_SERIES: f64 = 0.5;

/// Terms of `e^x - 1`'s Taylor series [`expm1`] sums: the remainder after them is below
/// `0.5^19 / 19!`, about `1.6e-23`, relative `3e-23` of the result at `|x| = 0.5`.
const EXPM1_TERMS: u32 = 18;

/// `e^x - 1` in f64 without the cancellation `exp(x) - 1` suffers for small `x`, which is where a
/// KL divergence between two nearly equal rows lives: the Taylor series below
/// [`EXPM1_SERIES`], and `infr_core::math::exp` (relative error 2^-24 of a result at least
/// 0.39 in magnitude) above it.
fn expm1(x: f64) -> f64 {
    if x.abs() < EXPM1_SERIES {
        // Horner on x(1 + x/2(1 + x/3(1 + ...))).
        (1..=EXPM1_TERMS)
            .rev()
            .fold(0.0, |acc, k| x / k as f64 * (1.0 + acc))
    } else {
        math::exp(x as f32) as f64 - 1.0
    }
}

/// KL(p ‖ q) for two logit rows `r` and `t` whose probabilities under the reference are `p`.
///
/// With `e = t - r`, `q_i = p_i e^{e_i} / Σ_j p_j e^{e_j}`, so `KL = ln E_p[e^e] - E_p[e]`, and
/// for any constant `c`, `KL = ln(1 + E_p[e^{e-c} - 1]) - E_p[e - c]`. Taking `c` near `E_p[e]`
/// leaves only the small deviations `e - c`, whose `e^x - 1` [`expm1`] gets to full relative
/// precision, so two rows a rounding apart score a KL of their real size rather than of the
/// sum's rounding. A row whose deviations reach the f32 exponential's overflow is far beyond any
/// tolerance; it is scored as infinite.
fn kl_divergence(p: &[f64], r: &[f32], t: &[f32]) -> f64 {
    let e = |i: usize| t[i] as f64 - r[i] as f64;
    let mass: f64 = p.iter().sum();
    let c = (0..p.len()).map(|i| p[i] * e(i)).sum::<f64>() / mass;
    if (0..p.len()).any(|i| p[i] > 0.0 && e(i) - c > OVERFLOW_ARG) {
        return f64::INFINITY;
    }
    let a: f64 = (0..p.len())
        .filter(|&i| p[i] > 0.0)
        .map(|i| p[i] * expm1(e(i) - c))
        .sum();
    let b: f64 = (0..p.len()).map(|i| p[i] * (e(i) - c)).sum();
    // Non-negative in exact arithmetic (Jensen); rounding can leave a tiny negative for rows
    // that differ by a constant, which is no distance.
    (math::ln_1p(a as f32) as f64 - b).max(0.0)
}

/// The largest argument [`kl_divergence`] takes an exponential of: below the f32 exponential's
/// overflow, so `e^x` stays finite.
const OVERFLOW_ARG: f64 = 80.0;

/// The argmax of `row`, the lowest index on an exact tie.
fn argmax(row: &[f32]) -> usize {
    (0..row.len()).fold(0, |b, i| if row[i] > row[b] { i } else { b })
}

/// Score `test` against `reference`, two logit rows over the same vocabulary.
pub fn score_row(reference: &[f32], test: &[f32]) -> Result<RowScore, RowError> {
    if reference.len() != test.len() {
        return Err(RowError::Shape {
            reference: reference.len(),
            test: test.len(),
        });
    }
    if reference.is_empty() {
        return Err(RowError::Empty);
    }
    check_finite(reference, "reference")?;
    check_finite(test, "test")?;
    let n = reference.len();
    let (pr, pt) = (softmax(reference), softmax(test));
    let kl = kl_divergence(&pr, reference, test);

    // The reference's top tokens, most probable first (index order on ties, so it is
    // deterministic), until they hold `TOPK_MASS` or `TOPK_MAX` is reached.
    // Only the first `TOPK_MAX` of that order are ever read, so select them (linear in the
    // vocabulary) and sort only those.
    let by_rank = |&a: &usize, &b: &usize| reference[b].total_cmp(&reference[a]).then(a.cmp(&b));
    let mut order: Vec<usize> = (0..n).collect();
    if n > TOPK_MAX {
        order.select_nth_unstable_by(TOPK_MAX - 1, by_rank);
        order.truncate(TOPK_MAX);
    }
    order.sort_by(by_rank);
    let (mut dot, mut nr, mut nt, mut mass) = (0f64, 0f64, 0f64, 0f64);
    for &i in order.iter().take(TOPK_MAX) {
        let (p, q) = (pr[i], pt[i]);
        dot += p * q;
        nr += p * p;
        nt += q * q;
        mass += p;
        if mass >= TOPK_MASS {
            break;
        }
    }
    // The test row can put no mass at all on the reference's top tokens (every probability
    // underflowing): that is as far apart as two rows get.
    let topk_cos = if nt > 0.0 {
        dot / (nr.sqrt() * nt.sqrt())
    } else {
        0.0
    };

    let mean = |row: &[f32]| row.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let (mr, mt) = (mean(reference), mean(test));
    let dmax = reference
        .iter()
        .zip(test)
        .map(|(&r, &t)| ((t as f64 - mt) - (r as f64 - mr)).abs())
        .fold(0f64, f64::max);
    let (rmin, rmax) = reference
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v as f64), hi.max(v as f64))
        });
    let range = rmax - rmin;
    // A constant reference row has no range: any deviation at all is then infinitely large
    // relative to it, and none is zero.
    let logit_dev = if range > 0.0 {
        dmax / range
    } else if dmax == 0.0 {
        0.0
    } else {
        f64::INFINITY
    };

    let top = argmax(reference);
    let second = (0..n)
        .filter(|&i| i != top)
        .fold(f64::NEG_INFINITY, |a, i| a.max(reference[i] as f64));
    let margin = reference[top] as f64 - second;
    let t_top = test[top];
    let ref_top1_rank = test.iter().filter(|&&v| v > t_top).count();
    Ok(RowScore {
        kl,
        topk_cos,
        logit_dev,
        margin,
        top1_agrees: argmax(test) == top,
        ref_top1_rank,
    })
}

/// The worst value of one metric over a run, and where it was.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Worst {
    pub value: f64,
    /// The position (token index in the scored sequence) it was measured at.
    pub pos: usize,
}

/// One metric's aggregate over a run: the worst position and the mean.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Agg {
    pub worst: Worst,
    pub sum: f64,
    pub n: usize,
}

impl Agg {
    /// An empty aggregate whose worst starts at `start` (the best value the metric can take).
    fn new(start: f64) -> Self {
        Agg {
            worst: Worst {
                value: start,
                pos: 0,
            },
            sum: 0.0,
            n: 0,
        }
    }

    /// Add `v` at `pos`; `worse(a, b)` says whether `a` is worse than `b`.
    fn add(&mut self, v: f64, pos: usize, worse: fn(f64, f64) -> bool) {
        if self.n == 0 || worse(v, self.worst.value) {
            self.worst = Worst { value: v, pos };
        }
        self.sum += v;
        self.n += 1;
    }

    /// The mean over the positions added (NaN when there were none).
    pub fn mean(&self) -> f64 {
        self.sum / self.n as f64
    }
}

/// Every metric's aggregate over the positions of one run (one prompt through one backend
/// configuration), with top-1 agreement counted only where the reference is not a near tie.
#[derive(Clone, Debug, PartialEq)]
pub struct RunScore {
    /// The reference top-2 margin (nats) at or below which a position is a near tie: top-1
    /// agreement is not counted there.
    pub near_tie: f64,
    pub kl: Agg,
    pub topk_cos: Agg,
    pub logit_dev: Agg,
    pub ref_top1_rank: Agg,
    /// Positions scored at all.
    pub positions: usize,
    /// Positions whose reference margin is above `near_tie`, where top-1 agreement is counted.
    pub top1_scored: usize,
    /// Of `top1_scored`, the positions whose top-1 disagrees, in order.
    pub top1_misses: Vec<usize>,
    /// Positions left out of the top-1 count as near ties.
    pub near_ties: usize,
    /// The largest reference margin at which the test's top-1 differed, counted or not (0 when it
    /// never did): how far from a tie a legitimate disagreement reaches is what a near-tie
    /// threshold has to clear.
    pub widest_miss: f64,
}

impl RunScore {
    pub fn new(near_tie: f64) -> Self {
        RunScore {
            near_tie,
            kl: Agg::new(0.0),
            topk_cos: Agg::new(1.0),
            logit_dev: Agg::new(0.0),
            ref_top1_rank: Agg::new(0.0),
            positions: 0,
            top1_scored: 0,
            top1_misses: Vec::new(),
            near_ties: 0,
            widest_miss: 0.0,
        }
    }

    /// Add the row score `s` measured at `pos`.
    pub fn add(&mut self, pos: usize, s: &RowScore) {
        let higher = |a: f64, b: f64| a > b;
        let lower = |a: f64, b: f64| a < b;
        self.kl.add(s.kl, pos, higher);
        self.topk_cos.add(s.topk_cos, pos, lower);
        self.logit_dev.add(s.logit_dev, pos, higher);
        self.ref_top1_rank.add(s.ref_top1_rank as f64, pos, higher);
        self.positions += 1;
        if !s.top1_agrees {
            self.widest_miss = self.widest_miss.max(s.margin);
        }
        if s.margin > self.near_tie {
            self.top1_scored += 1;
            if !s.top1_agrees {
                self.top1_misses.push(pos);
            }
        } else {
            self.near_ties += 1;
        }
    }

    /// Score every row of `test` against the same row of `reference` (both `[rows * vocab]`),
    /// row `r` recorded at position `positions[r]`. Fails on the first row that cannot be scored,
    /// naming its position.
    pub fn add_rows(
        &mut self,
        reference: &[f32],
        test: &[f32],
        vocab: usize,
        positions: &[usize],
    ) -> Result<(), String> {
        if reference.len() != test.len() || reference.len() != vocab * positions.len() {
            return Err(format!(
                "logit shapes differ: reference {}, test {}, expected {} rows of {vocab}",
                reference.len(),
                test.len(),
                positions.len()
            ));
        }
        for (r, &pos) in positions.iter().enumerate() {
            let row = r * vocab..(r + 1) * vocab;
            let s = score_row(&reference[row.clone()], &test[row])
                .map_err(|e| format!("position {pos}: {e}"))?;
            self.add(pos, &s);
        }
        Ok(())
    }

    /// Fold another run's positions into this one (both must use the same near-tie threshold).
    pub fn merge(&mut self, o: &RunScore) {
        assert_eq!(self.near_tie, o.near_tie, "merging runs scored differently");
        let fold = |a: &mut Agg, b: &Agg, worse: fn(f64, f64) -> bool| {
            if b.n > 0 && (a.n == 0 || worse(b.worst.value, a.worst.value)) {
                a.worst = b.worst;
            }
            a.sum += b.sum;
            a.n += b.n;
        };
        fold(&mut self.kl, &o.kl, |a, b| a > b);
        fold(&mut self.topk_cos, &o.topk_cos, |a, b| a < b);
        fold(&mut self.logit_dev, &o.logit_dev, |a, b| a > b);
        fold(&mut self.ref_top1_rank, &o.ref_top1_rank, |a, b| a > b);
        self.positions += o.positions;
        self.top1_scored += o.top1_scored;
        self.top1_misses.extend_from_slice(&o.top1_misses);
        self.near_ties += o.near_ties;
        self.widest_miss = self.widest_miss.max(o.widest_miss);
    }

    /// One line per metric, worst (with its position) and mean, for a test log.
    pub fn report(&self) -> String {
        let line = |name: &str, a: &Agg| {
            format!(
                "  {name:<14} worst {:>12.6e} at pos {:>4}  mean {:>12.6e}\n",
                a.worst.value,
                a.worst.pos,
                a.mean()
            )
        };
        format!(
            "{}{}{}{}  top-1          {} of {} agree above a {:.3}-nat margin ({} near ties left \
             out; widest miss at a {:.3}-nat margin){}\n",
            line("KL", &self.kl),
            line("top-k cos", &self.topk_cos),
            line("logit dev", &self.logit_dev),
            line("ref top-1 rank", &self.ref_top1_rank),
            self.top1_scored - self.top1_misses.len(),
            self.top1_scored,
            self.near_tie,
            self.near_ties,
            self.widest_miss,
            if self.top1_misses.is_empty() {
                String::new()
            } else {
                format!("; misses at {:?}", self.top1_misses)
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn identical_rows_are_at_distance_zero() {
        let r = [1.0f32, -2.0, 0.5, 3.0, 3.0 - 0.25];
        let s = score_row(&r, &r).unwrap();
        assert_eq!(s.kl, 0.0);
        assert!(close(s.topk_cos, 1.0, 1e-15), "{}", s.topk_cos);
        assert_eq!(s.logit_dev, 0.0);
        assert!(s.top1_agrees);
        assert_eq!(s.ref_top1_rank, 0);
        assert!(close(s.margin, 0.25, 1e-12), "{}", s.margin);
    }

    /// Softmax is shift-invariant, and so must every metric be: a backend that adds the same
    /// constant to every logit has made no error.
    #[test]
    fn a_constant_offset_is_no_distance() {
        let r = [1.0f32, -2.0, 0.5, 3.0, 2.0];
        let t: Vec<f32> = r.iter().map(|v| v + 7.0).collect();
        let s = score_row(&r, &t).unwrap();
        assert!(s.kl < 1e-15, "{}", s.kl);
        assert!(close(s.topk_cos, 1.0, 1e-15), "{}", s.topk_cos);
        assert!(s.logit_dev < 1e-12, "{}", s.logit_dev);
        assert!(s.top1_agrees);
    }

    /// Two tokens: KL between Bernoulli(p) and Bernoulli(q) has the closed form
    /// p ln(p/q) + (1−p) ln((1−p)/(1−q)).
    #[test]
    fn two_token_kl_matches_its_closed_form() {
        // Logits (0, a) give p1 = 1/(1+e^-a).
        let (a, b) = (1.0f64, 2.0f64);
        let p = 1.0 / (1.0 + (-a).exp());
        let q = 1.0 / (1.0 + (-b).exp());
        let want = p * (p / q).ln() + (1.0 - p) * ((1.0 - p) / (1.0 - q)).ln();
        let s = score_row(&[0.0, a as f32], &[0.0, b as f32]).unwrap();
        assert!(close(s.kl, want, want * 1e-6), "{} vs {want}", s.kl);
        // The probability cosine: both tokens are in the reference's top mass.
        let (pr, pt) = ([1.0 - p, p], [1.0 - q, q]);
        let cos = (pr[0] * pt[0] + pr[1] * pt[1])
            / ((pr[0] * pr[0] + pr[1] * pr[1]).sqrt() * (pt[0] * pt[0] + pt[1] * pt[1]).sqrt());
        assert!(close(s.topk_cos, cos, 1e-7), "{} vs {cos}", s.topk_cos);
        // Centred: reference (−a/2, a/2), test (−b/2, b/2); the difference is (b−a)/2 over a range
        // of a.
        assert!(
            close(s.logit_dev, (b - a) / 2.0 / a, 1e-7),
            "{}",
            s.logit_dev
        );
        assert!(s.top1_agrees);
    }

    /// Swapping the top two tokens is a top-1 miss, ranks the reference's choice second, and
    /// costs KL and cosine in proportion to the margin.
    #[test]
    fn a_permuted_top_two_is_seen_by_every_metric() {
        let r = [5.0f32, 3.0, 0.0, -1.0];
        let t = [3.0f32, 5.0, 0.0, -1.0];
        let s = score_row(&r, &t).unwrap();
        assert!(!s.top1_agrees);
        assert_eq!(s.ref_top1_rank, 1);
        assert!(close(s.margin, 2.0, 1e-12));
        // Closed form over the four tokens.
        let lse = |v: &[f64]| v.iter().map(|x| x.exp()).sum::<f64>().ln();
        let (rr, tt) = ([5.0, 3.0, 0.0, -1.0], [3.0, 5.0, 0.0, -1.0]);
        let (lr, lt) = (lse(&rr), lse(&tt));
        let want: f64 = (0..4)
            .map(|i| (rr[i] - lr).exp() * ((rr[i] - lr) - (tt[i] - lt)))
            .sum();
        assert!(close(s.kl, want, want * 1e-6), "{} vs {want}", s.kl);
        assert!(s.kl > 1.0);
        assert!(s.topk_cos < 0.5, "{}", s.topk_cos);
        // Means are equal; the largest centred difference is 2 over a range of 6.
        assert!(close(s.logit_dev, 2.0 / 6.0, 1e-12), "{}", s.logit_dev);
    }

    /// `e^x - 1` against reference values (to the digits written), on both sides of the switch
    /// from the series to the exponential.
    #[test]
    fn expm1_matches_known_values() {
        for (x, want) in [
            (1e-9, 1.000_000_000_5e-9),
            (1e-5, 1.000_005_000_016_666_7e-5),
            (-0.4, -0.329_679_953_964_360_7),
            (0.499, 0.647_073_373_515_35),
            (1.0, 1.718_281_828_459_045),
            (-3.0, -0.950_212_931_632_136),
        ] {
            let got = expm1(x);
            let rel = ((got - want) / want).abs();
            // The series is good to f64 rounding; the f32 exponential to 2^-24 of e^x.
            let tol = if x.abs() < EXPM1_SERIES { 1e-13 } else { 1e-7 };
            assert!(
                rel < tol,
                "expm1({x}) = {got}, want {want} (relative {rel:e})"
            );
        }
    }

    /// Two rows a hair apart: KL must come out at its true size, `Var_p(t - r) / 2` to second
    /// order, not at the size of the sum's rounding. A direct `Σ p (log p - log q)` built on f32
    /// exponentials would be off by its own rounding, about 1e-7, here a thousand times the
    /// answer.
    #[test]
    fn a_tiny_kl_is_measured_not_rounded() {
        let n = 2000;
        let r: Vec<f32> = (0..n)
            .map(|i| ((i * 37 % 101) as f32) * 0.15 - 7.0)
            .collect();
        let z = |i: usize| ((i * 53 % 97) as f64) / 48.0 - 1.0;
        let eps = 2e-4;
        let t: Vec<f32> = (0..n).map(|i| r[i] + (eps * z(i)) as f32).collect();
        let s = score_row(&r, &t).unwrap();
        // Second order, from the rows as stored (the f32 rounding of t is part of the input).
        let m = r.iter().fold(f64::NEG_INFINITY, |a, &v| a.max(v as f64));
        let w: Vec<f64> = r.iter().map(|&v| (v as f64 - m).exp()).collect();
        let sw: f64 = w.iter().sum();
        let e: Vec<f64> = (0..n).map(|i| t[i] as f64 - r[i] as f64).collect();
        let mean: f64 = (0..n).map(|i| w[i] / sw * e[i]).sum();
        let var: f64 = (0..n).map(|i| w[i] / sw * (e[i] - mean).powi(2)).sum();
        let want = var / 2.0;
        assert!(want > 1e-10 && want < 1e-7, "{want}");
        assert!(
            close(s.kl, want, want * 1e-2),
            "KL {} against {want} to second order",
            s.kl
        );
    }

    #[test]
    fn non_finite_logits_are_an_error_naming_the_index() {
        let r = [1.0f32, 2.0, 3.0];
        match score_row(&r, &[1.0, f32::NAN, 3.0]) {
            Err(RowError::NonFinite {
                which: "test",
                index: 1,
                value,
            }) if value.is_nan() => {}
            other => panic!("{other:?}"),
        }
        match score_row(&[f32::INFINITY, 0.0, 0.0], &r) {
            Err(RowError::NonFinite {
                which: "reference",
                index: 0,
                ..
            }) => {}
            other => panic!("{other:?}"),
        }
        let mut run = RunScore::new(0.1);
        let err = run
            .add_rows(
                &[0.0, 1.0, 0.0, 1.0],
                &[0.0, 1.0, f32::NEG_INFINITY, 1.0],
                2,
                &[7, 8],
            )
            .unwrap_err();
        assert!(
            err.contains("position 8") && err.contains("test logit 0"),
            "{err}"
        );
        assert!(score_row(&r, &[1.0, 2.0]).is_err());
    }

    /// Top-1 agreement is counted only above the near-tie margin, and the near ties are counted
    /// so a run that excluded everything is visible.
    #[test]
    fn near_ties_are_left_out_of_top1_and_counted() {
        let mut run = RunScore::new(0.5);
        // A near tie (margin 0.1) the test flips: not a miss.
        run.add_rows(&[1.0, 1.1], &[1.1, 1.0], 2, &[3]).unwrap();
        // A clear lead the test keeps, and one it flips.
        run.add_rows(&[0.0, 2.0, 0.0, 2.0], &[0.0, 2.0, 2.0, 0.0], 2, &[4, 5])
            .unwrap();
        assert_eq!(run.positions, 3);
        assert_eq!(run.near_ties, 1);
        assert_eq!(run.top1_scored, 2);
        assert_eq!(run.top1_misses, vec![5]);
        assert_eq!(run.widest_miss, 2.0);
        assert_eq!(run.ref_top1_rank.worst.value, 1.0);
        assert_eq!(run.kl.worst.pos, 5);
        assert!(run.report().contains("1 of 2 agree"), "{}", run.report());
    }

    /// The top-k cosine covers only the reference's top mass: a disagreement deep in the tail
    /// leaves it at 1 while KL and the logit deviation still see it.
    #[test]
    fn topk_cosine_ignores_the_tail() {
        let mut r = vec![-20.0f32; 200];
        r[0] = 10.0;
        r[1] = 9.0;
        let mut t = r.clone();
        t[150] = -5.0;
        let s = score_row(&r, &t).unwrap();
        assert!(close(s.topk_cos, 1.0, 1e-9), "{}", s.topk_cos);
        assert!(s.logit_dev > 0.4, "{}", s.logit_dev);
        assert!(s.kl > 0.0);
    }

    #[test]
    fn merging_keeps_the_worst_and_the_counts() {
        let mut a = RunScore::new(0.5);
        a.add_rows(&[0.0, 2.0], &[0.0, 1.9], 2, &[1]).unwrap();
        let mut b = RunScore::new(0.5);
        b.add_rows(&[0.0, 2.0], &[0.0, 1.0], 2, &[9]).unwrap();
        let kl_b = b.kl.worst.value;
        a.merge(&b);
        assert_eq!(a.positions, 2);
        assert_eq!(
            a.kl.worst,
            Worst {
                value: kl_b,
                pos: 9
            }
        );
        assert_eq!(a.kl.n, 2);
    }
}
