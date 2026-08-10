# Technical Progress and Findings Report

**Project:** Modeling Material Degradation with Fourier Neural Operators and Auditing Feature Causality

**Authors:** Fudail ibn Umar Farooq Khawaja (Purdue University) and Muhammad Ahmad (Thayer School of
Engineering, Dartmouth College). Both authors contributed equally.

**Date:** 2026-07-12 · **Status:** Phases 0 to 5 complete, all exit gates passed. Phases 6 and 7 pending.

**Scope:** Phases 0 and 1 were developed in a separate repository that is now frozen; they are
summarised here for continuity. The code in this repository covers Phases 2 to 5.

---

## 1. Executive Summary

This project set out to answer a deceptively simple question in a setting where the answer can
actually be checked: *when a neural network learns physics, does it rely on the causal drivers of the
system or on spurious correlates?* This is the "wolf vs. snow" shortcut-learning problem transplanted
into computational physics, where synthetic data gives us something photographs never can, a
ground-truth answer key, because we decide which factors enter the governing equations.

**The project's full synthetic arc is complete.** All five build phases passed their pre-registered
exit gates:

| Milestone | Result |
|---|---|
| Verified Allen–Cahn phase-field solver | 6/6 physics checks (energy decay, curvature law within 0.8% of theory, coarsening scaling, convergence, cross-scheme agreement) |
| FNO surrogate accuracy (claim 1) | 1.52% relative L2 in-distribution; **1.53% against targets from an independent numerical scheme**; 1.26% on a held-out parameter band |
| Spurious-correlation sandbox (Phase 4) | Shortcut adoption proven: biased model beats matched control 0.038 vs 0.083 severity MAE on identical held-out data |
| Causal audit (claim 2) | **21/21 gate-counted checks passed** across five independent diagnostics, all agreeing with the constructed answer key |
| Headline single number | Sobol sensitivity of the nuisance factor: **0.0000 in the physics, 0.883 in the model** |
| Throughput | ~100× solver speedup (CUDA backends), dataset generation from ~14 h to ~25 min |

The claims discipline held throughout: claim 1 (surrogate accuracy) and claim 2 (the audit separates
causal from spurious features) are supported by evidence tagged per-result; claim 3 (the PDE describes
*real* degradation) is explicitly not made, and awaits real experimental data.

Beyond detection, the audit demonstrated **mitigation**: surgically removing the identified nuisance
direction from the learned representation restored predictive validity on adversarial data (probe R²
from −1.03 to +0.47) while leaving the physical content 96% intact.

---

## 2. Phase-by-Phase Breakdown

### Phases 0 and 1: Environment, Problem Statement, and Diffusion Warm-up *(frozen repository)*

**(a) Objectives.** Reproducible environment; a written problem statement separating the three claims;
de-risk the full ML pipeline on the easiest target (2D heterogeneous diffusion) before the physics
gets interesting.

**(b) Deliverables.** Single-script pipeline: py-pde diffusion solver, Gaussian-random-field (GRF)
input sampling, 3,000-sample dataset with a held-out correlation-length band, trained FNO
(neuraloperator), evaluation suite, and a problem-statement document. These artifacts live in the
frozen Phase 0 and 1 repository and are not included here.

**(c) Findings and decisions.** GRFs adopted as the principled input distribution (controllable
correlation length and spectral decay, giving clean OOD holdouts). FNO (16 modes, 64 hidden channels,
2.4M params) reached 1.23% ID and 1.20% OOD relative L2, establishing the architecture and training
recipe reused in every later phase.

### Phase 2: Allen–Cahn Simulator and FNO Surrogate

**(a) Objectives.** A *trustworthy* degradation simulator (Allen–Cahn:
`∂u/∂t = M(x)·(ε²∇²u + u − u³ + g)`), a surrogate accurate enough to stand in for it, and a
generator≠validator independence check.

**(b) Deliverables.** The modular `fnocausal` package replacing the Phase 1 monolith: `common/`
(config and hashing, per-sample `SeedSequence` rng streams, GRFs, physics metrics, normalization),
`sim/` (primary SBDF2 pseudo-spectral solver, explicit-RK4 validator solver, py-pde cross-check,
Nyquist-correct Fourier resampling), `models/` (FNO factory, shared training loop), `analysis/`
(6-check verification suite, evaluation). A 6,000-sample dataset (64×64, T=1) plus a full validator
test set re-solved with the independent scheme at 128², dt/4. 16 unit tests.

**(c) Findings and decisions.**

- **Scheme choice was measured, not assumed.** First-order semi-implicit stepping produced
  1.8×10⁻³ cross-scheme disagreement, over tolerance, so the primary solver was upgraded to
  second-order SBDF2 (7×10⁻⁵ at dt=2.5×10⁻³). Variable mobility is handled by the standard stabilized
  splitting (implicit constant-coefficient part at `M_max`).
- **Solver verification is load-bearing.** Energy monotonicity, the shrinking-circle law
  `d(R²)/dt = −2ε²` (0.8% deviation), coarsening scaling, dt and grid convergence, and three-way
  scheme agreement (SBDF2 / RK4 / py-pde) all passed before a single training sample was generated.
- **Gate 2b:** ID test 1.52%, independent-scheme targets 1.53% (the two schemes agree to ~9×10⁻⁴ on
  targets, so the surrogate error budget is entirely model, not solver), held-out ε-band 1.26%.
- Two significant course corrections, horizon length and a missing input channel, are detailed in §3.

### Phase 3: Autoencoder Featurization

**(a) Objectives.** A compact latent representation `z` of the material state to serve as the
*analysis substrate* for the causal audit. The FNO always operates on full fields; the AE is
instrumentation, not pipeline.

**(b) Deliverables.** Convolutional autoencoder (4× stride-2 encoder, GroupNorm/GELU, mirrored
decoder); a 10-model sweep (2 field variants × z ∈ {8…128}); pixel-reconstruction and
functional-fidelity reports with figures.

**(c) Findings and decisions.** The planned pixel-L2 gate (≤5%) is **informationally impossible** for
saturated two-phase fields: reconstruction error is dominated by interface placement
(rel-L2 ≈ √(4δL)), so 5% would demand ~0.01-cell interface accuracy from any latent. The gate was
re-anchored on what the audit actually requires, preservation of physical observables. At z=128:
severity R² 0.975, interface-length R² 0.932, pixel 12%. The full reasoning is documented in-config
and in a dedicated evaluation script, and the pixel curve remains on the record.

### Phase 4: Spurious-Structure Injection (the Answer Key)

**(a) Objectives.** A dataset where a nuisance factor S is *provably* non-causal and a driver C is
*provably* causal, with training environments where S correlates with the outcome and evaluation
environments where the correlation is weakened, broken, or reversed.

**(b) Deliverables.** An 8,000-simulation pool (nucleation-growth regime, tilted well g=0.3, hidden
mobility `M(x)=m·GRF`); seven environments (train ρ∈{0.95, 0.8}, control ρ=0,
eval ρ∈{0.95, 0.5, 0, −0.95}); a per-sample answer-key CSV; five trained models, namely biased FNO
(u₀+s), control FNO (u₀+s in an uncorrelated environment), oracle FNO (u₀+M), and two severity heads,
together with the biased-environment AE audited in Phase 5.

**(c) Findings and decisions.**

- **The load-bearing design insight: hide the causal driver.** If a model sees all causal inputs, the
  target is deterministic in them and a competent network ignores any nuisance, leaving no shortcut
  and no experiment. With `M(x)` hidden, the nuisance genuinely carries information in biased
  environments, so the Bayes-optimal predictor *must* use it, which guarantees both adoption and
  collapse under reversal. This exactly mirrors wolf/snow: snow predicts wolf only because hidden
  habitat co-occurs with it.
- **Monotonicity by construction, not luck.** Traveling-wave analysis of the tilted Allen–Cahn
  equation gives front velocity `V = M·ε·3g/√2` and critical radius `R_c = ε√2/(3g)`, both
  *independent of m*. With g=0.3 and ε=0.03 this puts R_c ≈ 0.047, so seeding all nuclei in
  [0.07, 0.13] makes severity monotone in the hidden amplitude for every sample (confirmed 20/20 in
  matched-IC interventions; severity spread 0.04–0.88 with zero saturation). The tilt also stays below
  the spinodal g\* = 2/(3√3) ≈ 0.385, above which the −1 phase would destabilize.
- **Adoption confirmed (Gate 4iv).** On held-out same-distribution data, biased beats control
  (FNO 0.250 vs 0.351 rel-L2; head 0.038 vs 0.083 MAE) while the oracle sits at 0.0077, reproducing
  the full designed ordering.

### Phase 5: The Causal Audit

**(a) Objectives.** Quantitatively separate causal from spurious reliance using at least two
corroborating methods, and check every verdict against the Phase 4 answer key.

**(b) Deliverables.** `run_phase5_audit.py` orchestrating five analysis modules; a scorecard CSV
(every check: expected pattern, measured value, verdict); an auto-generated conclusions document;
five figures.

**(c) Findings.** All five diagnostics agree with the answer key (21/21 gate-counted checks):

| Diagnostic | Key result |
|---|---|
| Shortcut collapse | Biased head error **+327%** under flipped correlation (0.038 to 0.161, monotone in ρ-shift); control −4%; oracle flat at 0.0073 |
| Intervention probing | Latent responds to nuisance interventions with R²=0.976, to causal interventions with R²=0.495; directions at 80.4° |
| Invariance (ICP-style) | Causal-score coefficient stable across environments (rel. range 0.145); nuisance-score coefficient tracks ρ (~222× swing, sign-flips on flipped data) |
| Sobol sensitivity | Nuisance total index: **0.0000 (simulator) vs 0.883 (model)**; hidden driver: 0.644 (simulator) vs 0.0000 (model) |
| Latent ablation | Removing the S-direction: eval-ID probe R² 0.868 to 0.292, flipped R² **−1.028 to +0.470**; decoded physics channel changes only 4.3% |

One secondary check produced an honest **negative result**: an IRMv1-penalized probe generalized
*worse* than pooled ERM at every penalty weight (0.289 vs 0.425 flipped-environment R², λ selected on
an independent environment). This is recorded as informational with its likely explanation: the
pooled baseline is already implicitly invariance-regularized, since half its data comes from the ρ=0
environment, and linear IRMv1 is documented to underperform in low-environment-diversity settings
(Rosenfeld et al., 2021).

---

## 3. Technical Challenges and Explanation Report

**C1: Long-horizon operator learning is chaotic (Phase 2).** The first surrogate trained on a T=4
horizon memorized the training set (train MSE 5.5×10⁻⁴) but generalized at 17%, with best validation
at epoch 6 of 100. Diagnosis: a T=4 one-shot map spans an entire coarsening cascade whose merger and
annihilation outcomes depend sensitively on the input. *Resolution:* horizon shortened to T=1 where
the operator is well-conditioned; longer horizons are reachable by composing the surrogate
autoregressively (future work). Documented as a config-level DECISION.

**C2: A hidden-parameter bug that later became the experiment (Phase 2).** Even at T=1, 11% error
with instant overfitting. Root cause: the PDE parameter ε was not an input and is not inferable from
the other channels, so the target literally was not a function of the model's inputs. *Resolution:*
ε fed as a constant channel (standard parametric-FNO practice), collapsing the error 60× to 1.5%.
Instructive symmetry: the same hidden-variable structure that was a *bug* in Phase 2 is deliberately
*constructed* in Phase 4 (hidden mobility), where the early-plateau training signature reappears by
design.

**C3: Physics checks mis-specified for a finite box (Phase 2).** The textbook coarsening exponent
(ℓ ~ t^½) is unmeasurable in a unit box, since domains reach a quarter of the box within t≈1, leaving
no scaling decade, and the structure-factor length is additionally biased by the Porod tail.
*Resolution:* the check was restated in the reachable regime as the differential form of the same law
(ℓ² linear in ε²t; measured R²=0.98, slope ≈9ε²) using an interface-density length measure.

**C4: Nyquist handling in spectral resampling (Phase 2).** Naive Fourier truncation and zero-padding
broke the exact up/down-sample round trip (2.6% error) by mishandling the ambiguous Nyquist bin.
*Resolution:* Nyquist-splitting resampling matrices (coefficient split on upsample, re-summed on
downsample), after which the round trip is exact and validator targets are alias-free.

**C5: The compute wall (Phases 2 to 4).** Single-threaded NumPy solves projected ~14 h for one
dataset plus its validator set. *Resolution:* float32 CUDA backends for both solvers (~100×, 25 min
total), with the float64 NumPy path retained as the verified reference and cross-backend agreement
unit-tested (<10⁻⁴). This is also what made the audit's 1,280 extra Sobol simulations and the Phase 4
pool cheap.

**C6: Ecosystem breakage (Phase 5).** SALib 1.5.1 passes its problem names into `pd.unique`, which
pandas 3.0 no longer accepts for plain lists. *Resolution:* a one-line fix (names as `ndarray`),
noted for future SALib use.

**C7: Two audit checks were initially mis-formalized (Phase 5).** (i) Requiring the *multivariate*
probe's causal coefficient to be stable across environments is not what ICP predicts: when a shortcut
is available, a full-representation fit legitimately shifts weight from causal to spurious
directions, and that reweighting is itself evidence of adoption. Replaced with the ICP-correct
univariate score probes, which pass cleanly; the multivariate instability is retained on the record
as corroborating evidence. (ii) The IRM check tested a coefficient ratio rather than IRM's operative
promise, out-of-environment risk; it was re-formalized with λ selected on an independent shifted
environment, and remained negative, hence the documented informational finding above. Both
re-formalizations are explained inline in the audit script, and **thresholds were never adjusted to
fit results**.

---

## 4. Project Velocity and Technical Debt

**Velocity.** Phases 2 to 5, comprising the solvers, a roughly 30-module package, the full model
suite, and the five-diagnostic audit, were delivered in three working sessions, with all heavy
computation fitting a single consumer GPU (RTX 2070). The exit-gate discipline caught every
significant defect *before* it could contaminate downstream phases.

**Quality posture.**

- 16 unit tests (GRF statistics, solver invariants, resampling round trip, backend agreement,
  stability guards); all green.
- Every run is config-driven and seeded; every results CSV carries `run_id`, `config_hash`, and a
  `claim` tag, giving full provenance for the eventual write-up.
- Reference-versus-bulk numerical split: float64 NumPy is the verified path, float32 CUDA is the
  throughput path, and the two are agreement-tested.
- Per-sample `SeedSequence` streams make any sample or intervention twin exactly regenerable from two
  integers.

**Known limitations, ordered by priority.**

1. **Single-seed results.** All headline metrics come from one master seed per phase. Cheap to fix now
   that the GPU backends exist: a 5-seed replication with dispersion estimates should precede any
   publication claim.
2. **No continuous integration.** Tests and gates run manually; a minimal pipeline (pytest and ruff
   on push) would protect the reference solvers from silent regression.
3. **Single-direction ablation.** Diagnostic 5 ablates one latent direction; a small-subspace variant
   (QR of the top S-responding directions) is specified but not implemented, and is worth adding for
   robustness even though the single-direction result passed decisively.
4. **Surrogate horizon.** The Phase 2 surrogate covers T=1; autoregressive composition to longer
   horizons is unvalidated.
5. **Mitigation is demonstrated only post-hoc** via latent ablation. Training-time mitigation
   (invariance-regularized training) was probed only through the IRM linear probe, which was
   inconclusive.
6. **Portability and structure.** Some paths assume Windows, and the audit script is long (~450
   lines) and could be split per-diagnostic.

---

## 5. Next Steps and Recommendations

1. **Phase 6a, cross-PDE replication (highest priority).** Repeat Phases 4 and 5 on Fisher–KPP
   (`∂u/∂t = D∇²u + ru(1−u)`; traveling fronts, different dynamical character). The solver and audit
   infrastructure is designed for this. A method that only works on Allen–Cahn is overfit to one
   equation, and this is the clearest outstanding weakness.
2. **Phase 6b, robustness sweeps.** Re-run the audit under observation noise, sensor sparsity, and
   resolution changes; confirm the verdicts survive realistic measurement conditions.
3. **Multi-seed replication.** Five master seeds across the Phase 2 to 5 headline metrics, reporting
   means and dispersion. Addresses limitation 1, estimated at under a day of compute.
4. **Mitigation experiment.** Compare naive training against an invariance-trained or
   ablation-regularized model on the Phase 4 sandbox, upgrading the project from *diagnosing*
   shortcut learning to *fixing* it. Given the IRM linear-probe result, prioritize
   representation-level interventions such as the demonstrated ablation over IRMv1-style penalties.
5. **Phase 7, write-up.** Assemble the paper around the three-claims framing; every figure and CSV
   already carries provenance tags. The negative IRM result and the two documented check
   re-formalizations belong in the paper, since they strengthen rather than weaken the methodology
   narrative.
6. **Housekeeping.** Stand up CI, add linting, and write a short results-directory README mapping each
   CSV to the claim it supports.

---

*Evidence trail: `results/` (provenance-tagged CSVs), `figures/`,
`experiments/phase2_problem_statement.md` (claims discipline), and
`results/phase5_conclusions.md` (audit verdict).*
