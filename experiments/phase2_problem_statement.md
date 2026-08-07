# Phases 2-5 Problem Statement

Goal: model gradual material degradation with an Allen-Cahn phase-field FNO
surrogate, then test whether the features the pipeline relies on are causal
drivers of degradation or spurious correlates - the dogs-vs-wolves /
snow-in-the-background problem transplanted into a physics setting where the
data-generating factors are known exactly.

## Claims discipline (carried over from Phase 0)

Every result CSV in results/ carries a `claim` tag. The claims get different
verdicts on synthetic data:

1. **claim1 - the FNO is an accurate surrogate for the numerical solver.**
   Validatable now: train/val/test splits, a held-out epsilon band, and an
   independent-scheme test set (RK4 @128^2 vs the SBDF2 generator - the
   generator != validator discipline).

2. **claim2 - the audit separates causal from spurious features.**
   Validatable on synthetic data, and synthetic is the only clean way, because
   we build the answer key ourselves: the nuisance channel s(x) provably never
   enters the solver, while the hidden mobility amplitude m provably does.
   Detecting a nuisance we injected on purpose is not circular.

3. **claim3 - the PDE describes real degradation.**
   NOT validatable here and NOT claimed anywhere in this project. Requires
   real experiments.

## The load-bearing Phase 4 design choice: hide the causal driver

If the model sees all causal inputs (u0, M), the target u(T) is deterministic
in them and a competent surrogate simply ignores a nuisance channel - no
shortcut, no experiment. Instead the mobility field M(x) = m * lognormal-GRF
is HIDDEN from the audited models. Then u(T) is not determined by the visible
inputs, and in the biased environments the nuisance s(x) = s_level * P(x) + noise
carries real information about m (by constructed correlation with the severity
scalar sigma). The Bayes-optimal predictor on biased data uses s - guaranteeing
the shortcut is adopted, and guaranteeing collapse when the correlation flips.
This mirrors the wolf/husky example exactly: snow (S) predicts wolf only
because habitat (hidden C) co-occurs with snow.

Three models quantify it: biased FNO (u0, s; correlated envs), control FNO
(u0, s; uncorrelated env - its flip-drop should be ~0), oracle FNO (u0, M -
the accuracy upper bound).

## Phase scopes and exit gates

- **Phase 2**: verified Allen-Cahn solver (energy decay, shrinking circle,
  coarsening, convergence, cross-scheme agreement); 4000-sample dataset;
  FNO surrogate. Gate: ID rel-L2 <= 2%, validator-scheme error within 1.5x
  native, OOD epsilon band not catastrophic.
- **Phase 3**: conv autoencoder featurization (analysis tool only - the FNO
  operates on full fields). Gate (revised, see phase3_autoencoder.yaml):
  FUNCTIONAL fidelity of reconstructions at the chosen z_dim - severity R2
  >= 0.95 and interface-length R2 >= 0.90 - because pixel rel-L2 on saturated
  two-phase fields is dominated by interface placement and cannot reach the
  naive 5% target at any latent size; the audit needs the latent to preserve
  the physical observables, not pixel geometry. z-dim sweep documented.
- **Phase 4**: simulation pool with hidden m; environments at correlation
  rho in {0.95, 0.8 (train), 0.5, 0 (broken), -0.95 (flipped)}; answer-key CSV.
  Gate: sigma monotone in m; biased model beats control on ID (shortcut
  adopted); samples regenerable from stored seeds.
- **Phase 5**: five diagnostics - shortcut collapse, intervention-based latent
  probing, invariance across environments, Sobol sensitivity (simulator vs
  model contrast), latent ablation. Gate: diagnostics agree with each other
  and with the answer key.

## Core references

- Li et al., Fourier Neural Operator for Parametric PDEs, 2020.
- Geirhos et al., Shortcut Learning in Deep Neural Networks, 2020.
- Ribeiro et al., LIME / husky-vs-wolf example, 2016.
- Arjovsky et al., Invariant Risk Minimization, 2019.
- Peters et al., Invariant Causal Prediction, 2016.
- Allen & Cahn, 1979 (antiphase boundary motion); Bray, Theory of phase-ordering
  kinetics, 1994 (coarsening laws used in solver verification).
