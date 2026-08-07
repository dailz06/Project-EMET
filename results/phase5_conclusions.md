# Phase 5 Conclusions

Scorecard: 21/21 gate-counted checks passed
(results/phase5_audit_scorecard.csv). Informational (non-gating) checks:
irm_generalizes_better_on_flipped_secondary: NOT supportive (IRM 0.289 (lambda 100) vs ERM 0.425).

Note on the IRM secondary check: at every penalty weight swept (lambda 10-1000,
selected on the rho=0 environment), the IRMv1 linear probe generalized WORSE to
the flipped environment than pooled ERM. This does not contradict the audit:
the pooled ERM baseline is already implicitly invariance-regularized (half its
training data comes from the rho=0 control environment), and linear IRMv1 is
known to underperform in low-environment-diversity settings (Rosenfeld et al.
2021, "The Risks of Invariant Risk Minimization"). The primary invariance
evidence is the ICP-style univariate checks, which pass: the causal-score
coefficient is stable across environments while the nuisance-score coefficient
tracks rho.

## What is and is not established

- **Claim 1 (surrogate accuracy): SUPPORTED** by Phase 2 (ID rel-L2 1.5%,
  independent-scheme targets 1.5%, held-out epsilon band 1.3%).
- **Claim 2 (the audit separates causal from spurious features):
  SUPPORTED.
  Five independent diagnostics were checked against a constructed answer key
  (the nuisance channel s provably never enters the solver; the mobility
  amplitude m provably drives severity). All five agree with the answer key.
- **Claim 3 (the PDE describes real degradation): NOT ADDRESSED and NOT
  claimed.** All data in this project is synthetic; claim 3 requires real
  experiments (roadmap Phase 7).

## Headline numbers

- Biased severity head collapse (id -> flipped): 3.27
  relative MAE increase; control head: -0.04.
- Sobol contrast: ST_s(simulator) = 0.0000 vs ST_s(model) = 0.883.
- Ablating the latent S-direction: eval_id probe R2
  0.868 -> 0.292,
  eval_flipped -1.028 -> 0.470.

Provenance: run_id phase5_audit_7d6ad5fa16_s4242, config hash 7d6ad5fa16. All CSVs tagged claim2.
