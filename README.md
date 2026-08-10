# Causal Discovery for Materials Degradation

A verified phase-field simulator, a Fourier neural operator (FNO) surrogate, and a five-part causal
audit that tests whether the surrogate learned the governing physics or a correlated artifact.

Identifying which measurable properties govern how fast a material degrades normally requires slow,
expensive characterization. A learned surrogate can stand in for that, but only if it depends on
genuine causes rather than on correlations that happen to hold in the training set: a model that
predicts well for the wrong reasons fails precisely when it is extrapolated to a new material.

This repository builds a fully controlled synthetic testbed in which the correct answer is known in
advance, then measures whether intervention-based diagnostics recover it. Every claim in the results
below is checked against that ground truth.

---

## Authors

- **Fudail ibn Umar Farooq Khawaja**, Purdue University ([@dailz06](https://github.com/dailz06))
- **Muhammad Ahmad**, Dartmouth College ([@StrawberryMilkyCoder](https://github.com/StrawberryMilkyCoder))

Both authors contributed equally.

Research conducted at CEDAR Lab, Thayer School of Engineering, Dartmouth College, advised by
[Bijan Mazaheri](https://engineering.dartmouth.edu/community/faculty/bijan-mazaheri).

---

## Results

| Stage | Outcome |
| --- | --- |
| Solver verification | Three independent schemes brought into agreement before any training data was generated; SBDF2 cross-scheme agreement at `7e-5` |
| Curvature law | Shrinking-circle test recovers `d(R^2)/dt = -2 eps^2` to within 0.8% of theory |
| Coarsening | `l^2` grows linearly in `eps^2 t`, `R^2 = 0.979` |
| Dataset generation | About 100x faster after a float32 CUDA rewrite, roughly 14 h to 25 min (GPU versus CPU solver) |
| FNO surrogate | 1.52% relative L2 error in distribution, 1.26% on a held-out `epsilon` band |
| Causal audit | All five diagnostics agreed with the per-sample ground-truth key across 28,000 simulations |

Because the two independent solvers agree to within `~9e-4`, essentially all of the surrogate's
residual error is attributable to the model rather than the solver.

### Audit scorecard

| Diagnostic | Key result |
| --- | --- |
| Shortcut collapse | Biased head error rises 327% under a flipped correlation (`0.038 -> 0.161`); control changes by `-4%`, oracle stays flat |
| Intervention probing | Latent responds to the nuisance at `R^2 = 0.976` versus `R^2 = 0.495` for the causal driver; the two directions sit 80.4 degrees apart |
| Invariance (ICP-style) | Causal-score coefficient stable across environments; nuisance coefficient tracks `rho` over a `~222x` sign-flipping swing |
| Sobol sensitivity | Nuisance total index `0.0000` in the physics versus `0.883` in the model; hidden driver `0.644` in the physics versus `0.0000` in the model |
| Latent ablation | Removing the nuisance direction restores adversarial validity (probe `R^2: -1.03 -> +0.47`) while leaving decoded physics about 96% intact |

The clearest signal is the Sobol sensitivity: the nuisance factor is negligible in the physics yet
dominant in the model, so the model's sensitivity structure inverts the true physics. Latent ablation
doubles as a mitigation rather than only a diagnostic.

**Reported as found.** One secondary check returned a negative result: an IRM-penalized probe
generalized worse than plain empirical risk minimization at every penalty weight. Two diagnostics
were initially mis-specified and corrected without changing any threshold, a multivariate stability
test replaced with ICP-correct univariate probes, and the IRM check reformulated around
out-of-environment risk.

---

## Problem setup

The synthetic system is the Allen-Cahn equation with a spatially varying mobility `M(x)`, solved on a
periodic `[0,1]^2` domain at `64x64`:

$$\frac{\partial u}{\partial t} = M(x)\left[\varepsilon^{2}\nabla^{2}u + u - u^{3} + g\right]$$

Here `u` is the order parameter, `eps` the interface width, and `g` a constant tilt of the
double-well potential.

| Stage | Configuration |
| --- | --- |
| Surrogate (phases 2-3) | `g = 0` (textbook), `M(x)` visible as an input channel, an `epsilon` band held out for OOD evaluation |
| Causal audit (phases 4-5) | `g > 0` so nucleated damage grows, `M(x) = m * GRF` hidden; severity `sigma` is a transformed area fraction of `u(T)` |

Hiding the mobility field is what makes the audit non-trivial: the nuisance factor carries genuine
predictive information in the biased environments, so a model can score well while relying on it.
Traveling-wave analysis in the nucleation-growth regime gives a front velocity `V = 3*M*eps*g/sqrt(2)`
and a critical radius independent of the mobility amplitude, so seeding every nucleus above that
radius makes severity monotonic in the hidden amplitude.

Read [`experiments/phase2_problem_statement.md`](experiments/phase2_problem_statement.md) first. It
defines the three-claims discipline that every result in this repository is tagged against.

---

## Setup

Requires Python 3.10+ and, for the CUDA path, an NVIDIA GPU with CUDA 12.1 drivers.

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate
```

```bash
# CUDA 12.1 build (used for all reported runs)
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# CPU-only alternative: slower, but the float64 NumPy solver path is the verified reference
# pip install torch==2.5.1 torchvision==0.20.1

pip install -r requirements.txt
```

---

## Reproducing the results

Each script gates the next. Run them in order and stop at any failed gate.

```bash
# Unit tests: GRF statistics, solver sanity, resampling round trip
python -m pytest scripts/tests -q
```

**Phase 2: simulator and surrogate**

```bash
python scripts/run_phase2_verify_solver.py    # Gate 2a: six physics checks
python scripts/run_phase2_generate.py         # 4000 samples + validator targets
python scripts/run_phase2_train_fno.py
python scripts/run_phase2_eval.py             # Gate 2b: ID / OOD / independent-scheme
```

**Phase 3: latent featurization**

```bash
python scripts/run_phase3_train_ae.py         # z-dim sweep + pixel reconstruction report
python scripts/run_phase3_functional_eval.py  # Gate 3: severity / interface R^2
```

**Phase 4: spurious structure**

```bash
python scripts/run_phase4_build_envs.py       # pool + environments + answer key
python scripts/run_phase4_train_models.py     # biased / control / oracle (+ AE)
```

**Phase 5: causal audit**

```bash
python scripts/run_phase5_audit.py            # five diagnostics + scorecard
```

Results CSVs carry `run_id`, `config_hash`, and claim columns, so any row can be traced back to the
configuration that produced it.

---

## Repository layout

| Path | Contents |
| --- | --- |
| `fnocausal/common` | Config and seeding (per-sample `SeedSequence` streams), Gaussian random fields, normalization, physics metrics, IO |
| `fnocausal/sim` | SBDF2 spectral solver (primary), RK4 validator, py-pde cross-check, Nyquist-correct Fourier resampling, dataset generation |
| `fnocausal/models` | FNO factory, convolutional autoencoder, shared training loop |
| `fnocausal/analysis` | Solver verification, surrogate evaluation, causal audit |
| `experiments/` | YAML configs, one per phase, plus the problem statement |
| `scripts/` | Entry points for each phase and the unit tests |
| `data/`, `figures/`, `logs/`, `results/` | Generated outputs |

---

## Design notes

Two surrogate decisions were learned the hard way and are worth recording:

- **Horizon length.** The initial surrogate used `T = 4` and memorized its training set while
  generalizing at only 17% error, because a one-shot map across an entire coarsening cascade is
  poorly conditioned. Shortening to `T = 1` resolved it; longer horizons are left to autoregressive
  composition.
- **A hidden input.** At `T = 1` the model still overfit at 11% error because `epsilon` was not
  supplied as an input, so the target was not a well-defined function of the inputs. Adding
  `epsilon` as a constant channel reduced error roughly sixtyfold, to 1.5%. That same
  hidden-variable structure later became a deliberate design element of the causal audit.

Verification came before training data, not after. The textbook coarsening exponent `l ~ t^(1/2)` is
unmeasurable in a unit box, so that check is stated in its reachable differential form rather than
reported as a measurement.

---

## License

<!-- TODO: add a LICENSE file at the repository root before relying on this line.
     MIT is the usual choice for research code. Repository owner's call. -->

Released under the MIT License. See [`LICENSE`](LICENSE).
