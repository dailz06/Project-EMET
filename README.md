# FNO Degradation Project - Phases 2-5 (Allen-Cahn -> Causal Audit)

Continuation of `c:\fno_project` (Phase 0+1: heterogeneous-diffusion FNO
surrogate). This repo: Allen-Cahn phase-field simulator + FNO surrogate
(Phase 2), autoencoder featurization (Phase 3), injected spurious structure
(Phase 4), and an intervention-based causal audit (Phase 5).

Read `experiments/phase2_problem_statement.md` first - it defines the
three-claims discipline every result is tagged with.

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Run order

Each script gates the next; run them in order and stop at any failed gate.

```powershell
# Unit tests (GRF stats, solver sanity, resampling round trip)
python -m pytest scripts\tests -q

# Phase 2
python scripts\run_phase2_verify_solver.py    # Gate 2a: 6 physics checks
python scripts\run_phase2_generate.py         # 4000 samples + validator targets
python scripts\run_phase2_train_fno.py
python scripts\run_phase2_eval.py             # Gate 2b: ID/OOD/independent-scheme

# Phase 3
python scripts\run_phase3_train_ae.py         # z-dim sweep + pixel recon report
python scripts\run_phase3_functional_eval.py  # Gate 3: severity/interface R2

# Phase 4
python scripts\run_phase4_build_envs.py       # pool + environments + answer key
python scripts\run_phase4_train_models.py     # biased / control / oracle (+ AE)

# Phase 5
python scripts\run_phase5_audit.py            # 5 diagnostics + scorecard
```

## Layout

- `fnocausal/common` - config/seeding (per-sample SeedSequence streams), GRFs,
  normalization, physics metrics, IO.
- `fnocausal/sim` - SBDF2 spectral solver (primary), RK4 validator, py-pde
  cross-check, Nyquist-correct Fourier resampling, dataset generation.
- `fnocausal/models` - FNO factory, conv autoencoder, shared training loop.
- `fnocausal/analysis` - solver verification, surrogate eval, causal audit.
- `experiments/` - YAML configs (one per phase) + problem statement.
- `data/ checkpoints/ figures/ logs/ results/` - outputs. Results CSVs carry
  `run_id`, `config_hash`, and `claim` columns.

## PDE

du/dt = M(x) * (eps^2 lap(u) + u - u^3 + g), periodic [0,1]^2, 64x64.

- Phase 2: g=0 (textbook), M(x) visible, epsilon band held out OOD.
- Phase 4: g>0 (tilted well so nucleated damage grows), M(x) = m * GRF hidden;
  severity sigma = transformed area fraction of u(T).
