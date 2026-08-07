"""Gate 2a: run the Allen-Cahn solver verification suite.

Usage:
    python scripts/run_phase2_verify_solver.py [--config experiments/phase2_allen_cahn.yaml]

Writes results/phase2_solver_verification.csv (one row per check) and
figures/phase2_verification_*.png. Exits nonzero if any check fails - dataset
generation must not proceed on an unverified solver.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fnocausal.analysis.verification import (
    check_coarsening,
    check_dt_and_grid_convergence,
    check_energy_decay,
    check_scheme_agreement,
    check_shrinking_circle,
)
from fnocausal.common.config import config_hash, default_project_paths, load_yaml_config, make_run_id, tag_results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    paths = default_project_paths()
    config_path = args.config or (paths.experiments / "phase2_allen_cahn.yaml")
    config = load_yaml_config(config_path)
    run_id = make_run_id("phase2_verify", config)

    dt = float(config["solver_dt"])

    print("Running Allen-Cahn solver verification suite (Gate 2a)...")

    checks = []

    print("[1/6] energy decay (M=1)...")
    energy = check_energy_decay(dt=dt)
    checks.append(energy)

    print("[2/6] energy decay (variable mobility, tilted well)...")
    energy_mob = check_energy_decay(dt=dt, with_mobility=True, g=0.3)
    checks.append(energy_mob)

    print("[3/6] shrinking circle...")
    circle = check_shrinking_circle(dt=dt)
    checks.append(circle)

    print("[4/6] coarsening exponent...")
    coarsening = check_coarsening(dt=dt)
    checks.append(coarsening)

    print("[5/6] dt and grid convergence...")
    convergence = check_dt_and_grid_convergence(dt=dt)
    checks.append(convergence)

    print("[6/6] scheme agreement (IMEX vs RK4 vs py-pde)...")
    agreement = check_scheme_agreement(dt=dt)
    checks.append(agreement)

    rows = []
    for check in checks:
        rows.append(
            {
                "check": check["name"],
                "metric": check["metric"],
                "value": str(check["value"]),
                "threshold": str(check["threshold"]),
                "passed": check["passed"],
            }
        )

    results_df = tag_results(pd.DataFrame(rows), run_id, config_hash(config), "claim1")
    out_csv = paths.results / "phase2_solver_verification.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\nSaved verification results: {out_csv}")
    print(results_df[["check", "value", "threshold", "passed"]].to_string(index=False))

    # --- Figures ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    for b in range(energy["energies"].shape[0]):
        axes[0].plot(energy["times"], energy["energies"][b], alpha=0.6, lw=1)
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("Ginzburg-Landau energy")
    axes[0].set_title("Energy decay (M=1, g=0)")

    axes[1].plot(circle["times"], circle["r_squared"], "o", ms=4, label="measured")
    fit = circle["r_squared"][0] + circle["fitted_slope"] * circle["times"]
    expected = circle["r_squared"][0] + circle["expected_slope"] * circle["times"]
    axes[1].plot(circle["times"], fit, "-", label=f"fit slope {circle['fitted_slope']:.2e}")
    axes[1].plot(circle["times"], expected, "--", label=f"theory -2eps^2 = {circle['expected_slope']:.2e}")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("R^2")
    axes[1].set_title("Shrinking circle")
    axes[1].legend(fontsize=8)

    axes[2].plot(coarsening["times"], coarsening["lengths"] ** 2, "o", ms=4, label="l(t)^2")
    window = coarsening["fit_window"]
    t_fit = coarsening["times"][window]
    slope = coarsening["slope_norm"] * 0.03**2
    l2_start = (coarsening["lengths"][window] ** 2)[0]
    axes[2].plot(t_fit, l2_start + slope * (t_fit - t_fit[0]), "--",
                 label=f"fit: slope = {coarsening['slope_norm']:.1f} eps^2 (R2={coarsening['r_squared']:.3f})")
    axes[2].set_xlabel("t")
    axes[2].set_ylabel("interface length scale squared")
    axes[2].set_title("Coarsening: l^2 linear in t")
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = paths.figures / "phase2_verification_suite.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved verification figure: {fig_path}")

    all_passed = all(check["passed"] for check in checks)
    print(f"\nGate 2a {'PASSED' if all_passed else 'FAILED'}.")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
