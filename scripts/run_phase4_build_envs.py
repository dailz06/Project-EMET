"""Phase 4: simulate the pool, assemble environments, write the answer key.

Usage:
    python scripts/run_phase4_build_envs.py [--config ...]

Outputs (data/):
    phase4_pool.npz            u0, mobility, u_final, snapshots, severity
    phase4_answer_key.csv      per-sample: seeds, m, k, severity, s_level,
                               environment, target & realized rho
    phase4_generation_config.json

Gate 4 pre-checks performed here:
    (i)  severity is monotone in m (rank correlation on the pool, plus a
         matched-IC intervention check: same sample re-solved at m_lo/m_hi),
    (ii) severity has usable spread (not saturated at 0 or 1),
    (iii) 10 samples regenerate exactly from their stored (master_seed, id).
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scipy.stats import spearmanr

from fnocausal.common.config import config_hash, default_project_paths, load_yaml_config
from fnocausal.common.metrics import transformed_area_fraction
from fnocausal.common.seeding import set_seed
from fnocausal.sim.allen_cahn_spectral import solve_allen_cahn_imex
from fnocausal.sim.phase4_pool import (
    assign_environments,
    build_nuisance_channel,
    generate_phase4_pool,
    sample_phase4_inputs,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    paths = default_project_paths()
    config_path = args.config or (paths.experiments / "phase4_environments.yaml")
    config = load_yaml_config(config_path)
    set_seed(int(config["seed"]))

    pool_path = paths.data / "phase4_pool.npz"
    if pool_path.exists():
        print(f"Pool already exists: {pool_path}. Delete it to regenerate.")
        return 1

    pool = generate_phase4_pool(config)
    metadata = pool["metadata"]

    # --- Gate 4(i): severity monotone in m ---
    rho_pool, _ = spearmanr(metadata["m"], metadata["severity"])
    print(f"\nPool Spearman(m, severity) = {rho_pool:.3f} (marginal, mixed over ICs)")

    # Matched-IC interventions: re-solve 20 samples with m forced low/high,
    # everything else identical.
    n_check = 20
    deltas = []
    for i in range(n_check):
        sample = sample_phase4_inputs(config, i)
        texture = sample["mobility"] / sample["m"]
        results = {}
        for label, m_forced in (("lo", config["m_range"][0]), ("hi", config["m_range"][1])):
            solved = solve_allen_cahn_imex(
                sample["u0"],
                float(config["eps"]),
                float(config["t_final"]),
                float(config["solver_dt"]),
                domain_size=float(config["domain_size"]),
                mobility=(float(m_forced) * texture).astype(np.float32),
                g=float(config["g"]),
            )
            results[label] = float(transformed_area_fraction(solved["u_final"]))
        deltas.append(results["hi"] - results["lo"])

    deltas = np.array(deltas)
    monotone_ok = bool(np.all(deltas > 0))
    print(f"Matched-IC intervention: severity(m_hi) - severity(m_lo) > 0 for "
          f"{int((deltas > 0).sum())}/{n_check} samples (min delta {deltas.min():.4f})")

    # --- Gate 4(ii): severity spread ---
    sev = pool["severity"]
    spread_ok = bool((sev.std() > 0.05) and (sev.min() < 0.9) and (sev.max() > 0.1)
                     and ((sev > 0.98).mean() < 0.05) and ((sev < 0.02).mean() < 0.05))
    print(f"Severity: mean {sev.mean():.3f}, std {sev.std():.3f}, "
          f"range [{sev.min():.3f}, {sev.max():.3f}], "
          f"saturated high {(sev > 0.98).mean():.1%}, low {(sev < 0.02).mean():.1%}")

    # --- Gate 4(iii): regeneration from seeds ---
    regen_ok = True
    for i in np.linspace(0, int(config["pool_n"]) - 1, 10, dtype=int):
        sample = sample_phase4_inputs(config, int(i))
        if not (np.array_equal(sample["u0"], pool["u0"][i])
                and np.array_equal(sample["mobility"], pool["mobility"][i])):
            regen_ok = False
            print(f"Regeneration MISMATCH at sample {i}")
    print(f"Regeneration from (master_seed, sample_id): {'OK' if regen_ok else 'FAILED'}")

    # --- Environments + answer key ---
    answer_key = assign_environments(metadata, config)
    env_summary = (
        answer_key[answer_key["environment"] != "unused"]
        .groupby("environment")[["rho_target", "rho_realized_pearson", "rho_realized_spearman"]]
        .first()
    )
    counts = answer_key["environment"].value_counts()
    env_summary["n"] = counts
    print("\nEnvironments:")
    print(env_summary.to_string())

    # --- Nuisance channel, stored with the pool so every consumer sees the
    # exact same s fields (assigned samples only; unused rows get s = NaN
    # levels -> zero-filled fields and are never used downstream).
    s_levels = answer_key["s_level"].to_numpy()
    noise_rng = np.random.default_rng(int(config["seed"]) + 999)
    s_fields = build_nuisance_channel(np.nan_to_num(s_levels), config, noise_rng)

    # --- Save ---
    arrays = {
        "u0": pool["u0"],
        "mobility": pool["mobility"],
        "u_final": pool["u_final"],
        "severity": pool["severity"],
        "s_fields": s_fields,
    }
    for t, snap in pool["snapshots"].items():
        arrays[f"snapshot_t{t:g}"] = snap
    np.savez(pool_path, **arrays)
    answer_key.to_csv(paths.data / "phase4_answer_key.csv", index=False)

    generation_config = dict(config)
    generation_config["config_hash"] = config_hash(config)
    generation_config["pool_spearman_m_severity"] = float(rho_pool)
    generation_config["intervention_min_delta"] = float(deltas.min())
    with open(paths.data / "phase4_generation_config.json", "w") as f:
        json.dump({k: (list(v) if isinstance(v, tuple) else v) for k, v in generation_config.items()},
                  f, indent=2, default=str)

    print(f"\nSaved pool: {pool_path}")
    print(f"Saved answer key: {paths.data / 'phase4_answer_key.csv'}")

    # --- Figures: severity vs m, example fields ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sc = axes[0].scatter(metadata["m"], sev, c=metadata["n_seeds"], s=8, cmap="viridis", alpha=0.5)
    plt.colorbar(sc, ax=axes[0], label="n_seeds")
    axes[0].set_xlabel("hidden mobility amplitude m (causal driver C)")
    axes[0].set_ylabel("severity sigma = transformed area fraction")
    axes[0].set_title(f"sigma vs m (pool Spearman {rho_pool:.2f})")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(sev, bins=50)
    axes[1].set_xlabel("severity")
    axes[1].set_ylabel("count")
    axes[1].set_title("Severity distribution")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(paths.figures / "phase4_severity_vs_m.png", dpi=200)
    plt.close()

    ids = [0, 1, 2]
    fig, axes = plt.subplots(len(ids), 3, figsize=(10, 3 * len(ids)))
    for row, idx in enumerate(ids):
        for col, (field, title, cmap) in enumerate(
            [(pool["u0"][idx], f"u0 (k={metadata.loc[idx, 'n_seeds']})", "RdBu_r"),
             (pool["mobility"][idx], f"M(x) hidden (m={metadata.loc[idx, 'm']:.2f})", "viridis"),
             (pool["u_final"][idx], f"u(T), sigma={sev[idx]:.2f}", "RdBu_r")]
        ):
            im = axes[row, col].imshow(field, origin="lower", cmap=cmap)
            axes[row, col].set_title(title, fontsize=9)
            plt.colorbar(im, ax=axes[row, col], fraction=0.046)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    plt.tight_layout()
    plt.savefig(paths.figures / "phase4_pool_examples.png", dpi=200)
    plt.close()
    print(f"Saved figures: {paths.figures / 'phase4_severity_vs_m.png'}, "
          f"{paths.figures / 'phase4_pool_examples.png'}")

    passed = monotone_ok and spread_ok and regen_ok
    print(f"\nGate 4 pre-checks {'PASSED' if passed else 'FAILED'} "
          f"(monotone {monotone_ok}, spread {spread_ok}, regen {regen_ok})")
    print("Gate 4(iv) - shortcut adoption (biased beats control on ID) - is "
          "checked after run_phase4_train_models.py.")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
