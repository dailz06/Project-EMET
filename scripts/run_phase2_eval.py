"""Gate 2b: evaluate the Phase 2 FNO surrogate.

Scores the trained model on:
    - val / test (in-distribution, native SBDF2 targets)
    - ood_test (held-out high-epsilon band, native targets)
    - test_validator / ood_test_validator (SAME inputs, targets from the
      independent RK4@128^2 scheme) -- the generator != validator check.

Exit gate (from the plan): ID test rel-L2 <= 2%, validator-set error within
1.5x of the native test error, OOD band not catastrophically worse.

Usage:
    python scripts/run_phase2_eval.py [--config ...]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fnocausal.analysis.eval_surrogate import (
    plot_prediction_examples,
    predict_and_relative_errors,
    predict_fields,
)
from fnocausal.common.config import (
    config_hash,
    default_project_paths,
    load_yaml_config,
    make_run_id,
    tag_results,
)
from fnocausal.common.io_utils import load_dataset_npz
from fnocausal.common.normalization import (
    append_scalar_channel,
    load_normalizers,
    make_tensor_loader,
    normalize_array,
)
from fnocausal.common.seeding import get_device, set_seed
from fnocausal.models.fno import create_fno_model
from fnocausal.models.train_loop import load_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    paths = default_project_paths()
    config_path = args.config or (paths.experiments / "phase2_allen_cahn.yaml")
    config = load_yaml_config(config_path)
    set_seed(int(config["seed"]))
    device = get_device()
    run_id = make_run_id("phase2_eval", config)
    eps_norm = float(config["normalization_eps"])

    data = load_dataset_npz(
        paths.data / "phase2_allen_cahn_dataset.npz",
        paths.data / "phase2_allen_cahn_metadata.csv",
    )
    X, y, split = data["X"], data["y"], data["split"].astype(str)
    # epsilon is a required parametric input (not inferable from u0/M).
    X = append_scalar_channel(X, data["eps"])

    stats = load_normalizers(paths.data / "phase2_normalizers.npz")
    X_norm = normalize_array(X, stats["X_mean"], stats["X_std"])
    y_norm = normalize_array(y, stats["y_mean"], stats["y_std"])

    model = create_fno_model(config)
    checkpoint = load_checkpoint(paths.checkpoints / "phase2_fno_best.pt", device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    loader_kwargs = dict(batch_size=int(config["batch_size"]), num_workers=0, pin_memory=True)

    # --- Native-target evaluations ---
    rows = []
    for eval_name in ("val", "test", "ood_test"):
        mask = split == eval_name
        loader = make_tensor_loader(X_norm[mask], y_norm[mask], shuffle=False, **loader_kwargs)
        errors = predict_and_relative_errors(
            model, loader, stats["y_mean"], stats["y_std"], device, eps_norm
        )
        for sample_id, err in zip(np.where(mask)[0], errors):
            rows.append({"eval_set": eval_name, "sample_id": int(sample_id),
                         "relative_l2_error": float(err)})

    # --- Validator-target evaluations (same inputs, independent scheme) ---
    validator = np.load(paths.data / "phase2_validator_targets.npz")
    v_ids = validator["sample_ids"]
    y_v_norm = normalize_array(validator["y_validator"], stats["y_mean"], stats["y_std"])

    for eval_name in ("test", "ood_test"):
        in_set = np.isin(v_ids, np.where(split == eval_name)[0])
        loader = make_tensor_loader(
            X_norm[v_ids[in_set]], y_v_norm[in_set], shuffle=False, **loader_kwargs
        )
        errors = predict_and_relative_errors(
            model, loader, stats["y_mean"], stats["y_std"], device, eps_norm
        )
        for sample_id, err in zip(v_ids[in_set], errors):
            rows.append({"eval_set": f"{eval_name}_validator", "sample_id": int(sample_id),
                         "relative_l2_error": float(err)})

    results_df = tag_results(pd.DataFrame(rows), run_id, config_hash(config), "claim1")
    results_df.to_csv(paths.results / "phase2_fno_relative_l2_errors.csv", index=False)

    summary = (
        results_df.groupby("eval_set")["relative_l2_error"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    summary = tag_results(summary, run_id, config_hash(config), "claim1")
    summary.to_csv(paths.results / "phase2_fno_eval_summary.csv", index=False)
    print("\nRelative L2 summary:")
    print(summary.drop(columns=["run_id", "config_hash", "claim"]).to_string(index=False))

    # --- Exit-gate checks ---
    mean_err = summary.set_index("eval_set")["mean"]
    id_ok = mean_err["test"] <= 0.02
    validator_ok = mean_err["test_validator"] <= 1.5 * mean_err["test"]
    ood_ok = mean_err["ood_test"] <= 3.0 * mean_err["test"]

    print(f"\nGate 2b: ID test <= 2%: {id_ok} ({mean_err['test']:.4f})")
    print(f"Gate 2b: validator within 1.5x native: {validator_ok} "
          f"({mean_err['test_validator']:.4f} vs {mean_err['test']:.4f})")
    print(f"Gate 2b: OOD within 3x native: {ood_ok} ({mean_err['ood_test']:.4f})")

    # --- Figures ---
    test_ids = np.where(split == "test")[0][:2].tolist()
    ood_ids = np.where(split == "ood_test")[0][:2].tolist()
    example_ids = test_ids + ood_ids
    preds = predict_fields(model, X_norm[example_ids], stats["y_mean"], stats["y_std"], device)
    plot_prediction_examples(
        X[example_ids], y[example_ids], preds,
        sample_ids=list(range(len(example_ids))),
        output_path=paths.figures / "phase2_fno_prediction_examples.png",
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["val", "test", "test_validator", "ood_test", "ood_test_validator"]
    means = [mean_err[name] for name in order]
    plt.figure(figsize=(8, 5))
    plt.bar(order, means, color=["#4878d0", "#4878d0", "#6acc64", "#d65f5f", "#ee854a"])
    plt.ylabel("Mean relative L2 error")
    plt.title("Phase 2 FNO: ID vs OOD vs independent-scheme targets")
    plt.xticks(rotation=20)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(paths.figures / "phase2_fno_generalization.png", dpi=200)
    plt.close()
    print(f"Saved generalization figure: {paths.figures / 'phase2_fno_generalization.png'}")

    passed = bool(id_ok and validator_ok and ood_ok)
    print(f"\nGate 2b {'PASSED' if passed else 'FAILED'}.")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
