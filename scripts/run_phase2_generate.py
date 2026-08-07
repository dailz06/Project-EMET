"""Generate the Phase 2 Allen-Cahn dataset + validator test targets.

Usage:
    python scripts/run_phase2_generate.py [--config ...] [--skip-validator]

Outputs (data/):
    phase2_allen_cahn_dataset.npz        X, y, split, eps, snapshots
    phase2_allen_cahn_metadata.csv
    phase2_allen_cahn_generation_config.json
    phase2_validator_targets.npz         RK4@128^2 targets for test + ood_test
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fnocausal.common.config import config_hash, default_project_paths, load_yaml_config
from fnocausal.common.io_utils import save_dataset_npz
from fnocausal.common.seeding import set_seed
from fnocausal.sim.generate_dataset import generate_phase2_dataset, generate_validator_targets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--skip-validator", action="store_true")
    args = parser.parse_args()

    paths = default_project_paths()
    config_path = args.config or (paths.experiments / "phase2_allen_cahn.yaml")
    config = load_yaml_config(config_path)
    set_seed(int(config["seed"]))

    dataset_path = paths.data / "phase2_allen_cahn_dataset.npz"
    if dataset_path.exists():
        print(f"Dataset already exists: {dataset_path}. Delete it to regenerate.")
        return 1

    dataset = generate_phase2_dataset(config)

    split_counts = dataset["metadata"]["split"].value_counts().to_dict()
    print("Split counts:", split_counts)
    print("Held-out eps bounds:", dataset["heldout_bounds"])

    missing = [s for s in ("train", "val", "test", "ood_test") if split_counts.get(s, 0) == 0]
    if missing:
        raise RuntimeError(f"Empty splits: {missing}.")

    arrays = {
        "X": dataset["X"],
        "y": dataset["y"],
        "split": dataset["split"],
        "eps": dataset["metadata"]["eps"].to_numpy(dtype=np.float64),
    }
    for t, snap in dataset["snapshots"].items():
        arrays[f"snapshot_t{t:g}"] = snap

    generation_config = dict(config)
    generation_config["config_hash"] = config_hash(config)
    generation_config["heldout_eps_bounds"] = list(dataset["heldout_bounds"])
    generation_config["split_counts"] = split_counts

    save_dataset_npz(
        dataset_path,
        paths.data / "phase2_allen_cahn_metadata.csv",
        paths.data / "phase2_allen_cahn_generation_config.json",
        arrays,
        dataset["metadata"],
        generation_config,
    )

    # Example figure: one sample per split.
    metadata = dataset["metadata"]
    ids = metadata.groupby("split").head(1)["sample_id"].to_list()
    fig, axes = plt.subplots(len(ids), 3, figsize=(10, 3 * len(ids)))
    for row, idx in enumerate(ids):
        split_name = metadata.loc[idx, "split"]
        for col, (field, title, cmap) in enumerate(
            [
                (dataset["X"][idx, 0], f"u0 | {split_name} (eps={metadata.loc[idx, 'eps']:.3f})", "RdBu_r"),
                (dataset["X"][idx, 1], "M(x)", "viridis"),
                (dataset["y"][idx, 0], "u(T)", "RdBu_r"),
            ]
        ):
            im = axes[row, col].imshow(field, origin="lower", cmap=cmap)
            axes[row, col].set_title(title, fontsize=9)
            plt.colorbar(im, ax=axes[row, col], fraction=0.046)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    plt.tight_layout()
    plt.savefig(paths.figures / "phase2_dataset_examples.png", dpi=200)
    plt.close()
    print(f"Saved dataset examples: {paths.figures / 'phase2_dataset_examples.png'}")

    if not args.skip_validator:
        eval_mask = np.isin(dataset["split"], ["test", "ood_test"])
        eval_ids = np.where(eval_mask)[0]
        print(f"Generating validator targets for {len(eval_ids)} test/ood samples...")

        y_validator = generate_validator_targets(
            dataset["X"][eval_mask],
            arrays["eps"][eval_mask],
            config,
        )
        np.savez(
            paths.data / "phase2_validator_targets.npz",
            sample_ids=eval_ids,
            y_validator=y_validator,
        )
        print(f"Saved validator targets: {paths.data / 'phase2_validator_targets.npz'}")

        # Sanity: the two schemes should agree closely on the solved fields.
        native = dataset["y"][eval_mask]
        rel = np.linalg.norm(
            (native - y_validator).reshape(len(eval_ids), -1), axis=1
        ) / np.linalg.norm(native.reshape(len(eval_ids), -1), axis=1)
        print(f"Native-vs-validator target agreement: mean {rel.mean():.2e}, max {rel.max():.2e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
