"""Train the Phase 2 FNO surrogate on the Allen-Cahn dataset.

Usage:
    python scripts/run_phase2_train_fno.py [--config ...]

Outputs: checkpoints/phase2_fno_{best,last}.pt, logs/phase2_fno_training_log.csv,
figures/phase2_fno_training_curve.png, data/phase2_normalizers.npz.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fnocausal.analysis.eval_surrogate import plot_training_curve
from fnocausal.common.config import default_project_paths, load_yaml_config
from fnocausal.common.io_utils import load_dataset_npz
from fnocausal.common.normalization import (
    append_scalar_channel,
    compute_channel_stats,
    make_tensor_loader,
    normalize_array,
    save_normalizers,
)
from fnocausal.common.seeding import get_device, set_seed
from fnocausal.models.fno import create_fno_model
from fnocausal.models.train_loop import train_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    paths = default_project_paths()
    config_path = args.config or (paths.experiments / "phase2_allen_cahn.yaml")
    config = load_yaml_config(config_path)
    set_seed(int(config["seed"]))
    device = get_device()

    data = load_dataset_npz(
        paths.data / "phase2_allen_cahn_dataset.npz",
        paths.data / "phase2_allen_cahn_metadata.csv",
    )
    X, y, split = data["X"], data["y"], data["split"].astype(str)
    # epsilon is a required parametric input (not inferable from u0/M).
    X = append_scalar_channel(X, data["eps"])

    train_mask = split == "train"
    val_mask = split == "val"

    eps_norm = float(config["normalization_eps"])
    X_mean, X_std = compute_channel_stats(X[train_mask], eps=eps_norm)
    y_mean, y_std = compute_channel_stats(y[train_mask], eps=eps_norm)
    save_normalizers(paths.data / "phase2_normalizers.npz", X_mean, X_std, y_mean, y_std)

    X_norm = normalize_array(X, X_mean, X_std)
    y_norm = normalize_array(y, y_mean, y_std)

    loader_kwargs = dict(
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        pin_memory=bool(config["pin_memory"]),
    )
    train_loader = make_tensor_loader(X_norm[train_mask], y_norm[train_mask], shuffle=True, **loader_kwargs)
    val_loader = make_tensor_loader(X_norm[val_mask], y_norm[val_mask], shuffle=False, **loader_kwargs)

    model = create_fno_model(config)
    train_model(
        model,
        train_loader,
        val_loader,
        config,
        device,
        best_ckpt_path=paths.checkpoints / "phase2_fno_best.pt",
        last_ckpt_path=paths.checkpoints / "phase2_fno_last.pt",
        log_path=paths.logs / "phase2_fno_training_log.csv",
        checkpoint_dir=paths.checkpoints,
        log_prefix="[phase2_fno] ",
    )

    plot_training_curve(
        paths.logs / "phase2_fno_training_log.csv",
        paths.figures / "phase2_fno_training_curve.png",
        "Phase 2 FNO training curve (Allen-Cahn)",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
