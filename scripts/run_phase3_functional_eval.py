"""Gate 3 (revised): functional fidelity of the AE latent.

Why pixel rel-L2 is the wrong yardstick here: for saturated two-phase fields
the L2 error is dominated by interface placement - rel-L2 ~ sqrt(4 * delta * L)
for displacement delta over total interface length L, so a 5% target demands
delta ~ 0.01 grid cells from a z-dim-sized summary. That is informationally
impossible for GRF-tanh microstructures at any latent size swept (measured:
12%-70%, monotone in z, no elbow). It is also not what the Phase 5 audit
needs: the audit probes linear directions in z for causal/nuisance content,
so the latent must preserve the PHYSICAL OBSERVABLES the causal question is
about, not pixel geometry.

Revised gate (final_state AE at chosen z, val split):
    severity:        R^2( sigma(recon), sigma(true) )        >= 0.95
    interface length R^2( L(recon), L(true) )                >= 0.90

The pixel rel-L2 curve from run_phase3_train_ae.py remains part of the record.

Usage:
    python scripts/run_phase3_functional_eval.py [--config ...]
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fnocausal.common.config import (
    config_hash,
    default_project_paths,
    load_yaml_config,
    make_run_id,
    tag_results,
)
from fnocausal.common.io_utils import load_dataset_npz
from fnocausal.common.metrics import interface_length, transformed_area_fraction
from fnocausal.common.normalization import normalize_array
from fnocausal.common.seeding import get_device, set_seed
from fnocausal.models.autoencoder import ConvAutoencoder
from fnocausal.models.train_loop import load_checkpoint


def r_squared(pred: np.ndarray, true: np.ndarray) -> float:
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2)) + 1e-30
    return 1.0 - ss_res / ss_tot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    paths = default_project_paths()
    config_path = args.config or (paths.experiments / "phase3_autoencoder.yaml")
    config = load_yaml_config(config_path)
    set_seed(int(config["seed"]))
    device = get_device()
    run_id = make_run_id("phase3_functional", config)

    data = load_dataset_npz(
        paths.data / "phase2_allen_cahn_dataset.npz",
        paths.data / "phase2_allen_cahn_metadata.csv",
    )
    split = data["split"].astype(str)
    val_mask = split == "val"
    fields = data["y"][val_mask]              # (N, 1, nx, nx) u(T)
    domain = 1.0

    stats = np.load(paths.data / "phase3_ae_final_state_normalizers.npz")
    fields_norm = normalize_array(fields, stats["mean"], stats["std"])

    sigma_true = transformed_area_fraction(fields[:, 0])
    length_true = interface_length(fields[:, 0], domain)

    rows = []
    for z_dim in config["z_dims"]:
        ckpt_path = paths.checkpoints / f"phase3_ae_final_state_z{z_dim}_best.pt"
        model = ConvAutoencoder(in_channels=1, z_dim=int(z_dim),
                                base_channels=int(config["base_channels"]))
        ckpt = load_checkpoint(ckpt_path, device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device).eval()

        recons = []
        with torch.no_grad():
            for start in range(0, fields_norm.shape[0], 128):
                xb = torch.from_numpy(fields_norm[start:start + 128]).to(device)
                recons.append(model(xb).cpu().numpy())
        recon = np.concatenate(recons) * stats["std"] + stats["mean"]

        sigma_recon = transformed_area_fraction(recon[:, 0])
        length_recon = interface_length(recon[:, 0], domain)

        rows.append(
            {
                "variant": "final_state",
                "z_dim": int(z_dim),
                "severity_r2": r_squared(sigma_recon, sigma_true),
                "interface_length_r2": r_squared(length_recon, length_true),
                "severity_mae": float(np.abs(sigma_recon - sigma_true).mean()),
            }
        )
        print(f"z={z_dim}: severity R2 {rows[-1]['severity_r2']:.4f}, "
              f"interface-length R2 {rows[-1]['interface_length_r2']:.4f}, "
              f"severity MAE {rows[-1]['severity_mae']:.4f}")

    results_df = tag_results(pd.DataFrame(rows), run_id, config_hash(config), "infrastructure")
    out_csv = paths.results / "phase3_ae_functional_fidelity.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\nSaved functional-fidelity results: {out_csv}")

    plt.figure(figsize=(7, 5))
    plt.semilogx(results_df["z_dim"], results_df["severity_r2"], "o-", base=2, label="severity R2")
    plt.semilogx(results_df["z_dim"], results_df["interface_length_r2"], "s-", base=2,
                 label="interface-length R2")
    plt.axhline(0.95, color="gray", ls="--", lw=1, label="0.95 gate (severity)")
    plt.axhline(0.90, color="lightgray", ls=":", lw=1, label="0.90 gate (interface)")
    plt.xlabel("latent dimension z")
    plt.ylabel("R2 of reconstructed observable")
    plt.title("Phase 3 AE functional fidelity (val, final-state fields)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(paths.figures / "phase3_ae_functional_fidelity.png", dpi=200)
    plt.close()
    print(f"Saved figure: {paths.figures / 'phase3_ae_functional_fidelity.png'}")

    # Gate: smallest z meeting both criteria becomes the recommended chosen_z_dim.
    ok = results_df[(results_df["severity_r2"] >= 0.95)
                    & (results_df["interface_length_r2"] >= 0.90)]
    if len(ok):
        chosen = int(ok["z_dim"].min())
        print(f"\nGate 3 (functional) PASSED. Smallest passing z_dim: {chosen} "
              f"(update chosen_z_dim in phase3_autoencoder.yaml if different).")
        return 0

    print("\nGate 3 (functional) FAILED at every z_dim.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
