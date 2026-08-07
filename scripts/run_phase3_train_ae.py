"""Gate 3: train the convolutional autoencoder sweep on Phase 2 fields.

For each variant (input_stack: (u0, M); final_state: u(T)) and each z_dim,
trains a ConvAutoencoder and reports validation reconstruction relative L2.

Usage:
    python scripts/run_phase3_train_ae.py [--config ...]

Outputs: checkpoints/phase3_ae_<variant>_z<dim>_{best,last}.pt,
results/phase3_ae_reconstruction.csv, figures/phase3_ae_sweep.png,
figures/phase3_ae_examples.png.
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
from fnocausal.common.metrics import relative_l2_error_batch
from fnocausal.common.normalization import compute_channel_stats, make_tensor_loader, normalize_array
from fnocausal.common.seeding import get_device, set_seed
from fnocausal.models.autoencoder import ConvAutoencoder
from fnocausal.models.train_loop import train_model


def reconstruction_errors(model, fields_norm, mean, std, device, batch_size=64):
    """Per-sample relative L2 of AE reconstructions in unnormalized units."""
    model.eval()
    errors = []
    with torch.no_grad():
        for start in range(0, fields_norm.shape[0], batch_size):
            xb = torch.from_numpy(fields_norm[start:start + batch_size]).to(device)
            recon = model(xb)
            mean_t = torch.from_numpy(mean).to(device)
            std_t = torch.from_numpy(std).to(device)
            err = relative_l2_error_batch(recon * std_t + mean_t, xb * std_t + mean_t, eps=1e-6)
            errors.append(err.cpu().numpy())
    return np.concatenate(errors)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    paths = default_project_paths()
    config_path = args.config or (paths.experiments / "phase3_autoencoder.yaml")
    config = load_yaml_config(config_path)
    set_seed(int(config["seed"]))
    device = get_device()
    run_id = make_run_id("phase3_ae", config)

    data = load_dataset_npz(
        paths.data / "phase2_allen_cahn_dataset.npz",
        paths.data / "phase2_allen_cahn_metadata.csv",
    )
    split = data["split"].astype(str)
    train_mask = split == "train"
    val_mask = split == "val"

    variant_fields = {
        "input_stack": data["X"],                # (N, 2, nx, nx): u0, M
        "final_state": data["y"],                # (N, 1, nx, nx): u(T)
    }

    rows = []
    eps_norm = float(config["normalization_eps"])
    loader_kwargs = dict(
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        pin_memory=bool(config["pin_memory"]),
    )

    for variant in config["variants"]:
        fields = variant_fields[variant]
        mean, std = compute_channel_stats(fields[train_mask], eps=eps_norm)
        fields_norm = normalize_array(fields, mean, std)
        np.savez(paths.data / f"phase3_ae_{variant}_normalizers.npz", mean=mean, std=std)

        train_loader = make_tensor_loader(
            fields_norm[train_mask], fields_norm[train_mask], shuffle=True, **loader_kwargs
        )
        val_loader = make_tensor_loader(
            fields_norm[val_mask], fields_norm[val_mask], shuffle=False, **loader_kwargs
        )

        for z_dim in config["z_dims"]:
            tag = f"{variant}_z{z_dim}"
            print(f"\n=== Training AE {tag} ===")
            model = ConvAutoencoder(
                in_channels=fields.shape[1],
                z_dim=int(z_dim),
                base_channels=int(config["base_channels"]),
            )
            model = train_model(
                model,
                train_loader,
                val_loader,
                config,
                device,
                best_ckpt_path=paths.checkpoints / f"phase3_ae_{tag}_best.pt",
                last_ckpt_path=paths.checkpoints / f"phase3_ae_{tag}_last.pt",
                log_path=paths.logs / f"phase3_ae_{tag}_training_log.csv",
                log_prefix=f"[{tag}] ",
            )

            val_errors = reconstruction_errors(model, fields_norm[val_mask], mean, std, device)
            rows.append(
                {
                    "variant": variant,
                    "z_dim": int(z_dim),
                    "val_recon_rel_l2_mean": float(val_errors.mean()),
                    "val_recon_rel_l2_median": float(np.median(val_errors)),
                    "val_recon_rel_l2_std": float(val_errors.std()),
                }
            )
            print(f"[{tag}] val reconstruction rel-L2: {val_errors.mean():.4f}")

    results_df = tag_results(pd.DataFrame(rows), run_id, config_hash(config), "infrastructure")
    out_csv = paths.results / "phase3_ae_reconstruction.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\nSaved reconstruction sweep: {out_csv}")

    # Sweep figure
    plt.figure(figsize=(7, 5))
    for variant in config["variants"]:
        sub = results_df[results_df["variant"] == variant]
        plt.semilogx(sub["z_dim"], sub["val_recon_rel_l2_mean"], "o-", base=2, label=variant)
    plt.axhline(0.05, color="gray", ls="--", lw=1, label="5% gate")
    plt.xlabel("latent dimension z")
    plt.ylabel("val reconstruction relative L2")
    plt.title("Phase 3 AE latent-dimension sweep")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(paths.figures / "phase3_ae_sweep.png", dpi=200)
    plt.close()
    print(f"Saved sweep figure: {paths.figures / 'phase3_ae_sweep.png'}")

    # Example reconstructions at the chosen z_dim (final_state variant).
    chosen = int(config["chosen_z_dim"])
    fields = variant_fields["final_state"]
    mean, std = compute_channel_stats(fields[train_mask], eps=eps_norm)
    fields_norm = normalize_array(fields, mean, std)
    model = ConvAutoencoder(in_channels=1, z_dim=chosen, base_channels=int(config["base_channels"]))
    ckpt = torch.load(
        paths.checkpoints / f"phase3_ae_final_state_z{chosen}_best.pt",
        map_location=device, weights_only=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    val_ids = np.where(val_mask)[0][:4]
    with torch.no_grad():
        recon = model(torch.from_numpy(fields_norm[val_ids]).to(device)).cpu().numpy()
    recon = recon * std + mean

    fig, axes = plt.subplots(len(val_ids), 3, figsize=(9, 3 * len(val_ids)))
    for row, idx in enumerate(val_ids):
        for col, (img, title) in enumerate(
            [(fields[idx, 0], "u(T)"), (recon[row, 0], f"AE recon (z={chosen})"),
             (recon[row, 0] - fields[idx, 0], "error")]
        ):
            cmap = "RdBu_r" if col < 2 else "coolwarm"
            im = axes[row, col].imshow(img, origin="lower", cmap=cmap)
            axes[row, col].set_title(title, fontsize=9)
            plt.colorbar(im, ax=axes[row, col], fraction=0.046)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    plt.tight_layout()
    plt.savefig(paths.figures / "phase3_ae_examples.png", dpi=200)
    plt.close()
    print(f"Saved AE examples: {paths.figures / 'phase3_ae_examples.png'}")

    # Gate 3 is decided by FUNCTIONAL fidelity, not pixel rel-L2 (see
    # run_phase3_functional_eval.py and the chosen_z_dim note in the config):
    # pixel L2 on saturated two-phase fields is dominated by interface
    # placement and cannot reach the naive 5% at any swept latent size.
    print("\nSweep complete. Run run_phase3_functional_eval.py for the Gate 3 "
          "decision (severity / interface-length R2 of reconstructions).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
