"""Phase 4: train biased / control / oracle FNOs, severity heads, and the
biased-environment autoencoder.

Model table (all tagged claim2):
    biased FNO    inputs (u0, s)  trained on train_rho95 + train_rho80
    control FNO   inputs (u0, s)  trained on control_train (rho = 0)
    oracle FNO    inputs (u0, M)  trained on train_rho95 + train_rho80
    biased head   (u0, s) -> sigma, same envs as biased FNO
    control head  (u0, s) -> sigma, control env
    biased AE     (u0, s) reconstruction, biased envs (Phase 5 probing target)

Gate 4(iv): the biased models must BEAT the control models on eval_id -
proof the shortcut was adopted. Checked at the end of this script.

Usage:
    python scripts/run_phase4_train_models.py [--config ...]
"""

import argparse
import sys
from pathlib import Path

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
from fnocausal.common.metrics import relative_l2_error_batch
from fnocausal.common.normalization import (
    compute_channel_stats,
    make_tensor_loader,
    normalize_array,
    save_normalizers,
)
from fnocausal.common.seeding import get_device, set_seed
from fnocausal.models.autoencoder import ConvAutoencoder
from fnocausal.models.fno import create_fno_model
from fnocausal.models.severity_head import SeverityHead
from fnocausal.models.train_loop import train_model

BIASED_TRAIN_ENVS = ("train_rho95", "train_rho80")
CONTROL_TRAIN_ENVS = ("control_train",)


def load_pool(paths):
    """Load the Phase 4 pool arrays and answer key."""
    npz = np.load(paths.data / "phase4_pool.npz")
    answer_key = pd.read_csv(paths.data / "phase4_answer_key.csv")
    return npz, answer_key


def build_inputs(npz, ids: np.ndarray, input_kind: str) -> np.ndarray:
    """
    Assemble model input stacks for the given sample ids.

    Inputs:
        npz: loaded pool npz with u0, mobility, s_fields.
        ids: np.ndarray of sample ids.
        input_kind: "u0_s" (audited models) or "u0_M" (oracle).

    Outputs:
        X: np.ndarray, (len(ids), 2, nx, nx) float32.
    """
    u0 = npz["u0"][ids]
    second = npz["s_fields"][ids] if input_kind == "u0_s" else npz["mobility"][ids]
    return np.stack([u0, second], axis=1)


def train_val_ids(answer_key, envs, val_fraction, rng):
    """Split the samples of the given environments into train/val ids."""
    ids = answer_key.loc[answer_key["environment"].isin(envs), "sample_id"].to_numpy()
    ids = ids.copy()
    rng.shuffle(ids)
    n_val = int(round(val_fraction * len(ids)))
    return ids[n_val:], ids[:n_val]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--skip-fno", action="store_true", help="only heads + AE")
    args = parser.parse_args()

    paths = default_project_paths()
    config_path = args.config or (paths.experiments / "phase4_environments.yaml")
    config = load_yaml_config(config_path)
    set_seed(int(config["seed"]))
    device = get_device()
    run_id = make_run_id("phase4_train", config)
    eps_norm = float(config["normalization_eps"])

    npz, answer_key = load_pool(paths)
    split_rng = np.random.default_rng(int(config["seed"]) + 321)
    val_fraction = float(config["val_fraction"])

    loader_kwargs = dict(
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        pin_memory=bool(config["pin_memory"]),
    )

    summary_rows = []

    model_specs = [
        ("biased", "u0_s", BIASED_TRAIN_ENVS),
        ("control", "u0_s", CONTROL_TRAIN_ENVS),
        ("oracle", "u0_M", BIASED_TRAIN_ENVS),
    ]

    # --- FNO field surrogates ---
    if not args.skip_fno:
        for role, input_kind, envs in model_specs:
            print(f"\n=== Training {role} FNO (inputs {input_kind}, envs {envs}) ===")
            train_ids, val_ids = train_val_ids(answer_key, envs, val_fraction, split_rng)

            X_train = build_inputs(npz, train_ids, input_kind)
            X_val = build_inputs(npz, val_ids, input_kind)
            y_train = npz["u_final"][train_ids][:, np.newaxis]
            y_val = npz["u_final"][val_ids][:, np.newaxis]

            X_mean, X_std = compute_channel_stats(X_train, eps=eps_norm)
            y_mean, y_std = compute_channel_stats(y_train, eps=eps_norm)
            save_normalizers(
                paths.data / f"phase4_fno_{role}_normalizers.npz", X_mean, X_std, y_mean, y_std
            )

            train_loader = make_tensor_loader(
                normalize_array(X_train, X_mean, X_std),
                normalize_array(y_train, y_mean, y_std),
                shuffle=True, **loader_kwargs,
            )
            val_loader = make_tensor_loader(
                normalize_array(X_val, X_mean, X_std),
                normalize_array(y_val, y_mean, y_std),
                shuffle=False, **loader_kwargs,
            )

            model = create_fno_model(config)
            model = train_model(
                model, train_loader, val_loader, config, device,
                best_ckpt_path=paths.checkpoints / f"phase4_fno_{role}_best.pt",
                last_ckpt_path=paths.checkpoints / f"phase4_fno_{role}_last.pt",
                log_path=paths.logs / f"phase4_fno_{role}_training_log.csv",
                log_prefix=f"[fno_{role}] ",
            )

            # Val relative L2 in physical units.
            model.eval()
            errs = []
            with torch.no_grad():
                X_val_norm = normalize_array(X_val, X_mean, X_std)
                for start in range(0, len(val_ids), 64):
                    xb = torch.from_numpy(X_val_norm[start:start + 64]).to(device)
                    pred = model(xb).cpu().numpy() * y_std + y_mean
                    err = relative_l2_error_batch(
                        torch.from_numpy(pred), torch.from_numpy(y_val[start:start + 64]), eps=1e-6
                    )
                    errs.append(err.numpy())
            val_rel = float(np.concatenate(errs).mean())
            summary_rows.append({"model": f"fno_{role}", "val_metric": "rel_l2", "val_value": val_rel})
            print(f"[fno_{role}] val rel-L2: {val_rel:.4f}")

    # --- Severity heads ---
    for role, envs in (("biased", BIASED_TRAIN_ENVS), ("control", CONTROL_TRAIN_ENVS)):
        print(f"\n=== Training {role} severity head ===")
        train_ids, val_ids = train_val_ids(answer_key, envs, val_fraction, split_rng)

        X_train = build_inputs(npz, train_ids, "u0_s")
        X_val = build_inputs(npz, val_ids, "u0_s")
        sev = answer_key.set_index("sample_id")["severity"]
        y_train = sev.loc[train_ids].to_numpy(dtype=np.float32)[:, np.newaxis]
        y_val = sev.loc[val_ids].to_numpy(dtype=np.float32)[:, np.newaxis]

        X_mean, X_std = compute_channel_stats(X_train, eps=eps_norm)
        np.savez(paths.data / f"phase4_head_{role}_normalizers.npz", X_mean=X_mean, X_std=X_std)

        head_config = dict(config)
        head_config["epochs"] = int(config["head_epochs"])

        train_loader = make_tensor_loader(
            normalize_array(X_train, X_mean, X_std), y_train, shuffle=True, **loader_kwargs
        )
        val_loader = make_tensor_loader(
            normalize_array(X_val, X_mean, X_std), y_val, shuffle=False, **loader_kwargs
        )

        head = SeverityHead(in_channels=2, base_channels=int(config["ae_base_channels"]))
        head = train_model(
            head, train_loader, val_loader, head_config, device,
            best_ckpt_path=paths.checkpoints / f"phase4_head_{role}_best.pt",
            last_ckpt_path=paths.checkpoints / f"phase4_head_{role}_last.pt",
            log_path=paths.logs / f"phase4_head_{role}_training_log.csv",
            log_prefix=f"[head_{role}] ",
        )

        head.eval()
        with torch.no_grad():
            preds = []
            X_val_norm = normalize_array(X_val, X_mean, X_std)
            for start in range(0, len(val_ids), 128):
                xb = torch.from_numpy(X_val_norm[start:start + 128]).to(device)
                preds.append(head(xb).cpu().numpy())
        preds = np.concatenate(preds)
        val_mae = float(np.abs(preds - y_val).mean())
        summary_rows.append({"model": f"head_{role}", "val_metric": "sigma_mae", "val_value": val_mae})
        print(f"[head_{role}] val sigma MAE: {val_mae:.4f}")

    # --- Biased-environment AE (Phase 5 probing target) ---
    print("\n=== Training biased-environment AE on (u0, s) ===")
    train_ids, val_ids = train_val_ids(answer_key, BIASED_TRAIN_ENVS, val_fraction, split_rng)
    X_train = build_inputs(npz, train_ids, "u0_s")
    X_val = build_inputs(npz, val_ids, "u0_s")

    X_mean, X_std = compute_channel_stats(X_train, eps=eps_norm)
    np.savez(paths.data / "phase4_ae_biased_normalizers.npz", X_mean=X_mean, X_std=X_std)

    ae_config = dict(config)
    ae_config["epochs"] = int(config["ae_epochs"])

    train_loader = make_tensor_loader(
        normalize_array(X_train, X_mean, X_std), normalize_array(X_train, X_mean, X_std),
        shuffle=True, **loader_kwargs,
    )
    val_loader = make_tensor_loader(
        normalize_array(X_val, X_mean, X_std), normalize_array(X_val, X_mean, X_std),
        shuffle=False, **loader_kwargs,
    )

    ae = ConvAutoencoder(
        in_channels=2, z_dim=int(config["ae_z_dim"]), base_channels=int(config["ae_base_channels"])
    )
    ae = train_model(
        ae, train_loader, val_loader, ae_config, device,
        best_ckpt_path=paths.checkpoints / "phase4_ae_biased_best.pt",
        last_ckpt_path=paths.checkpoints / "phase4_ae_biased_last.pt",
        log_path=paths.logs / "phase4_ae_biased_training_log.csv",
        log_prefix="[ae_biased] ",
    )

    summary = tag_results(pd.DataFrame(summary_rows), run_id, config_hash(config), "claim2")
    summary.to_csv(paths.results / "phase4_training_summary.csv", index=False)
    print(f"\nSaved training summary: {paths.results / 'phase4_training_summary.csv'}")
    print("Gate 4(iv) - shortcut adoption on eval_id - is verified inside "
          "run_phase5_audit.py (shortcut-collapse table).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
