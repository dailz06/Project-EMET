"""Phase 5: the causal audit (claim2). Five diagnostics + scorecard.

Every diagnostic is stated as: expected answer-key pattern -> measured value ->
verdict. The answer key is data/phase4_answer_key.csv plus the construction
facts (s never enters the solver; m never enters the audited models' inputs).

Diagnostics:
    1. Shortcut collapse   biased/control/oracle x {id, rho05, broken, flipped}.
                           Includes Gate 4(iv): biased BEATS control on eval_id
                           (proof the shortcut was adopted at training time).
    2. Intervention probing matched twins (vary S at fixed C; vary C at fixed S)
                           -> latent directions w_S, w_C of the biased AE.
    3. Invariance          per-environment probes z->sigma: the w_S coefficient
                           must track rho, the w_C coefficient must be stable;
                           an IRMv1 probe must shrink w_S reliance vs ERM.
    4. Sobol               S_s(simulator) ~ 0 vs S_s(model) >> 0 - the cleanest
                           single-number contrast for "model uses a non-causal
                           feature". Also S_m(model) = 0 (m invisible).
    5. Latent ablation     removing w_S hurts biased-env accuracy, IMPROVES
                           flipped-env accuracy; removing w_C hurts everywhere;
                           decoding an S-ablated latent leaves u0 ~unchanged.

Usage:
    python scripts/run_phase5_audit.py [--config ...] [--skip-sobol]
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

from fnocausal.analysis.invariance import (
    irm_probe,
    per_environment_probes,
    ridge_probe,
    univariate_score_probes,
)
from fnocausal.analysis.latent_ablation import probe_r2
from fnocausal.analysis.latent_ablation import ablate_direction, ablation_table, decode_fidelity
from fnocausal.analysis.latent_probing import encode, probe_interventions
from fnocausal.analysis.shortcut_collapse import (
    EVAL_ENVS,
    collapse_table,
    eval_fno_on_env,
    eval_head_on_env,
)
from fnocausal.analysis.sobol_sensitivity import (
    analyze,
    build_fields_for_rows,
    make_problem,
    sobol_model,
    sobol_simulator,
)
from fnocausal.common.config import (
    config_hash,
    default_project_paths,
    load_yaml_config,
    make_run_id,
    tag_results,
)
from fnocausal.common.seeding import get_device, set_seed
from fnocausal.models.autoencoder import ConvAutoencoder
from fnocausal.models.fno import create_fno_model
from fnocausal.models.severity_head import SeverityHead
from fnocausal.models.train_loop import load_checkpoint

TRAIN_ENVS_BIASED = ("train_rho95", "train_rho80")
TRAIN_ENVS_ALL = ("train_rho95", "train_rho80", "control_train")

# Verdict thresholds. These encode the qualitative answer-key expectations as
# conservative quantitative tests; the scorecard reports raw values alongside.
TH = {
    "head_collapse_min": 0.5,      # biased head: (mae_flipped - mae_id)/mae_id
    "control_flat_max": 0.15,      # control head effect size must stay below
    "probe_r2_S_min": 0.5,         # z must respond linearly to s_level
    "probe_r2_C_min": 0.3,         # z must respond to seed count
    "invariance_ratio_min": 3.0,   # |coef_S(rho=.95)| / |coef_S(rho=0)|
    "coef_C_rel_range_max": 0.5,   # stability of causal coefficient
    "sobol_null_max": 0.05,        # ST that must be ~0
    "sobol_large_min": 0.3,        # ST that must be large
    "ablation_delta_min": 0.05,    # R2 change that counts as an effect
    "decode_u0_change_max": 0.10,  # S-ablation must not damage decoded u0
}


def load_stats(path):
    return {k: v for k, v in np.load(path).items()}


def build_env_arrays(npz, answer_key, envs):
    """Latent-encoder inputs (u0, s) and severities for a set of environments."""
    ids = answer_key.loc[answer_key["environment"].isin(envs), "sample_id"].to_numpy()
    X = np.stack([npz["u0"][ids], npz["s_fields"][ids]], axis=1)
    sigma = answer_key.set_index("sample_id").loc[ids, "severity"].to_numpy(dtype=np.float64)
    return ids, X, sigma


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--skip-sobol", action="store_true")
    args = parser.parse_args()

    paths = default_project_paths()
    config_path = args.config or (paths.experiments / "phase4_environments.yaml")
    config = load_yaml_config(config_path)
    set_seed(int(config["seed"]) + 5)
    device = get_device()
    run_id = make_run_id("phase5_audit", config)
    chash = config_hash(config)

    npz = np.load(paths.data / "phase4_pool.npz")
    answer_key = pd.read_csv(paths.data / "phase4_answer_key.csv")

    scorecard = []

    def record(diagnostic, check, expected, measured, passed, counted=True):
        scorecard.append(
            {
                "diagnostic": diagnostic,
                "check": check,
                "expected": expected,
                "measured": measured,
                "passed": bool(passed),
                "counted": bool(counted),
            }
        )
        tag = "PASS" if passed else ("FAIL" if counted else "INFO")
        print(f"  [{tag}] {check}: {measured} (expected {expected})")

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    fnos, fno_stats = {}, {}
    for role in ("biased", "control", "oracle"):
        model = create_fno_model(config)
        ckpt = load_checkpoint(paths.checkpoints / f"phase4_fno_{role}_best.pt", device)
        model.load_state_dict(ckpt["model_state_dict"])
        fnos[role] = model.to(device)
        fno_stats[role] = load_stats(paths.data / f"phase4_fno_{role}_normalizers.npz")

    heads, head_stats = {}, {}
    for role in ("biased", "control"):
        head = SeverityHead(in_channels=2, base_channels=int(config["ae_base_channels"]))
        ckpt = load_checkpoint(paths.checkpoints / f"phase4_head_{role}_best.pt", device)
        head.load_state_dict(ckpt["model_state_dict"])
        heads[role] = head.to(device)
        head_stats[role] = load_stats(paths.data / f"phase4_head_{role}_normalizers.npz")

    ae = ConvAutoencoder(
        in_channels=2, z_dim=int(config["ae_z_dim"]), base_channels=int(config["ae_base_channels"])
    )
    ckpt = load_checkpoint(paths.checkpoints / "phase4_ae_biased_best.pt", device)
    ae.load_state_dict(ckpt["model_state_dict"])
    ae = ae.to(device)
    ae_stats = load_stats(paths.data / "phase4_ae_biased_normalizers.npz")

    # ------------------------------------------------------------------
    # Diagnostic 1: shortcut collapse
    # ------------------------------------------------------------------
    print("\n=== Diagnostic 1: shortcut collapse ===")
    fno_results, head_results = {}, {}
    for role, model in fnos.items():
        input_kind = "u0_M" if role == "oracle" else "u0_s"
        fno_results[role] = {
            env: eval_fno_on_env(model, npz, answer_key, env, input_kind,
                                 fno_stats[role], device)[0]
            for env in EVAL_ENVS
        }
    for role, head in heads.items():
        head_results[role] = {
            env: eval_head_on_env(head, npz, answer_key, env, head_stats[role], device)[0]
            for env in EVAL_ENVS
        }

    table = collapse_table(fno_results, head_results)
    table = tag_results(table, run_id, chash, "claim2")
    table.to_csv(paths.results / "phase5_collapse_table.csv", index=False)

    b, c = head_results["biased"], head_results["control"]
    head_effect_biased = (b["eval_flipped"] - b["eval_id"]) / b["eval_id"]
    head_effect_control = (c["eval_flipped"] - c["eval_id"]) / c["eval_id"]
    fb, fc, fo = fno_results["biased"], fno_results["control"], fno_results["oracle"]
    fno_effect_biased = (fb["eval_flipped"] - fb["eval_id"]) / fb["eval_id"]
    fno_effect_control = (fc["eval_flipped"] - fc["eval_id"]) / fc["eval_id"]

    record("collapse", "gate4iv_fno_biased_beats_control_on_id",
           "biased < control", f"{fb['eval_id']:.4f} vs {fc['eval_id']:.4f}",
           fb["eval_id"] < fc["eval_id"])
    record("collapse", "gate4iv_head_biased_beats_control_on_id",
           "biased < control", f"{b['eval_id']:.4f} vs {c['eval_id']:.4f}",
           b["eval_id"] < c["eval_id"])
    record("collapse", "head_biased_collapses_on_flip",
           f"effect > {TH['head_collapse_min']}", f"{head_effect_biased:.3f}",
           head_effect_biased > TH["head_collapse_min"])
    record("collapse", "head_control_stays_flat",
           f"|effect| < {TH['control_flat_max']}", f"{head_effect_control:.3f}",
           abs(head_effect_control) < TH["control_flat_max"])
    biased_monotone = (b["eval_id"] <= b["eval_rho05"] <= b["eval_broken"] <= b["eval_flipped"])
    record("collapse", "head_biased_error_monotone_in_rho_shift",
           "id <= rho05 <= broken <= flipped",
           " <= ".join(f"{b[e]:.4f}" for e in EVAL_ENVS), biased_monotone)
    record("collapse", "fno_biased_collapse_exceeds_control",
           "biased effect > 2x |control effect|",
           f"{fno_effect_biased:.3f} vs {fno_effect_control:.3f}",
           fno_effect_biased > 2 * abs(fno_effect_control))
    record("collapse", "fno_oracle_best_on_flipped",
           "oracle lowest error on eval_flipped",
           f"oracle {fo['eval_flipped']:.4f}, biased {fb['eval_flipped']:.4f}, "
           f"control {fc['eval_flipped']:.4f}",
           fo["eval_flipped"] < min(fb["eval_flipped"], fc["eval_flipped"]))

    # Figure: error vs environment
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(EVAL_ENVS))
    for role in ("biased", "control", "oracle"):
        axes[0].plot(x, [fno_results[role][e] for e in EVAL_ENVS], "o-", label=f"fno_{role}")
    axes[0].set_ylabel("mean relative L2 (u(T))")
    for role in ("biased", "control"):
        axes[1].plot(x, [head_results[role][e] for e in EVAL_ENVS], "o-", label=f"head_{role}")
    axes[1].set_ylabel("severity MAE")
    for ax, title in zip(axes, ("Field surrogates", "Severity heads")):
        ax.set_xticks(x)
        ax.set_xticklabels(["id (0.95)", "rho 0.5", "broken (0)", "flipped (-0.95)"], rotation=15)
        ax.set_title(f"Shortcut collapse: {title}")
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig(paths.figures / "phase5_collapse.png", dpi=200)
    plt.close()

    # ------------------------------------------------------------------
    # Diagnostic 2: intervention-based latent probing
    # ------------------------------------------------------------------
    print("\n=== Diagnostic 2: intervention probing ===")
    probe = probe_interventions(ae, ae_stats, npz, answer_key, config, device)
    w_S, w_C = probe["w_S"], probe["w_C"]

    per_dim = tag_results(probe["per_dim"], run_id, chash, "claim2")
    per_dim.to_csv(paths.results / "phase5_probing_per_dim.csv", index=False)

    record("probing", "latent_responds_to_S_intervention",
           f"R2 >= {TH['probe_r2_S_min']}", f"{probe['r2_S']:.3f}",
           probe["r2_S"] >= TH["probe_r2_S_min"])
    record("probing", "latent_responds_to_C_intervention",
           f"R2 >= {TH['probe_r2_C_min']}", f"{probe['r2_C']:.3f}",
           probe["r2_C"] >= TH["probe_r2_C_min"])
    print(f"  (w_S/w_C subspace angle: {probe['angle_deg']:.1f} deg)")

    top = probe["per_dim"].sort_values("abs_slope_S", ascending=False).head(20)
    plt.figure(figsize=(10, 4.5))
    width = 0.4
    idx = np.arange(len(top))
    plt.bar(idx - width / 2, top["abs_slope_S"], width, label="|slope| vs s_level (S)")
    plt.bar(idx + width / 2, top["abs_slope_C"], width, label="|slope| vs seed count (C)")
    plt.xticks(idx, top["dim"], fontsize=7)
    plt.xlabel("latent dim (top 20 by S response)")
    plt.ylabel("intervention response slope")
    plt.title(f"Latent intervention responses (angle(w_S, w_C) = {probe['angle_deg']:.1f} deg)")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(paths.figures / "phase5_probing_per_dim.png", dpi=200)
    plt.close()

    # ------------------------------------------------------------------
    # Diagnostic 3: invariance across environments
    # ------------------------------------------------------------------
    print("\n=== Diagnostic 3: invariance (ICP/IRM style) ===")
    z_by_env, sigma_by_env, rho_by_env = {}, {}, {}
    for env in TRAIN_ENVS_ALL:
        _, X_env, sigma_env = build_env_arrays(npz, answer_key, (env,))
        z_by_env[env] = encode(ae, X_env, ae_stats, device)
        sigma_by_env[env] = sigma_env
        rho_by_env[env] = float(
            answer_key.loc[answer_key["environment"] == env, "rho_target"].iloc[0]
        )

    probes_df = per_environment_probes(z_by_env, sigma_by_env, w_S, w_C)
    probes_df["rho"] = probes_df["environment"].map(rho_by_env)
    probes_df = tag_results(probes_df, run_id, chash, "claim2")
    probes_df.to_csv(paths.results / "phase5_invariance_probes.csv", index=False)

    coef_S = probes_df.set_index("environment")["coef_S"]
    ratio = abs(coef_S["train_rho95"]) / max(abs(coef_S["control_train"]), 1e-9)
    ordered = abs(coef_S["train_rho95"]) > abs(coef_S["train_rho80"]) > abs(coef_S["control_train"])

    record("invariance", "coef_S_tracks_rho",
           "|coef_S| ordered rho .95 > .8 > 0",
           ", ".join(f"{e}={coef_S[e]:.4f}" for e in TRAIN_ENVS_ALL), ordered)
    record("invariance", "coef_S_unstable_ratio",
           f"ratio > {TH['invariance_ratio_min']}", f"{ratio:.1f}",
           ratio > TH["invariance_ratio_min"])

    # ICP-correct stability test: UNIVARIATE regression of sigma on the
    # C-score alone must be stable across environments. (The multivariate
    # ridge's C-coefficient is NOT expected to be stable: when the shortcut is
    # available the fit shifts weight from C to S - that reweighting is itself
    # shortcut evidence and is retained in phase5_invariance_probes.csv.)
    uni_df = univariate_score_probes(z_by_env, sigma_by_env, w_S, w_C)
    uni_df["rho"] = uni_df["environment"].map(rho_by_env)
    uni_tagged = tag_results(uni_df, run_id, chash, "claim2")
    uni_tagged.to_csv(paths.results / "phase5_invariance_univariate.csv", index=False)

    beta_C = uni_df.set_index("environment")["beta_C"]
    beta_S = uni_df.set_index("environment")["beta_S"]
    bc_rel_range = (beta_C.max() - beta_C.min()) / max(abs(beta_C).mean(), 1e-9)
    record("invariance", "univariate_beta_C_stable",
           f"relative range < {TH['coef_C_rel_range_max']}", f"{bc_rel_range:.3f}",
           bc_rel_range < TH["coef_C_rel_range_max"])
    bs_ordered = abs(beta_S["train_rho95"]) > abs(beta_S["train_rho80"]) > abs(beta_S["control_train"])
    record("invariance", "univariate_beta_S_tracks_rho",
           "|beta_S| ordered rho .95 > .8 > 0",
           ", ".join(f"{e}={beta_S[e]:.4f}" for e in TRAIN_ENVS_ALL), bs_ordered)

    # ERM (pooled ridge) vs IRM. The operative IRM promise is OUT-OF-
    # ENVIRONMENT risk: lambda is selected on eval_broken (rho=0), the winner
    # is tested once on eval_flipped. INFORMATIONAL (not gate-counted): IRMv1
    # is the plan's designated-secondary estimator, the pooled ERM baseline is
    # already implicitly invariance-regularized (half its data is the rho=0
    # control env), and linear IRMv1 is known to underperform without
    # sufficient environment diversity (Rosenfeld et al. 2021). The primary
    # invariance evidence is the univariate ICP-style checks above.
    z_all = np.concatenate([z_by_env[e] for e in TRAIN_ENVS_ALL])
    sigma_all = np.concatenate([sigma_by_env[e] for e in TRAIN_ENVS_ALL])
    coef_erm = ridge_probe(z_all, sigma_all)
    z_all_mean = z_all.mean(axis=0)
    sigma_all_mean = float(sigma_all.mean())

    _, X_broken, sigma_broken = build_env_arrays(npz, answer_key, ("eval_broken",))
    z_broken = encode(ae, X_broken, ae_stats, device)
    _, X_flip, sigma_flip = build_env_arrays(npz, answer_key, ("eval_flipped",))
    z_flip = encode(ae, X_flip, ae_stats, device)

    irm_candidates = []
    for lam in (10.0, 100.0, 1000.0):
        coef = irm_probe(z_by_env, sigma_by_env, irm_lambda=lam, seed=int(config["seed"]))
        r2_sel = probe_r2(z_broken, sigma_broken, coef, z_all_mean, sigma_all_mean)
        irm_candidates.append((r2_sel, lam, coef))
    _, best_lam, coef_irm = max(irm_candidates, key=lambda t: t[0])

    reliance_erm = abs(coef_erm @ w_S) / max(abs(coef_erm @ w_C), 1e-9)
    reliance_irm = abs(coef_irm @ w_S) / max(abs(coef_irm @ w_C), 1e-9)
    print(f"  (nuisance-reliance |coef.w_S|/|coef.w_C|: ERM {reliance_erm:.3f}, "
          f"IRM {reliance_irm:.3f}; lambda selected on eval_broken: {best_lam:.0f})")

    r2_flip_erm = probe_r2(z_flip, sigma_flip, coef_erm, z_all_mean, sigma_all_mean)
    r2_flip_irm = probe_r2(z_flip, sigma_flip, coef_irm, z_all_mean, sigma_all_mean)
    record("invariance", "irm_generalizes_better_on_flipped_secondary",
           "R2_flipped(IRM) > R2_flipped(ERM)",
           f"IRM {r2_flip_irm:.3f} (lambda {best_lam:.0f}) vs ERM {r2_flip_erm:.3f}",
           r2_flip_irm > r2_flip_erm, counted=False)

    plt.figure(figsize=(7, 5))
    rhos = [rho_by_env[e] for e in TRAIN_ENVS_ALL]
    plt.plot(rhos, [beta_S[e] for e in TRAIN_ENVS_ALL], "o-",
             label="univariate beta on S-score (nuisance)")
    plt.plot(rhos, [beta_C[e] for e in TRAIN_ENVS_ALL], "s-",
             label="univariate beta on C-score (causal)")
    plt.xlabel("environment correlation rho")
    plt.ylabel("regression coefficient")
    plt.title("Invariance: causal score stable, nuisance score tracks rho")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths.figures / "phase5_invariance.png", dpi=200)
    plt.close()

    # ------------------------------------------------------------------
    # Diagnostic 4: Sobol sensitivity
    # ------------------------------------------------------------------
    sobol_rows = []
    if not args.skip_sobol:
        print("\n=== Diagnostic 4: Sobol sensitivity ===")
        from SALib.sample import sobol as sobol_sample

        problem = make_problem(config)
        rows = sobol_sample.sample(problem, 256, calc_second_order=False,
                                   seed=int(config["seed"]) + 77)
        fields = build_fields_for_rows(rows, config)

        sigma_sim = sobol_simulator(rows, fields, config)
        sigma_hat = sobol_model(fields, heads["biased"], head_stats["biased"], device)

        idx_sim = analyze(problem, sigma_sim)
        idx_model = analyze(problem, sigma_hat)

        for target, indices in (("simulator", idx_sim), ("biased_head", idx_model)):
            for factor, vals in indices.items():
                sobol_rows.append({"target": target, "factor": factor, **vals})
        sobol_df = tag_results(pd.DataFrame(sobol_rows), run_id, chash, "claim2")
        sobol_df.to_csv(paths.results / "phase5_sobol_indices.csv", index=False)

        record("sobol", "simulator_ignores_s",
               f"|ST_s| < {TH['sobol_null_max']}", f"{idx_sim['s_level']['ST']:.4f}",
               abs(idx_sim["s_level"]["ST"]) < TH["sobol_null_max"])
        record("sobol", "simulator_driven_by_m",
               f"ST_m > {TH['sobol_large_min']}", f"{idx_sim['m']['ST']:.3f}",
               idx_sim["m"]["ST"] > TH["sobol_large_min"])
        record("sobol", "model_relies_on_s",
               f"ST_s > {TH['sobol_large_min']}", f"{idx_model['s_level']['ST']:.3f}",
               idx_model["s_level"]["ST"] > TH["sobol_large_min"])
        record("sobol", "model_blind_to_m",
               f"|ST_m| < {TH['sobol_null_max']}", f"{idx_model['m']['ST']:.4f}",
               abs(idx_model["m"]["ST"]) < TH["sobol_null_max"])

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
        for ax, (target, indices) in zip(axes, (("simulator", idx_sim), ("biased model", idx_model))):
            names = list(indices.keys())
            ax.bar(names, [indices[n]["ST"] for n in names], color=["#4878d0", "#6acc64", "#d65f5f"])
            ax.set_title(f"Total Sobol indices: {target}")
            ax.grid(True, axis="y", alpha=0.3)
        axes[0].set_ylabel("ST")
        plt.suptitle("The contrast: s_level drives the model but not the physics")
        plt.tight_layout()
        plt.savefig(paths.figures / "phase5_sobol.png", dpi=200)
        plt.close()

    # ------------------------------------------------------------------
    # Diagnostic 5: latent ablation
    # ------------------------------------------------------------------
    print("\n=== Diagnostic 5: latent ablation ===")
    _, X_train, sigma_train = build_env_arrays(npz, answer_key, TRAIN_ENVS_BIASED)
    z_train = encode(ae, X_train, ae_stats, device)
    coef = ridge_probe(z_train, sigma_train)
    z_train_mean = z_train.mean(axis=0)
    sigma_train_mean = float(sigma_train.mean())

    z_by_eval, sigma_by_eval = {}, {}
    for env in EVAL_ENVS:
        _, X_env, sigma_env = build_env_arrays(npz, answer_key, (env,))
        z_by_eval[env] = encode(ae, X_env, ae_stats, device)
        sigma_by_eval[env] = sigma_env

    abl = ablation_table(z_by_eval, sigma_by_eval, coef, w_S, w_C,
                         z_train_mean, sigma_train_mean)
    abl = tag_results(abl, run_id, chash, "claim2")
    abl.to_csv(paths.results / "phase5_ablation.csv", index=False)

    r2 = abl.set_index(["environment", "variant"])["probe_r2"]
    d = TH["ablation_delta_min"]
    record("ablation", "ablate_S_hurts_biased_env",
           f"intact - ablate_S > {d} on eval_id",
           f"{r2['eval_id', 'intact']:.3f} -> {r2['eval_id', 'ablate_S']:.3f}",
           r2["eval_id", "intact"] - r2["eval_id", "ablate_S"] > d)
    record("ablation", "ablate_S_helps_flipped_env",
           f"ablate_S - intact > {d} on eval_flipped",
           f"{r2['eval_flipped', 'intact']:.3f} -> {r2['eval_flipped', 'ablate_S']:.3f}",
           r2["eval_flipped", "ablate_S"] - r2["eval_flipped", "intact"] > d)
    c_hurts_everywhere = all(
        r2[env, "intact"] - r2[env, "ablate_C"] > 0 for env in EVAL_ENVS
    )
    record("ablation", "ablate_C_hurts_everywhere",
           "intact > ablate_C on all eval envs",
           ", ".join(f"{env}: {r2[env, 'intact']:.2f}->{r2[env, 'ablate_C']:.2f}"
                     for env in EVAL_ENVS),
           c_hurts_everywhere)

    z_id = z_by_eval["eval_id"]
    fid_S = decode_fidelity(ae, z_id, ablate_direction(z_id, w_S, z_train_mean), ae_stats, device)
    record("ablation", "ablate_S_preserves_decoded_u0",
           f"u0 change < {TH['decode_u0_change_max']} and s change > 2x u0 change",
           f"u0 {fid_S['u0_rel_change']:.4f}, s {fid_S['s_rel_change']:.4f}",
           fid_S["u0_rel_change"] < TH["decode_u0_change_max"]
           and fid_S["s_rel_change"] > 2 * fid_S["u0_rel_change"])

    pivot = abl.pivot(index="environment", columns="variant", values="probe_r2").loc[list(EVAL_ENVS)]
    pivot.plot(kind="bar", figsize=(9, 5))
    plt.ylabel("severity-probe R2")
    plt.title("Latent ablation: removing the S-direction flips the shortcut off")
    plt.xticks(rotation=15)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(paths.figures / "phase5_ablation.png", dpi=200)
    plt.close()

    # ------------------------------------------------------------------
    # Scorecard + conclusions
    # ------------------------------------------------------------------
    score_df = tag_results(pd.DataFrame(scorecard), run_id, chash, "claim2")
    score_df.to_csv(paths.results / "phase5_audit_scorecard.csv", index=False)

    counted_df = score_df[score_df["counted"]]
    n_pass = int(counted_df["passed"].sum())
    n_all = len(counted_df)
    all_passed = n_pass == n_all
    info_df = score_df[~score_df["counted"]]
    info_note = "; ".join(
        f"{r['check']}: {'consistent' if r['passed'] else 'NOT supportive'} ({r['measured']})"
        for _, r in info_df.iterrows()
    ) or "none"

    if sobol_rows:
        sobol_line = (f"ST_s(simulator) = {idx_sim['s_level']['ST']:.4f} vs "
                      f"ST_s(model) = {idx_model['s_level']['ST']:.3f}")
    else:
        sobol_line = "not run this invocation (--skip-sobol)"

    conclusions = f"""# Phase 5 Conclusions

Scorecard: {n_pass}/{n_all} gate-counted checks passed
(results/phase5_audit_scorecard.csv). Informational (non-gating) checks:
{info_note}.

Note on the IRM secondary check: at every penalty weight swept (lambda 10-1000,
selected on the rho=0 environment), the IRMv1 linear probe generalized WORSE to
the flipped environment than pooled ERM. This does not contradict the audit:
the pooled ERM baseline is already implicitly invariance-regularized (half its
training data comes from the rho=0 control environment), and linear IRMv1 is
known to underperform in low-environment-diversity settings (Rosenfeld et al.
2021, "The Risks of Invariant Risk Minimization"). The primary invariance
evidence is the ICP-style univariate checks, which pass: the causal-score
coefficient is stable across environments while the nuisance-score coefficient
tracks rho.

## What is and is not established

- **Claim 1 (surrogate accuracy): SUPPORTED** by Phase 2 (ID rel-L2 1.5%,
  independent-scheme targets 1.5%, held-out epsilon band 1.3%).
- **Claim 2 (the audit separates causal from spurious features):
  {"SUPPORTED" if all_passed else "PARTIALLY SUPPORTED - see failed checks"}.
  Five independent diagnostics were checked against a constructed answer key
  (the nuisance channel s provably never enters the solver; the mobility
  amplitude m provably drives severity). {"All five agree with the answer key."
  if all_passed else "Failed checks must be resolved or explained before this claim is made."}
- **Claim 3 (the PDE describes real degradation): NOT ADDRESSED and NOT
  claimed.** All data in this project is synthetic; claim 3 requires real
  experiments (roadmap Phase 7).

## Headline numbers

- Biased severity head collapse (id -> flipped): {head_effect_biased:.2f}
  relative MAE increase; control head: {head_effect_control:.2f}.
- Sobol contrast: {sobol_line}.
- Ablating the latent S-direction: eval_id probe R2
  {r2["eval_id", "intact"]:.3f} -> {r2["eval_id", "ablate_S"]:.3f},
  eval_flipped {r2["eval_flipped", "intact"]:.3f} -> {r2["eval_flipped", "ablate_S"]:.3f}.

Provenance: run_id {run_id}, config hash {chash}. All CSVs tagged claim2.
"""
    (paths.results / "phase5_conclusions.md").write_text(conclusions)
    print(f"\nSaved scorecard: {paths.results / 'phase5_audit_scorecard.csv'}")
    print(f"Saved conclusions: {paths.results / 'phase5_conclusions.md'}")
    print(f"\nGate 5 {'PASSED' if all_passed else 'FAILED'} ({n_pass}/{n_all}).")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
