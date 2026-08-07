"""Config loading, hashing, run identification, and project paths.

Every experiment is driven by a YAML file in experiments/. Results CSVs carry
run_id, config_hash, and a claim tag so each number can be traced to the exact
configuration that produced it and to the claim it supports.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml


VALID_CLAIMS = ("claim1", "claim2", "infrastructure")


def load_yaml_config(path: Path) -> dict:
    """
    Load a YAML config file.

    Inputs:
        path: Path to a .yaml file.

    Outputs:
        config: dict.
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config at {path} did not parse to a dict.")

    return config


def config_hash(config: dict) -> str:
    """
    Deterministic short hash of a config dict.

    Inputs:
        config: dict (must be JSON-serializable after tuple->list coercion).

    Outputs:
        digest: str, first 10 hex chars of sha256 over canonical JSON.
    """
    def _coerce(obj):
        if isinstance(obj, tuple):
            return [_coerce(v) for v in obj]
        if isinstance(obj, list):
            return [_coerce(v) for v in obj]
        if isinstance(obj, dict):
            return {k: _coerce(v) for k, v in sorted(obj.items())}
        return obj

    canonical = json.dumps(_coerce(config), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:10]


def make_run_id(name: str, config: dict) -> str:
    """
    Deterministic run identifier: <name>_<config-hash>_s<seed>.

    Inputs:
        name: str, experiment name (e.g. "phase2_fno").
        config: dict with a "seed" key.

    Outputs:
        run_id: str.
    """
    seed = config.get("seed", "NA")
    return f"{name}_{config_hash(config)}_s{seed}"


def tag_results(df: pd.DataFrame, run_id: str, cfg_hash: str, claim: str) -> pd.DataFrame:
    """
    Attach provenance columns to a results DataFrame.

    Inputs:
        df: pd.DataFrame.
        run_id: str.
        cfg_hash: str.
        claim: str, one of VALID_CLAIMS. "infrastructure" marks tool-validity
            results (e.g. AE reconstruction) that support neither claim directly.

    Outputs:
        tagged: pd.DataFrame copy with run_id, config_hash, claim columns.
    """
    if claim not in VALID_CLAIMS:
        raise ValueError(f"claim must be one of {VALID_CLAIMS}, got {claim!r}.")

    tagged = df.copy()
    tagged["run_id"] = run_id
    tagged["config_hash"] = cfg_hash
    tagged["claim"] = claim
    return tagged


@dataclass(frozen=True)
class ProjectPaths:
    """
    Standard output locations rooted at the project directory.

    Inputs:
        root: Path, project root (the directory containing fnocausal/).

    Outputs:
        Dataclass with data/checkpoints/figures/logs/results/experiments Paths.
    """

    root: Path
    data: Path
    checkpoints: Path
    figures: Path
    logs: Path
    results: Path
    experiments: Path

    @classmethod
    def from_root(cls, root: Path) -> "ProjectPaths":
        root = Path(root).resolve()
        paths = cls(
            root=root,
            data=root / "data",
            checkpoints=root / "checkpoints",
            figures=root / "figures",
            logs=root / "logs",
            results=root / "results",
            experiments=root / "experiments",
        )
        for p in (paths.data, paths.checkpoints, paths.figures, paths.logs, paths.results):
            p.mkdir(parents=True, exist_ok=True)
        return paths


def default_project_paths() -> ProjectPaths:
    """
    Resolve project paths from this file's location (…/fnocausal/common/config.py).

    Outputs:
        ProjectPaths for the repository root, independent of the CWD the
        script was launched from (the root path contains a space, so relying
        on CWD is fragile on Windows).
    """
    return ProjectPaths.from_root(Path(__file__).resolve().parents[2])
