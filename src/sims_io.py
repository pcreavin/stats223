"""Persist simulated indicator sequences and run lengths (data/derived/sims/{config}.npz + .json).

Later detectors (e-detectors) must run on exactly these paths, so every simulation configuration used by an
experiment is saved here and reloaded instead of re-simulated.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone

import numpy as np

from . import paths


def config_name(K: int, level: float, t_star: int, beta_post: float, break_in_X: bool, M: int,
                seed: int, n: int = 1000, m: int = 250, E: int = 5001) -> str:
    return (f"K{K}_a{level:g}_b{level:g}_t{t_star}_bp{beta_post:g}_bx{int(bool(break_in_X))}"
            f"_M{M}_n{n}_m{m}_E{E}_s{seed}")


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=paths.ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def sim_path(name: str):
    return paths.SIMS / f"{name}.npz"


def exists(name: str) -> bool:
    return sim_path(name).exists() and sim_path(name).with_suffix(".json").exists()


def save_sim(name: str, arrays: dict, meta: dict) -> None:
    paths.ensure_dirs()
    np.savez_compressed(sim_path(name), **arrays)
    meta = dict(meta, name=name, arrays=sorted(arrays), git_commit=_git_commit(),
                created=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    sim_path(name).with_suffix(".json").write_text(json.dumps(meta, indent=2, default=float))


def load_sim(name: str) -> tuple[dict, dict]:
    with np.load(sim_path(name)) as z:
        arrays = {k: z[k] for k in z.files}
    return arrays, json.loads(sim_path(name).with_suffix(".json").read_text())


def list_sims(**filters) -> list[dict]:
    """Metadata of all saved simulations whose settings equal the given filters."""
    out = []
    for f in sorted(paths.SIMS.glob("*.json")):
        meta = json.loads(f.read_text())
        if all(meta.get(k) == v for k, v in filters.items()):
            out.append(meta)
    return out
