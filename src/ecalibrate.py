"""Calibrated thresholds for the learned e-detectors (Extension 2, K = 1).

Under correct forecasts Z_t is iid categorical(p) (Prop. 1), so the null law of any detector path can be simulated
exactly from iid draws, as in Algorithm 1: gamma_hat = (1 - iota)-quantile of max_{t in window} M_t over B null
paths, so that a fraction iota of null paths ever alarms. Z is drawn from shared uniforms (common random numbers
across levels and detectors) in fixed seed chunks.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from . import edetectors as ed, oracle, paths

N, IOTA = 1000, 0.1
CAL_SEED, CAL_B, CHUNK = 11, 20_000, 2000


def null_Z(level: float, B: int, seed: int, n: int = N, chunk: int = CHUNK):
    """Yield iid null band paths Z [b, n] in chunks; chunk c uses SeedSequence(seed).spawn(...)[c]."""
    cp = np.cumsum(oracle.p_null(level))[:-1]
    for c, ss in enumerate(np.random.SeedSequence(seed).spawn(-(-B // chunk))):
        b = min(chunk, B - c * chunk)
        yield np.searchsorted(cp, np.random.default_rng(ss).random((b, n)), side="right").astype(np.int8)


def null_max(name: str, level: float, B: int, seed: int, hp: dict | None = None,
             windows: dict = ed.WINDOWS) -> dict:
    """{window: max_t M_t [B]} of a learned detector on B iid null paths."""
    p = oracle.p_null(level)
    out = {w: [] for w in windows}
    for Z in null_Z(level, B, seed):
        for w, v in ed.window_max(ed.learned_path(name, Z, p, hp), windows).items():
            out[w].append(v)
    return {w: np.concatenate(v) for w, v in out.items()}


def gamma_hat(maxima: np.ndarray, iota: float = IOTA) -> float:
    return float(np.quantile(maxima, 1 - iota))


def bootstrap_se(maxima: np.ndarray, iota: float = IOTA, n_boot: int = 200, seed: int = 1) -> float:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, maxima.size, (n_boot, maxima.size))
    return float(np.quantile(maxima[idx], 1 - iota, axis=1).std(ddof=1))


def _hp_tag(hp: dict | None) -> str:
    full = {k: dict(v, **(hp or {}).get(k, {})) for k, v in ed.HP.items()}
    s = json.dumps(dict(hp=full, lam_s=ed.LAM_S, lam_j=ed.LAM_J), sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:8]


def calibrate(names=ed.LEARNED, levels=(0.9, 0.95), B: int = CAL_B, seed: int = CAL_SEED, iota: float = IOTA,
              hp: dict | None = None, force: bool = False) -> pd.DataFrame:
    """gamma_hat (and bootstrap SE, Ville size) per detector, level and window; cached in data/derived/ecal/."""
    f = paths.DERIVED / "ecal" / f"gamma_B{B}_s{seed}_i{iota:g}_{_hp_tag(hp)}.csv"
    if f.exists() and not force:
        return pd.read_csv(f)
    rows = []
    for lv in levels:
        for name in names:
            for w, mx in null_max(name, lv, B, seed, hp).items():
                rows.append(dict(detector=name, level=lv, window=w, gamma_hat=gamma_hat(mx, iota),
                                 se=bootstrap_se(mx, iota), size_ville=float((mx >= N / iota).mean()), B=B,
                                 seed=seed, iota=iota))
    df = pd.DataFrame(rows)
    f.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(f, index=False)
    return df


def lookup(cal: pd.DataFrame, name: str, level: float, window: str) -> float:
    r = cal[(cal.detector == name) & np.isclose(cal.level, level) & (cal.window == window)]
    if len(r) != 1:
        raise KeyError((name, level, window))
    return float(r.gamma_hat.iloc[0])


__all__ = ["null_Z", "null_max", "gamma_hat", "bootstrap_se", "calibrate", "lookup", "CAL_SEED", "CAL_B"]
