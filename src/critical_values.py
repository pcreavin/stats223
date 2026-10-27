"""Port of R/CriticalValues.R::Critical_values (VaR / CoVaR / reverse CoVaR only).

Algorithm 1 as coded: B paths of length n with shared uniforms
    I = 1{U1 > beta},  I1 = 1{U1 > beta} * 1{U2 > alpha};
record the sup over T of each combined detector; take (1-nu)-quantiles over the grid
nu = 0, 1e-4, ..., 1 (R quantile type 7 == numpy default 'linear'); pick nu by
which.min(|target(nu) - iota|) (closest, not <= iota).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .detectors import Moments, null_moments, rolling_gini, rolling_uc

NU_GRID = np.arange(10001) / 10000.0   # == seq.int(0, 1, by = 1e-4)


def _standardised_sup(uc: np.ndarray, gini: np.ndarray, ms: Moments, a: float) -> np.ndarray:
    # same arithmetic as combined_detector(covar_detector(...)), applied to raw paths
    return (a * ((uc - ms.mu_UC) / ms.sd_UC) + (1 - a) * ((gini - ms.mu_IID) / ms.sd_IID)).max(axis=1)


def simulate_sups_multi(n: int, m: int, alpha: float, beta: float, B: int, moment_sets,
                        a: float = 0.5, rng: np.random.Generator | None = None,
                        chunk: int = 1000) -> list[tuple[np.ndarray, np.ndarray]]:
    """Steps 1-3 on one set of paths, standardised with each (ms_var, ms_covar) in
    `moment_sets`: returns [(sup VaR, sup CoVaR), ...], each array of length B."""
    rng = np.random.default_rng(rng)
    out = [([], []) for _ in moment_sets]
    for start in range(0, B, chunk):
        b = min(chunk, B - start)
        U1, U2 = rng.random((b, n)), rng.random((b, n))
        I = (U1 > beta).astype(np.int8)
        I1 = ((U1 > beta) & (U2 > alpha)).astype(np.int8)
        raw_v = rolling_uc(I, m, 1 - beta), rolling_gini(I, m)
        raw_c = rolling_uc(I1, m, (1 - alpha) * (1 - beta)), rolling_gini(I1, m)
        for (ms_v, ms_c), (sv, sc) in zip(moment_sets, out):
            sv.append(_standardised_sup(*raw_v, ms_v, a))
            sc.append(_standardised_sup(*raw_c, ms_c, a))
    return [(np.concatenate(sv), np.concatenate(sc)) for sv, sc in out]


def simulate_sups(n: int, m: int, alpha: float, beta: float, B: int,
                  ms_var: Moments, ms_covar: Moments, a: float = 0.5,
                  rng: np.random.Generator | None = None, chunk: int = 1000):
    """Steps 1-3: arrays of sup VaR and sup CoVaR detectors, each of length B."""
    return simulate_sups_multi(n, m, alpha, beta, B, [(ms_var, ms_covar)], a, rng, chunk)[0]


def _exceed(sup: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Boolean matrix [grid, B]: sup >= c_nu."""
    return sup[None, :] >= c[:, None]


def _exceed_probs(sup_v: np.ndarray, sup_c: np.ndarray):
    """Grid quantiles (type 7) and exceedance probabilities; independent of K."""
    q = 1 - NU_GRID
    c_v = np.quantile(sup_v, q)
    c_c = np.quantile(sup_c, q)
    ev, ec = _exceed(sup_v, c_v), _exceed(sup_c, c_c)
    return c_v, c_c, ev.mean(1), ec.mean(1), (ev & ec).mean(1), (ev | ec).mean(1)


def _pick(probs, K: int, iota: float) -> dict:
    c_v, c_c, p_v, p_c, p_both, p_or = probs
    target = p_v + K * p_c - K * p_both
    i = int(np.argmin(np.abs(target - iota)))          # first minimum, like which.min
    target_rev = p_or
    j = int(np.argmin(np.abs(target_rev - iota / K)))
    return dict(c_VaR=c_v[i], c_SysRisk=c_c[i], nu=NU_GRID[i], size=target[i],
                c_VaR_rev=c_v[j], c_SysRisk_rev=c_c[j], nu_rev=NU_GRID[j], size_rev=target_rev[j])


def select_thresholds(sup_v: np.ndarray, sup_c: np.ndarray, K: int, iota: float) -> dict:
    """Steps 4-5 (i) and (iv). Returns thresholds for CoVaR and reverse CoVaR monitoring."""
    return _pick(_exceed_probs(sup_v, sup_c), K, iota)


def select_thresholds_multi(sup_v: np.ndarray, sup_c: np.ndarray, K_set, iota: float) -> dict:
    """select_thresholds for every K in K_set, sharing the K-free work: {K: dict}."""
    probs = _exceed_probs(sup_v, sup_c)
    return {K: _pick(probs, K, iota) for K in K_set}


def critical_values(m: int, n: int, K_set, alpha: float, beta: float, B: int, iota: float,
                    a: float = 0.5, seed: int = 0, reps_moments: int = 10**5,
                    return_sups: bool = False):
    """Same columns as their table, for SysRisk in {'CoVaR', 'CoVaR_Reverse'} and each K.

    Moments and sup samples are simulated once per (alpha, beta) and shared across K
    (their code re-simulates per K; this is distributionally identical)."""
    rng = np.random.default_rng(seed)
    ms_v = null_moments(m, 1 - beta, reps_moments, rng)
    ms_c = null_moments(m, (1 - alpha) * (1 - beta), reps_moments, rng)
    sup_v, sup_c = simulate_sups(n, m, alpha, beta, B, ms_v, ms_c, a, rng)
    rows = []
    for K in K_set:
        th = select_thresholds(sup_v, sup_c, K, iota)
        for sysrisk, cv_, cs_ in (("CoVaR", th["c_VaR"], th["c_SysRisk"]),
                                  ("CoVaR_Reverse", th["c_VaR_rev"], th["c_SysRisk_rev"])):
            rows.append(dict(SysRisk=sysrisk, c_VaR=cv_, c_SysRisk=cs_,
                             mu_VaR_UC=ms_v.mu_UC, sd_VaR_UC=ms_v.sd_UC,
                             mu_VaR_IID=ms_v.mu_IID, sd_VaR_IID=ms_v.sd_IID,
                             mu_SysRisk_UC=ms_c.mu_UC, sd_SysRisk_UC=ms_c.sd_UC,
                             mu_SysRisk_IID=ms_c.mu_IID, sd_SysRisk_IID=ms_c.sd_IID,
                             alpha=alpha, beta=beta, n=n, m=m, K=K, iota=iota, B=B, a=a))
    df = pd.DataFrame(rows)
    return (df, sup_v, sup_c) if return_sups else df


def bootstrap_se(sup_v: np.ndarray, sup_c: np.ndarray, K: int, iota: float,
                 n_boot: int = 50, seed: int = 1) -> dict:
    """Bootstrap SE of the selected thresholds (resampling (sup_v, sup_c) pairs)."""
    rng = np.random.default_rng(seed)
    B = sup_v.size
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, B, B)
        draws.append(select_thresholds(sup_v[idx], sup_c[idx], K, iota))
    keys = ("c_VaR", "c_SysRisk", "c_VaR_rev", "c_SysRisk_rev")
    return {k: float(np.std([d[k] for d in draws], ddof=1)) for k in keys}


def bootstrap_se_multi(sup_v: np.ndarray, sup_c: np.ndarray, K_set, iota: float,
                       n_boot: int = 50, seed: int = 1) -> dict:
    """bootstrap_se for every K in K_set from one set of resamples: {K: dict}."""
    rng = np.random.default_rng(seed)
    B = sup_v.size
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, B, B)
        draws.append(select_thresholds_multi(sup_v[idx], sup_c[idx], K_set, iota))
    keys = ("c_VaR", "c_SysRisk", "c_VaR_rev", "c_SysRisk_rev")
    return {K: {k: float(np.std([d[K][k] for d in draws], ddof=1)) for k in keys} for K in K_set}
