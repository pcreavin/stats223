"""Port of R/Detectors.R::CoVaR_Detector and R/mu.R::musigma_CoVaR_fct.

Conventions reproduced from their code (not the paper):
  * rolling windows of length m ending at T = m..n  ->  n-m+1 values;
  * UC part is |window mean(I) - p| (absolute value);
  * IID part is the Gini coefficient of the durations d_i = diff(c(0, t_i)), with t_i the
    1-based violation positions inside the window; the censored gap after the last
    violation is NOT included; Gini := 0 when S in {0, m} and (automatically) S == 1;
  * both parts are standardised by simulated null moments; combined = a*UC + (1-a)*IID.
"""
from __future__ import annotations

from dataclasses import dataclass

import numba
import numpy as np


@dataclass(frozen=True)
class Moments:
    mu_UC: float
    sd_UC: float
    mu_IID: float
    sd_IID: float


# --------------------------------------------------------------------------- Gini
def gini_durations(t_i: np.ndarray, m: int) -> float:
    """Reference (scalar) Gini of durations for 1-based violation positions t_i in a window.

    Literal transcription of
        (1 / S^2) * 2 * sum(dist(d_i, "manhattan")) / (2 * mean(d_i)).
    """
    t_i = np.asarray(t_i, dtype=np.int64)
    S = t_i.size
    if S == 0 or S >= m:
        return 0.0
    d = np.diff(np.concatenate(([0], t_i)))
    sum_abs = np.abs(d[:, None] - d[None, :]).sum() // 2      # sum_{i<j} |d_i - d_j|
    return (1.0 / S**2) * 2.0 * float(sum_abs) / (2.0 * (d.sum() / S))


@numba.njit(cache=True)
def _gini_from_sorted(d_sorted, S):
    # sum_{i<j} |d_i - d_j| = sum_k (2k - S - 1) d_(k), k = 1..S   (integer arithmetic)
    acc = 0
    tot = 0
    for k in range(S):
        acc += (2 * (k + 1) - S - 1) * d_sorted[k]
        tot += d_sorted[k]
    return (1.0 / (S * S)) * 2.0 * acc / (2.0 * (tot / S))


@numba.njit(cache=True)
def _rolling_gini_row(I, m, out):
    n = I.shape[0]
    pos = np.empty(n, dtype=np.int64)
    npos = 0
    for i in range(n):
        if I[i] != 0:
            pos[npos] = i
            npos += 1
    d = np.empty(m, dtype=np.int64)
    lo = 0  # first violation index >= window start
    hi = 0  # first violation index > window end
    for t in range(m - 1, n):
        start = t - m + 1
        while lo < npos and pos[lo] < start:
            lo += 1
        while hi < npos and pos[hi] <= t:
            hi += 1
        S = hi - lo
        if S == 0 or S >= m:
            out[start] = 0.0
            continue
        prev = start - 1  # position "0" in window coordinates
        for k in range(S):
            d[k] = pos[lo + k] - prev
            prev = pos[lo + k]
        ds = np.sort(d[:S])
        out[start] = _gini_from_sorted(ds, S)


@numba.njit(cache=True, parallel=True)
def _rolling_gini_2d(I, m, out):
    for r in numba.prange(I.shape[0]):
        _rolling_gini_row(I[r], m, out[r])


def rolling_gini(I: np.ndarray, m: int) -> np.ndarray:
    """Gini detector (unstandardised) for every window; I has shape (..., n)."""
    I = np.ascontiguousarray(I, dtype=np.int8)
    lead, n = I.shape[:-1], I.shape[-1]
    flat = I.reshape(-1, n)
    out = np.empty((flat.shape[0], n - m + 1))
    _rolling_gini_2d(flat, m, out)
    return out.reshape(*lead, n - m + 1)


def rolling_gini_reference(I: np.ndarray, m: int) -> np.ndarray:
    """Slow, obviously-correct version of rolling_gini for a 1-D sequence."""
    I = np.asarray(I)
    n = I.size
    return np.array([gini_durations(np.flatnonzero(I[t - m + 1:t + 1]) + 1, m)
                     for t in range(m - 1, n)])


# --------------------------------------------------------------------------- UC
def rolling_uc(I: np.ndarray, m: int, p: float) -> np.ndarray:
    """|rolling mean of I - p| (unstandardised); I has shape (..., n)."""
    I = np.asarray(I, dtype=np.int64)
    c = np.concatenate([np.zeros(I.shape[:-1] + (1,), np.int64), np.cumsum(I, axis=-1)], axis=-1)
    counts = c[..., m:] - c[..., :-m]
    return np.abs(counts / m - p)


# --------------------------------------------------------------------------- detectors
def covar_detector(I: np.ndarray, m: int, p: float, ms: Moments) -> tuple[np.ndarray, np.ndarray]:
    """Standardised (UC, IID) detector sequences, as CoVaR_Detector in R/Detectors.R."""
    uc = (rolling_uc(I, m, p) - ms.mu_UC) / ms.sd_UC
    iid = (rolling_gini(I, m) - ms.mu_IID) / ms.sd_IID
    return uc, iid


def combined_detector(I: np.ndarray, m: int, p: float, ms: Moments, a: float = 0.5) -> np.ndarray:
    uc, iid = covar_detector(I, m, p, ms)
    return a * uc + (1 - a) * iid


# --------------------------------------------------------------------------- null moments
def null_moments(m: int, p: float, reps: int = 10**5, rng: np.random.Generator | None = None,
                 chunk: int = 20_000) -> Moments:
    """Port of musigma_CoVaR_fct: moments over `reps` iid windows with I = 1{U <= p}."""
    rng = np.random.default_rng(rng)
    uc_all, iid_all = [], []
    for start in range(0, reps, chunk):
        r = min(chunk, reps - start)
        I = (rng.random((r, m)) <= p).astype(np.int8)
        uc_all.append(rolling_uc(I, m, p)[:, 0])
        iid_all.append(rolling_gini(I, m)[:, 0])
    uc, iid = np.concatenate(uc_all), np.concatenate(iid_all)
    return Moments(uc.mean(), uc.std(ddof=1), iid.mean(), iid.std(ddof=1))


def moments_from_cv_row(row, which: str) -> Moments:
    """Moments stored in their critical-value table; which in {'VaR', 'SysRisk'}."""
    return Moments(float(row[f"mu_{which}_UC"]), float(row[f"sd_{which}_UC"]),
                   float(row[f"mu_{which}_IID"]), float(row[f"sd_{which}_IID"]))
