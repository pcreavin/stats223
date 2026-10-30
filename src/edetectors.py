"""Joint three-band e-Shiryaev-Roberts detector.

Bands Z_t in {0, 1, 2}: 0 = no sector VaR miss, 1 = sector miss with the bank within its CoVaR, 2 = joint miss.
Under correct forecasts P(Z_t = j | F_{t-1}) = p = (beta, (1-beta) alpha, (1-beta)(1-alpha)). Betting q_t
(predictable) gives the e-values L_t = q_t(Z_t) / p(Z_t), E[L_t | F_{t-1}] = 1 under the null, and the SR statistic
M_t = (1 + M_{t-1}) L_t = sum_{k <= t} prod_{s=k..t} L_s, M_0 = 0, so E[M_t] = t under the null. Alarm when
M_t >= gamma (gamma = n / iota gives ARL >= gamma by Ville / Pollak).
"""
from __future__ import annotations

import numpy as np

WINDOWS = {"matched": (250, 1000), "day1": (1, 1000)}


def categories(I_VaR, I_CoVaR) -> np.ndarray:
    """Z = I_VaR + I_CoVaR (int8); I_CoVaR is a joint miss, so it implies I_VaR."""
    I_VaR, I_CoVaR = np.asarray(I_VaR), np.asarray(I_CoVaR)
    if np.any(I_CoVaR > I_VaR):
        raise ValueError("I_CoVaR = 1 with I_VaR = 0")
    return (I_VaR + I_CoVaR).astype(np.int8)


def e_values(Z, q, p) -> np.ndarray:
    """L_t = q_t(Z_t) / p(Z_t); Z [..., T]; q constant (3,) or per-day [..., T, 3]."""
    Z, q, p = np.asarray(Z), np.asarray(q, float), np.asarray(p, float)
    qz = q[Z] if q.ndim == 1 else np.take_along_axis(q, Z[..., None].astype(np.intp), -1)[..., 0]
    return qz / p[Z]


def e_sr(Z, q, p, gamma: float = 1e4, windows: dict = WINDOWS, M0=0.0, t0: int = 0, return_path: bool = False):
    """SR recursion over the last axis of Z (days t0+1 .. t0+T, 1-based), vectorised over leading axes.

    Pass M0 = previous `M_last` and t0 = days already processed to continue over time chunks; combine the alarms
    of successive chunks with np.fmin (first chunk wins). Returns dict(alarm={window: first 1-based day in the
    window with M_t >= gamma, else inf}, M_last, gamma[, M = path])."""
    L = e_values(Z, q, p)
    T = L.shape[-1]
    M = np.empty(L.shape)
    m = np.broadcast_to(np.asarray(M0, float), L.shape[:-1]).copy()
    with np.errstate(over="ignore", invalid="ignore"):                # overflow to inf stays >= gamma
        for t in range(T):
            m = (1.0 + m) * L[..., t]
            M[..., t] = m
    days = t0 + 1 + np.arange(T)
    hit = M >= gamma
    alarm = {}
    for w, (lo, hi) in windows.items():
        h = hit & ((days >= lo) & (days <= hi))
        alarm[w] = np.where(h.any(-1), days[h.argmax(-1)], np.inf)
    out = dict(alarm=alarm, M_last=m, gamma=gamma)
    if return_path:
        out["M"] = M
    return out


__all__ = ["categories", "e_values", "e_sr", "WINDOWS"]
