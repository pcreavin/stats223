"""Joint three-band e-Shiryaev-Roberts detector.

Bands Z_t in {0, 1, 2}: 0 = no sector VaR miss, 1 = sector miss with the bank within its CoVaR, 2 = joint miss.
Under correct forecasts P(Z_t = j | F_{t-1}) = p = (beta, (1-beta) alpha, (1-beta)(1-alpha)). Betting q_t
(predictable) gives the e-values L_t = q_t(Z_t) / p(Z_t), E[L_t | F_{t-1}] = 1 under the null, and the SR statistic
M_t = (1 + M_{t-1}) L_t = sum_{k <= t} prod_{s=k..t} L_s, M_0 = 0, so E[M_t] = t under the null. Alarm when
M_t >= gamma (gamma = n / iota gives ARL >= gamma by Ville / Pollak).

Learned bets (Extension 2) use only Z_1..Z_{t-1}, so they stay valid e-values:
  rate     average of SR processes over fixed alternatives with inflated miss rates (mixture_q, e_sr_mix);
  plugin   shrunk EWMA band frequencies (plugin_q);
  cluster  separate shrunk frequencies after a recent miss and otherwise (duration_q);
  combined 1/2 rate + 1/2 cluster (an average of SR processes is again one: E[M_t] = t under the null).
"""
from __future__ import annotations

import numpy as np
from scipy import signal

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
    out = dict(alarm=alarms(M, gamma, windows, t0), M_last=m, gamma=gamma)
    if return_path:
        out["M"] = M
    return out


def _window_mask(T: int, windows: dict, t0: int = 0) -> tuple[dict, np.ndarray]:
    days = t0 + 1 + np.arange(T)
    return {w: (days >= lo) & (days <= hi) for w, (lo, hi) in windows.items()}, days


def alarms(M, gamma: float, windows: dict = WINDOWS, t0: int = 0) -> dict:
    """{window: first 1-based day in the window with M_t >= gamma, else inf}; M [..., T]."""
    masks, days = _window_mask(M.shape[-1], windows, t0)
    out = {}
    for w, inw in masks.items():
        h = (M >= gamma) & inw
        out[w] = np.where(h.any(-1), days[h.argmax(-1)], np.inf)
    return out


def window_max(M, windows: dict = WINDOWS, t0: int = 0) -> dict:
    """{window: max of M_t over the days in the window}; P(window_max >= gamma) is the alarm probability."""
    masks, _ = _window_mask(M.shape[-1], windows, t0)
    return {w: np.where(inw, M, -np.inf).max(-1) for w, inw in masks.items()}


def e_sr_mix(Z, qs, p, w=None, gamma: float = 1e4, windows: dict = WINDOWS, return_path: bool = False):
    """Weighted average of SR processes, one per bet in `qs` (each constant (3,) or per-day [..., T, 3]); the alarm
    is on the average. Uniform weights by default."""
    L = np.stack([e_values(Z, q, p) for q in qs], -1)                  # [..., T, J]
    J = L.shape[-1]
    w = np.full(J, 1.0 / J) if w is None else np.asarray(w, float)
    m = np.zeros(L.shape[:-2] + (J,))
    M = np.empty(L.shape[:-1])
    with np.errstate(over="ignore", invalid="ignore"):
        for t in range(L.shape[-2]):
            m = (1.0 + m) * L[..., t, :]
            M[..., t] = m @ w
    out = dict(alarm=alarms(M, gamma, windows), M_last=m, gamma=gamma)
    if return_path:
        out["M"] = M
    return out


# --------------------------------------------------------------------------- learned bets
LAM_S, LAM_J = (1.5, 2.0, 3.0), (1.5, 2.0, 3.0, 5.0)


def mixture_q(p, lam_s=LAM_S, lam_j=LAM_J) -> np.ndarray:
    """Constant alternatives [J, 3]: sector-miss probability p1 + p2 scaled by lam_s, joint-miss probability p2 by
    lam_j; pairs that would make the middle band negative are dropped."""
    p = np.asarray(p, float)
    ps, pj = p[1] + p[2], p[2]
    return np.array([(1 - a * ps, a * ps - b * pj, b * pj) for a in lam_s for b in lam_j
                     if a * ps < 1 and a * ps >= b * pj])


def _one_sided(q, p):
    """Never bet less than p on a miss band (1, 2); band 0 takes the remainder."""
    q = q.copy()
    q[..., 1:] = np.maximum(q[..., 1:], p[1:])
    q[..., 0] = 1 - q[..., 1:].sum(-1)
    if np.any(q[..., 0] <= 0):
        raise ValueError("one-sided bet left no mass on band 0")
    return q


def plugin_q(Z, p, lam: float = 0.98, kappa: float = 50.0, one_sided: bool = True) -> np.ndarray:
    """q_t [..., T, 3] = (n_t + kappa p) / (N_t + kappa), n_t = sum_{s<t} lam^(t-1-s) onehot(Z_s) (q_1 = p)."""
    p = np.asarray(p, float)
    x = np.eye(3)[np.asarray(Z)]
    x = np.concatenate([np.zeros_like(x[..., :1, :]), x[..., :-1, :]], axis=-2)   # day t sees days < t
    n = signal.lfilter([1.0], [1.0, -lam], x, axis=-2)
    q = (n + kappa * p) / (n.sum(-1, keepdims=True) + kappa)
    return _one_sided(q, p) if one_sided else q


def duration_q(Z, p, h: int = 1, lam: float = 0.98, kappa: float = 50.0, one_sided: bool = True) -> np.ndarray:
    """State-dependent bet q_t [..., T, 3]: state s_t = 1{a miss (Z >= 1) on one of the days t-h .. t-1}. Each state
    keeps its own shrunk EWMA band frequencies, updated (and discounted by lam) only on days spent in that state, so
    under clustering the after-a-miss state learns to bet on more misses."""
    Z, p = np.asarray(Z), np.asarray(p, float)
    T = Z.shape[-1]
    c = np.concatenate([np.zeros(Z.shape[:-1] + (1,), np.int64), np.cumsum(Z >= 1, -1)], -1)
    idx = np.arange(T)
    state = (c[..., idx] - c[..., np.maximum(idx - h, 0)]) > 0               # misses on days t-h .. t-1
    x = np.eye(3)[Z]
    n0, n1 = np.zeros(Z.shape[:-1] + (3,)), np.zeros(Z.shape[:-1] + (3,))
    q = np.empty(Z.shape + (3,))
    for t in range(T):
        st = state[..., t, None]
        cur = np.where(st, n1, n0)
        q[..., t, :] = (cur + kappa * p) / (cur.sum(-1, keepdims=True) + kappa)
        n1 = np.where(st, lam * n1 + x[..., t, :], n1)
        n0 = np.where(st, n0, lam * n0 + x[..., t, :])
    return _one_sided(q, p) if one_sided else q


LEARNED = ("rate", "plugin", "cluster", "combined")
HP = dict(plugin=dict(lam=0.99, kappa=50.0), cluster=dict(h=1, lam=0.99, kappa=50.0))   # frozen: exp 12a --tune (seed 7)


def learned_path(name: str, Z, p, hp: dict | None = None) -> np.ndarray:
    """SR path M [..., T] of a learned detector (names in LEARNED); hp overrides HP."""
    hp = {k: dict(v, **(hp or {}).get(k, {})) for k, v in HP.items()}
    if name == "rate":
        return e_sr_mix(Z, list(mixture_q(p)), p, return_path=True)["M"]
    if name == "plugin":
        return e_sr(Z, plugin_q(Z, p, **hp["plugin"]), p, return_path=True)["M"]
    if name == "cluster":
        return e_sr(Z, duration_q(Z, p, **hp["cluster"]), p, return_path=True)["M"]
    if name == "combined":
        qs = list(mixture_q(p))
        J = len(qs)
        return e_sr_mix(Z, qs + [duration_q(Z, p, **hp["cluster"])], p,
                        w=[0.5 / J] * J + [0.5], return_path=True)["M"]
    raise KeyError(name)


__all__ = ["categories", "e_values", "e_sr", "e_sr_mix", "alarms", "window_max", "mixture_q", "plugin_q",
           "duration_q", "learned_path", "LEARNED", "HP", "WINDOWS"]
