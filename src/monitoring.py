"""Port of R/SysRisk_Monitoring.R (CoVaR and reverse CoVaR), built on a generic core.

`monitor` works for any detector: give it detector paths [..., T, D] and thresholds [D].
Conventions from their code: normalised = path / threshold; alarm when normalised > 1
(strict); run length = first 1-based window index with an alarm (index 1 = window ending
at day m), else inf; Joint = min over detectors.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detectors import combined_detector, moments_from_cv_row


@dataclass
class MonitorResult:
    normalised: np.ndarray     # [..., T, D]
    run_length: np.ndarray     # [..., D]  (float, inf if no alarm)
    joint: np.ndarray          # [...]
    names: list[str]


def run_lengths(alarms: np.ndarray) -> np.ndarray:
    """First 1-based index along axis -2 where alarms is True, else inf. alarms: [..., T, D]."""
    any_ = alarms.any(axis=-2)
    first = alarms.argmax(axis=-2) + 1.0
    return np.where(any_, first, np.inf)


def monitor(paths: np.ndarray, thresholds, names=None) -> MonitorResult:
    paths = np.asarray(paths, dtype=float)
    thr = np.asarray(thresholds, dtype=float)
    if thr.shape != paths.shape[-1:]:
        raise ValueError("thresholds must have shape [D] matching paths[..., T, D]")
    norm = paths / thr
    rl = run_lengths(norm > 1)
    names = list(names) if names is not None else [str(i) for i in range(thr.size)]
    return MonitorResult(norm, rl, rl.min(axis=-1), names)


def first_detector_shares(run_length: np.ndarray) -> np.ndarray:
    """Share of replications [R, D] in which each detector raised the (possibly tied) first
    alarm -- tie rule of simulations/eval_sim_process_files.R (all tied minima count)."""
    rl = np.asarray(run_length)
    mn = rl.min(axis=-1, keepdims=True)
    first = (rl == mn) & np.isfinite(rl)
    return first.mean(axis=0)


# ----------------------------------------------------------------- CoVaR-specific wrappers
def covar_paths(I_VaR, I_CoVaR, cv_row, m: int, a: float = 0.5):
    """Detector paths [..., T, 1+K] (VaR, CoVaR_1..K) and thresholds [1+K] for CoVaR mode.

    I_VaR: [..., n];  I_CoVaR: [..., K, n];  cv_row: the 'CoVaR' row of the CV table."""
    alpha, beta = float(cv_row["alpha"]), float(cv_row["beta"])
    ms_v, ms_c = moments_from_cv_row(cv_row, "VaR"), moments_from_cv_row(cv_row, "SysRisk")
    dv = combined_detector(I_VaR, m, 1 - beta, ms_v, a)[..., None, :]          # [...,1,T]
    dc = combined_detector(I_CoVaR, m, (1 - alpha) * (1 - beta), ms_c, a)       # [...,K,T]
    paths = np.swapaxes(np.concatenate([dv, dc], axis=-2), -1, -2)              # [...,T,1+K]
    K = dc.shape[-2]
    thr = np.array([cv_row["c_VaR"]] + [cv_row["c_SysRisk"]] * K, dtype=float)
    return paths, thr


def reverse_paths(I_VaR_Rev, I_CoVaR_Rev, cv_rev_row, m: int, a: float = 0.5):
    """Detector paths [..., T, 2K] (VaR_1..K, CoVaR_1..K) for reverse CoVaR mode."""
    alpha, beta = float(cv_rev_row["alpha"]), float(cv_rev_row["beta"])
    ms_v, ms_c = moments_from_cv_row(cv_rev_row, "VaR"), moments_from_cv_row(cv_rev_row, "SysRisk")
    dv = combined_detector(I_VaR_Rev, m, 1 - beta, ms_v, a)
    dc = combined_detector(I_CoVaR_Rev, m, (1 - alpha) * (1 - beta), ms_c, a)
    paths = np.swapaxes(np.concatenate([dv, dc], axis=-2), -1, -2)
    K = dv.shape[-2]
    thr = np.array([cv_rev_row["c_VaR"]] * K + [cv_rev_row["c_SysRisk"]] * K, dtype=float)
    return paths, thr


def monitor_covar(I_VaR, I_CoVaR, cv_row, m: int, asset_names=None, a: float = 0.5) -> MonitorResult:
    paths, thr = covar_paths(I_VaR, I_CoVaR, cv_row, m, a)
    K = thr.size - 1
    banks = list(asset_names[1:]) if asset_names is not None else [str(k) for k in range(1, K + 1)]
    return monitor(paths, thr, ["VaR"] + [f"CoVaR_{b}" for b in banks])


def monitor_reverse(I_VaR_Rev, I_CoVaR_Rev, cv_rev_row, m: int, asset_names=None,
                    a: float = 0.5) -> MonitorResult:
    paths, thr = reverse_paths(I_VaR_Rev, I_CoVaR_Rev, cv_rev_row, m, a)
    K = thr.size // 2
    banks = list(asset_names[1:]) if asset_names is not None else [str(k) for k in range(1, K + 1)]
    return monitor(paths, thr, [f"VaR_{b}" for b in banks] + [f"CoVaR_{b}" for b in banks])
