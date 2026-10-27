"""Thresholds from our simulated sups standardised with THEIR stored null moments.

Their code re-simulates the moments for every K row of the CV table, so the K-row moments
differ slightly. Selecting K's thresholds on sups standardised with K's own stored moments
isolates moment Monte Carlo noise from everything else in the comparison."""
from __future__ import annotations

from . import data
from .critical_values import select_thresholds
from .detectors import Moments, moments_from_cv_row


def stored_moment_sets(cv, alpha: float, beta: float, K_set, n: int = 1000, m: int = 250,
                       iota: float = 0.1) -> dict[int, tuple[Moments, Moments]]:
    """{K: (ms_var, ms_covar)} from their CoVaR rows (reverse rows carry the same moments)."""
    out = {}
    for K in K_set:
        row = data.crit_value_row(cv, "CoVaR", alpha, beta, K, n=n, m=m, iota=iota)
        out[K] = (moments_from_cv_row(row, "VaR"), moments_from_cv_row(row, "SysRisk"))
    return out


def thresholds_with_stored_moments(sup_pairs: dict, K_set, iota: float) -> dict:
    """sup_pairs = {K: (sup_v, sup_c)} standardised with K's stored moments -> {K: thresholds}."""
    return {K: select_thresholds(*sup_pairs[K], K, iota) for K in K_set}
