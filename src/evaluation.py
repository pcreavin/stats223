"""Comparison statistics for alarm days on common paths (copied from experiments/11_oracle_k1.py, unchanged)."""
from __future__ import annotations

import numpy as np
from scipy import stats


def summarise(day, t_star, M):
    rej = np.isfinite(day)
    d = day[rej & (day > t_star)] - t_star
    q1, q3 = np.percentile(d, [25, 75]) if d.size else (np.nan, np.nan)
    return dict(rejection=rej.mean(), se=np.sqrt(rej.mean() * (1 - rej.mean()) / M),
                prebreak_share=(rej & (day <= t_star)).mean(), share_detected=d.size / M, n_detected=d.size,
                delay_median=np.median(d) if d.size else np.nan, delay_mean=d.mean() if d.size else np.nan,
                delay_q1=q1, delay_q3=q3)


def mcnemar(r_theirs, r_other):
    b, c = int((r_theirs & ~r_other).sum()), int((~r_theirs & r_other).sum())
    p = stats.binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    return b, c, p


def paired_delay(d_theirs, d_other, t_star, B=2000, seed=0):
    both = np.isfinite(d_theirs) & np.isfinite(d_other) & (d_theirs > t_star) & (d_other > t_star)
    diff = d_other[both] - d_theirs[both]
    if diff.size < 2:
        return dict(n_pairs=int(diff.size), diff_median=np.nan, ci_median=(np.nan, np.nan), diff_mean=np.nan,
                    ci_mean=(np.nan, np.nan), wilcoxon_p=np.nan)
    rng = np.random.default_rng(seed)
    bs = diff[rng.integers(0, diff.size, (B, diff.size))]
    ci = lambda x: np.percentile(x, [2.5, 97.5])
    w = stats.wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
    return dict(n_pairs=int(both.sum()), diff_median=np.median(diff), ci_median=ci(np.median(bs, 1)),
                diff_mean=diff.mean(), ci_mean=ci(bs.mean(1)), wilcoxon_p=w)


__all__ = ["summarise", "mcnemar", "paired_delay"]
