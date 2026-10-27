"""Shared reporting for experiments 02 and 03 (Tables 1-2): ours-vs-theirs tables and the figure."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.monitoring import first_detector_shares

BAND, BAND_INFO = 1.55, 1.1          # pp; PASS band = difference of two MC estimates at M = 5000
OURS, THEIRS = "#2a78d6", "#eb6834"  # categorical slots 1-2 of the default validated palette
INK, MUTED = "#0b0b0b", "#52514e"


def joint_table(sims, key, theirs, measure, K_set, levels):
    """sims[(K, level)] = (arrays, meta); key = 'rl_covar' | 'rl_rev'; theirs = their_size.joint_txt()."""
    rows = []
    for K in K_set:
        for lv in levels:
            rl = sims[(K, lv)][0][key]
            ours = 100 * np.mean(np.isfinite(rl[:, -1]))
            th = theirs[(theirs.measure == measure) & (theirs.K == K) & (theirs.level == lv)].joint.iloc[0]
            d = ours - th
            rows.append(dict(level=lv, K=K, ours=ours, theirs=th, diff=d,
                             within_1_1=abs(d) <= BAND_INFO, check="PASS" if abs(d) <= BAND else "FAIL"))
    return pd.DataFrame(rows)


def first_table(sims, key, theirs, names_fn, K_set, levels):
    """First-detector shares (%) vs theirs; band = 3 sd of the difference of two binomial estimates."""
    rows = []
    for K in K_set:
        for lv in levels:
            rl = sims[(K, lv)][0][key]
            M = rl.shape[0]
            sh = 100 * first_detector_shares(rl[:, :-1])
            for nm, o in zip(names_fn(K), sh):
                th = theirs[(theirs.K == K) & (theirs.level == lv) & (theirs.detector == nm)].share.iloc[0]
                p = (o + th) / 200
                band = 100 * 3 * np.sqrt(2 * p * (1 - p) / M)
                rows.append(dict(level=lv, K=K, detector=nm, ours=o, theirs=th, diff=o - th, band3sd=band,
                                 inside=abs(o - th) <= band))
    return pd.DataFrame(rows)


def plot_joint(tab, title, path, iota=10.0):
    levels = sorted(tab.level.unique())
    fig, axes = plt.subplots(1, len(levels), figsize=(4.2 * len(levels), 3.4), sharey=True)
    for ax, lv in zip(np.atleast_1d(axes), levels):
        t = tab[tab.level == lv]
        x = np.arange(len(t))
        ax.axhline(iota, color=MUTED, lw=1, ls="--")
        ax.errorbar(x - 0.08, t.ours, yerr=BAND, fmt="o", color=OURS, ms=7, lw=2, capsize=0, label="ours (± 1.55pp)")
        ax.plot(x + 0.08, t.theirs, "D", color=THEIRS, ms=7, label="theirs")
        ax.set_xticks(x, [f"K={k}" for k in t.K])
        ax.set_title(f"α = β = {lv}", color=INK, fontsize=10)
        ax.grid(axis="y", color="#e6e6e3", lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED)
    first = np.atleast_1d(axes)[0]
    first.set_ylabel("Joint rejection rate (%)", color=INK)
    last = np.atleast_1d(axes)[-1]
    last.text(len(tab[tab.level == levels[-1]]) - 0.5, iota + 0.12, "ι = 10%", color=MUTED, fontsize=8, ha="right")
    first.legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle(title, color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
