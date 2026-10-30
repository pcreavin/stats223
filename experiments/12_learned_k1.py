"""Extension 2, M3: learned e-detectors vs their Joint detector (and the oracle) on the Section 3.4 break paths, K = 1.

Same grid and paths as experiment 11 (common random numbers: every detector sees the same saved indicators).
Learned detectors (src/edetectors.LEARNED, hyperparameters frozen in HP) alarm from day 1 (headline; the
matched window, days 250-1000 like theirs, is also reported) at the calibrated gamma_hat
(experiments/12a_calibrate.py, B = 20,000, seed 11) and, for reference, at Ville's gamma = n / iota. Their Joint
detector: saved run lengths (alarm day = m - 1 + rl). Oracle: experiment 11's cached alarms (gamma = n / iota).

Usage: python experiments/12_learned_k1.py [--jobs 3] [--force] [--plot-only | --compute-only]
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import data, ecalibrate as ec, edetectors as ed, evaluation as ev, monitoring, oracle, paths, power_sims, \
    sims_io

N, M_WIN, IOTA, M, SEED = 1000, 250, 0.1, 2000, 2026
LEVELS = (0.9, 0.95)
BETA_PRE = 0.7
BREAK_T = (0, 400, 700, 800, 850, 900, 950, 1000)
BREAK_BP = (0.70, 0.75, 0.80, 0.85)
GRID = [(t, 0.85) for t in BREAK_T] + [(0, bp) for bp in BREAK_BP if bp != 0.85]
WINDOWS = tuple(ed.WINDOWS)
HEADLINE = "day1"                      # e-detectors may alarm from day 1; theirs can only from day m = 250


def head(window=HEADLINE):
    """{detector: column} of the detectors shown in tables and figures, for one e-detector alarm window."""
    return {"theirs": "theirs", "oracle": f"oracle_{window}", **{n: f"{n}_cal_{window}" for n in ed.LEARNED}}


LABEL = dict(theirs="theirs (Joint)", oracle="oracle (true q, γ = n/ι)", rate="rate mixture", plugin="plug-in",
             cluster="cluster", combined="combined")
STYLE = dict(theirs=dict(color="#6b6b6b", marker="o", ls="-"), oracle=dict(color="#1f1f1f", marker="", ls="--"),
             rate=dict(color="#2a78d6", marker="s", ls="-"), plugin=dict(color="#eb6834", marker="^", ls="-"),
             cluster=dict(color="#1baf7a", marker="D", ls="-"), combined=dict(color="#eda100", marker="v", ls="-"))


def is_break(t, bp):
    return t < N and not np.isclose(bp, BETA_PRE)


def scenario_name(t, bp):
    return f"t{t}_b{round(bp * 100):03d}"


def cache_path(t, bp, lv):
    return paths.DERIVED / "exp12" / f"days_{scenario_name(t, bp)}_a{lv:g}_M{M}_s{SEED}.npz"


def oracle_days(t, bp, lv):
    f = paths.DERIVED / "exp11" / f"oracle_{scenario_name(t, bp)}_a{lv:g}_M{M}_s{SEED}.npz"
    with np.load(f) as z:
        return {f"oracle_{w}": z[f"alarm_{w}"] for w in WINDOWS}


def compute(args):
    """Alarm days [M] of theirs and the learned detectors for one scenario and level, cached (oracle: see load)."""
    t, bp, lv, force = args
    f = cache_path(t, bp, lv)
    if f.exists() and not force:
        return
    cal = ec.calibrate()
    a, meta = sims_io.load_sim(power_sims.name(1, lv, t, bp, M, SEED))
    assert meta["break_in_X"] and meta["t_star"] == t
    Z = ed.categories(a["I_VaR"], a["I_CoVaR"][:, 0])
    p = oracle.p_null(lv)
    days = dict(theirs=M_WIN - 1 + a["rl_covar"][:, -1])
    for name in ed.LEARNED:
        path = ed.learned_path(name, Z, p)
        for w in WINDOWS:
            win = {w: ed.WINDOWS[w]}
            days[f"{name}_cal_{w}"] = ed.alarms(path, ec.lookup(cal, name, lv, w), win)[w]
            days[f"{name}_ville_{w}"] = ed.alarms(path, N / IOTA, win)[w]
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **days)


def load(t, bp, lv):
    with np.load(cache_path(t, bp, lv)) as z:
        return {k: z[k] for k in z.files} | oracle_days(t, bp, lv)


def analyse():
    rows, tests = [], []
    for t, bp in GRID:
        for lv in LEVELS:
            days = load(t, bp, lv)
            key = dict(scenario=scenario_name(t, bp), t_star=t, beta_post=bp, level=lv)
            for det, d in days.items():
                rows.append(dict(key, detector=det, **ev.summarise(d, t, M)))
                if det == "theirs":
                    continue
                b, c, p_mc = ev.mcnemar(np.isfinite(days["theirs"]), np.isfinite(d))
                pdl = ev.paired_delay(days["theirs"], d, t)
                tests.append(dict(key, detector=det, theirs_only=b, other_only=c, mcnemar_p=p_mc,
                                  n_pairs=pdl["n_pairs"], delay_diff_median=pdl["diff_median"],
                                  ci_lo=pdl["ci_median"][0], ci_hi=pdl["ci_median"][1],
                                  delay_diff_mean=pdl["diff_mean"], wilcoxon_p=pdl["wilcoxon_p"]))
    return pd.DataFrame(rows), pd.DataFrame(tests)


def checks(tab):
    out = []
    ref = pd.read_csv(paths.DERIVED / f"exp11_grid_M{M}_s{SEED}_summary.csv")
    for w in WINDOWS:
        a = tab[tab.detector == f"oracle_{w}"].set_index(["t_star", "beta_post", "level"]).rejection.sort_index()
        b = ref[ref.detector == f"oracle_{w}"].set_index(["t_star", "beta_post", "level"]).rejection.sort_index()
        out.append(("oracle rejection == experiment 11", w, bool(np.allclose(a.values, b.values))))
    nulls = tab[np.array([not is_break(t, b) for t, b in zip(tab.t_star, tab.beta_post)]) & tab.detector.str.contains("_cal_")]
    se = np.sqrt(IOTA * (1 - IOTA) / M)
    out.append(("calibrated size at the null scenarios within 3 SE of 10%",
                f"max |z| = {np.abs(nulls.rejection - IOTA).max() / se:.2f}",
                bool((np.abs(nulls.rejection - IOTA) < 3 * se).all())))
    their = tab[(tab.detector == "theirs")].set_index(["t_star", "beta_post", "level"])
    ref_t = ref[ref.detector == "theirs"].set_index(["t_star", "beta_post", "level"])
    out.append(("theirs rejection == experiment 11", "",
                bool(np.allclose(their.rejection.sort_index().values, ref_t.rejection.sort_index().values))))
    return pd.DataFrame(out, columns=["check", "detail", "ok"]).assign(result=lambda d: d.ok.map({True: "PASS",
                                                                                                  False: "FAIL"}))


# --------------------------------------------------------------------------- figures
def _style_ax(ax):
    ax.grid(True, color="#e6e6e3", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9a9a96")
    ax.tick_params(colors="#55554f", labelsize=8)


def _panel(tab, lv, j):
    if j == 0:
        g = tab[np.isclose(tab.beta_post, 0.85) & np.isclose(tab.level, lv)].sort_values("t_star")
        return g, "t_star", f"Break date (β_post = 0.85), α = β = {lv}", "break day t*"
    g = tab[(tab.t_star == 0) & np.isclose(tab.level, lv)].sort_values("beta_post")
    return g, "beta_post", f"Break size (t* = 0), α = β = {lv}", "post-break GARCH/DCC β"


def _legend(fig, axes):
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8.5)


def plot_power(tab, f, window=HEADLINE):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharey=True)
    for i, lv in enumerate(LEVELS):
        for j in range(2):
            ax = axes[i, j]
            g, x, title, xl = _panel(tab, lv, j)
            for det, key in head(window).items():
                o = g[g.detector == key]
                ax.plot(o[x], o.rejection, lw=2, ms=5, mfc="white", label=LABEL[det], **STYLE[det])
            ax.axhline(IOTA, color="#9a9a96", ls=":", lw=1, label="nominal size 10%")
            ax.set_title(title, fontsize=9, color="#2b2b28")
            ax.set_xlabel(xl, fontsize=8.5)
            _style_ax(ax)
        axes[i, 0].set_ylabel("rejection rate by day 1000", fontsize=8.5)
    _legend(fig, axes)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(f, dpi=140)
    plt.close(fig)


def plot_delays(tab, f, window=HEADLINE):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharey=True)
    for i, lv in enumerate(LEVELS):
        for j in range(2):
            ax = axes[i, j]
            g, x, title, xl = _panel(tab, lv, j)
            g = g[[is_break(t, b) for t, b in zip(g.t_star, g.beta_post)]]
            dets = list(head(window).items())
            span = (np.ptp(g[x]) or 1) * 0.012
            for k, (det, key) in enumerate(dets):
                o = g[g.detector == key]
                off = (k - (len(dets) - 1) / 2) * span
                yerr = np.vstack([o.delay_median - o.delay_q1, o.delay_q3 - o.delay_median])
                st = STYLE[det]
                ax.errorbar(o[x] + off, o.delay_median, yerr=yerr, color=st["color"], marker=st["marker"] or "o",
                            ls=st["ls"], lw=1.5, ms=4.5, mfc="white", capsize=2, elinewidth=0.9, label=LABEL[det])
            ax.set_title(title, fontsize=9, color="#2b2b28")
            ax.set_xlabel(xl, fontsize=8.5)
            _style_ax(ax)
        axes[i, 0].set_ylabel("detection delay: alarm day − t* (median, IQR)", fontsize=8.5)
    _legend(fig, axes)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(f, dpi=140)
    plt.close(fig)


CDF_SCEN = [(0, 0.85), (400, 0.85), (800, 0.85), (900, 0.85), (0, 0.75), (1000, 0.85)]


def plot_cdf(lv, f, window=HEADLINE):
    """Share of paths with an alarm by each day; the break day and day 250 marked."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.8), sharey=True)
    day = np.arange(1, N + 1)
    for ax, (t, bp) in zip(axes.flat, CDF_SCEN):
        d = load(t, bp, lv)
        for det, key in head(window).items():
            st = STYLE[det]
            share = (d[key][:, None] <= day[None, :]).mean(0)
            ax.plot(day, share, color=st["color"], ls=st["ls"], lw=2, label=LABEL[det])
        if is_break(t, bp):
            ax.axvline(t, color="#e34948", lw=1.2)
            ax.text(t + 8, 0.97, f"break t* = {t}", color="#e34948", fontsize=8, va="top")
        ax.axvline(M_WIN, color="#9a9a96", lw=0.8, ls=":")              # first day theirs can alarm
        ax.axhline(IOTA, color="#9a9a96", lw=0.6, ls=":")
        name = "no break (null)" if not is_break(t, bp) else f"t* = {t}, β_post = {bp:g}"
        ax.set_title(name, fontsize=9, color="#2b2b28")
        ax.set_xlabel("monitoring day", fontsize=8.5)
        _style_ax(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("share of paths alarmed by this day", fontsize=8.5)
    _legend(fig, axes)
    fig.suptitle(f"When does each detector fire?  α = β = {lv}, M = {M} paths", fontsize=10, color="#2b2b28")
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(f, dpi=140)
    plt.close(fig)


def plot_example(t, bp, lv, f, window=HEADLINE):
    """One path: band outcomes, their normalised detector, and the learned e-detectors' log10 M_t vs gamma_hat."""
    a, _ = sims_io.load_sim(power_sims.name(1, lv, t, bp, M, SEED))
    days = load(t, bp, lv)
    ok = np.isfinite(days["theirs"]) & (days["theirs"] > t)
    for n in ed.LEARNED:
        ok &= np.isfinite(days[f"{n}_cal_{window}"]) & (days[f"{n}_cal_{window}"] > t)
    i = int(np.flatnonzero(ok)[np.argsort(days["theirs"][ok])[ok.sum() // 2]])      # median-ish path for theirs
    Z = ed.categories(a["I_VaR"][i:i + 1], a["I_CoVaR"][i:i + 1, 0])
    p, cal = oracle.p_null(lv), ec.calibrate()
    cv = data.load_crit_values()
    row = data.crit_value_row(cv, "CoVaR", lv, lv, 1, n=N, m=M_WIN, iota=IOTA)
    pth, thr = monitoring.covar_paths(a["I_VaR"][i:i + 1], a["I_CoVaR"][i:i + 1], row, M_WIN)
    norm = (pth / thr)[0]
    day = np.arange(1, N + 1)

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8), sharex=True, gridspec_kw=dict(height_ratios=[0.8, 2, 2.6]))
    ax = axes[0]
    for band, col, lab in ((1, "#eb6834", "market VaR miss"), (2, "#e34948", "joint (CoVaR) miss")):
        dd = day[Z[0] == band]
        ax.vlines(dd, band - 1.4, band - 0.6, color=col, lw=1.2, label=lab)
    ax.set_yticks([])
    ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=2)
    ax.set_title(f"Example path {i}: t* = {t}, β_post = {bp:g}, α = β = {lv}", fontsize=9.5, color="#2b2b28")

    ax = axes[1]
    dd = np.arange(M_WIN, N + 1)
    ax.plot(dd, norm[:, 0], color="#6b6b6b", lw=1.5, label="theirs: VaR detector / threshold")
    ax.plot(dd, norm[:, 1], color="#1f1f1f", lw=1.5, ls="--", label="theirs: CoVaR detector / threshold")
    ax.axhline(1, color="#e34948", lw=0.9, ls=":")
    ax.set_ylabel("normalised detector\n(alarm when > 1)", fontsize=8.5)
    ax.legend(loc="upper left", fontsize=8, frameon=False)

    ax = axes[2]
    for n in ed.LEARNED:
        M_t = ed.learned_path(n, Z, p)[0]
        g = ec.lookup(cal, n, lv, window)
        ax.plot(day, np.log10(np.maximum(M_t, 1e-3) / g), color=STYLE[n]["color"], lw=1.6, label=LABEL[n])
    ax.axhline(0, color="#e34948", lw=0.9, ls=":")
    ax.set_ylabel("log10(M_t / γ̂)\n(alarm when ≥ 0)", fontsize=8.5)
    ax.set_xlabel("monitoring day", fontsize=8.5)
    ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=2)
    ax.set_ylim(-4, 1.5)
    for ax in axes:
        ax.axvline(t, color="#e34948", lw=1.2)
        ax.axvline(M_WIN, color="#9a9a96", lw=0.8, ls=":")
        _style_ax(ax)
    axes[2].text(t + 6, 1.3, f"break t* = {t}", color="#e34948", fontsize=8, va="top")
    first = {k: days[k][i] for k in ["theirs"] + [f"{n}_cal_{window}" for n in ed.LEARNED]}
    fig.tight_layout()
    fig.savefig(f, dpi=140)
    plt.close(fig)
    return i, first


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--compute-only", action="store_true", help="learned/theirs alarm days only (no oracle needed)")
    args = ap.parse_args()
    stem = paths.DERIVED / f"exp12_grid_M{M}_s{SEED}"
    if not args.plot_only:
        ec.calibrate()                                            # make sure the cache exists before forking
        t0 = time.time()
        jobs = [(t, bp, lv, args.force) for t, bp in GRID for lv in LEVELS]
        with ProcessPoolExecutor(args.jobs) as ex:
            list(ex.map(compute, jobs))
        print(f"alarm days ready in {time.time() - t0:.0f}s", flush=True)
        if args.compute_only:
            return
        tab, tests = analyse()
        tab.to_csv(f"{stem}_summary.csv", index=False)
        tests.to_csv(f"{stem}_tests.csv", index=False)
    tab, tests = pd.read_csv(f"{stem}_summary.csv"), pd.read_csv(f"{stem}_tests.csv")

    idx = ["level", "t_star", "beta_post"]
    paths.ensure_dirs()
    for window in (HEADLINE, "matched"):
        H = head(window)
        h = tab[tab.detector.isin(H.values())].assign(detector=lambda d: d.detector.map({v: k for k, v in H.items()}))
        with pd.option_context("display.width", 220, "display.max_rows", 400, "display.max_columns", 40):
            print(f"\n##### e-detector alarm window: {window} (theirs always from day {M_WIN}) #####")
            print("\n=== rejection rate by day 1000 (learned at calibrated gamma_hat) ===")
            print(h.pivot_table(index=idx, columns="detector", values="rejection")[list(H)].to_string(
                float_format=lambda v: f"{v:.3f}"))
            print("\n=== median delay after the break [alarm day - t*] ===")
            hb = h[[is_break(t, b) for t, b in zip(h.t_star, h.beta_post)]]
            print(hb.pivot_table(index=idx, columns="detector", values="delay_median")[list(H)].to_string(
                float_format=lambda v: f"{v:.0f}"))
            print("\n=== paired median delay difference vs theirs (negative = earlier) ===")
            tt = tests[tests.detector.isin(list(H.values())[1:])]
            print(tt.pivot_table(index=idx, columns="detector", values="delay_diff_median").to_string(
                float_format=lambda v: f"{v:.0f}"))
        sfx = "" if window == HEADLINE else f"_{window}"
        plot_power(tab, paths.FIGURES / f"exp12_power{sfx}.png", window)
        plot_delays(tab, paths.FIGURES / f"exp12_delays{sfx}.png", window)
        for lv in LEVELS:
            plot_cdf(lv, paths.FIGURES / f"exp12_detection_cdf_a{lv:g}{sfx}.png", window)
        i, first = plot_example(400, 0.85, 0.95, paths.FIGURES / f"exp12_example_path{sfx}.png", window)
        print(f"\nexample path {i} ({window}): first alarms {first}")
    print("\n=== checks ===")
    print(checks(tab).to_string(index=False))
    print(f"wrote {stem.name}_{{summary,tests}}.csv and figures/exp12_*.png")


if __name__ == "__main__":
    main()
