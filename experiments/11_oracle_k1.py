"""Time-varying ORACLE joint e-SR detector (K = 1) vs their Joint detector on the saved Section 3.4 break paths.

The oracle bets q_t = P_true(Z_t = j | F_{t-1}) (src/oracle.py) in the three-band SR recursion (src/edetectors.py),
gamma = n / iota = 10,000, windows: matched (accumulate from day 1, alarm in days 250..1000) and from day 1.
Their Joint detector: the saved run lengths (day = m - 1 + rl). One DGP pass per scenario serves both levels;
q_t is computed per level (a_t, b_t depend on it). Checks A-F print PASS/FAIL.

Under the null the oracle bets p, so E_t = 1 and M_t = t < gamma: its false-alarm rate is 0 by construction (not
calibrated).

Usage: python experiments/11_oracle_k1.py --scenario t400_b085 --level 0.95 [--smoke]     # one scenario, full report
       python experiments/11_oracle_k1.py --grid [--jobs 3] [--force] [--plot-only]       # Step 5 grid
"""
import argparse
import re
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src import dgp, edetectors as ed, oracle, paths, power_sims, sims_io

N, M_WIN, IOTA, NU = power_sims.N, power_sims.M_WIN, 0.1, 5.0
GAMMA = N / IOTA
BETA_PRE = dgp.DGPParams().garch_pre[2]                   # beta_post = BETA_PRE: no break (a null)
LEVELS = power_sims.LEVELS
N_EXAMPLES = 20
WINDOWS = ("matched", "day1")
DETECTORS = ("theirs", "oracle_matched", "oracle_day1")
BREAK_T = (0, 400, 700, 800, 850, 900, 950, 1000)        # beta_post = 0.85; 850 and 950 added to Fig 2's grid
BREAK_BP = (0.70, 0.75, 0.80, 0.85)                       # t* = 0
GRID = [(t, 0.85) for t in BREAK_T] + [(0, bp) for bp in BREAK_BP if bp != 0.85]
PATH_KEYS = [f"{k}_{w}" for k in ("alarm", "kl", "loglr") for w in WINDOWS] + ["resid", "qmean", "freq"]


def is_break(t, bp):
    return t < N and not np.isclose(bp, BETA_PRE)


def scenario_name(t, bp):
    return f"t{t}_b{round(bp * 100):03d}"


def parse_scenario(s):
    t, b = re.fullmatch(r"t(\d+)_b(\d{3})", s).groups()
    return int(t), int(b) / 100


def run_oracle(t_star, bp, levels, M, seed, n_chunks, saved):
    """{level: (per-path results, check statistics, example paths)}, chunk by chunk (q_t is not persisted)."""
    out = {}
    for lv in levels:
        res = {k: [] for k in PATH_KEYS}
        out[lv] = (res, dict(A=True, C=0.0, E_maxdev=0.0, E_alarms_pre=0), {})
    for sl, tr in oracle.truth_chunks(t_star, bp, M, seed, n_chunks=n_chunks):
        t0 = time.time()
        for lv in levels:
            res, chk, ex = out[lv]
            p = oracle.p_null(lv)
            var, covar = oracle.forecasts(tr, lv, NU)
            iv, ic = oracle.indicators(tr, var, covar)
            chk["A"] &= bool(np.array_equal(iv, saved[lv]["I_VaR"][sl])
                             and np.array_equal(ic, saved[lv]["I_CoVaR"][sl]))
            Z = ed.categories(iv, ic[:, 0])
            q = oracle.oracle_q(tr, var, covar, NU)
            L = ed.e_values(Z, q, p)
            if t_star > 0:
                chk["C"] = max(chk["C"], float(np.abs(q[:, :t_star] - p).max()))
                chk["E_maxdev"] = max(chk["E_maxdev"], float(np.abs(L[:, :t_star] - 1).max()))
            onehot, post = np.eye(3)[Z], slice(t_star, N)
            if t_star < N:
                res["resid"].append((onehot[:, post] - q[:, post]).mean(1))
                res["qmean"].append(q[:, post].mean(1))
                res["freq"].append(onehot[:, post].mean(1))
            r = ed.e_sr(Z, q, p, gamma=GAMMA)
            z0 = np.zeros((Z.shape[0], 1))
            cum = np.concatenate([z0, np.cumsum(oracle.kl(q, p), 1)], 1)            # cum[:, t] = days 1..t
            cll = np.concatenate([z0, np.cumsum(np.log(L), 1)], 1)
            for w in WINDOWS:
                a = r["alarm"][w]
                chk["E_alarms_pre"] += int((a <= t_star).sum())
                res[f"alarm_{w}"].append(a)
                idx = np.where(np.isfinite(a), a, 0).astype(int)
                at = lambda c: np.where(np.isfinite(a), c[np.arange(len(a)), idx] - c[:, t_star], np.nan)
                res[f"kl_{w}"].append(at(cum))
                res[f"loglr_{w}"].append(at(cll))
            if not ex:
                ex.update(q=q[:N_EXAMPLES].astype(np.float32), Z=Z[:N_EXAMPLES],
                          M=ed.e_sr(Z[:N_EXAMPLES], q[:N_EXAMPLES], p, gamma=GAMMA, return_path=True)["M"])
        print(f"  {scenario_name(t_star, bp)} paths {sl.start}..{sl.stop - 1}: {time.time() - t0:.0f}s "
              f"(q + detector, {len(levels)} levels)", flush=True)
    return {lv: ({k: np.concatenate(v) if v else np.empty((0, 3)) for k, v in res.items()}, chk, ex)
            for lv, (res, chk, ex) in out.items()}


def load_saved(t_star, bp, lv, M, seed):
    saved, meta = sims_io.load_sim(power_sims.name(1, lv, t_star, bp, M, seed))
    assert meta["draw_dim"] == power_sims.DRAW_DIM and meta["chunk"] == 500 and meta["break_in_X"]
    return saved


def cache_path(t, bp, lv, M, seed):
    return paths.DERIVED / "exp11" / f"oracle_{scenario_name(t, bp)}_a{lv:g}_M{M}_s{seed}.npz"


def compute_scenario(args):
    """Run the oracle for one scenario at all levels and cache the per-path results."""
    t, bp, M, seed, force = args
    if not force and all(cache_path(t, bp, lv, M, seed).exists() for lv in LEVELS):
        return
    saved = {lv: load_saved(t, bp, lv, M, seed) for lv in LEVELS}
    for lv, (res, chk, ex) in run_oracle(t, bp, LEVELS, M, seed, None, saved).items():
        f = cache_path(t, bp, lv, M, seed)
        f.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f, **res, **{f"chk_{k}": v for k, v in chk.items()},
                            **{f"ex_{k}": v for k, v in ex.items()})


def load_scenario(t, bp, lv, M, seed):
    with np.load(cache_path(t, bp, lv, M, seed)) as z:
        res = {k: z[k] for k in PATH_KEYS}
        chk = {k[4:]: z[k].item() for k in z.files if k.startswith("chk_")}
    return res, chk


# --------------------------------------------------------------------------- statistics
def summarise(day, t_star, M):
    rej = np.isfinite(day)
    d = day[rej & (day > t_star)] - t_star
    q1, q3 = np.percentile(d, [25, 75]) if d.size else (np.nan, np.nan)
    return dict(rejection=rej.mean(), se=np.sqrt(rej.mean() * (1 - rej.mean()) / M),
                prebreak_share=(rej & (day <= t_star)).mean(), share_detected=d.size / M, n_detected=d.size,
                delay_median=np.median(d) if d.size else np.nan, delay_mean=d.mean() if d.size else np.nan,
                delay_q1=q1, delay_q3=q3)


def mcnemar(r_theirs, r_oracle):
    b, c = int((r_theirs & ~r_oracle).sum()), int((~r_theirs & r_oracle).sum())
    p = stats.binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    return b, c, p


def paired_delay(d_theirs, d_oracle, t_star, B=2000, seed=0):
    both = np.isfinite(d_theirs) & np.isfinite(d_oracle) & (d_theirs > t_star) & (d_oracle > t_star)
    diff = d_oracle[both] - d_theirs[both]
    if diff.size < 2:
        return dict(n_pairs=int(diff.size), diff_median=np.nan, ci_median=(np.nan, np.nan), diff_mean=np.nan,
                    ci_mean=(np.nan, np.nan), wilcoxon_p=np.nan)
    rng = np.random.default_rng(seed)
    bs = diff[rng.integers(0, diff.size, (B, diff.size))]
    ci = lambda x: np.percentile(x, [2.5, 97.5])
    w = stats.wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
    return dict(n_pairs=int(both.sum()), diff_median=np.median(diff), ci_median=ci(np.median(bs, 1)),
                diff_mean=diff.mean(), ci_mean=ci(bs.mean(1)), wilcoxon_p=w)


def analyse(t_star, bp, lv, res, chk, saved):
    """Summary rows (per detector), paired tests (per oracle window), checks and information, as DataFrames."""
    M = res["alarm_day1"].size
    theirs = M_WIN - 1 + saved["rl_covar"][:M, -1]
    days = {"theirs": theirs, "oracle_matched": res["alarm_matched"], "oracle_day1": res["alarm_day1"]}
    key = dict(scenario=scenario_name(t_star, bp), t_star=t_star, beta_post=bp, level=lv)
    tab = pd.DataFrame([dict(key, detector=k, **summarise(v, t_star, M)) for k, v in days.items()])
    tests = []
    for w in WINDOWS:
        o = res[f"alarm_{w}"]
        b, c, p_mc = mcnemar(np.isfinite(theirs), np.isfinite(o))
        pdl = paired_delay(theirs, o, t_star)
        tests.append(dict(key, oracle=w, mcnemar_theirs_only=b, mcnemar_oracle_only=c, mcnemar_p=p_mc,
                          n_pairs=pdl["n_pairs"], delay_diff_median=pdl["diff_median"],
                          ci_median_lo=pdl["ci_median"][0], ci_median_hi=pdl["ci_median"][1],
                          delay_diff_mean=pdl["diff_mean"], ci_mean_lo=pdl["ci_mean"][0],
                          ci_mean_hi=pdl["ci_mean"][1], wilcoxon_p=pdl["wilcoxon_p"]))
    tests = pd.DataFrame(tests)
    r = res["resid"]
    zD = r.mean(0) / (r.std(0, ddof=1) / np.sqrt(r.shape[0])) if r.size else np.full(3, np.nan)
    nanmed = lambda x: np.median(x[np.isfinite(x)]) if np.isfinite(x).any() else np.nan
    info = pd.DataFrame([dict(key, oracle=w, median_cum_KL=nanmed(res[f"kl_{w}"]),
                              median_realised_logLR=nanmed(res[f"loglr_{w}"]), log_gamma=np.log(GAMMA),
                              log_gamma_headstart=np.log(GAMMA / (1 + t_star))) for w in WINDOWS])
    o, t = tab.set_index("detector").loc["oracle_matched"], tab.set_index("detector").loc["theirs"]
    rows = [("A", "recomputed indicators == saved (exact)", str(chk["A"]), bool(chk["A"])),
            ("C", "max |q_t - p|, t <= t* (< 1e-6)", f"{chk['C']:.1e}" if t_star else "n/a", chk["C"] < 1e-6),
            ("D", "post-break max |z| band freq vs mean q (< 3)",
             f"{np.abs(zD).max():.2f}" if r.size else "n/a", bool(not r.size or np.abs(zD).max() < 3)),
            ("E", "oracle alarms at day <= t* (== 0); max |E_t - 1|",
             f"{chk['E_alarms_pre']}; {chk['E_maxdev']:.1e}", chk["E_alarms_pre"] == 0)]
    if is_break(t_star, bp):
        rows += [("F", "oracle(matched) rejection >= theirs", f"{o.rejection:.4f} vs {t.rejection:.4f}",
                  bool(o.rejection >= t.rejection)),
                 ("F", "oracle(matched) detected after t* >= theirs",
                  f"{o.share_detected:.4f} vs {t.share_detected:.4f}", bool(o.share_detected >= t.share_detected)),
                 ("F", "oracle(matched) median delay <= theirs", f"{o.delay_median:.0f} vs {t.delay_median:.0f}",
                  bool(not np.isfinite(t.delay_median) or o.delay_median <= t.delay_median))]
    checks = pd.DataFrame([dict(key, check=c, what=w, value=v, result="PASS" if ok else "FAIL")
                           for c, w, v, ok in rows])
    band = pd.DataFrame(dict(band=[0, 1, 2], p_null=oracle.p_null(lv),
                             mean_q=res["qmean"].mean(0) if r.size else np.nan,
                             empirical=res["freq"].mean(0) if r.size else np.nan, z=zD))
    return tab, tests, checks, info, band


# --------------------------------------------------------------------------- figures
COLS = {"oracle_matched": "#2a78d6", "oracle_day1": "#eb6834"}
LABELS = {"oracle_matched": "oracle, matched window (days 250-1000)", "oracle_day1": "oracle, from day 1"}


def _panels(fig_axes, s, lv, j):
    if j == 0:
        g = s[np.isclose(s.beta_post, 0.85) & (s.level == lv)].sort_values("t_star")
        return g, "t_star", "Break date (β_post = 0.85)", "break date t*"
    g = s[(s.t_star == 0) & (s.level == lv)].sort_values("beta_post")
    return g, "beta_post", "Break size (t* = 0)", "post-break β"


def plot_power(tab, path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    for i, lv in enumerate(LEVELS):
        for j in range(2):
            ax = axes[i, j]
            g, x, title, xlabel = _panels(axes, tab, lv, j)
            th = g[g.detector == "theirs"]
            ax.plot(th[x], th.rejection, "-o", color="grey", lw=1.5, ms=3, label="theirs (Joint)")
            for k, (det, dx) in enumerate((("oracle_matched", -1), ("oracle_day1", 1))):
                o = g[g.detector == det]
                off = dx * (8 if j == 0 else 0.004)
                ax.errorbar(o[x] + off, o.rejection, yerr=2 * o.se, fmt="o", mfc="white", mec=COLS[det],
                            ecolor=COLS[det], ms=5, capsize=2, lw=1, label=f"{LABELS[det]} ± 2 SE")
            ax.axhline(0.1, color="grey", ls="--", lw=0.8, label="nominal size 10%")
            ax.set_title(f"{title}, α = β = {lv}", fontsize=9)
            ax.set_xlabel(xlabel)
        axes[i, 0].set_ylabel("rejection rate by day 1000")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_delays(tab, path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    for i, lv in enumerate(LEVELS):
        for j in range(2):
            ax = axes[i, j]
            g, x, title, xlabel = _panels(axes, tab, lv, j)
            g = g[[is_break(t, b) for t, b in zip(g.t_star, g.beta_post)]]
            for det, dx, col in (("theirs", 0, "grey"), ("oracle_matched", -1, COLS["oracle_matched"]),
                                 ("oracle_day1", 1, COLS["oracle_day1"])):
                o = g[g.detector == det]
                off = dx * (8 if j == 0 else 0.004)
                yerr = np.vstack([o.delay_median - o.delay_q1, o.delay_q3 - o.delay_median])
                lab = "theirs (Joint)" if det == "theirs" else LABELS[det]
                ax.errorbar(o[x] + off, o.delay_median, yerr=yerr, fmt="o-" if det == "theirs" else "o",
                            mfc="white" if det != "theirs" else col, mec=col, ecolor=col, color=col, ms=5,
                            capsize=2, lw=1, label=f"{lab}: median, IQR")
            ax.set_title(f"{title}, α = β = {lv}", fontsize=9)
            ax.set_xlabel(xlabel)
        axes[i, 0].set_ylabel("delay (alarm day − t*), alarms after t*")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- main
def single(args):
    t_star, bp = parse_scenario(args.scenario)
    lv, t_all = args.level, time.time()
    saved = load_saved(t_star, bp, lv, args.M, args.seed)
    print(f"scenario t*={t_star}, beta_post={bp:g}, alpha=beta={lv:g}, M={args.M}{' (smoke)' if args.smoke else ''}")
    res, chk, ex = run_oracle(t_star, bp, [lv], args.M, args.seed, 1 if args.smoke else None, {lv: saved})[lv]
    tag = f"{args.scenario}_a{lv:g}{'_smoke' if args.smoke else ''}"
    np.savez_compressed(paths.DERIVED / f"exp11_examples_{tag}.npz", **ex, t_star=t_star, level=lv)
    tab, tests, checks, info, band = analyse(t_star, bp, lv, res, chk, saved)
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
        print("\n=== rejection rate by day 1000 and delays (alarm day - t*, alarms after t*) ===")
        print(tab.drop(columns=["scenario", "t_star", "beta_post", "level"]).to_string(index=False))
        print("\n=== paired tests vs theirs (McNemar on rejection; delay diff = oracle - theirs, bootstrap 95% CI) ===")
        print(tests.drop(columns=["scenario", "t_star", "beta_post", "level"]).to_string(index=False))
        print("\n=== CHECK D: post-break band frequencies vs mean q_t ===")
        print(band.to_string(index=False))
        print("\n=== information: post-break cumulative KL(q||p) and realised log LR at the alarm (median) ===")
        print(info.drop(columns=["scenario", "t_star", "beta_post", "level"]).to_string(index=False))
    print("\n=== checks (B: tests/test_oracle.py) ===")
    print(checks[["check", "what", "value", "result"]].to_string(index=False))
    stem = paths.DERIVED / f"exp11_{tag}"
    tab.to_csv(f"{stem}_summary.csv", index=False)
    tests.to_csv(f"{stem}_tests.csv", index=False)
    checks.to_csv(f"{stem}_checks.csv", index=False)
    print(f"\nwrote {stem.name}_{{summary,tests,checks}}.csv; total {time.time() - t_all:.0f}s")


def grid(args):
    stem = paths.DERIVED / f"exp11_grid_M{args.M}_s{args.seed}"
    if not args.plot_only:
        t0 = time.time()
        with ProcessPoolExecutor(args.jobs) as ex:
            list(ex.map(compute_scenario, [(t, bp, args.M, args.seed, args.force) for t, bp in GRID]))
        print(f"oracle runs ready in {time.time() - t0:.0f}s", flush=True)
        parts = [analyse(t, bp, lv, *load_scenario(t, bp, lv, args.M, args.seed),
                         load_saved(t, bp, lv, args.M, args.seed)) for t, bp in GRID for lv in LEVELS]
        for i, name in enumerate(("summary", "tests", "checks", "info")):
            pd.concat([p[i] for p in parts], ignore_index=True).to_csv(f"{stem}_{name}.csv", index=False)
    tab, tests, checks, info = (pd.read_csv(f"{stem}_{n}.csv") for n in ("summary", "tests", "checks", "info"))

    f = lambda v: f"{v:.3f}"
    with pd.option_context("display.width", 220, "display.max_rows", 400, "display.max_columns", 40):
        idx = ["level", "t_star", "beta_post"]
        print("\n=== rejection rate by day 1000 (pre-break alarm share in brackets: theirs) ===")
        w = tab.pivot_table(index=idx, columns="detector", values=["rejection", "prebreak_share", "share_detected"])
        print(w.to_string(float_format=f))
        print("\n=== delays (alarm day - t*, alarms after t*): median [IQR] ===")
        d = tab[[is_break(t, b) for t, b in zip(tab.t_star, tab.beta_post)]].assign(
            delay=lambda x: x.apply(lambda r: f"{r.delay_median:.0f} [{r.delay_q1:.0f}, {r.delay_q3:.0f}]", 1))
        print(d.pivot_table(index=idx, columns="detector", values="delay", aggfunc="first").to_string())
        print("\n=== paired tests vs theirs ===")
        print(tests.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
        print("\n=== information: median post-break cumulative KL / realised log LR at the alarm ===")
        print(info.to_string(index=False, float_format=f))
        print("\n=== checks ===")
        print(checks.pivot_table(index=["level", "t_star", "beta_post"], columns="what", values="result",
                                 aggfunc="first").to_string())
        bad = checks[checks.result == "FAIL"]
        print(f"\n{len(checks) - len(bad)}/{len(checks)} checks PASS")
        if len(bad):
            print(bad[["level", "t_star", "beta_post", "check", "what", "value"]].to_string(index=False))
    paths.ensure_dirs()
    plot_power(tab, paths.FIGURES / "exp11_power.png")
    plot_delays(tab, paths.FIGURES / "exp11_delays.png")
    print(f"\nwrote {stem.name}_{{summary,tests,checks,info}}.csv, figures/exp11_power.png, exp11_delays.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="t400_b085")
    ap.add_argument("--level", type=float, default=0.95)
    ap.add_argument("--M", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--smoke", action="store_true", help="first seed chunk only (500 paths)")
    ap.add_argument("--grid", action="store_true", help="all Step 5 scenarios and both levels")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    grid(args) if args.grid else single(args)


if __name__ == "__main__":
    main()
