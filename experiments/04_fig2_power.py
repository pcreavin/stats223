"""Figures 2-3 (power) from OUR simulations with THEIR stored thresholds, vs the exact targets of exp 08.

Reduced scope (Milestone C): K in {1, 5}, alpha = beta in {0.9, 0.95}, CoVaR plus reverse for K = 5,
break_in_X = TRUE. Fig 2 left: beta_post = 0.85, t* in {0, 400, 700, 800, 900, 1000}; Fig 2 right: t* = 0,
beta_post in {0.70, 0.75, 0.80, 0.85}. Fig 3: K = 5, alpha = 0.9 first-detector shares. Delays: Joint,
t* in {400, 700}. PASS iff |ours - theirs| <= 3 sqrt(SE_ours^2 + SE_theirs^2).

Usage: python experiments/04_fig2_power.py [--M 2000] [--seed 2026] [--jobs 3] [--only T,BP] [--force]
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import null_sims, paths, power_sims, sims_io
from src.monitoring import first_detector_shares

LEVELS, K_SET = power_sims.LEVELS, power_sims.K_SET
M_WIN, N = power_sims.M_WIN, power_sims.N
DELAY_T = (400, 700)
MEASURES = (("CoVaR", "rl_covar"), ("RevCoVaR", "rl_rev"))


def _run(args):
    t, bp, M, seed, force = args
    power_sims.get(t, bp, M, seed, force=force)


def compare(df):
    """Adds diff, tol = 3 sqrt(se_ours^2 + se_theirs^2), z and check (both SEs 0: PASS only if equal)."""
    df = df.copy()
    df["diff"] = df.ours - df.theirs
    se = np.sqrt(df.se_ours ** 2 + df.se_theirs ** 2)
    df["tol"] = 3 * se
    df["both_se0"] = se == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        df["z"] = np.where(se > 0, df["diff"] / se, np.where(df["diff"] == 0, 0.0, np.inf))
    df["check"] = np.where(df["diff"].abs() <= df.tol, "PASS", "FAIL")
    return df


def binom_se(p, n):
    return np.sqrt(p * (1 - p) / n)


def fig2_table(sims, tg, M):
    rows = []
    for (t, bp), cfg in sims.items():
        for (K, lv), (a, _) in cfg.items():
            for meas, key in MEASURES:
                if key not in a:
                    continue
                p = np.isfinite(a[key][:, -1]).mean()
                for fig, ok in (("fig2_left", np.isclose(bp, 0.85)), ("fig2_right", t == 0)):
                    if not ok:
                        continue
                    th = tg[(tg.figure == fig) & (tg.Measure == meas) & (tg.K == K) & (tg.alpha == lv)
                            & (tg.Break_point == t) & np.isclose(tg.DCC_coef_post, bp)]
                    assert len(th) == 1, (fig, meas, K, lv, t, bp)
                    rows.append(dict(figure=fig, Measure=meas, alpha=lv, K=K, t_star=t, beta_post=bp,
                                     x=t if fig == "fig2_left" else bp, ours=p, se_ours=binom_se(p, M),
                                     theirs=th.Rate.iloc[0], se_theirs=th.SE.iloc[0]))
    return compare(pd.DataFrame(rows)).sort_values(["figure", "Measure", "alpha", "K", "x"])


def fig3_table(sims, tg, M, lv=0.9, K=5):
    rows = []
    for t in power_sims.FIG2_LEFT_T:
        a, _ = sims[(t, 0.85)][(K, lv)]
        for meas, key in MEASURES:
            rl = a[key]
            if meas == "CoVaR":
                assets = ["X"] + [str(k) for k in range(1, K + 1)]
            else:
                assets = [f"VaR_{k}" for k in range(1, K + 1)] + [f"CoVaR_{k}" for k in range(1, K + 1)]
            shares = dict(zip(assets, first_detector_shares(rl[:, :-1])))
            shares["Joint"] = np.isfinite(rl[:, -1]).mean()
            for asset, p in shares.items():
                th = tg[(tg.figure == "fig3") & (tg.Measure == meas) & (tg.alpha == lv)
                        & (tg.Break_point == t) & (tg.Asset == asset)]
                assert len(th) == 1, (meas, t, asset)
                rows.append(dict(Measure=meas, alpha=lv, K=K, t_star=t, Asset=asset, Series=th.Series.iloc[0],
                                 Detector=th.Detector.iloc[0], ours=p, se_ours=binom_se(p, M),
                                 theirs=th.Rate.iloc[0], se_theirs=th.SE.iloc[0]))
    return compare(pd.DataFrame(rows))


def delay_table(sims, dl):
    rows = []
    for t in DELAY_T:
        for (K, lv), (a, _) in sims[(t, 0.85)].items():
            for meas, key in MEASURES:
                if key not in a:
                    continue
                rl = a[key][:, -1]
                day = M_WIN - 1 + rl[np.isfinite(rl)]
                d = day[day > t] - t
                th = dl[(dl.sim_setting == "BreakPoint") & (dl.Measure == meas) & (dl.Asset == "Joint")
                        & dl.break_in_X & (dl.E == 5001) & (dl.K == K) & (dl.alpha == lv) & (dl.Break_point == t)
                        & np.isclose(dl.DCC_coef_post, 0.85)]
                assert len(th) == 1, (meas, K, lv, t)
                th = th.iloc[0]
                n_th = th.N * th.share_detected_after
                base = dict(Measure=meas, alpha=lv, K=K, t_star=t, n_detected=d.size, n_detected_theirs=n_th)
                rows.append(dict(base, stat="mean", ours=d.mean(), se_ours=d.std(ddof=1) / np.sqrt(d.size),
                                 theirs=th.delay_mean, se_theirs=th.delay_mean_se))
                # their bootstrap SE of the median, rescaled to our number of detections (no bootstrap here)
                rows.append(dict(base, stat="median", ours=np.median(d),
                                 se_ours=th.delay_median_se * np.sqrt(n_th / d.size),
                                 theirs=th.delay_median, se_theirs=th.delay_median_se))
    return compare(pd.DataFrame(rows))


def null_equality(sims, M, seed):
    """t* = 1000 (no break) must equal the first M paths of Milestone B's null simulation exactly."""
    out = []
    for (K, lv), (a, _) in sims[(1000, 0.85)].items():
        b, _ = sims_io.load_sim(null_sims.name(K, lv, 5000, seed))
        for k, v in a.items():
            w = b[k][:M]
            if k.startswith("I_") and k != "I_VaR":
                w = w[:, :K]
            out.append(dict(K=K, level=lv, array=k, equal=np.array_equal(v, w)))
    return pd.DataFrame(out)


def pass_table(df, idx, col):
    return df.pivot_table(index=idx, columns=col, values="check", aggfunc="first", sort=False)


def plot_fig2(t2, path):
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharey=True)
    cols = {1: "#2a78d6", 5: "#eb6834"}
    for i, lv in enumerate(LEVELS):
        for j, (figname, meas) in enumerate([(f, m) for f in ("fig2_left", "fig2_right")
                                             for m in ("CoVaR", "RevCoVaR")]):
            ax = axes[i, j]
            for K in K_SET:
                s = t2[(t2.figure == figname) & (t2.Measure == meas) & (t2.alpha == lv) & (t2.K == K)]
                if s.empty:
                    continue
                ax.plot(s.x, s.theirs, "-", color=cols[K], label=f"theirs K={K}")
                ax.errorbar(s.x, s.ours, yerr=2 * s.se_ours, fmt="o", color=cols[K], ms=4, label=f"ours K={K}")
            ax.axhline(0.1, color="grey", ls="--", lw=0.8)
            ax.set_title(f"{figname} {meas} α=β={lv}", fontsize=9)
            ax.set_xlabel("break point t*" if figname == "fig2_left" else "post-break β")
    axes[0, 0].set_ylabel("rejection rate")
    axes[1, 0].set_ylabel("rejection rate")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_fig3(t3, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, meas in zip(axes, ("CoVaR", "RevCoVaR")):
        s = t3[t3.Measure == meas]
        cmap = plt.get_cmap("tab20")
        for c, (asset, g) in enumerate(s.groupby("Asset", sort=False)):
            g = g.sort_values("t_star")
            ax.plot(g.t_star, g.theirs, "-", color=cmap(c), lw=1, label=asset)
            ax.errorbar(g.t_star, g.ours, yerr=2 * g.se_ours, fmt="o", color=cmap(c), ms=3)
        ax.axhline(0.1, color="grey", ls="--", lw=0.8)
        ax.set_title(f"Fig 3 first detection, {meas}, K=5, α=β=0.9 (lines theirs, points ours ±2SE)", fontsize=9)
        ax.set_xlabel("break point t*")
        ax.legend(fontsize=7, ncol=2)
    axes[0].set_ylabel("first-detection share")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--only", type=str, default=None, help="one scenario 'T,BP' (smoke test)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    paths.ensure_dirs()
    pd.set_option("display.width", 220, "display.max_rows", 400, "display.max_columns", 40)
    t0 = time.time()

    scen = power_sims.SCENARIOS
    if args.only:
        t, bp = args.only.split(",")
        scen = [(int(t), float(bp))]
    print(f"{len(scen)} scenarios, M={args.M}, seed={args.seed}, jobs={args.jobs}", flush=True)
    with ProcessPoolExecutor(args.jobs) as ex:
        list(ex.map(_run, [(t, bp, args.M, args.seed, args.force) for t, bp in scen]))
    sims = {(t, bp): power_sims.get(t, bp, args.M, args.seed) for t, bp in scen}
    print(f"all scenarios ready in {time.time() - t0:.0f}s", flush=True)

    if (1000, 0.85) in sims:
        if args.M % 500:
            print("\n(null equality check skipped: M must be a multiple of the 500-path chunk)")
        else:
            eq = null_equality(sims, args.M, args.seed)
            print(f"\n=== t* = 1000 vs Milestone B null arrays (first {args.M} paths): "
                  f"{eq.equal.sum()}/{len(eq)} arrays identical -> {'PASS' if eq.equal.all() else 'FAIL'}")
            if not eq.equal.all():
                print(eq[~eq.equal].to_string(index=False))
    if args.only:
        print(f"smoke test done in {time.time() - t0:.0f}s")
        return

    tg = pd.read_csv(paths.TARGETS / "targets_fig2_fig3_fig4.csv")
    dl = pd.read_csv(paths.TARGETS / "targets_delays.csv")
    f3 = lambda v: f"{v:.3f}"

    t2 = fig2_table(sims, tg, args.M)
    for figname in ("fig2_left", "fig2_right"):
        s = t2[t2.figure == figname]
        print(f"\n=== {figname}: Joint rejection rate, ours (M={args.M}) vs theirs (M=5000) ===")
        print(s.pivot_table(index=["Measure", "alpha", "K"], columns="x", values=["ours", "theirs"], sort=False)
              .to_string(float_format=f3))
        print(pass_table(s, ["Measure", "alpha", "K"], "x").to_string())

    t3 = fig3_table(sims, tg, args.M)
    print(f"\n=== fig3: first-detector shares, K=5, α=β=0.9 ===")
    print(t3.pivot_table(index=["Measure", "Asset"], columns="t_star", values=["ours", "theirs"], sort=False)
          .to_string(float_format=f3))
    print(pass_table(t3, ["Measure", "Asset"], "t_star").to_string())

    td = delay_table(sims, dl)
    print("\n=== Joint detection delays (days after t*, alarms after t* only) ===")
    print(td[["Measure", "alpha", "K", "t_star", "stat", "n_detected", "ours", "theirs", "tol", "z", "check"]]
          .to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    allc = pd.concat([t2.assign(table=t2.figure), t3.assign(table="fig3"), td.assign(table="delays")],
                     ignore_index=True)
    print("\n=== Summary ===")
    summ = allc.groupby("table").agg(cells=("check", "size"), PASS=("check", lambda c: (c == "PASS").sum()),
                                     both_se0=("both_se0", "sum"), max_abs_z=("z", lambda z: z.abs().max()))
    print(summ.to_string(float_format=lambda v: f"{v:.2f}"))
    print(f"TOTAL: {(allc.check == 'PASS').sum()}/{len(allc)} PASS")
    worst = allc.assign(abs_z=allc.z.abs()).sort_values("abs_z", ascending=False).head(10)
    print("\n10 worst cells by |z|:")
    print(worst[["table", "Measure", "alpha", "K", "t_star", "beta_post", "Asset", "stat", "ours", "theirs",
                 "tol", "z", "check"]].to_string(index=False, float_format=lambda v: f"{v:.4f}", na_rep=""))

    tag = f"M{args.M}_s{args.seed}"
    t2.to_csv(paths.DERIVED / f"exp04_fig2_{tag}.csv", index=False)
    t3.to_csv(paths.DERIVED / f"exp04_fig3_{tag}.csv", index=False)
    td.to_csv(paths.DERIVED / f"exp04_delays_{tag}.csv", index=False)
    plot_fig2(t2, paths.FIGURES / "fig2_ours_vs_theirs.png")
    plot_fig3(t3, paths.FIGURES / "fig3_ours_vs_theirs.png")
    print(f"\nwrote exp04_*_{tag}.csv, figures/fig2_ours_vs_theirs.png, fig3_ours_vs_theirs.png; "
          f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
