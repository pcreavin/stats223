"""Their reduced critical-value table over (n, m) (df_CritVal_reduced_20250805_mn.rds, B=2000).

For every (n, m, alpha=beta):
  1. null moments vs theirs;
  2. thresholds (B=2000, as theirs) vs theirs, tolerance 3*sqrt(2)*bootstrap SE;
  3. achieved size on fresh null sups (B=20,000): (a) our thresholds + our moments,
     (b) their thresholds + their per-K moments, on the same paths;
  4. granularity of the sup distribution (discrete cells are judged by check 3 only);
  5. pooled tilt of ours - theirs, with our moments and with their stored moments.

Usage: python experiments/01c_reduced_table.py [--seeds 2026] [--B 2000] [--B-fresh 20000]
       [--n-boot 100] [--workers 8] [--plot-only]
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src import critical_values as CV, data, paths
from src.detectors import null_moments, rolling_gini, rolling_uc
from src.their_moments import stored_moment_sets, thresholds_with_stored_moments

ALPHAS = [0.75, 0.9, 0.95]
K_SET = [1, 2, 5, 10]
IOTA, REPS = 0.1, 10**5
Z99 = stats.norm.ppf(0.995)
ATOL = 1e-9                     # >= tolerance: their R atom values may differ in the last bits
SE_EPS = 1e-9                   # bootstrap SE below this: every resample picked the same atom
ST_MOM, ST_SEL, ST_FRESH, ST_BOOT = 0, 1, 2, 3
COLS = {"CoVaR": ("c_VaR", "c_SysRisk"), "CoVaR_Reverse": ("c_VaR_rev", "c_SysRisk_rev")}


def _init_worker():
    import numba
    numba.set_num_threads(1)


def moments_task(args):
    seed, m, ia = args
    al = ALPHAS[ia]
    rng = np.random.default_rng([seed, ST_MOM, 0, m, ia])
    return (seed, m, ia), (null_moments(m, 1 - al, REPS, rng),
                           null_moments(m, (1 - al) ** 2, REPS, rng))


def _size_paths(sv, sc, c_v, c_c, K, typ, atol):
    ev, ec = sv >= c_v - atol, sc >= c_c - atol
    if typ == "CoVaR":
        return ev + K * ec - K * (ev & ec)
    return K * (ev | ec).astype(float)


def _atom(sup, c):
    return float(np.mean(np.abs(sup - c) < ATOL))


def _spacing(u, c):
    """Width of the support gap around c: next distinct sup value above c minus the last below
    (u = sorted distinct fresh sups). Large relative to the tolerance means the threshold
    cannot be pinned down more finely than the support allows."""
    lo, hi = u[u < c - ATOL], u[u > c + ATOL]
    return (hi[0] if hi.size else np.inf) - (lo[-1] if lo.size else -np.inf)


def cell_task(args):
    seed, n, m, ia, ms_ours, B, B_fresh, n_boot = args
    al = ALPHAS[ia]
    cv = data.load_crit_values_reduced()
    ms_th = stored_moment_sets(cv, al, al, K_SET, n=n, m=m)
    sets = [ms_ours] + [ms_th[K] for K in K_SET]

    sel = CV.simulate_sups_multi(n, m, al, al, B, sets, rng=np.random.default_rng([seed, ST_SEL, n, m, ia]))
    ours = CV.select_thresholds_multi(*sel[0], K_SET, IOTA)
    se = CV.bootstrap_se_multi(*sel[0], K_SET, IOTA, n_boot=n_boot, seed=[seed, ST_BOOT, n, m, ia])
    tm = thresholds_with_stored_moments(dict(zip(K_SET, sel[1:])), K_SET, IOTA)

    fresh = CV.simulate_sups_multi(n, m, al, al, B_fresh, sets,
                                   rng=np.random.default_rng([seed, ST_FRESH, n, m, ia]))
    fv, fc = fresh[0]
    uv, uc = np.unique(np.round(fv, 9)), np.unique(np.round(fc, 9))
    n_distinct_v, n_distinct_c = len(uv), len(uc)
    p_s01 = stats.binom.cdf(1, m, (1 - al) ** 2)

    rows = []
    for k, K in enumerate(K_SET):
        tv, tc = fresh[1 + k]
        for typ, (kv, kc) in COLS.items():
            t = data.crit_value_row(cv, typ, al, al, K, n=n, m=m)
            o_v, o_c = ours[K][kv], ours[K][kc]
            row = dict(seed=seed, n=n, m=m, level=al, K=K, type=typ,
                       n_windows=n - m + 1, distinct_sup_VaR=n_distinct_v,
                       distinct_sup_CoVaR=n_distinct_c, share_S01=p_s01)
            for col, o, th, s, tmv in (("c_VaR", o_v, t.c_VaR, se[K][kv], tm[K][kv]),
                                       ("c_SysRisk", o_c, t.c_SysRisk, se[K][kc], tm[K][kc])):
                tol = 3 * np.sqrt(2) * s
                row.update({f"{col}_ours": o, f"{col}_theirs": th, f"{col}_diff": o - th,
                            f"{col}_se": s, f"{col}_tol": tol,
                            f"{col}_z": (o - th) / (np.sqrt(2) * s) if s > SE_EPS else np.nan,
                            f"{col}_check": "PASS" if abs(o - th) <= tol else "FAIL",
                            f"{col}_theirmom": tmv, f"{col}_theirmom_diff": tmv - th,
                            f"{col}_spacing_ratio": _spacing(uv if col == "c_VaR" else uc, o) / tol
                            if tol > 0 else np.inf})
            # check 3: achieved size, each threshold set with its own moments, same paths
            Xa = _size_paths(fv, fc, o_v, o_c, K, typ, ATOL)
            Xb = _size_paths(tv, tc, t.c_VaR, t.c_SysRisk, K, typ, ATOL)
            Va, Vb, D = Xa.var(ddof=1), Xb.var(ddof=1), (Xa - Xb).var(ddof=1)
            band_a = Z99 * np.sqrt(Va / B + Va / B_fresh)
            band_b = Z99 * np.sqrt(Vb / B + Vb / B_fresh)
            band_d = Z99 * np.sqrt((Va + Vb) / B + D / B_fresh)
            sa, sb = Xa.mean(), Xb.mean()
            nom_a, nom_b, diff_ok = abs(sa - IOTA) <= band_a, abs(sb - IOTA) <= band_b, abs(sa - sb) <= band_d
            if nom_a and nom_b and diff_ok:
                c3 = "PASS"
            elif diff_ok and not nom_a and not nom_b and np.sign(sa - IOTA) == np.sign(sb - IOTA):
                c3 = "SHARED"
            else:
                c3 = "FAIL"
            atom = max(_atom(fv, o_v), _atom(fc, o_c), _atom(tv, t.c_VaR), _atom(tc, t.c_SysRisk))
            on_jump = atom >= 0.005 or min(se[K][kv], se[K][kc]) < SE_EPS
            discrete = n_distinct_v < 50 or n_distinct_c < 50 or on_jump
            row.update(size_ours=sa, size_theirs=sb, size_diff=sa - sb,
                       band_nominal_ours=band_a, band_nominal_theirs=band_b, band_diff=band_d,
                       size_ours_exact=_size_paths(fv, fc, o_v, o_c, K, typ, 0).mean(),
                       size_theirs_exact=_size_paths(tv, tc, t.c_VaR, t.c_SysRisk, K, typ, 0).mean(),
                       atom_mass_max=atom, discrete=discrete, check3=c3)
            c2 = "PASS" if row["c_VaR_check"] == row["c_SysRisk_check"] == "PASS" else "FAIL"
            row["check2"] = c2
            row["coarse"] = max(row["c_VaR_spacing_ratio"], row["c_SysRisk_spacing_ratio"]) > 1
            row["verdict"] = c3 if discrete else ("FAIL" if c2 == "FAIL" else c3)
            rows.append(row)
    return rows


def moment_rows(seed, n, m, ia, ms_ours, cv):
    al = ALPHAS[ia]
    th_rows = cv[(cv.SysRisk == "CoVaR") & (cv.n == n) & (cv.m == m)
                 & (cv.alpha.round(6) == al) & (cv.beta.round(6) == al)]
    th1 = th_rows[th_rows.K == 1].iloc[0]
    out = []
    for which, ms in zip(("VaR", "SysRisk"), ms_ours):
        for stat in ("UC", "IID"):
            sd_o = getattr(ms, f"sd_{stat}")
            for q in ("mu", "sd"):
                col = f"{q}_{which}_{stat}"
                o, th = getattr(ms, f"{q}_{stat}"), th1[col]
                tol = max(0.02 * abs(th), 4 * sd_o / np.sqrt(REPS))
                out.append(dict(seed=seed, n=n, m=m, level=al,
                                detector="VaR" if which == "VaR" else "CoVaR", stat=stat, quantity=q,
                                ours=o, theirs=th, their_spread_K=np.ptp(th_rows[col]),
                                rel_diff=o / th - 1 if th != 0 else np.nan, tol=tol,
                                edge="m50" if m == 50 else ("m999" if m == 999 else ""),
                                check="PASS" if abs(o - th) <= tol else "FAIL"))
    return out


def deep_moment(m, al, det, stat, mom, cv, reps=10**7, chunk=50_000):
    p = (1 - al) * ((1 - al) if det == "CoVaR" else 1)
    rng = np.random.default_rng([99, m, int(round(al * 100))])
    x = []
    for _ in range(reps // chunk):
        I = (rng.random((chunk, m)) <= p).astype(np.int8)
        x.append(rolling_gini(I, m)[:, 0] if stat == "IID" else rolling_uc(I, m, p)[:, 0])
    x = np.concatenate(x)
    sd = x.std(ddof=1)
    kurt = np.mean((x - x.mean()) ** 4) / sd**4
    which = "SysRisk" if det == "CoVaR" else "VaR"
    th = cv[(cv.SysRisk == "CoVaR") & (cv.m == m) & (cv.alpha.round(6) == al)][f"sd_{which}_{stat}"]
    ours = mom[(mom.m == m) & (mom.level == al) & (mom.detector == det) & (mom.stat == stat)
               & (mom.quantity == "sd")].ours.iloc[0]
    se = sd * np.sqrt((kurt - 1) / (4 * REPS))
    return (f"  m={m} level={al} {det} {stat}: reference sd {sd:.5f} (mu {x.mean():.5f}, kurtosis {kurt:.0f}); "
            f"SE of a 1e5 sd {se:.5f}; ours {ours:.5f} ({(ours - sd) / se:+.1f} SE); "
            f"theirs over {len(th)} (n, K) rows: mean {th.mean():.5f}, range [{th.min():.5f}, {th.max():.5f}]")


def tilt(res, which):
    """Pooled tilt over non-duplicate cells; per-cell mean over seeds first, as in 01b."""
    r = res[~((res.type == "CoVaR_Reverse") & (res.K == 1))]
    key = ["n", "m", "level", "K", "type"]
    out = []
    for col in ("c_VaR", "c_SysRisk"):
        d = r.groupby(key).agg(diff=(f"{col}_{which}", "mean"), se=(f"{col}_se", "mean"),
                               cnt=("seed", "size"), discrete=("discrete", "any")).reset_index()
        d["z"] = d["diff"] / np.sqrt(d.se**2 + d.se**2 / d.cnt)
        for label, s in (("all", d), ("non-discrete", d[~d.discrete & (d.se > SE_EPS)])):
            zz = s.z.replace([np.inf, -np.inf], np.nan).dropna()
            # cells in one (n, m, level) share our sups: average z per group, t-test over groups
            gm = s.assign(z=zz).groupby(["n", "m", "level"]).z.mean().dropna()
            out.append(dict(moments="ours" if which == "diff" else "theirs", threshold=col, cells=label,
                            N=len(s), share_pos=(s["diff"] > 0).mean(), mean_diff=s["diff"].mean(),
                            pooled_z=zz.sum() / np.sqrt(len(zz)) if len(zz) else np.nan,
                            groups=len(gm), group_t=gm.mean() / (gm.std(ddof=1) / np.sqrt(len(gm)))))
    return pd.DataFrame(out)


ZLIM = 6


def plot_tilt(res):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    nd = res[~res.discrete & ~((res.type == "CoVaR_Reverse") & (res.K == 1))]
    colors = dict(zip(ALPHAS, ("C0", "C1", "C2")))
    for col, mk in (("c_VaR", "o"), ("c_SysRisk", "^")):
        for which, fill in (("diff", True), ("theirmom_diff", False)):
            z = nd[f"{col}_{which}"] / (np.sqrt(2) * nd[f"{col}_se"])
            for xi, (xcol, other) in enumerate((("m", "n"), ("n", "m"))):
                jit = (nd[other].rank(method="dense") - 2) * (8 if xcol == "m" else 15)
                for al in ALPHAS:
                    s = nd.level == al
                    ax[xi].scatter(nd[xcol][s] + jit[s], z[s], s=10, marker=mk, alpha=.5,
                                   facecolors=colors[al] if fill else "none", edgecolors=colors[al],
                                   label=f"{col}, {'our' if fill else 'their'} moments, {al}")
    for xi, xcol in enumerate(("m", "n")):
        z = nd["c_VaR_diff"] / (np.sqrt(2) * nd["c_VaR_se"])
        mean_by = z.groupby(nd[xcol]).mean()
        ax[xi].plot(mean_by.index, mean_by.values, "k-", lw=1.2, label="mean z, c_VaR, our moments")
        ax[xi].axhline(0, color="grey", lw=.8)
        ax[xi].set(xlabel=xcol, title=f"(ours - theirs) / (sqrt(2) SE) vs {xcol}")
    clipped = 0
    for col in ("c_VaR", "c_SysRisk"):
        for which in ("diff", "theirmom_diff"):
            clipped += int((nd[f"{col}_{which}"] / (np.sqrt(2) * nd[f"{col}_se"])).abs().gt(ZLIM).sum())
    for a in ax:
        a.set_ylim(-ZLIM, ZLIM)
    ax[0].set_ylabel("z")
    ax[0].text(0.01, 0.01, f"{clipped} points beyond |z| = {ZLIM} not shown (bootstrap SE breakdown,"
               " see 01d)", transform=ax[0].transAxes, fontsize=7, va="bottom")
    ax[1].legend(fontsize=6, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(paths.FIGURES / "exp01c_tilt_vs_m.png", dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026])
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--B-fresh", type=int, default=20000)
    ap.add_argument("--n-boot", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--plot-only", action="store_true", help="redraw the figure from exp01c_reduced.csv")
    args = ap.parse_args()
    paths.ensure_dirs()
    if args.plot_only:
        plot_tilt(pd.read_csv(paths.DERIVED / "exp01c_reduced.csv"))
        return
    cv = data.load_crit_values_reduced()
    nm = sorted(cv[["n", "m"]].drop_duplicates().itertuples(index=False, name=None))
    print(f"{len(nm)} (n, m) pairs x {len(ALPHAS)} levels x {len(args.seeds)} seed(s)")
    t0 = time.time()

    with ProcessPoolExecutor(args.workers, initializer=_init_worker) as ex:
        mom_jobs = [(s, m, ia) for s in args.seeds for m in sorted({m for _, m in nm})
                    for ia in range(len(ALPHAS))]
        moments = dict(ex.map(moments_task, mom_jobs))
        print(f"moments done ({time.time() - t0:.0f}s)")
        jobs = [(s, n, m, ia, moments[(s, m, ia)], args.B, args.B_fresh, args.n_boot)
                for s in args.seeds for n, m in nm for ia in range(len(ALPHAS))]
        jobs.sort(key=lambda j: -(j[1] - j[2] + 1) * np.sqrt(j[2]))     # longest first
        res = pd.DataFrame([r for rows in ex.map(cell_task, jobs) for r in rows])
    print(f"cells done ({time.time() - t0:.0f}s)")
    res = res.sort_values(["seed", "n", "m", "level", "type", "K"]).reset_index(drop=True)
    mom = pd.DataFrame([r for s in args.seeds for n, m in nm for ia in range(len(ALPHAS))
                        for r in moment_rows(s, n, m, ia, moments[(s, m, ia)], cv)])
    res.to_csv(paths.DERIVED / "exp01c_reduced.csv", index=False)
    mom.to_csv(paths.DERIVED / "exp01c_reduced_moments.csv", index=False)

    pd.set_option("display.width", 220, "display.max_rows", 300)
    f4 = lambda x: f"{x:.4g}"
    print("\n=== 1. Null moments (10^5 windows) vs their K=1 row ===")
    print(f"{(mom.check == 'PASS').sum()}/{len(mom)} PASS; worst |rel diff| "
          f"{mom.rel_diff.abs().max():.2%}")
    print(mom.groupby(["edge"]).check.value_counts().unstack(fill_value=0).to_string())
    print("\nEdge cases m=50 and m=999:")
    print(mom[mom.edge != ""].drop(columns=["seed"]).to_string(index=False, float_format=f4))
    if (mom.check == "FAIL").any():
        print("\nMoment FAILs:\n" + mom[mom.check == "FAIL"].to_string(index=False, float_format=f4))
        print("\nReference for the failing (m, p): 10^7 windows; SE of a 10^5-window sd uses the kurtosis")
        for (m, al, det, stat) in mom[mom.check == "FAIL"][["m", "level", "detector", "stat"]] \
                .drop_duplicates().itertuples(index=False, name=None):
            print(deep_moment(m, al, det, stat, mom, cv))

    print("\n=== 2-4. Thresholds and achieved size, by granularity ===")
    print(res.groupby("discrete")[["check2", "check3", "verdict"]]
          .agg(lambda s: s.value_counts().to_dict()).to_string())
    print(f"\nOverall verdict: {res.verdict.value_counts().to_dict()}  (of {len(res)} cells)")
    print("\nCheck 2 in non-discrete cells, split by coarse support at the threshold "
          "(gap in the sup support wider than the tolerance; diagnostic only):")
    print(res[~res.discrete].groupby("coarse").check2.value_counts().unstack(fill_value=0).to_string())
    disc = res[res.discrete]
    if len(disc):
        print("\nDiscrete (n, m, level) groups:")
        print(disc.groupby(["n", "m", "level"]).agg(
            n_windows=("n_windows", "first"), distinct_VaR=("distinct_sup_VaR", "first"),
            distinct_CoVaR=("distinct_sup_CoVaR", "first"), share_S01=("share_S01", "first"),
            cells=("K", "size"), max_atom=("atom_mass_max", "max"),
            check3=("check3", lambda s: s.value_counts().to_dict())).to_string(float_format=f4))
    show = ["seed", "n", "m", "level", "K", "type", "discrete", "c_VaR_diff", "c_VaR_tol", "c_SysRisk_diff",
            "c_SysRisk_tol", "size_ours", "size_theirs", "band_nominal_ours", "band_diff", "check2", "check3"]
    bad = res[res.verdict != "PASS"]
    if len(bad):
        print("\nCells not PASS:\n" + bad[show].to_string(index=False, float_format=f4))
    print("\nAchieved size: mean (a) ours %.4f, (b) theirs %.4f, mean paired diff %+.4f"
          % (res.size_ours.mean(), res.size_theirs.mean(), res.size_diff.mean()))

    print("\n=== 5. Tilt (ours - theirs), pooled ===")
    tl = pd.concat([tilt(res, "diff"), tilt(res, "theirmom_diff")])
    print(tl.to_string(index=False, float_format=f4))

    plot_tilt(res)
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
