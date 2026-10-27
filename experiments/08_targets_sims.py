"""Exact simulation targets (Tables 1-2, Figures 2-4, detection delays) from DH's stored run lengths.

Checks our aggregation against their processed files (sim_eval_*processed_20251117.rds) and their
size tables (simulations/output/size_*.txt), then writes the targets and re-plots the figures.

Usage: python experiments/08_targets_sims.py [--n-boot 500] [--seed 2026]
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src import ggstyle as gg, paths, their_results as T

MEASURES = ["CoVaR", "RevCoVaR"]
MEASURE_LABEL = {"CoVaR": "CoVaR", "RevCoVaR": "Reverse CoVaR"}
K_SET = [1, 2, 5, 10]
N, M, IOTA = 1000, 250, 0.1


def verdict(ok):
    return "PASS" if ok else "FAIL"


# --------------------------------------------------------------------------- aggregation
def aggregate(settings):
    """Per-detector rates for any detection and first detection, all settings and measures."""
    out = []
    for name, df in settings.items():
        first = T.first_detection(df)
        for meas in MEASURES:
            for kind, d in (("any", df), ("first", first)):
                r = T.rates(d, meas)
                r["sim_setting"], r["kind"] = name, kind
                out.append(r[r["N"] > 0])
        del first
    res = pd.concat(out, ignore_index=True)
    res["Asset"] = res["Asset"].astype(str)
    det = [T.detector_of(a, m) for a, m in zip(res["Asset"], res["Measure"])]
    res["Detector"] = [d for d, _ in det]
    res["Series"] = [s for _, s in det]
    return res


def check_processed(res):
    """Exact comparison with their processed files (DetectionRate, ARL, sim_replications)."""
    rows = []
    for kind, first in (("any", False), ("first", True)):
        th = T.load_processed(first)
        th = th[th.sim_setting.isin(["BreakPoint", "ParamMisspec", "EstWindow"])]
        for meas in MEASURES:
            # their first-detection file only holds BreakPoint, so compare on the settings it has
            ours = res[(res.kind == kind) & (res.Measure == meas)
                       & res.sim_setting.isin(th.sim_setting.unique())]
            nrep = "sim_replications" if meas == "CoVaR" else "sim_replications_RevCoVaR"
            sub = th[th[nrep] > 0]
            sub = sub[sub[f"{meas}_DetectionRate"].notna()]
            mg = sub.merge(ours, on=["sim_setting", "Asset", *T.DESIGN], how="outer", indicator=True)
            both = mg[mg._merge == "both"]
            d_rate = (both["Rate"] - both[f"{meas}_DetectionRate"]).abs().max()
            arl_t, arl_o = both[f"{meas}_ARL"], both["ARL"]
            arl_nan_ok = (arl_t.isna() == arl_o.isna()).all()
            d_arl = (arl_t - arl_o).abs().max()
            d_n = (both[nrep] - both["N"]).abs().max()
            rows.append(dict(kind=kind, measure=meas, settings=",".join(sorted(sub.sim_setting.unique())),
                             rows_theirs=len(sub), rows_ours=len(ours),
                             unmatched=int((mg._merge != "both").sum()), max_d_rate=d_rate,
                             max_d_ARL=d_arl, max_d_N=d_n, ARL_nan_pattern=arl_nan_ok))
    t = pd.DataFrame(rows)
    t["verdict"] = [verdict(r.unmatched == 0 and r.max_d_rate <= 1e-12 and r.max_d_N == 0
                            and r.max_d_ARL <= 1e-9 and r.ARL_nan_pattern) for r in t.itertuples()]
    return t


# --------------------------------------------------------------------------- size tables
def _fmt(x):
    return "" if pd.isna(x) else f"{100 * x:.2f}"


def _null_rate(res, kind, meas, K, alpha, asset):
    sel = res[(res.sim_setting == "BreakPoint") & (res.kind == kind) & (res.Measure == meas)
              & (res.Break_point == 1000) & (~res.break_in_X) & (res.K == K)
              & (res.alpha == alpha) & (res.Asset == asset)]
    return sel["Rate"].iloc[0] if len(sel) == 1 else np.nan


def check_size_tables(res):
    out = T.SIM_OUTPUT
    cells = []  # (file, row, col, theirs, ours)
    lab = {"CoVaR": "CoVaR", "Reverse CoVaR": "RevCoVaR"}

    # Table 1: Joint null rejection rates (any detection; the first-detection Joint rate is equal)
    t1 = T.parse_xtable(out / "size_accum_rate.txt")
    for _, r in t1[t1.Measure.isin(lab)].iterrows():
        for col in t1.columns:
            if "_0." in col:
                K, al = col.split("_")
                cells.append(("size_accum_rate.txt", r.Measure, col, r[col],
                              _fmt(_null_rate(res, "any", lab[r.Measure], int(K), float(al), "Joint"))))

    # Table 2 variants at K=5, alpha=0.9
    def cov_asset(col):  # CoVaR rows: VaR_1 = 'X', CoVaR_j = 'j'
        return "Joint" if col == "Joint" else ("X" if col == "VaR_1" else col.split("_")[1])

    for fname, kind in (("size_accumrate_individual.txt", "any"), ("size_individual.txt", "any"),
                        ("size_first_individual.txt", "first")):
        t2 = T.parse_xtable(out / fname)
        for _, r in t2[t2.Measure.isin(lab)].iterrows():
            meas = lab[r.Measure]
            for col in [c for c in t2.columns if c == "Joint" or c.startswith(("VaR_", "CoVaR_"))]:
                if meas == "CoVaR":
                    asset = cov_asset(col)
                    ours = "" if col.startswith("VaR_") and col != "VaR_1" else \
                        _fmt(_null_rate(res, kind, meas, 5, 0.9, asset))
                else:
                    ours = _fmt(_null_rate(res, kind, meas, 5, 0.9, col))
                    # their eval_sim_AnyDetection.R filters Measure != "Reversed CoVaR" (typo),
                    # so the reverse row only keeps Joint in the any-detection tables
                    if kind == "any" and col != "Joint":
                        ours = ""
                cells.append((fname, r.Measure, col, r[col], ours))

    # Per-K first-detection tables
    tc = T.parse_xtable(out / "size_first_individual_CoVaR.txt")
    for _, r in tc.iterrows():
        K, al = int(r.K), float(r.alpha)
        for col in [c for c in tc.columns if c == "Joint" or c.startswith(("VaR_", "CoVaR_"))]:
            asset = cov_asset(col)
            ours = _fmt(_null_rate(res, "first", "CoVaR", K, al, asset)) \
                if asset in ("Joint", "X") or int(asset) <= K else ""
            cells.append(("size_first_individual_CoVaR.txt", f"{al}/K={K}", col, r[col], ours))
    tr = T.parse_xtable(out / "size_first_individual_RevCoVaR.txt")
    for _, r in tr.iterrows():
        K, al = int(r.K), float(r.alpha)
        cells.append(("size_first_individual_RevCoVaR.txt", f"{al}/K={K}", "Joint", r.Joint,
                      _fmt(_null_rate(res, "first", "RevCoVaR", K, al, "Joint"))))
        for j in range(1, 11):
            ours = _fmt(_null_rate(res, "first", "RevCoVaR", K, al, f"{r.Type}_{j}")) if j <= K else ""
            cells.append(("size_first_individual_RevCoVaR.txt", f"{al}/K={K}", f"{r.Type}_{j}",
                          r[str(j)], ours))

    c = pd.DataFrame(cells, columns=["file", "row", "col", "theirs", "ours"])
    c["match"] = c.theirs == c.ours
    return c


def table2_reverse_corrected(res):
    """Reverse CoVaR row of Table 2 (any detection) without their label typo."""
    cols = ["Joint"] + [f"VaR_{j}" for j in range(1, 6)] + [f"CoVaR_{j}" for j in range(1, 6)]
    return pd.DataFrame([{c: _fmt(_null_rate(res, "any", "RevCoVaR", 5, 0.9, c)) for c in cols}])


# --------------------------------------------------------------------------- targets
def figure_targets(res):
    keep = ["Measure", "sim_setting", "kind", "K", "alpha", "beta", "E", "Break_point",
            "DCC_coef_post", "break_in_X", "Asset", "Detector", "Series", "Rate", "SE", "N", "ARL"]
    any_, first = res[res.kind == "any"], res[res.kind == "first"]
    parts = {
        "table1": any_[(any_.sim_setting == "BreakPoint") & (any_.Break_point == 1000)
                       & (~any_.break_in_X) & (any_.Asset == "Joint")],
        "fig2_left": any_[(any_.sim_setting == "BreakPoint") & any_.break_in_X
                          & (any_.DCC_coef_post == 0.85) & (any_.Asset == "Joint")],
        "fig2_right": any_[(any_.sim_setting == "ParamMisspec") & (any_.Asset == "Joint")],
        "fig3": first[(first.sim_setting == "BreakPoint") & (first.K == 5) & first.break_in_X
                      & (first.Series.isin(["Joint", "X", "1", "2", "3", "4", "5"]))],
        "fig4": any_[(any_.sim_setting == "EstWindow") & (any_.Asset == "Joint")],
    }
    out = pd.concat([p[keep].assign(figure=k) for k, p in parts.items()], ignore_index=True)
    return out[["figure", *keep]].sort_values(["figure", "Measure", "alpha", "K", "E", "DCC_coef_post",
                                               "Break_point", "Asset"]).reset_index(drop=True)


# --------------------------------------------------------------------------- delays
DAYS = np.arange(M, N + 1)  # alarm days 250..1000; one extra bin for 'no alarm by n'


def _q_counts(vals, C, q):
    """Type-7 quantile of values 'vals' (sorted) with counts C (B x L), row-wise."""
    tot = C.sum(1)
    if len(vals) == 0:
        return np.full(len(C), np.nan)
    cum = np.cumsum(C, 1)
    h = (tot - 1) * q
    lo, hi = np.floor(h), np.ceil(h)
    idx_lo = (cum <= lo[:, None]).sum(1)
    idx_hi = (cum <= hi[:, None]).sum(1)
    L = len(vals) - 1
    v_lo, v_hi = vals[np.minimum(idx_lo, L)], vals[np.minimum(idx_hi, L)]
    out = v_lo + (h - lo) * (v_hi - v_lo)
    return np.where(tot > 0, out, np.nan)


def _delay_stats(C, tstar):
    """C: B x 752 counts over alarm days 250..1000 and 'none'."""
    Ntot = C.sum(1)
    pre = DAYS <= tstar
    Cf, Cp = C[:, :-1][:, pre], C[:, :-1][:, ~pre]
    nf, np_, nc = Cf.sum(1), Cp.sum(1), C[:, -1]
    delays = DAYS[~pre] - tstar
    with np.errstate(invalid="ignore", divide="ignore"):
        st = dict(
            share_false_alarm=nf / Ntot, share_detected_after=np_ / Ntot, share_censored=nc / Ntot,
            power_given_no_false_alarm=np_ / (Ntot - nf),
            censored_given_no_false_alarm=nc / (Ntot - nf),
            delay_mean=(Cp * delays).sum(1) / np_,
            delay_median=_q_counts(delays, Cp, 0.5),
            delay_q25=_q_counts(delays, Cp, 0.25), delay_q75=_q_counts(delays, Cp, 0.75),
            false_day_mean=(Cf * DAYS[pre]).sum(1) / nf if pre.any() else np.full(len(C), np.nan),
            false_day_median=_q_counts(DAYS[pre], Cf, 0.5) if pre.any() else np.full(len(C), np.nan),
            false_day_q25=_q_counts(DAYS[pre], Cf, 0.25) if pre.any() else np.full(len(C), np.nan),
            false_day_q75=_q_counts(DAYS[pre], Cf, 0.75) if pre.any() else np.full(len(C), np.nan),
        )
    st["delay_iqr"] = st["delay_q75"] - st["delay_q25"]
    st["false_day_iqr"] = st["false_day_q75"] - st["false_day_q25"]
    return st


def delay_table(settings, n_boot, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for name, df in settings.items():
        for meas in MEASURES:
            col = T.RL_COLS[meas]
            d = df[df[col].notna()]
            if name == "BreakPoint":  # Joint everywhere, every own detector for the Figure 3 design
                d = d[(d.Asset == "Joint") | ((d.K == 5) & d.break_in_X)]
            else:
                d = d[d.Asset == "Joint"]
            for key, g in d.groupby(["Asset", *T.DESIGN], observed=True):
                rl = g[col].to_numpy()
                day = np.where(np.isfinite(rl), M - 1 + rl, N + 1).astype(int)
                cnt = np.bincount(day - M, minlength=len(DAYS) + 1)[: len(DAYS) + 1].astype(float)
                tstar = key[1]
                point = _delay_stats(cnt[None, :], tstar)
                boot = _delay_stats(rng.multinomial(len(rl), cnt / cnt.sum(), size=n_boot).astype(float),
                                    tstar)
                row = dict(sim_setting=name, Measure=meas, **dict(zip(["Asset", *T.DESIGN], key)),
                           N=len(rl), break_ge_m=tstar >= M)
                for k, v in point.items():
                    row[k] = v[0]
                    if k.startswith(("delay_", "false_day_")):
                        row[f"{k}_se"] = np.nanstd(boot[k], ddof=1) if np.isfinite(boot[k]).sum() > 1 else np.nan
                    else:
                        p = v[0]
                        denom = len(rl) if not k.endswith("given_no_false_alarm") \
                            else len(rl) * (1 - point["share_false_alarm"][0])
                        row[f"{k}_se"] = np.sqrt(p * (1 - p) / denom) if denom > 0 else np.nan
                rows.append(row)
    out = pd.DataFrame(rows)
    out["Detector"] = [T.detector_of(a, m)[0] for a, m in zip(out.Asset, out.Measure)]
    return out


# --------------------------------------------------------------------------- plots
def _hlines(ax, iota=IOTA):
    for y in (0, 1):
        ax.axhline(y, color="black", lw=gg.LW_DEFAULT)
    ax.axhline(iota, color="black", lw=gg.LW_DEFAULT, ls=gg.LINETYPES["dashed"])


def plot_fig2(tg):
    cols = gg.hue_pal(4)
    fig = plt.figure(figsize=(9, 5))
    outer = fig.add_gridspec(1, 2, left=0.07, right=0.95, top=0.92, bottom=0.2, wspace=0.18)
    for p, (fname, xcol, xlab, rowcol, rowname) in enumerate(
            [("fig2_left", "Break_point", "Break point", "alpha", "alpha"),
             ("fig2_right", "DCC_coef_post", "Post break DCC parameter", "beta", "beta")]):
        d = tg[tg.figure == fname]
        levels = sorted(d[rowcol].unique())
        axes = gg.facet_grid(fig, outer[p], levels, MEASURES, [str(a) for a in levels],
                             [MEASURE_LABEL[m] for m in MEASURES])
        for i, lv in enumerate(levels):
            for j, meas in enumerate(MEASURES):
                ax = axes[i][j]
                _hlines(ax)
                for k, c in zip(K_SET, cols):
                    s = d[(d[rowcol] == lv) & (d.Measure == meas) & (d.K == k)].sort_values(xcol)
                    ax.plot(s[xcol], s.Rate, color=c, lw=gg.LW_DEFAULT)
                if i == len(levels) - 1:
                    ax.set_xlabel(xlab if j == 0 else "", fontsize=10)
        axes[0][0].set_ylabel("Rejection rate", fontsize=10, y=0)
        for ax in (axes[-1][0], axes[-1][1]):
            ax.xaxis.label.set_x(1.0)
    handles = [Line2D([], [], color=c, lw=gg.LW_DEFAULT) for c in cols]
    gg.bottom_legend(fig, handles, [str(k) for k in K_SET], "K")
    fig.savefig(paths.FIGURES / "targets_fig2.png", dpi=200)
    plt.close(fig)


def plot_fig3(tg, alpha=0.9):
    d = tg[(tg.figure == "fig3") & (tg.alpha == alpha)]
    series = ["Joint", "X", "1", "2", "3", "4", "5"]
    cmap = dict(zip(series, ["black", "blue", *gg.hue_pal(5)]))
    lt = {"Joint": "solid", "VaR": "longdash", "Systemic Risk": "dotdash"}
    fig = plt.figure(figsize=(8, 4))
    gs = fig.add_gridspec(1, 1, left=0.08, right=0.98, top=0.9, bottom=0.3)
    axes = gg.facet_grid(fig, gs[0], [0], MEASURES, None, [MEASURE_LABEL[m] for m in MEASURES])[0]
    lw = 1 * gg.PT_PER_MM * 0.75
    for ax, meas in zip(axes, MEASURES):
        ax.axvline(0.7, color="grey", lw=gg.LW_DEFAULT)  # as in their script (a leftover at x=0.7)
        ax.axhline(IOTA, color="black", lw=gg.LW_DEFAULT, ls=gg.LINETYPES["dashed"])
        sub = d[d.Measure == meas]
        for (det, ser), s in sub.groupby(["Detector", "Series"]):
            s = s.sort_values("Break_point")
            ax.plot(s.Break_point, s.Rate, color=cmap[ser], ls=gg.LINETYPES[lt[det]], lw=lw)
        ax.set_xlabel("Break point", fontsize=10)
    axes[0].set_ylabel("Rejection rate", fontsize=10)
    axes[0].xaxis.label.set_x(1.0)
    axes[1].set_xlabel("")
    h1 = [Line2D([], [], color=cmap[s], lw=lw) for s in series]
    h2 = [Line2D([], [], color="black", lw=lw, ls=gg.LINETYPES[lt[k]]) for k in lt]
    gg.bottom_legend(fig, h1, series, "Series", y=0.08)
    gg.bottom_legend(fig, h2, list(lt), "Detector", y=0.0)
    fig.savefig(paths.FIGURES / "targets_fig3.png", dpi=200)
    plt.close(fig)


def plot_fig4(tg, alpha=0.9):
    d = tg[(tg.figure == "fig4") & (tg.alpha == alpha)]
    E_levels = [1000, 2000, 5000, 5001]
    cols = gg.hue_pal(4)
    fig = plt.figure(figsize=(8, 5))
    gs = fig.add_gridspec(1, 1, left=0.08, right=0.95, top=0.92, bottom=0.18)
    axes = gg.facet_grid(fig, gs[0], MEASURES, K_SET, [MEASURE_LABEL[m] for m in MEASURES],
                         [f"K = {k}" for k in K_SET])
    for i, meas in enumerate(MEASURES):
        for j, k in enumerate(K_SET):
            ax = axes[i][j]
            _hlines(ax)
            for e, c in zip(E_levels, cols):
                s = d[(d.Measure == meas) & (d.K == k) & (d.E == e)].sort_values("Break_point")
                ax.plot(s.Break_point, s.Rate, color=c, lw=gg.LW_DEFAULT)
            ax.set_xlim(-50, 1050)
    axes[1][0].set_xlabel("Break point", fontsize=10)
    axes[1][0].xaxis.label.set_x(2.0)
    axes[0][0].set_ylabel("Rejection rate", fontsize=10, y=0)
    handles = [Line2D([], [], color=c, lw=gg.LW_DEFAULT) for c in cols]
    gg.bottom_legend(fig, handles, ["1000", "2000", "5000", "True parameter value"], "E")
    fig.savefig(paths.FIGURES / "targets_fig4.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    paths.ensure_dirs()
    pd.set_option("display.width", 200, "display.max_columns", 30)

    settings = {k: f() for k, f in T.SIM_LOADERS.items()}
    for k, df in settings.items():
        ab = df[["alpha", "beta"]].drop_duplicates()
        print(f"{k}: {len(df):,} rows, (alpha, beta) pairs {sorted(map(tuple, ab.to_numpy()))}, "
              f"reps {df.i_MC.nunique()}")

    res = aggregate(settings)

    print("\n== Exact check vs their processed files (sim_eval_*processed_20251117.rds)")
    cp = check_processed(res)
    print(cp.to_string(index=False))

    print("\n== Size tables vs simulations/output/*.txt (strings at 2 dp)")
    st = check_size_tables(res)
    summ = st.groupby("file").agg(cells=("match", "size"), matched=("match", "sum"))
    summ["verdict"] = [verdict(a == b) for a, b in zip(summ.cells, summ.matched)]
    print(summ.to_string())
    if (~st.match).any():
        print(st[~st.match].to_string(index=False))
    print("\nTable 2 reverse row (any detection), corrected for their 'Reversed CoVaR' typo:")
    print(table2_reverse_corrected(res).to_string(index=False))

    tg = figure_targets(res)
    tg.to_csv(paths.TARGETS / "targets_fig2_fig3_fig4.csv", index=False)
    print(f"\nwrote targets_fig2_fig3_fig4.csv ({len(tg)} rows)")

    show = tg[tg.figure == "table1"].pivot_table(index="Measure", columns=["K", "alpha"], values="Rate")
    print("\nTable 1 (Joint null rejection rate, %):\n", (100 * show).round(2).to_string())
    for fname, xcol in (("fig2_left", "Break_point"), ("fig2_right", "DCC_coef_post")):
        p = tg[tg.figure == fname].pivot_table(index=["Measure", "alpha", "K"], columns=xcol,
                                               values="Rate")
        print(f"\n{fname} rejection rates:\n", p.round(3).to_string())
    p = tg[(tg.figure == "fig3") & (tg.alpha == 0.9)].pivot_table(
        index=["Measure", "Detector", "Series"], columns="Break_point", values="Rate")
    print("\nfig3 first-detection shares (K=5, alpha=0.9):\n", p.round(3).to_string())
    p = tg[(tg.figure == "fig4") & (tg.alpha == 0.9)].pivot_table(
        index=["Measure", "K", "E"], columns="Break_point", values="Rate")
    print("\nfig4 rejection rates (alpha=0.9; E=5001 is the true parameter):\n", p.round(3).to_string())
    print(f"\nmax MC SE over all rates: {tg.SE.max():.4f}")

    dl = delay_table(settings, args.n_boot, args.seed)
    dl.to_csv(paths.TARGETS / "targets_delays.csv", index=False)
    print(f"\nwrote targets_delays.csv ({len(dl)} rows, bootstrap B={args.n_boot})")
    j = dl[(dl.sim_setting == "BreakPoint") & (dl.Asset == "Joint") & dl.break_in_X
           & (dl.alpha == 0.9) & dl.break_ge_m]
    cols = ["Measure", "K", "Break_point", "share_false_alarm", "share_detected_after",
            "share_censored", "delay_median", "delay_mean", "delay_mean_se", "delay_iqr",
            "false_day_median"]
    print("\nJoint delays, BreakPoint, break in all series, alpha=0.9:\n",
          j[cols].round(3).to_string(index=False))

    plot_fig2(tg)
    plot_fig3(tg)
    plot_fig4(tg)
    print("\nwrote figures/targets_fig2.png, targets_fig3.png, targets_fig4.png")


if __name__ == "__main__":
    main()
