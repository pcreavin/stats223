"""Milestone D: the application (Figures 5-6) on the R fits of r/fit_forecasts_application.R.

(a) our monitoring on the R indicators vs their unchanged SysRisk_Monitoring (rmon_*), (b) vs their stored
appl_res_20251024.rds, (c) indicators recomputed from R's losses, D_t, R_t and nu with our c(rho, nu),
(d) Figures 5-6, (e) qualitative PASS/FAIL checks. Everything is aligned by window index; `date` is the
correct date of the na.omit-ed data row, their labels (shifted in COVID, see exp 10) are kept separately.

Usage: python experiments/05_application_fig5_fig6.py [--recompute-specs CCC_norm]
"""
import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from src import data, dgp, ggstyle as gg, paths, their_results
from src.monitoring import monitor_covar, monitor_reverse

APPL = paths.DERIVED / "appl"
SPECS = ["CCC_norm", "DCC_std"]
SPEC_LABEL = {"CCC_norm": "CCC-GARCH Gaussian", "DCC_std": "DCC-GARCH Student-t"}
WINDOWS = ["FinCrisis", "CalmTime", "COVID", "TrumpTariff"]
WINDOW_LABEL = {"FinCrisis": "Financial Crisis", "CalmTime": "Calm Time", "COVID": "COVID",
                "TrumpTariff": "Trumps Tariffs"}
EVENTS = pd.to_datetime(["2008-09-15", "2020-03-13", "2023-03-10", "2025-04-02"])
INDEX, BANKS = "SPF", ["BAC", "C", "GS", "JPM", "WFC"]
K, M_WIN, LEVEL = 5, 250, 0.95
TOL = 1e-10


def load_run(spec, win):
    df = pd.read_csv(APPL / f"{spec}_{win}.csv", parse_dates=["date", "date_theirs"])
    nu = pd.read_csv(APPL / f"nu_{spec}_{win}.csv")
    rmon = {mode: (pd.read_csv(APPL / f"rmon_{spec}_{win}_{mode}.csv", parse_dates=["date", "date_theirs"]),
                   pd.read_csv(APPL / f"rmon_{spec}_{win}_{mode}_rl.csv"))
            for mode in ("covar", "rev")}
    return df, nu, rmon


def indicators(df):
    return dict(I_VaR=df["I_VaR"].to_numpy(),
                **{s: df[[f"{s}_{b}" for b in BANKS]].to_numpy().T for s in ("I_CoVaR", "I_VaR_Rev", "I_CoVaR_Rev")})


def our_monitoring(ind, cv_row, cv_rev_row):
    # SysRisk_Monitoring takes the moments from the CoVaR row in both modes; only the thresholds differ
    rev_row = cv_row.copy()
    rev_row[["c_VaR", "c_SysRisk"]] = cv_rev_row[["c_VaR", "c_SysRisk"]]
    names = [INDEX] + BANKS
    return {"covar": monitor_covar(ind["I_VaR"], ind["I_CoVaR"], cv_row, M_WIN, names),
            "rev": monitor_reverse(ind["I_VaR_Rev"], ind["I_CoVaR_Rev"], rev_row, M_WIN, names)}


def first_alarms(norm: pd.DataFrame) -> pd.Series:
    """0-based window index of the first normalised value > 1 per detector (NaN if none)."""
    alarm = norm.to_numpy() > 1
    return pd.Series(np.where(alarm.any(0), alarm.argmax(0), np.nan), index=norm.columns)


def their_paths(appl, spec, win, mode):
    d = appl[(appl.GARCH_spec == spec) & (appl.Setting == win) & (appl.alpha == LEVEL)
             & (appl.SysRiskMeasure == ("CoVaR" if mode == "covar" else "RevCoVaR"))].copy()
    if mode == "covar":
        d["det"] = np.where(d["Monitoring Sequence"] == "VaR", "VaR", "CoVaR_" + d.Asset.astype(str))
    else:
        d["det"] = d["Monitoring Sequence"].astype(str) + "_" + d.Asset.astype(str)
    return d.pivot(index="Date", columns="det", values="value").sort_index()


# --------------------------------------------------------------------------- (c) recomputed indicators
def recompute(df, nu):
    """violation_sequences with their nu indexing: nu[0] (= their nu[1]) for X, nu[k] for bank k."""
    nu = nu["nu"].to_numpy(float)
    Y = df[[f"loss_{a}" for a in [INDEX] + BANKS]].to_numpy()
    S = df[[f"sd_{a}" for a in [INDEX] + BANKS]].to_numpy()
    R = df[[f"r_{INDEX}_{b}" for b in BANKS]].to_numpy()
    ours = dgp.violation_sequences(Y, S, R, nu, LEVEL, LEVEL)
    # thresholds, for listing disagreeing days
    thr = {"I_VaR": S[:, 0] * dgp.var_unit(LEVEL, nu[0])}
    for k, b in enumerate(BANKS):
        thr[f"I_CoVaR_{b}"] = S[:, k + 1] * dgp._c_of(R[:, k], LEVEL, LEVEL, nu[k + 1])
        thr[f"I_VaR_Rev_{b}"] = S[:, k + 1] * dgp.var_unit(LEVEL, nu[k + 1])
        thr[f"I_CoVaR_Rev_{b}"] = S[:, 0] * dgp._c_of(R[:, k], LEVEL, LEVEL, nu[0])
    rows, rates = [], {}
    for col in [c for c in df.columns if c.startswith("I_")]:
        seq, _, bank = col.rpartition("_") if col != "I_VaR" else ("I_VaR", "", None)
        o = ours["I_VaR"] if col == "I_VaR" else ours[seq][BANKS.index(bank)]
        r = df[col].to_numpy()
        rates[col] = (o == r).mean()
        loss = Y[:, 0] if col in ("I_VaR",) or col.startswith("I_CoVaR_Rev") else Y[:, BANKS.index(bank) + 1]
        for i in np.flatnonzero(o != r):
            rows.append(dict(seq=col, day=i, date=df.date[i].date(), R=int(r[i]), ours=int(o[i]),
                             loss=loss[i], thr=thr[col][i], rel_gap=abs(loss[i] - thr[col][i]) / abs(thr[col][i])))
    return rates, rows


# --------------------------------------------------------------------------- (d) figures
def figure(paths_by_cell, mode, out):
    fig = plt.figure(figsize=(8.5, 5))
    gs = fig.add_gridspec(1, 1, left=0.07, right=0.95, top=0.92, bottom=0.2)
    axes = gg.facet_grid(fig, gs[0], SPECS, WINDOWS, [SPEC_LABEL[s] for s in SPECS],
                         [WINDOW_LABEL[w] for w in WINDOWS])
    cols = dict(zip([INDEX] + BANKS, gg.hue_pal(6)))
    for i, spec in enumerate(SPECS):
        for j, win in enumerate(WINDOWS):
            ax = axes[i][j]
            if (spec, win) not in paths_by_cell:
                continue
            dates, norm = paths_by_cell[(spec, win)]
            for name in norm.columns:
                det, _, asset = name.partition("_")
                asset = asset or INDEX
                ax.plot(dates, norm[name], color=cols[asset], lw=0.6,
                        ls=gg.LINETYPES["dashed" if det == "VaR" else "solid"])
            ax.axhline(1, color="black", lw=0.9)
            for e in EVENTS:
                if dates.min() <= e <= dates.max():
                    ax.axvline(e, color="darkgrey", ls=gg.LINETYPES["dashed"], lw=0.6)
            ax.set_ylim(-0.5, 2)
            ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
            ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    fig.supylabel("Normalized Detector Value", fontsize=9)
    sys_label = "CoVaR" if mode == "covar" else "RCoVaR"
    gg.bottom_legend(fig, [Line2D([], [], color="k", ls=gg.LINETYPES[l]) for l in ("dashed", "solid")],
                     ["VaR", sys_label], "Monitoring Sequence", y=0.07)
    assets = [INDEX] + BANKS if mode == "covar" else BANKS
    gg.bottom_legend(fig, [Line2D([], [], color=cols[a]) for a in assets], assets, "Asset", y=0.0)
    fig.savefig(out, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- (e) qualitative checks
def checks(fa, dates):
    """fa[(spec, win, mode)] = first-alarm index per detector; dates[(spec, win)] = window dates."""
    out = []
    crisis = ["FinCrisis", "COVID", "TrumpTariff"]

    def first_set(key):
        s = fa[key].dropna()
        return list(s[s == s.min()].index) if len(s) else []

    if all(("CCC_norm", w, "covar") in fa for w in crisis):
        firsts = {w: first_set(("CCC_norm", w, "covar")) for w in crisis}
        ok = all(f and all(d.startswith("CoVaR") for d in f) for f in firsts.values())
        out.append(("E1 CCC_norm crisis windows: first alarm is CoVaR", ok,
                    "; ".join(f"{w}: {','.join(f) or 'none'}" for w, f in firsts.items())))
    key = ("CCC_norm", "FinCrisis", "covar")
    if key in fa:
        f = first_set(key)
        wfc = fa[key].get("CoVaR_WFC", np.nan)
        d = dates[key[:2]][int(wfc)] if np.isfinite(wfc) else None
        out.append(("E2(i) CCC_norm FinCrisis: WFC among first CoVaR alarms", "CoVaR_WFC" in f,
                    f"first: {','.join(f)} on {dates[key[:2]][int(fa[key][f[0]])].date() if f else '-'}"))
        out.append(("E2(ii) CCC_norm FinCrisis: WFC alarm in 2008", d is not None and d.year == 2008,
                    f"WFC first alarm {d.date() if d is not None else 'none'}"))
    if all(("DCC_std", w, "covar") in fa for w in WINDOWS):
        alarms = {(w, det): int(i) for w in WINDOWS for det, i in fa[("DCC_std", w, "covar")].dropna().items()}
        ok = list(alarms) == [("COVID", "VaR")]
        detail = ", ".join(f"{w}/{d}:{dates[('DCC_std', w)][i].date()}" for (w, d), i in alarms.items()) or "none"
        if ok:
            dv = dates[("DCC_std", "COVID")]
            ev = int(np.searchsorted(dv.to_numpy(), EVENTS[1].to_datetime64()))
            gap = alarms[("COVID", "VaR")] - ev
            ok = abs(gap) <= 20
            detail += f" ({gap:+d} trading days from 2020-03-13)"
        out.append(("E3 DCC_std: only VaR_SPF alarms, in COVID, within 20 days of 2020-03-13", ok, detail))
    calm = [k for k in fa if k[1] == "CalmTime"]
    if calm:
        n_al = {k: int(fa[k].notna().sum()) for k in calm}
        out.append(("E4 CalmTime: no alarms (all specs, both modes)", all(v == 0 for v in n_al.values()),
                    ", ".join(f"{k[0]}/{k[2]}:{v}" for k, v in n_al.items())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute-specs", nargs="*", default=["CCC_norm"])
    args = ap.parse_args()
    paths.ensure_dirs()
    pd.set_option("display.width", 220, "display.max_rows", 400, "display.max_columns", 30)
    cv = data.load_crit_values()
    cv_row = data.crit_value_row(cv, "CoVaR", LEVEL, LEVEL, K)
    cv_rev_row = data.crit_value_row(cv, "CoVaR_Reverse", LEVEL, LEVEL, K)
    appl = their_results.load_appl_res()
    cells = [(s, w) for s in SPECS for w in WINDOWS if (APPL / f"{s}_{w}.csv").exists()]
    print(f"cells: {cells}\n")

    a_rows, b_rows, fa_rows, c_rates, c_days = [], [], [], [], []
    fa_ours, fa_theirs, win_dates, fig_paths = {}, {}, {}, {"covar": {}, "rev": {}}
    for spec, win in cells:
        df, nu, rmon = load_run(spec, win)
        res = our_monitoring(indicators(df), cv_row, cv_rev_row)
        dates = rmon["covar"][0]["date"]
        win_dates[(spec, win)] = dates
        for mode, r in res.items():
            ours = pd.DataFrame(r.normalised, columns=r.names)
            rpath, rl = rmon[mode]
            # (a) vs their unchanged SysRisk_Monitoring
            diff = np.abs(ours.to_numpy() - rpath[r.names].to_numpy()).max()
            rl_ok = np.array_equal(np.append(r.run_length, r.joint), rl.iloc[:, 2].to_numpy(float))
            a_rows.append(dict(spec=spec, window=win, mode=mode, max_abs_diff=diff, rl_equal=rl_ok,
                               check="PASS" if diff <= TOL and rl_ok else "FAIL"))
            # (b) vs their stored appl_res
            th = their_paths(appl, spec, win, mode)[r.names]
            db = np.abs(ours.to_numpy() - th.to_numpy())
            fo, ft = first_alarms(ours), first_alarms(th.reset_index(drop=True))
            b_rows.append(dict(spec=spec, window=win, mode=mode, max_abs_diff=db.max(),
                               n_days_diff_gt_1e8=int((db.max(1) > 1e-8).sum()),
                               first_idx_equal=fo.fillna(-1).equals(ft.fillna(-1)),
                               labels_shifted=int((rpath.date != rpath.date_theirs).sum())))
            for det in r.names:
                io, it = fo[det], ft[det]
                if np.isnan(io) and np.isnan(it):
                    continue
                fa_rows.append(dict(spec=spec, window=win, mode=mode, detector=det,
                                    ours_idx=io, ours_date=dates[int(io)].date() if io == io else None,
                                    theirs_idx=it, theirs_label=th.index[int(it)].date() if it == it else None,
                                    same=io == it))
            fa_ours[(spec, win, mode)], fa_theirs[(spec, win, mode)] = fo, ft
            fig_paths[mode][(spec, win)] = (dates, ours)
        # (c) recomputed indicators
        if spec in args.recompute_specs:
            t0 = time.time()
            rates, days = recompute(df, nu)
            c_rates.append(dict(spec=spec, window=win, nu=" ".join(f"{v:g}" for v in nu.nu),
                                min_agreement=min(rates.values()),
                                n_disagree=len(days), sec=round(time.time() - t0, 1)))
            c_days += [dict(spec=spec, window=win, **d) for d in days]

    fmt = lambda v: f"{v:.3g}"
    A = pd.DataFrame(a_rows)
    print("=== (a) our monitoring on R indicators vs their unchanged SysRisk_Monitoring (tol 1e-10) ===")
    print(A.to_string(index=False, float_format=fmt))
    B = pd.DataFrame(b_rows)
    print("\n=== (b) our normalised paths vs their stored appl_res_20251024.rds (alpha=beta=0.95) ===")
    print(B.to_string(index=False, float_format=fmt))
    FA = pd.DataFrame(fa_rows)
    print("\n=== (b) first alarms per detector (window index 0..750; ours = correct date, theirs = their label) ===")
    print(FA.to_string(index=False))
    if c_rates:
        C = pd.DataFrame(c_rates)
        print("\n=== (c) indicators recomputed with our c(rho, nu) vs R's (their nu indexing) ===")
        print(C.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        if c_days:
            print("disagreeing days:\n" + pd.DataFrame(c_days).to_string(index=False, float_format=lambda v: f"{v:.6g}"))
        C.to_csv(paths.DERIVED / "exp05_recompute.csv", index=False)
        pd.DataFrame(c_days).to_csv(paths.DERIVED / "exp05_recompute_days.csv", index=False)

    # (d)
    figure(fig_paths["covar"], "covar", paths.FIGURES / "fig5_covar.png")
    figure(fig_paths["rev"], "rev", paths.FIGURES / "fig6_revcovar.png")
    print("\nfigures: figures/fig5_covar.png, figures/fig6_revcovar.png")

    # (e)
    print("\n=== (e) qualitative checks (ours | theirs) ===")
    E = [dict(check=c, ours="PASS" if ok else "FAIL", ours_detail=d, theirs="PASS" if ok2 else "FAIL",
              theirs_detail=d2)
         for (c, ok, d), (_, ok2, d2) in zip(checks(fa_ours, win_dates), checks(fa_theirs, win_dates))]
    for e in E:
        print(f"{e['check']}\n    ours   {e['ours']}: {e['ours_detail']}\n    theirs {e['theirs']}: {e['theirs_detail']}")

    print(f"\n(a) {(A.check == 'PASS').sum()}/{len(A)} PASS; (b) first-alarm indices equal in "
          f"{B.first_idx_equal.sum()}/{len(B)} cells, max |diff| {B.max_abs_diff.max():.3g}")
    A.to_csv(paths.DERIVED / "exp05_vs_rmon.csv", index=False)
    B.to_csv(paths.DERIVED / "exp05_vs_appl_res.csv", index=False)
    FA.to_csv(paths.DERIVED / "exp05_first_alarms.csv", index=False)
    pd.DataFrame(E).to_csv(paths.DERIVED / "exp05_checks.csv", index=False)


if __name__ == "__main__":
    main()
