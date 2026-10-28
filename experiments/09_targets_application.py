"""Application targets (Figures 5-6) from DH's stored detector paths (appl_res_20251024.rds).

Per GARCH spec x window x measure (CoVaR, reverse CoVaR) x detector at alpha = beta = 0.95: first
alarm (first normalised value > 1), peak, days above 1, and offsets to the event dates; plus the
Joint first alarm with their tie rule (every detector with RL == min RL counts). Checks the paper's
qualitative claims for CoVaR monitoring and re-plots Figures 5 (CoVaR) and 6 (reverse CoVaR).

Dates: appl_res labels windows with tail(data_Assets_wide, n), which is not na.omit'd. We keep
that label ('first_alarm_label') and add the date actually monitored ('first_alarm_date', see
exp 10); offsets use the latter.

Usage: python experiments/09_targets_application.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import ggstyle as gg, paths, their_results as T

LEVEL = 0.95
MEASURES = ["CoVaR", "RevCoVaR"]
SETTINGS = ["FinCrisis", "CalmTime", "COVID", "TrumpTariff"]
SPECS = ["DCC_norm", "DCC_std", "CCC_norm", "CCC_std"]
EVENTS = pd.to_datetime(["2008-09-15", "2020-03-13", "2023-03-10", "2025-04-02"])
PLOT_SPECS = ["CCC_norm", "DCC_std"]


def verdict(ok):
    return "PASS" if ok else "FAIL"


def detector_name(seq, asset, measure):
    if seq == "VaR":
        return f"VaR_{asset}"
    return f"{'RCoVaR' if measure == 'RevCoVaR' else 'CoVaR'}_{asset}"


# --------------------------------------------------------------------------- per-detector table
def alarm_table(appl, win):
    rows = []
    for (spec, setting, meas, seq, asset), g in appl.groupby(
            ["GARCH_spec", "Setting", "SysRiskMeasure", "Monitoring Sequence", "Asset"], observed=True):
        g = g.sort_values("Date")
        lab = pd.DatetimeIndex(g["Date"])
        if not lab.equals(win[setting]["labelled"]):
            raise AssertionError(f"{spec}/{setting}: appl_res dates differ from tail(wide, n)")
        used = win[setting]["used"]
        v = g["value"].to_numpy()
        above = np.flatnonzero(v > 1)
        rl = above[0] + 1 if len(above) else np.inf
        ip = int(np.argmax(v))
        r = dict(GARCH_spec=spec, Setting=setting, SysRiskMeasure=meas, alpha=LEVEL, beta=LEVEL,
                 Detector=detector_name(seq, asset, meas), Type="VaR" if seq == "VaR" else "SysRisk",
                 first_alarm_window=rl,
                 first_alarm_label=lab[above[0]] if len(above) else pd.NaT,
                 first_alarm_date=used[above[0]] if len(above) else pd.NaT,
                 peak=v[ip], peak_label=lab[ip], peak_date=used[ip], days_above_1=len(above))
        rows.append(r)
    det = pd.DataFrame(rows)
    # Joint: RL = min over detectors; all detectors attaining it count as first
    joint = []
    for keys, g in det.groupby(["GARCH_spec", "Setting", "SysRiskMeasure"]):
        rl = g.first_alarm_window.min()
        first = g[g.first_alarm_window == rl] if np.isfinite(rl) else g.iloc[:0]
        j = dict(zip(["GARCH_spec", "Setting", "SysRiskMeasure"], keys), alpha=LEVEL, beta=LEVEL,
                 Detector="Joint", Type="Joint", first_alarm_window=rl,
                 first_alarm_label=first.first_alarm_label.min() if len(first) else pd.NaT,
                 first_alarm_date=first.first_alarm_date.min() if len(first) else pd.NaT,
                 joint_first_by=";".join(sorted(first.Detector)),
                 n_alarming=int((g.days_above_1 > 0).sum()))
        joint.append(j)
    out = pd.concat([det, pd.DataFrame(joint)], ignore_index=True)
    out["joint_first_by"] = out["joint_first_by"].fillna("")
    return add_offsets(out, win)


def add_offsets(df, win):
    """Offsets of the (monitored) first-alarm date to every event date inside the window's
    monitored date range: trading days (positions in the 751 monitored dates) and calendar days."""
    for ev in EVENTS:
        tag = ev.strftime("%Y%m%d")
        td, cd = [], []
        for s, d in zip(df.Setting, df.first_alarm_date):
            used = win[s]["used"]
            if not (used[0] <= ev <= used[-1]) or pd.isna(d):
                td.append(np.nan), cd.append(np.nan)
                continue
            td.append(used.get_loc(d) - used.searchsorted(ev))
            cd.append((d - ev).days)
        df[f"off_td_{tag}"], df[f"off_cd_{tag}"] = td, cd
    return df


# --------------------------------------------------------------------------- claims
def claims(tab):
    c = tab[tab.SysRiskMeasure == "CoVaR"]
    det = c[c.Type != "Joint"]
    joint = c[c.Type == "Joint"].set_index(["GARCH_spec", "Setting"])
    out = []

    # (a) CCC-Gaussian, crisis windows: Joint alarms, first alarm caused by a CoVaR detector,
    #     and more CoVaR than VaR detectors alarm
    for s in ["FinCrisis", "COVID"]:
        j = joint.loc[("CCC_norm", s)]
        d = det[(det.GARCH_spec == "CCC_norm") & (det.Setting == s) & (det.days_above_1 > 0)]
        n_cov, n_var = (d.Type == "SysRisk").sum(), (d.Type == "VaR").sum()
        ok = (np.isfinite(j.first_alarm_window) and any(x.startswith("CoVaR") for x in j.joint_first_by.split(";"))
              and n_cov > n_var)
        out.append(("a", f"CCC_norm {s}: alarm mainly via CoVaR", ok,
                    f"first by {j.joint_first_by or '-'} on {_d(j.first_alarm_date)}; alarming CoVaR {n_cov}/5, VaR {n_var}/1"))

    # (b) CCC-Gaussian, FinCrisis: WFC first
    j = joint.loc[("CCC_norm", "FinCrisis")]
    first = j.joint_first_by.split(";")
    out.append(("b", "CCC_norm FinCrisis: CoVaR_WFC first", "CoVaR_WFC" in first,
                f"first set {first} ({'alone' if first == ['CoVaR_WFC'] else 'tied/other'})"))

    # (c) DCC-Student-t: only the sector VaR detector alarms, around COVID
    d = det[(det.GARCH_spec == "DCC_std") & (det.days_above_1 > 0)]
    var_cov = d[(d.Detector == "VaR_SPF") & (d.Setting == "COVID")]
    ok = set(d.Detector) == {"VaR_SPF"} and len(var_cov) == 1
    ev = var_cov.iloc[0] if len(var_cov) else None
    out.append(("c", "DCC_std: alarms only via VaR_SPF, around COVID", ok,
                f"alarming: {sorted(zip(d.Setting, d.Detector))}"
                + (f"; COVID first alarm {_d(ev.first_alarm_date)} ({ev.off_td_20200313:+.0f} td, "
                   f"{ev.off_cd_20200313:+.0f} cd vs 2020-03-13)" if ev is not None else "")))

    # (d) calm window: no alarms
    for spec in PLOT_SPECS:
        n_al = (det[(det.GARCH_spec == spec) & (det.Setting == "CalmTime")].days_above_1 > 0).sum()
        out.append(("d", f"{spec} CalmTime: no alarms", n_al == 0, f"{n_al} alarming detectors"))
    all_specs = det[(det.Setting == "CalmTime") & (det.days_above_1 > 0)]
    print("  (info) CalmTime alarms over all 4 specs, CoVaR:",
          sorted(zip(all_specs.GARCH_spec, all_specs.Detector)) or "none")
    return pd.DataFrame(out, columns=["claim", "statement", "ok", "evidence"])


def _d(x):
    return "-" if pd.isna(x) else pd.Timestamp(x).date().isoformat()


def print_table(tab, meas):
    t = tab[(tab.SysRiskMeasure == meas)].copy()
    t = t[(t.days_above_1 > 0) | (t.Type == "Joint")]
    cols = ["GARCH_spec", "Setting", "Detector", "first_alarm_window", "first_alarm_date",
            "first_alarm_label", "peak", "peak_date", "days_above_1", "n_alarming", "joint_first_by",
            "off_td_20080915", "off_td_20200313", "off_td_20230310", "off_td_20250402"]
    t["first_alarm_date"] = t.first_alarm_date.map(_d)
    t["first_alarm_label"] = t.first_alarm_label.map(_d)
    t["peak_date"] = t.peak_date.map(_d)
    with pd.option_context("display.width", 250, "display.max_rows", 500, "display.max_columns", 30):
        print(t.sort_values(["GARCH_spec", "Setting", "Type", "Detector"])[cols]
              .to_string(index=False, float_format=lambda x: f"{x:.3g}"))


# --------------------------------------------------------------------------- figures
def plot(appl, meas, fname):
    d = appl[(appl.SysRiskMeasure == meas) & appl.GARCH_spec.isin(PLOT_SPECS)]
    assets = ["SPF", *sorted(set(d.Asset) - {"SPF"})]
    colours = dict(zip(T.APPL_ASSETS[:1] + sorted(T.APPL_ASSETS[1:]), gg.hue_pal(6)))
    fig = plt.figure(figsize=(11, 5.5))
    gs = fig.add_gridspec(1, 1, left=0.06, right=0.95, top=0.92, bottom=0.16)
    axes = gg.facet_grid(fig, gs[0], PLOT_SPECS, SETTINGS, PLOT_SPECS, SETTINGS)
    for i, spec in enumerate(PLOT_SPECS):
        for j, s in enumerate(SETTINGS):
            ax = axes[i][j]
            g = d[(d.GARCH_spec == spec) & (d.Setting == s)]
            for (a, seq), h in g.groupby(["Asset", "Monitoring Sequence"]):
                ax.plot(h.Date, h.value, color=colours[a], lw=0.8,
                        ls=gg.LINETYPES["dashed"] if seq == "VaR" else "-")
            ax.axhline(1, color="black", lw=1.2)
            lo, hi = g.Date.min(), g.Date.max()
            for ev in EVENTS:
                if lo <= ev <= hi:
                    ax.axvline(ev, color="darkgrey", ls=gg.LINETYPES["dashed"], lw=0.8)
            ax.set_ylim(-0.5, 2)
            ax.tick_params(axis="x", labelsize=6, rotation=30)
    axes[0][0].set_ylabel("Normalized Detector Value", y=0)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color="k", ls=gg.LINETYPES["dashed"]), Line2D([], [], color="k")]
    h += [Line2D([], [], color=colours[a]) for a in assets]
    sys_lab = "RCoVaR" if meas == "RevCoVaR" else "CoVaR"
    gg.bottom_legend(fig, h, ["VaR", sys_lab, *assets], "")
    fig.savefig(paths.FIGURES / fname, dpi=200)
    plt.close(fig)


def main():
    paths.ensure_dirs()
    appl = T.load_appl_res()
    appl = appl[(appl.alpha == LEVEL) & (appl.beta == LEVEL) & appl.SysRiskMeasure.isin(MEASURES)]
    win = T.appl_window_dates()
    tab = alarm_table(appl, win)

    # consistency: Joint RL = min of detector RLs; 751 dates per window
    det = tab[tab.Type != "Joint"].groupby(["GARCH_spec", "Setting", "SysRiskMeasure"]).first_alarm_window.min()
    jt = tab[tab.Type == "Joint"].set_index(["GARCH_spec", "Setting", "SysRiskMeasure"]).first_alarm_window
    print(f"Joint RL == min detector RL: {verdict(det.equals(jt.loc[det.index]))}")
    print(f"detectors per (spec, window): CoVaR {6}, RevCoVaR {10}; rows {len(tab)} "
          f"(expected {4 * 4 * (6 + 10 + 2)}): {verdict(len(tab) == 4 * 4 * 18)}")

    print("\n=== Claims, CoVaR monitoring, alpha = beta = 0.95 ===")
    cl = claims(tab)
    for r in cl.itertuples():
        print(f"  ({r.claim}) {verdict(r.ok)}  {r.statement}: {r.evidence}")

    for meas in MEASURES:
        print(f"\n=== {meas}: Joint rows and alarming detectors (td = trading days, monitored dates) ===")
        print_table(tab, meas)

    out = tab.copy()
    for c in ["first_alarm_label", "first_alarm_date", "peak_label", "peak_date"]:
        out[c] = out[c].map(lambda x: "" if pd.isna(x) else pd.Timestamp(x).date().isoformat())
    out.to_csv(paths.TARGETS / "targets_application_alarms.csv", index=False)
    print(f"\nwrote {paths.TARGETS / 'targets_application_alarms.csv'} ({len(out)} rows)")

    plot(appl, "CoVaR", "targets_fig5.png")
    plot(appl, "RevCoVaR", "targets_fig6.png")
    print(f"wrote {paths.FIGURES}/targets_fig5.png, targets_fig6.png")


if __name__ == "__main__":
    main()
