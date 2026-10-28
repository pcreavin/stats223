"""Date alignment of appl_res: are the stored date labels those of the days actually monitored?

apply_SysRisk_Monitoring.R labels the detector values with tail(data_Assets_wide, n) %>% pull(Date),
where data_Assets_wide is only date-filtered, but monitors the last n rows of the na.omit'd matrix.
Any missing value among the six assets in a window's last n rows therefore shifts the labels.
Quick check for every window; full comparison (shift, where it starts, affected alarms from exp 09)
for every window where it fails.

Usage: python experiments/10_data_alignment.py   (run exp 09 first)
"""
import numpy as np
import pandas as pd

from src import data, paths, their_results as T

N, M = 1000, 250


def verdict(ok):
    return "PASS" if ok else "FAIL"


def main():
    # pivot_wider orders rows by first appearance; with every asset sorted and the first asset
    # (BAC) covering all dates, that is date order, so the pandas sorted pivot is exact
    long = data.load_assets_long()
    long = long[long.Asset.isin(T.APPL_ASSETS)]
    first = long.Asset.iloc[0]
    order_ok = (long.groupby("Asset").Date.apply(lambda s: s.is_monotonic_increasing).all()
                and set(long.Date) == set(long[long.Asset == first].Date))
    print(f"pivot_wider row order == date order: {verdict(order_ok)}")

    win = T.appl_window_dates(N, M)
    appl = T.load_appl_res()
    print("\n=== Quick check per window ===")
    bad = []
    for name, w in win.items():
        tail = w["window"].tail(N)
        na_tail = tail.isna().sum()
        na_win = w["window"].isna().sum()
        stored = pd.DatetimeIndex(sorted(appl.loc[appl.Setting == name, "Date"].unique()))
        ok = na_tail.sum() == 0
        print(f"{name:12s} start {w['start'].date()} (floor binds: {w['start'] == pd.Timestamp('1999-05-05')}), "
              f"raw rows {len(w['window'])}, nE {w['nE']}, E {w['E']} [{verdict(w['E'] == 1500)}], "
              f"NA in window {na_win[na_win > 0].to_dict() or 0}, NA in last n {na_tail[na_tail > 0].to_dict() or 0}, "
              f"appl_res dates == tail(wide, n): {verdict(stored.equals(w['labelled']))}, "
              f"labels correct: {verdict(ok)}")
        if not ok:
            bad.append(name)

    tab = pd.read_csv(paths.TARGETS / "targets_application_alarms.csv",
                      parse_dates=["first_alarm_label", "first_alarm_date", "peak_label", "peak_date"])
    for name in bad:
        w = win[name]
        lab, used = w["labelled"], w["used"]
        tail = w["window"].tail(N)
        na_rows = tail.index[tail.isna().any(axis=1)]
        # shift in rows of the raw (date-filtered) frame between label and the day monitored
        pos = pd.Series(np.arange(len(w["window"])), index=w["window"].index)
        shift_rows = pos[lab].to_numpy() - pos[used].to_numpy()
        shift_cd = (lab - used).days
        wrong = np.flatnonzero(lab != used)
        print(f"\n=== {name}: full comparison ===")
        print(f"NA rows in the last n: {[d.date().isoformat() for d in na_rows]} "
              f"({tail.loc[na_rows].isna().sum()[lambda s: s > 0].to_dict()})")
        print(f"mislabelled windows: {len(wrong)}/{len(lab)}, positions 1..{wrong.max() + 1} "
              f"(labels {lab[0].date()}..{lab[wrong.max()].date()}); correct from label {lab[wrong.max() + 1].date()}")
        print(f"label - true date: {sorted(set(shift_rows))} rows, "
              f"{shift_cd.min()}..{shift_cd.max()} calendar days (label is later than the day monitored)")
        for lo, hi in [(0, 3), (wrong.max() - 3, wrong.max() + 3)]:
            for k in range(max(lo, 0), hi):
                print(f"  window {k + 1:3d}: label {lab[k].date()}  monitored {used[k].date()}  "
                      f"shift {shift_rows[k]} rows")
        t = tab[(tab.Setting == name) & (tab.alpha == 0.95)]
        aff = t[t.first_alarm_label.notna() & (t.first_alarm_label != t.first_alarm_date)]
        ok_al = t[t.first_alarm_label.notna() & (t.first_alarm_label == t.first_alarm_date)]
        print(f"\nfirst alarms (alpha=beta=0.95) with a wrong label: {len(aff)}, correct: {len(ok_al)}")
        print(aff[["GARCH_spec", "SysRiskMeasure", "Detector", "first_alarm_window", "first_alarm_label",
                   "first_alarm_date"]].to_string(index=False))
        print("correctly labelled first alarms:")
        print(ok_al[["GARCH_spec", "SysRiskMeasure", "Detector", "first_alarm_label"]].to_string(index=False))
        pk = t[t.peak_label.notna() & (t.peak_label != t.peak_date)]
        print(f"peaks with a wrong label: {len(pk)}: "
              f"{sorted(set(zip(pk.GARCH_spec, pk.SysRiskMeasure, pk.Detector)))}")


if __name__ == "__main__":
    main()
