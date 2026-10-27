"""Table 1: size of CoVaR monitoring under the null DCC-GARCH-t DGP (t* = n, break_in_X = FALSE), with their
stored thresholds. Simulations are persisted to data/derived/sims/ and reused by 03 and later detectors.

Usage: python experiments/02_table1_size.py [--M 5000] [--seed 2026] [--force]
"""
import argparse
import time

import numpy as np
import pandas as pd

from _size_report import BAND, first_table, joint_table, plot_joint
from src import data, null_sims, paths, their_size
from src.monitoring import covar_paths

K_SET, LEVELS = null_sims.K_SET, null_sims.LEVELS


def iid_sanity(cv, M, seed):
    """K = 1 with iid indicators from shared uniforms (Algorithm 1): Joint rate should be ~ iota."""
    rng = np.random.default_rng(seed)
    rows = []
    for lv in LEVELS:
        U1, U2 = rng.random((M, null_sims.N)), rng.random((M, null_sims.N))
        I = (U1 > lv).astype(np.int8)
        ind = dict(I_VaR=I, I_CoVaR=((U1 > lv) & (U2 > lv)).astype(np.int8)[:, None, :])
        rl = null_sims.run_lengths(ind, cv, 1, lv)["rl_covar"]
        r = 100 * np.mean(np.isfinite(rl[:, -1]))
        rows.append(dict(level=lv, K=1, joint=r, target=100 * null_sims.IOTA, diff=r - 100 * null_sims.IOTA,
                         check="PASS" if abs(r - 100 * null_sims.IOTA) <= BAND else "FAIL"))
    return pd.DataFrame(rows)


def diagnostics(sims, cv, n_paths=1000):
    """Miss rates vs nominal, the CoVaR-implies-VaR property and cross-bank dependence (K = 10)."""
    rows = []
    for lv in LEVELS:
        a, _ = sims[(10, lv)]
        row, _ = null_sims.cv_rows(cv, 10, lv)
        pth, _ = covar_paths(a["I_VaR"][:n_paths], a["I_CoVaR"][:n_paths], row, null_sims.M_WIN)
        sup = pth.max(axis=-2)[:, 1:]                                   # [paths, K] bank CoVaR sups
        c = np.corrcoef(sup.T)
        ic = a["I_CoVaR"][:n_paths].astype(float)                        # [paths, K, n]
        x = ic.transpose(1, 0, 2).reshape(10, -1)
        ci = np.corrcoef(x)
        off = ~np.eye(10, dtype=bool)
        rows.append(dict(level=lv,
                         I_VaR=a["I_VaR"].mean(), nominal_VaR=1 - lv,
                         I_CoVaR=a["I_CoVaR"].mean(), nominal_CoVaR=(1 - lv) ** 2,
                         I_VaR_Rev=a["I_VaR_Rev"].mean(), I_CoVaR_Rev=a["I_CoVaR_Rev"].mean(),
                         CoVaR_without_VaR=int((a["I_CoVaR"] > a["I_VaR"][:, None, :]).sum()),
                         corr_bank_CoVaR_sups=c[off].mean(), corr_bank_CoVaR_indicators=ci[off].mean()))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    paths.ensure_dirs()
    pd.set_option("display.width", 200, "display.max_rows", 300)
    t0 = time.time()
    sims = null_sims.get(args.M, args.seed, force=args.force)
    cv = data.load_crit_values()

    joint = joint_table(sims, "rl_covar", their_size.joint_txt(), "CoVaR", K_SET, LEVELS)
    first = first_table(sims, "rl_covar", their_size.first_covar_txt(),
                        lambda K: ["VaR"] + [f"CoVaR_{k}" for k in range(1, K + 1)], K_SET, LEVELS)
    print(f"\n=== Table 1: Joint rejection rate (%), CoVaR monitoring, M={args.M} (theirs M=5000) ===")
    print(joint.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\n=== First-detector shares (%) vs theirs (ties count for every tied detector) ===")
    agg = first.groupby(["level", "K"]).agg(detectors=("inside", "size"), inside_3sd=("inside", "sum"),
                                            max_abs_diff=("diff", lambda d: d.abs().max()))
    print(agg.to_string(float_format=lambda v: f"{v:.2f}"))
    wide = first.pivot_table(index=["level", "K"], columns="detector", values="ours", sort=False)
    print("ours:\n" + wide.to_string(float_format=lambda v: f"{v:.2f}"))

    print("\n=== (a) iid-indicator sanity run, K = 1 (their thresholds; target = iota) ===")
    iid = iid_sanity(cv, args.M, args.seed + 1)
    print(iid.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    print("\n=== (b) Their tables recomputed from sim_BreakPoint_20251006.rds vs their .txt ===")
    rds = their_size.check_rds_vs_txt()
    print(rds.to_string(index=False))

    print("\n=== Diagnostics (K = 10) ===")
    diag = diagnostics(sims, cv)
    print(diag.T.to_string(header=False, float_format=lambda v: f"{v:.5f}"))

    n_pass = (joint.check == "PASS").sum()
    print(f"\nTable 1 Joint: {n_pass}/{len(joint)} PASS (±{BAND}pp); {joint.within_1_1.sum()}/{len(joint)} "
          f"within ±1.1pp; iid sanity {(iid.check == 'PASS').sum()}/{len(iid)} PASS; rds == txt: "
          f"{'PASS' if rds.exact.all() else 'FAIL'}; total {time.time() - t0:.0f}s")

    tag = f"M{args.M}_s{args.seed}"
    joint.to_csv(paths.DERIVED / f"exp02_joint_{tag}.csv", index=False)
    first.to_csv(paths.DERIVED / f"exp02_first_{tag}.csv", index=False)
    iid.to_csv(paths.DERIVED / f"exp02_iid_{tag}.csv", index=False)
    diag.to_csv(paths.DERIVED / f"exp02_diagnostics_{tag}.csv", index=False)
    plot_joint(joint, "Table 1: CoVaR monitoring size under the null", paths.FIGURES / f"exp02_size_{tag}.png")


if __name__ == "__main__":
    main()
