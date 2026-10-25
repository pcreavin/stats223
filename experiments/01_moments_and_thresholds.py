"""Step 2: null moments and critical values vs their stored table (df_CritVal_full_20250805.rds).

Usage: python experiments/01_moments_and_thresholds.py [--B 5000] [--n-boot 30] [--seed 2026]
"""
import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import critical_values as CV, data, paths

ALPHAS = [0.75, 0.9, 0.95]
K_SET = [1, 2, 5, 10]
N, M, IOTA, REPS = 1000, 250, 0.1, 10**5


def verdict(diff, tol):
    return "PASS" if abs(diff) <= tol else "FAIL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--n-boot", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    paths.ensure_dirs()
    theirs = data.load_crit_values()
    mom_rows, thr_rows = [], []

    for ia, al in enumerate(ALPHAS):
        t0 = time.time()
        ours, sv, sc = CV.critical_values(M, N, K_SET, al, al, args.B, IOTA,
                                          seed=args.seed + ia, return_sups=True)
        print(f"alpha=beta={al}: simulated B={args.B} in {time.time() - t0:.1f}s")

        # ---- moments (ours computed once per alpha; theirs once per K row)
        th_rows = theirs[(theirs.SysRisk == "CoVaR") & (theirs.alpha == al) & (theirs.beta == al)
                         & (theirs.n == N) & (theirs.m == M)]
        o = ours.iloc[0]
        for which in ("VaR", "SysRisk"):
            for stat in ("UC", "IID"):
                mu_c, sd_c = f"mu_{which}_{stat}", f"sd_{which}_{stat}"
                th_mu, th_sd = th_rows[mu_c].iloc[0], th_rows[sd_c].iloc[0]
                se_mu = np.sqrt(2) * o[sd_c] / np.sqrt(REPS)
                tol_mu = max(0.02 * abs(th_mu), 4 * se_mu)
                tol_sd = 0.02 * th_sd
                lbl = "VaR" if which == "VaR" else "CoVaR"
                mom_rows.append(dict(level=al, detector=lbl, stat=stat, quantity="mu",
                                     ours=o[mu_c], theirs=th_mu,
                                     their_spread_K=np.ptp(th_rows[mu_c]),
                                     rel_diff=o[mu_c] / th_mu - 1, tol_rel=tol_mu / abs(th_mu),
                                     check=verdict(o[mu_c] - th_mu, tol_mu)))
                mom_rows.append(dict(level=al, detector=lbl, stat=stat, quantity="sd",
                                     ours=o[sd_c], theirs=th_sd,
                                     their_spread_K=np.ptp(th_rows[sd_c]),
                                     rel_diff=o[sd_c] / th_sd - 1, tol_rel=0.02,
                                     check=verdict(o[sd_c] - th_sd, tol_sd)))

        # ---- thresholds
        for K in K_SET:
            se = CV.bootstrap_se(sv, sc, K, IOTA, n_boot=args.n_boot, seed=K)
            for sysrisk, keys in (("CoVaR", ("c_VaR", "c_SysRisk")),
                                  ("CoVaR_Reverse", ("c_VaR_rev", "c_SysRisk_rev"))):
                o_row = ours[(ours.SysRisk == sysrisk) & (ours.K == K)].iloc[0]
                t_row = data.crit_value_row(theirs, sysrisk, al, al, K)
                for col, sek in zip(("c_VaR", "c_SysRisk"), keys):
                    se_o = se[sek]
                    se_t = se_o * np.sqrt(args.B / 5000)      # their B = 5000
                    tol = 3 * np.hypot(se_o, se_t)
                    d = o_row[col] - t_row[col]
                    thr_rows.append(dict(level=al, K=K, type=sysrisk, threshold=col,
                                         ours=o_row[col], theirs=t_row[col], diff=d,
                                         se_ours=se_o, tol=tol, z=d / np.hypot(se_o, se_t),
                                         check=verdict(d, tol)))

    pd.set_option("display.width", 200, "display.max_rows", 200)
    mom = pd.DataFrame(mom_rows)
    thr = pd.DataFrame(thr_rows)
    print("\n=== Null moments (10^5 windows, m=250): ours vs theirs ===")
    print(mom.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print("\n=== Critical values (n=1000, m=250, iota=0.1): ours (B=%d) vs theirs (B=5000) ===" % args.B)
    print(thr.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nMoments: {(mom.check == 'PASS').sum()}/{len(mom)} PASS;"
          f" thresholds: {(thr.check == 'PASS').sum()}/{len(thr)} PASS")
    mom.to_csv(paths.DERIVED / f"exp01_moments_B{args.B}.csv", index=False)
    thr.to_csv(paths.DERIVED / f"exp01_thresholds_B{args.B}.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    for (typ, col), mk in zip(thr.groupby(["type", "threshold"]).groups, "osD^"):
        sub = thr[(thr.type == typ) & (thr.threshold == col)]
        ax[0].errorbar(sub.theirs, sub.ours, yerr=sub.tol, fmt=mk, ms=4, label=f"{typ} {col}", alpha=.8)
    lim = [thr[["ours", "theirs"]].min().min() - .2, thr[["ours", "theirs"]].max().max() + .2]
    ax[0].plot(lim, lim, "k-", lw=.8)
    ax[0].set(xlabel="theirs", ylabel="ours (± tol)", title="Critical values")
    ax[0].legend(fontsize=7)
    ax[1].bar(range(len(mom)), mom.rel_diff * 100)
    ax[1].axhline(2, color="grey", ls="--"); ax[1].axhline(-2, color="grey", ls="--")
    ax[1].set(ylabel="relative diff (%)", title="Null moments", xticks=[])
    fig.tight_layout()
    fig.savefig(paths.FIGURES / f"exp01_moments_thresholds_B{args.B}.png", dpi=150)


if __name__ == "__main__":
    main()
