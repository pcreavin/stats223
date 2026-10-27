"""Table 2: size of reverse CoVaR monitoring under the null, on the simulations persisted by experiment 02
(each path is reused for both levels and for both monitoring modes, as in their sim_monitoring_function.R).

Usage: python experiments/03_table2_size_reverse.py [--M 5000] [--seed 2026]
"""
import argparse

import numpy as np
import pandas as pd

from _size_report import BAND, first_table, joint_table, plot_joint
from src import data, null_sims, paths, their_size

K_SET, LEVELS = (2, 5, 10), null_sims.LEVELS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    paths.ensure_dirs()
    pd.set_option("display.width", 200, "display.max_rows", 300)
    sims = null_sims.get(args.M, args.seed, simulate=False)

    # recompute the reverse run lengths from the stored indicators and check they equal the stored ones
    cv = data.load_crit_values()
    recomputed = all(np.array_equal(null_sims.run_lengths(sims[(K, lv)][0], cv, K, lv)["rl_rev"],
                                    sims[(K, lv)][0]["rl_rev"]) for K in K_SET for lv in LEVELS)

    joint = joint_table(sims, "rl_rev", their_size.joint_txt(), "Reverse CoVaR", K_SET, LEVELS)
    first = first_table(sims, "rl_rev", their_size.first_rev_txt(),
                        lambda K: [f"VaR_{k}" for k in range(1, K + 1)] + [f"CoVaR_{k}" for k in range(1, K + 1)],
                        K_SET, LEVELS)
    print(f"=== Table 2: Joint rejection rate (%), reverse CoVaR monitoring, M={args.M} (theirs M=5000) ===")
    print(joint.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\n=== First-detector shares (%) vs theirs (ties count for every tied detector) ===")
    agg = first.groupby(["level", "K"]).agg(detectors=("inside", "size"), inside_3sd=("inside", "sum"),
                                            max_abs_diff=("diff", lambda d: d.abs().max()))
    print(agg.to_string(float_format=lambda v: f"{v:.2f}"))
    first["type"] = first.detector.str.split("_").str[0]
    by_type = first.groupby(["level", "K", "type"])[["ours", "theirs"]].sum()
    print("summed over banks:\n" + by_type.to_string(float_format=lambda v: f"{v:.2f}"))

    n_pass = (joint.check == "PASS").sum()
    print(f"\nTable 2 Joint: {n_pass}/{len(joint)} PASS (±{BAND}pp); {joint.within_1_1.sum()}/{len(joint)} "
          f"within ±1.1pp; stored reverse run lengths reproduced from indicators: "
          f"{'PASS' if recomputed else 'FAIL'}")

    tag = f"M{args.M}_s{args.seed}"
    joint.to_csv(paths.DERIVED / f"exp03_joint_{tag}.csv", index=False)
    first.to_csv(paths.DERIVED / f"exp03_first_{tag}.csv", index=False)
    plot_joint(joint, "Table 2: reverse CoVaR monitoring size under the null",
               paths.FIGURES / f"exp03_size_{tag}.png")


if __name__ == "__main__":
    main()
