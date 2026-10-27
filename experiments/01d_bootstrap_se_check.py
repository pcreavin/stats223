"""Is the bootstrap SE of experiment 01c right? Across-seed sd of our B=2000 thresholds vs it.

For every (n, m, level) group with a check-2 FAIL in a CoVaR row of exp01c, plus three control
groups, re-run our whole selection procedure (moments + sups + thresholds) for 20 independent
seeds and compare the sd of the thresholds with 01c's bootstrap SE.

Usage: python experiments/01d_bootstrap_se_check.py [--n-seeds 20] [--workers 8]   (~35 min)
Needs data/derived/exp01c_reduced.csv (run experiment 01c first).
"""
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from src import critical_values as CV, paths
from src.detectors import null_moments

K_SET = [1, 2, 5, 10]
KEY = ["n", "m", "level", "K"]
CONTROLS = [(1000, 250, 0.9), (1500, 500, 0.75), (500, 150, 0.9)]


def job(a):
    n, m, al, s = a
    import numba
    numba.set_num_threads(1)
    rng = np.random.default_rng([555, s, n, m, int(al * 100)])
    ms = (null_moments(m, 1 - al, 10**5, rng), null_moments(m, (1 - al) ** 2, 10**5, rng))
    sv, sc = CV.simulate_sups_multi(n, m, al, al, 2000, [ms], rng=rng)[0]
    th = CV.select_thresholds_multi(sv, sc, K_SET, 0.1)
    return [dict(n=n, m=m, level=al, seed=s, K=K, c_VaR=th[K]["c_VaR"], c_SysRisk=th[K]["c_SysRisk"])
            for K in K_SET]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    r = pd.read_csv(paths.DERIVED / "exp01c_reduced.csv")
    fails = r[(r.check2 == "FAIL") & (r.type == "CoVaR")][["n", "m", "level"]].drop_duplicates()
    ctrl = pd.DataFrame(CONTROLS, columns=["n", "m", "level"])
    groups = pd.concat([fails, ctrl]).drop_duplicates()
    jobs = [(int(g.n), int(g.m), float(g.level), s) for g in groups.itertuples() for s in range(args.n_seeds)]
    with ProcessPoolExecutor(args.workers) as ex:
        d = pd.DataFrame([x for rows in ex.map(job, jobs) for x in rows])

    sd = d.groupby(KEY)[["c_VaR", "c_SysRisk"]].std().reset_index()
    b = r[r.type == "CoVaR"][KEY + ["c_VaR_se", "c_SysRisk_se", "check2"]]
    x = sd.merge(b, on=KEY)
    x["ratio_VaR"] = x.c_VaR / x.c_VaR_se
    x["ratio_SysRisk"] = x.c_SysRisk / x.c_SysRisk_se
    x["control"] = x.set_index(["n", "m", "level"]).index.isin(ctrl.set_index(["n", "m", "level"]).index)
    x.to_csv(paths.DERIVED / "exp01d_se_check.csv", index=False)
    pd.set_option("display.width", 200)
    print("c_VaR / c_SysRisk columns: across-seed sd of our thresholds; *_se: 01c bootstrap SE")
    print(x[KEY + ["control", "check2", "c_SysRisk", "c_SysRisk_se", "ratio_SysRisk", "ratio_VaR"]]
          .round(3).to_string(index=False))
    print("\nmedian sd/SE ratio (rows: control group, level == 0.95):")
    print(x.groupby(["control", x.level == 0.95])[["ratio_VaR", "ratio_SysRisk"]].median().round(2))

    # 01c cells re-judged with the across-seed sd where the bootstrap SE breaks down (ratio > 3)
    bad = x[x.ratio_SysRisk > 3][KEY]
    m = r.merge(bad, on=KEY).merge(sd[KEY + ["c_SysRisk"]].rename(columns={"c_SysRisk": "sd_seed"}), on=KEY)
    m["z_seed"] = m.c_SysRisk_diff / (np.sqrt(2) * m.sd_seed)
    print(f"\n{len(bad)} (n, m, level, K) cells with sd/SE > 3; their 01c rows re-judged with the across-seed sd:")
    print(m[KEY + ["type", "c_SysRisk_diff", "c_SysRisk_se", "sd_seed", "z_seed", "check2", "check3"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
