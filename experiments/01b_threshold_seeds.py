"""Seed study of the c_VaR tilt in experiment 01 (main table, n=1000, m=250, B=5000).

Part 1 re-runs our full procedure for several seeds (seeding as in 01: seed + level index,
so seed 2026 reproduces exp01). Part 2 standardises our sups with THEIR stored per-K moments
(src/their_moments.py) to see how much of the tilt is moment Monte Carlo noise.

Usage: python experiments/01b_threshold_seeds.py [--seeds 2026 101 ...] [--their-seeds 101 ...]
Needs data/derived/exp01_thresholds_B5000.csv (run experiment 01 first) for bootstrap SEs.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from src import critical_values as CV, data, paths
from src.their_moments import stored_moment_sets, thresholds_with_stored_moments

ALPHAS = [0.75, 0.9, 0.95]
K_SET = [1, 2, 5, 10]
N, M, IOTA, B = 1000, 250, 0.1, 5000
KEY = ["level", "K", "type", "threshold"]


def run_ours(seed):
    theirs = data.load_crit_values()
    rows = []
    for ia, al in enumerate(ALPHAS):
        ours = CV.critical_values(M, N, K_SET, al, al, B, IOTA, seed=seed + ia)
        for K in K_SET:
            for typ in ("CoVaR", "CoVaR_Reverse"):
                o = ours[(ours.SysRisk == typ) & (ours.K == K)].iloc[0]
                t = data.crit_value_row(theirs, typ, al, al, K)
                for col in ("c_VaR", "c_SysRisk"):
                    rows.append(dict(seed=seed, level=al, K=K, type=typ, threshold=col,
                                     ours=o[col], theirs=t[col], diff=o[col] - t[col]))
    return pd.DataFrame(rows)


def run_their_moments(seed):
    theirs = data.load_crit_values()
    rows = []
    for ia, al in enumerate(ALPHAS):
        ms = stored_moment_sets(theirs, al, al, K_SET)
        sups = CV.simulate_sups_multi(N, M, al, al, B, [ms[K] for K in K_SET],
                                      rng=np.random.default_rng([seed, ia]))
        th = thresholds_with_stored_moments(dict(zip(K_SET, sups)), K_SET, IOTA)
        for K in K_SET:
            t = data.crit_value_row(theirs, "CoVaR", al, al, K)
            for col in ("c_VaR", "c_SysRisk"):
                rows.append(dict(seed=seed, level=al, K=K, type="CoVaR", threshold=col,
                                 ours=th[K][col], theirs=t[col], diff=th[K][col] - t[col]))
    return pd.DataFrame(rows)


def pooled(g, z):
    """Pooled z over cells; reverse K=1 duplicates CoVaR K=1 and is dropped."""
    g = g[~((g.type == "CoVaR_Reverse") & (g.K == 1))]
    return g.groupby(["type", "threshold"]).apply(
        lambda s: pd.Series(dict(cells=len(s), mean_pos=int((s["mean"] > 0).sum()),
                                 avg_diff=s["mean"].mean(),
                                 pooled_z=s[z].sum() / np.sqrt(len(s)),
                                 max_abs_z=s[z].abs().max())), include_groups=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 101, 202, 303, 404, 505, 606])
    ap.add_argument("--their-seeds", type=int, nargs="+", default=[101, 202, 303, 404, 505, 606, 707, 808])
    args = ap.parse_args()
    paths.ensure_dirs()
    se_file = paths.DERIVED / "exp01_thresholds_B5000.csv"
    if not se_file.exists():
        raise SystemExit(f"{se_file} missing: run experiments/01_moments_and_thresholds.py --B 5000 first")
    se = pd.read_csv(se_file)[KEY + ["se_ours"]]
    pd.set_option("display.width", 200)

    with ProcessPoolExecutor(min(8, len(args.seeds))) as ex:
        ours = pd.concat(ex.map(run_ours, args.seeds))
    ours.to_csv(paths.DERIVED / "exp01b_seeds_ours.csv", index=False)
    print("=== Part 1: our moments. Per seed: cells with ours > theirs, and mean diff ===")
    for (typ, col), g in ours.groupby(["type", "threshold"]):
        s = g.groupby("seed", sort=False)["diff"].agg(n_pos=lambda x: int((x > 0).sum()), mean="mean")
        print(f"\n{typ} {col}\n{s.T.round(4).to_string()}")
    g = ours.groupby(KEY)["diff"].agg(["mean", "std", "count"]).reset_index().merge(se, on=KEY)
    g["z_boot"] = g["mean"] / np.sqrt(g.se_ours**2 + g["std"]**2 / g["count"])   # SE_theirs = bootstrap SE
    g["z_sd"] = g["mean"] / (g["std"] * np.sqrt(1 + 1 / g["count"]))            # SE_theirs = across-seed sd
    print("\nPooled over cells, SE_theirs = bootstrap SE:\n" + pooled(g, "z_boot").round(3).to_string())
    print("\nPooled over cells, SE_theirs = across-seed sd:\n" + pooled(g, "z_sd").round(3).to_string())

    with ProcessPoolExecutor(min(8, len(args.their_seeds))) as ex:
        tm = pd.concat(ex.map(run_their_moments, args.their_seeds))
    tm.to_csv(paths.DERIVED / "exp01b_seeds_theirmoments.csv", index=False)
    h = tm.groupby(KEY)["diff"].agg(["mean", "std", "count"]).reset_index()
    h["z_sd"] = h["mean"] / (h["std"] * np.sqrt(1 + 1 / h["count"]))
    print("\n=== Part 2: their stored moments (CoVaR rows) ===")
    print(pooled(h, "z_sd").round(4).to_string())
    print(h.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
