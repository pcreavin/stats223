"""Their size results: the xtable .txt outputs and a recomputation from sim_BreakPoint_20251006.rds
(as in simulations/eval_sim_process_files.R + eval_sim_AnyDetection.R / eval_sim_FirstDetection.R)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyreadr

from . import paths
from .monitoring import first_detector_shares

OUT = paths.EXT / "simulations" / "output"


def parse_xtable(path) -> list[list[str]]:
    """Body rows (between \\midrule and \\bottomrule) of an xtable booktabs table, split on '&'."""
    lines = path.read_text().splitlines()
    i0 = next(i for i, l in enumerate(lines) if "\\midrule" in l)
    i1 = next(i for i, l in enumerate(lines) if "\\bottomrule" in l)
    return [[c.strip() for c in l.replace("\\\\", "").split("&")] for l in lines[i0 + 1:i1]]


def _num(s):
    return float(s) if s else np.nan


def joint_txt() -> pd.DataFrame:
    """size_accum_rate.txt: Joint rejection rate (%) for CoVaR and Reverse CoVaR."""
    cols = [(1, .9), (1, .95), None, (2, .9), (2, .95), None, (5, .9), (5, .95), None, (10, .9), (10, .95)]
    rows = []
    for r in parse_xtable(OUT / "size_accum_rate.txt"):
        if r[0] not in ("CoVaR", "Reverse CoVaR"):
            continue
        for c, v in zip(cols, r[2:]):
            if c is not None and v:
                rows.append(dict(measure=r[0], K=c[0], level=c[1], joint=float(v)))
    return pd.DataFrame(rows)


def first_covar_txt() -> pd.DataFrame:
    """size_first_individual_CoVaR.txt: first-detector shares (%), detectors VaR, CoVaR_1..K."""
    rows = []
    for r in parse_xtable(OUT / "size_first_individual_CoVaR.txt"):
        lv, K = float(r[0]), int(r[2])
        names = ["VaR"] + [f"CoVaR_{k}" for k in range(1, 11)]
        for nm, v in zip(names, r[6:]):
            if v:
                rows.append(dict(K=K, level=lv, detector=nm, share=float(v)))
    return pd.DataFrame(rows)


def first_rev_txt() -> pd.DataFrame:
    """size_first_individual_RevCoVaR.txt: first-detector shares (%), detectors VaR_k, CoVaR_k."""
    rows = []
    for r in parse_xtable(OUT / "size_first_individual_RevCoVaR.txt"):
        lv, K, typ = float(r[0]), int(r[2]), r[6]
        for k, v in enumerate(r[8:], start=1):
            if v:
                rows.append(dict(K=K, level=lv, detector=f"{typ}_{k}", share=float(v)))
    return pd.DataFrame(rows)


def load_breakpoint_null() -> pd.DataFrame:
    df = next(iter(pyreadr.read_r(str(paths.SIM_BREAKPOINT_RDS)).values()))
    return df[(df.Break_point == 1000) & (~df.break_in_X)].copy()


def recompute_from_rds(null: pd.DataFrame | None = None):
    """(joint, first_covar, first_rev) DataFrames in the same layout as the *_txt() functions."""
    null = load_breakpoint_null() if null is None else null
    joint, fc, fr = [], [], []
    for (K, lv), g in null.groupby(["K", "alpha"]):
        K = int(K)
        cov = g[g.CoVaR_RunLength.notna()].pivot(index="i_MC", columns="Asset", values="CoVaR_RunLength")
        joint.append(dict(measure="CoVaR", K=K, level=lv, joint=100 * np.mean(cov["Joint"] < np.inf)))
        dets = ["X"] + [str(k) for k in range(1, K + 1)]
        sh = first_detector_shares(cov[dets].to_numpy())
        fc += [dict(K=K, level=lv, detector=("VaR" if d == "X" else f"CoVaR_{d}"), share=100 * s)
               for d, s in zip(dets, sh)]
        rev = g[g.RevCoVaR_RunLength.notna()]
        if len(rev):
            rev = rev.pivot(index="i_MC", columns="Asset", values="RevCoVaR_RunLength")
            joint.append(dict(measure="Reverse CoVaR", K=K, level=lv, joint=100 * np.mean(rev["Joint"] < np.inf)))
            dets = [f"VaR_{k}" for k in range(1, K + 1)] + [f"CoVaR_{k}" for k in range(1, K + 1)]
            sh = first_detector_shares(rev[dets].to_numpy())
            fr += [dict(K=K, level=lv, detector=d, share=100 * s) for d, s in zip(dets, sh)]
    return pd.DataFrame(joint), pd.DataFrame(fc), pd.DataFrame(fr)


def check_rds_vs_txt() -> pd.DataFrame:
    """Exact comparison of the rds recomputation with their .txt tables (one row per table)."""
    j, fc, fr = recompute_from_rds()
    out = []
    for name, ours, theirs, keys, val in (
            ("joint", j, joint_txt(), ["measure", "K", "level"], "joint"),
            ("first CoVaR", fc, first_covar_txt(), ["K", "level", "detector"], "share"),
            ("first Reverse", fr, first_rev_txt(), ["K", "level", "detector"], "share")):
        mg = theirs.merge(ours, on=keys, how="outer", suffixes=("_txt", "_rds"), indicator=True)
        d = (mg[f"{val}_rds"].round(2) - mg[f"{val}_txt"]).abs()
        out.append(dict(table=name, cells=len(mg), unmatched=int((mg["_merge"] != "both").sum()),
                        max_abs_diff=float(d.max()), exact=bool((mg["_merge"] == "both").all() and d.max() < 1e-9)))
    return pd.DataFrame(out)
