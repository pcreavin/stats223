"""Loaders and aggregation for DH's stored simulation and application results.

The aggregation copies simulations/eval_sim_process_files.R:
- a run length is the first window (1..n-m+1) whose normalised detector exceeds 1, and Inf if none;
- rows where a measure does not apply are NA (CoVaR columns on the reverse 'VaR_k'/'CoVaR_k'
  rows, RevCoVaR columns on the 'X'/'1..K' rows), and every mean uses na.rm=TRUE;
- 'Joint' is a row of its own whose run length is the minimum over the detectors;
- first detection: within (i_MC, design), keep RL == min(RL, na.rm) and set the rest to Inf,
  so all tied detectors (and always Joint) count as first.
"""
import numpy as np
import pandas as pd

from . import data, paths

SIM_DIR = paths.EXT / "simulations" / "data"
ESTWINDOW_RDS = [SIM_DIR / f"sim_EstWindow_20251107_{i}.rds" for i in range(1, 7)]
# i_MC offsets used in eval_sim_process_files.R when binding the six EstWindow files
ESTWINDOW_OFFSETS = [0, 1000, 3000, 4000, 5000, 6000]
PROCESSED_RDS = SIM_DIR / "sim_eval_processed_20251117.rds"
PROCESSED_FIRST_RDS = SIM_DIR / "sim_eval_firstdetection_processed_20251117.rds"
SIM_OUTPUT = paths.EXT / "simulations" / "output"
CACHE = paths.DERIVED / "their_results"

DESIGN = ["Break_point", "n", "m", "E", "K", "alpha", "beta", "DCC_coef_post", "break_in_X"]
RL_COLS = {"CoVaR": "CoVaR_RunLength", "RevCoVaR": "RevCoVaR_RunLength"}
KEEP = ["Asset", "RiskMeasure", *RL_COLS.values(), "i_MC", *DESIGN]


def describe(df: pd.DataFrame, name: str, max_unique: int = 40) -> None:
    print(f"===== {name}: {len(df):,} rows")
    for c in df.columns:
        u = pd.unique(df[c])
        if len(u) <= max_unique:
            print(f"  {c:<20} {sorted(map(str, u))}")
        else:
            print(f"  {c:<20} {len(u)} unique, min {df[c].min()}, max {df[c].max()}")
    for col in RL_COLS.values():
        if col in df:
            x = df[col]
            print(f"  {col}: NA {x.isna().mean():.4f}, Inf {np.isinf(x).mean():.4f}")


def _tidy_sim(df: pd.DataFrame) -> pd.DataFrame:
    df = df[KEEP].copy()
    df["Asset"] = df["Asset"].astype("category")
    df["RiskMeasure"] = df["RiskMeasure"].astype("category")
    for c in ("Break_point", "n", "m", "E", "K"):
        df[c] = df[c].astype(np.int32)
    return df


def _cached(name: str, build) -> pd.DataFrame:
    f = CACHE / f"{name}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    CACHE.mkdir(parents=True, exist_ok=True)
    df = build()
    df.to_parquet(f, index=False)
    return df


def load_breakpoint() -> pd.DataFrame:
    return _cached("breakpoint", lambda: _tidy_sim(data._read_rds(paths.SIM_BREAKPOINT_RDS)))


def load_parammisspec() -> pd.DataFrame:
    return _cached("parammisspec", lambda: _tidy_sim(data._read_rds(paths.SIM_PARAMMISSPEC_RDS)))


def load_estwindow() -> pd.DataFrame:
    def build():
        parts = []
        for f, off in zip(ESTWINDOW_RDS, ESTWINDOW_OFFSETS):
            d = _tidy_sim(data._read_rds(f))
            d["i_MC"] = d["i_MC"] + off
            parts.append(d)
        out = pd.concat(parts, ignore_index=True)
        out["Asset"] = out["Asset"].astype("category")
        out["RiskMeasure"] = out["RiskMeasure"].astype("category")
        return out
    return _cached("estwindow", build)


SIM_LOADERS = {"BreakPoint": load_breakpoint, "ParamMisspec": load_parammisspec,
               "EstWindow": load_estwindow}


def load_processed(first: bool = False) -> pd.DataFrame:
    df = data._read_rds(PROCESSED_FIRST_RDS if first else PROCESSED_RDS)
    for c in ("Break_point", "n", "m", "E", "K"):
        df[c] = df[c].astype(np.int32)
    return df


def load_appl_res() -> pd.DataFrame:
    """appl_res_20251024.rds in long format. 'Detector' is VaR or SysRisk; for reverse CoVaR the
    systemic sequence is labelled 'CoVaR' in the file and 'RCoVaR' in the figure."""
    df = data.load_appl_res()
    df["Detector"] = np.where(df["Monitoring Sequence"] == "VaR", "VaR", "SysRisk")
    return df


def rates(df: pd.DataFrame, measure: str) -> pd.DataFrame:
    """Per-detector rejection rate and ARL (mean finite run length), as eval_sim_process_files.R."""
    rl = df[RL_COLS[measure]]
    tmp = df[["Asset", *DESIGN]].copy()
    tmp["valid"] = rl.notna()
    tmp["rej"] = np.isfinite(rl)
    tmp["rl_fin"] = rl.where(np.isfinite(rl))
    g = tmp.groupby(["Asset", *DESIGN], observed=True)
    out = g.agg(N=("valid", "sum"), n_rej=("rej", "sum"), ARL=("rl_fin", "mean")).reset_index()
    out["Rate"] = out["n_rej"] / out["N"].where(out["N"] > 0)
    out["SE"] = np.sqrt(out["Rate"] * (1 - out["Rate"]) / out["N"].where(out["N"] > 0))
    out["Measure"] = measure
    return out


def first_detection(df: pd.DataFrame) -> pd.DataFrame:
    """Their tie rule: within (i_MC, design) keep RL == min(RL, na.rm=TRUE), else Inf; NA stays NA."""
    out = df.copy()
    keys = ["i_MC", *DESIGN]
    for col in RL_COLS.values():
        gmin = out.groupby(keys, observed=True)[col].transform("min")
        x = out[col]
        out[col] = np.where(x.isna(), np.nan, np.where(x == gmin, x, np.inf))
    return out


def detector_of(asset: str, measure: str) -> tuple[str, str]:
    """(Detector, Series) as in eval_sim_FirstDetection.R: 'X' / 'VaR_k' -> VaR, '1..K' /
    'CoVaR_k' -> Systemic Risk, 'Joint' -> Joint."""
    if asset == "Joint":
        return "Joint", "Joint"
    if measure == "CoVaR":
        return ("VaR", "X") if asset == "X" else ("Systemic Risk", asset)
    det, series = asset.split("_")
    return ("VaR" if det == "VaR" else "Systemic Risk"), series


def parse_xtable(path) -> pd.DataFrame:
    """Parse a booktabs xtable .txt into a string DataFrame (blank cells -> '')."""
    lines = [l.strip() for l in open(path) if "&" in l]
    rows = [[c.strip().rstrip("\\").strip() for c in l.split("&")] for l in lines]
    header = [h.replace("\\_", "_") for h in rows[0]]
    return pd.DataFrame(rows[1:], columns=header)
