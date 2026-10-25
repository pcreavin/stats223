"""Loaders for the data and stored results shipped with the reference repo."""
import pandas as pd
import pyreadr

from . import paths


def _read_rds(path) -> pd.DataFrame:
    return next(iter(pyreadr.read_r(str(path)).values()))


def load_assets_long() -> pd.DataFrame:
    """Long format: Date, Asset, NegReturn (negative log-returns in %)."""
    df = _read_rds(paths.ASSETS_RDS)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def load_assets() -> pd.DataFrame:
    """Wide format, one column per asset, indexed by Date (NaN before listing)."""
    long = load_assets_long()
    wide = long.pivot(index="Date", columns="Asset", values="NegReturn").sort_index()
    wide.columns.name = None
    return wide


def load_crit_values() -> pd.DataFrame:
    """Their stored critical-value table (B=5000), with 'CoVaR_Exposure' recoded to
    'CoVaR_Reverse' exactly as in code_UDE/sim_szenarios.R."""
    cv = _read_rds(paths.CRIT_VALUES_RDS)
    cv["SysRisk"] = cv["SysRisk"].replace({"CoVaR_Exposure": "CoVaR_Reverse"})
    return cv


def crit_value_row(cv: pd.DataFrame, sysrisk: str, alpha: float, beta: float, K: int,
                   n: int = 1000, m: int = 250, iota: float = 0.1) -> pd.Series:
    sel = cv[(cv.SysRisk == sysrisk) & (cv.beta.round(6) == round(beta, 6))
             & (cv.alpha.round(6) == round(alpha, 6)) & (cv.K == K)
             & (cv.n == n) & (cv.m == m) & ((cv.iota - iota).abs() < 1e-12)]
    if len(sel) != 1:
        raise KeyError(f"expected one CV row, got {len(sel)}")
    return sel.iloc[0]


def load_appl_res() -> pd.DataFrame:
    df = _read_rds(paths.APPL_RES_RDS)
    df["Date"] = pd.to_datetime(df["Date"])
    return df
