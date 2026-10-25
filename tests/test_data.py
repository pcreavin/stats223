import pandas as pd

from src import data


def test_assets_wide():
    w = data.load_assets()
    for a in ["SPF", "BAC", "C", "GS", "JPM", "WFC"]:
        assert a in w.columns
    assert 8_500 < len(w) < 9_500
    assert w.index.min() == pd.Timestamp("1990-01-02")
    assert w.index.max() == pd.Timestamp("2025-10-23")
    assert w["SPF"].first_valid_index().year == 1993
    assert w["GS"].first_valid_index() >= pd.Timestamp("1999-05-01")


def test_crit_values_recoded():
    cv = data.load_crit_values()
    assert set(cv.SysRisk) == {"CoVaR", "CoES", "MES", "CoVaR_Reverse"}
    row = data.crit_value_row(cv, "CoVaR", 0.95, 0.95, K=1)
    assert abs(row.c_VaR - 2.35) < 0.01 and abs(row.c_SysRisk - 4.82) < 0.01
