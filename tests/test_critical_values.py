import numpy as np

from src import critical_values as CV
from src.detectors import Moments


def test_nu_grid_matches_r_seq():
    assert CV.NU_GRID.size == 10001 and CV.NU_GRID[0] == 0 and CV.NU_GRID[-1] == 1
    assert CV.NU_GRID[1234] == 0.1234


def test_select_thresholds_independent_sups():
    rng = np.random.default_rng(0)
    sv, sc = rng.normal(size=4000), rng.normal(size=4000)
    th = CV.select_thresholds(sv, sc, K=1, iota=0.1)
    assert abs(th["size"] - 0.1) < 2e-3
    # with independent sups, P(V or C) = 0.1 -> each marginal ~ 0.0513 -> c ~ z_{0.9487}
    assert abs(th["c_VaR"] - 1.632) < 0.08
    # reverse with K=2 targets iota/K = 0.05
    th2 = CV.select_thresholds(sv, sc, K=2, iota=0.1)
    assert abs(th2["size_rev"] - 0.05) < 2e-3


def test_closest_not_below_rule():
    # which.min(|target - iota|) may land above iota; argmin is first-min like which.min
    x = np.abs(np.array([0.3, 0.1004, 0.0996, 0.2]) - 0.1)
    assert int(np.argmin(x)) == 1


def test_critical_values_deterministic_small_B():
    ms = Moments(0.01, 0.008, 0.44, 0.08)
    a = CV.simulate_sups(300, 100, 0.9, 0.9, 50, ms, ms, rng=5)
    b = CV.simulate_sups(300, 100, 0.9, 0.9, 50, ms, ms, rng=5)
    np.testing.assert_array_equal(a[0], b[0])
    df = CV.critical_values(100, 300, [1, 2], 0.9, 0.9, B=100, iota=0.1, seed=1, reps_moments=2000)
    assert list(df.SysRisk) == ["CoVaR", "CoVaR_Reverse"] * 2
    assert (df.c_SysRisk > 0).all()
