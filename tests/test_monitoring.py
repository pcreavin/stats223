import numpy as np
import pandas as pd

from src import data, monitoring as M
from src.detectors import combined_detector, moments_from_cv_row


def test_run_length_strict_and_one_based():
    paths = np.array([[0.5, 2.0], [1.0, 0.1], [3.0, 0.1]])   # T=3, D=2
    res = M.monitor(paths, [1.0, 1.0])
    # col0: 1.0 is not > 1 -> first alarm at index 3; col1: index 1
    np.testing.assert_array_equal(res.run_length, [3.0, 1.0])
    assert res.joint == 1.0


def test_inf_and_broadcast():
    paths = np.zeros((4, 10, 3))
    paths[2, 5, 1] = 5.0
    res = M.monitor(paths, [2.0, 2.0, 2.0])
    assert res.run_length.shape == (4, 3)
    assert np.isinf(res.joint[0]) and res.joint[2] == 6.0


def test_first_detector_tie_rule():
    rl = np.array([[3, 3, np.inf], [np.inf, np.inf, np.inf], [5, 2, 9]])
    np.testing.assert_allclose(M.first_detector_shares(rl), [1 / 3, 2 / 3, 0])


def test_monitor_covar_wrapper_consistent():
    cv = data.load_crit_values()
    row = data.crit_value_row(cv, "CoVaR", 0.95, 0.95, K=2)
    rng = np.random.default_rng(0)
    U1, U2 = rng.random((2, 1000)), rng.random((2, 2, 1000))
    I_v = (U1[0] > 0.95).astype(np.int8)
    I_c = ((U1[0] > 0.95) & (U2[0] > 0.95)).astype(np.int8)
    res = M.monitor_covar(I_v, I_c, row, 250, asset_names=["X", "A", "B"])
    assert res.names == ["VaR", "CoVaR_A", "CoVaR_B"]
    assert res.normalised.shape == (751, 3)
    dv = combined_detector(I_v, 250, 0.05, moments_from_cv_row(row, "VaR"))
    np.testing.assert_allclose(res.normalised[:, 0], dv / row.c_VaR)
