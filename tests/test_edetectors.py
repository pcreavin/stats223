import numpy as np
import pytest

from src import edetectors as ed

P = np.array([0.95, 0.0475, 0.0025])


def test_categories():
    np.testing.assert_array_equal(ed.categories([0, 1, 1], [0, 0, 1]), [0, 1, 2])
    with pytest.raises(ValueError):
        ed.categories([0], [1])


def test_null_mean_is_t():
    rng = np.random.default_rng(0)
    n, T = 200_000, 50
    Z = rng.choice(3, size=(n, T), p=P)
    q = np.array([0.9, 0.08, 0.02])
    M = ed.e_sr(Z, q, P, return_path=True)["M"]
    for t in (1, 10, 50):
        m = M[:, t - 1]
        assert abs(m.mean() - t) < 4 * m.std() / np.sqrt(n)


def test_recursion_equals_sum_over_start_dates():
    rng = np.random.default_rng(1)
    Z = rng.choice(3, size=(3, 12), p=[0.6, 0.3, 0.1])
    q = rng.dirichlet(np.ones(3), size=(3, 12))
    L = ed.e_values(Z, q, P)
    M = ed.e_sr(Z, q, P, return_path=True)["M"]
    expl = np.array([[sum(np.prod(L[i, k:t + 1]) for k in range(t + 1)) for t in range(12)] for i in range(3)])
    np.testing.assert_allclose(M, expl, rtol=1e-12)


def test_q_equal_p_gives_t():
    Z = np.random.default_rng(2).choice(3, size=(4, 1000), p=P)
    M = ed.e_sr(Z, np.broadcast_to(P, (4, 1000, 3)), P, return_path=True)["M"]
    np.testing.assert_array_equal(M, np.broadcast_to(np.arange(1.0, 1001), M.shape))


def test_windows_and_time_chunks():
    # cross gamma on day 5, fall back on day 6, cross again on day 9; matched window starts on day 7
    L = np.array([1, 1, 1, 1, 1000, 1e-9, 1, 1, 1e5, 1.0])
    q = np.stack([P * l for l in L])          # q_t(0) / p(0) = L_t (only band 0 observed)
    Z = np.zeros(10, np.int8)
    r = ed.e_sr(Z, q, P, gamma=1e3, windows={"matched": (7, 10), "day1": (1, 10)})
    assert r["alarm"]["day1"] == 5 and r["alarm"]["matched"] == 9
    w = {"matched": (7, 10), "day1": (1, 10)}
    r1 = ed.e_sr(Z[:6], q[:6], P, gamma=1e3, windows=w)
    r2 = ed.e_sr(Z[6:], q[6:], P, gamma=1e3, windows=w, M0=r1["M_last"], t0=6)
    for k in w:
        assert np.fmin(r1["alarm"][k], r2["alarm"][k]) == r["alarm"][k]
    np.testing.assert_allclose(r2["M_last"], r["M_last"])
