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


# --------------------------------------------------------------------------- learned bets (Extension 2)
def _valid(q):
    assert np.all(q > 0)
    np.testing.assert_allclose(q.sum(-1), 1.0, rtol=0, atol=1e-12)


def test_mixture_q_rows_are_inflated_alternatives():
    Q = ed.mixture_q(P)
    _valid(Q)
    assert len(Q) == 12
    assert np.all(Q[:, 1] + Q[:, 2] > P[1] + P[2]) and np.all(Q[:, 2] > P[2])


@pytest.mark.parametrize("bet", ["plugin", "cluster"])
def test_learned_bets_are_valid_and_predictable(bet):
    rng = np.random.default_rng(3)
    Z = rng.choice(3, size=(5, 300), p=[0.8, 0.15, 0.05])
    f = (lambda z: ed.plugin_q(z, P)) if bet == "plugin" else (lambda z: ed.duration_q(z, P, h=3))
    q = f(Z)
    _valid(q)
    np.testing.assert_array_equal(q[:, 0], np.broadcast_to(P, (5, 3)))      # nothing observed yet: bet p
    assert np.all(q[..., 1:] >= P[1:])                                       # one-sided
    for t in (0, 1, 150, 299):                                               # q_s for s <= t ignores Z_t
        Z2 = Z.copy()
        Z2[:, t] = (Z2[:, t] + 1) % 3
        np.testing.assert_array_equal(f(Z2)[:, :t + 1], q[:, :t + 1])


def test_plugin_shrinks_to_p_and_learns():
    Z = np.random.default_rng(4).choice(3, size=(2, 400), p=[0.7, 0.2, 0.1])
    np.testing.assert_allclose(ed.plugin_q(Z, P, kappa=1e12), np.broadcast_to(P, (2, 400, 3)), atol=1e-9)
    q = ed.plugin_q(Z, P, lam=0.98, kappa=50, one_sided=False)
    assert np.all(q[:, -50:, 2].mean() > 3 * P[2])                          # tracks the elevated joint-miss rate


def test_cluster_bets_more_after_a_miss():
    rng = np.random.default_rng(5)
    Z = np.zeros((1, 4000), np.int64)
    for s in rng.choice(3990, 150, replace=False):                           # bursts of misses
        Z[0, s:s + 4] = 1
    q = ed.duration_q(Z, P, h=1)
    after = np.r_[False, Z[0, :-1] >= 1]
    assert q[0, after, 1].mean() > 3 * q[0, ~after, 1].mean()


def test_mix_with_one_bet_equals_e_sr():
    Z = np.random.default_rng(6).choice(3, size=(3, 100), p=P)
    q = ed.plugin_q(Z, P)
    np.testing.assert_allclose(ed.e_sr_mix(Z, [q], P, return_path=True)["M"],
                               ed.e_sr(Z, q, P, return_path=True)["M"], rtol=1e-12)


@pytest.mark.parametrize("name", ed.LEARNED)
def test_learned_null_mean_is_t(name):
    rng = np.random.default_rng(7)
    n, T = 100_000, 60
    Z = rng.choice(3, size=(n, T), p=P)
    M = ed.learned_path(name, Z, P)
    for t in (1, 20, 60):
        m = M[:, t - 1]
        assert abs(m.mean() - t) <= 4 * m.std() / np.sqrt(n)


def test_alarms_and_window_max():
    M = np.array([[1, 5, 20, 3, 50.0]])
    w = {"late": (3, 5), "all": (1, 5)}
    assert ed.alarms(M, 10, w)["late"][0] == 3 and ed.alarms(M, 60, w)["all"][0] == np.inf
    mx = ed.window_max(M, {"early": (1, 2), "all": (1, 5)})
    assert mx["early"][0] == 5 and mx["all"][0] == 50
