import numpy as np
import pytest
from scipy import stats

from src import dgp, paths, sims_io

SMALL = dict(E=301, n=400, n_trans=100)


def test_unit_variance_t_marginals_and_correlation():
    # constant sigma^2 = 1 and R = Qbar, so Y = eps ~ unit-variance t_5 with correlation 0.5
    p = dgp.DGPParams(K=2, E=1, n=40_000, t_break=40_000, garch_pre=(1.0, 0.0, 0.0), dcc_pre=(0.0, 0.0))
    Y, S, R = dgp.simulate_and_filter(4, p, np.random.default_rng(0))
    y = Y.reshape(-1, 3)
    np.testing.assert_allclose(y.var(0), 1.0, rtol=0.04)
    c = np.corrcoef(y.T)
    np.testing.assert_allclose(c[np.triu_indices(3, 1)], 0.5, atol=0.015)
    np.testing.assert_allclose(S, 1.0)
    np.testing.assert_allclose(R, 0.5)
    # standardised margin is t_5: compare a tail probability
    assert np.mean(y[:, 0] * np.sqrt(5 / 3) > stats.t.ppf(0.99, 5)) == pytest.approx(0.01, abs=0.0015)


def test_garch_unconditional_variance():
    p = dgp.DGPParams(K=1, **SMALL)
    Y, S, _ = dgp.simulate_and_filter(200, p, np.random.default_rng(1))
    target = 0.1 / (1 - 0.1 - 0.7)
    assert (S ** 2).mean() == pytest.approx(target, rel=0.03)
    assert (Y ** 2).mean() == pytest.approx(target, rel=0.05)


@pytest.mark.parametrize("break_in_X", [False, True])
def test_break_index(break_in_X):
    t_star = 150
    null = dgp.DGPParams(K=2, t_break=SMALL["n"], **SMALL)
    brk = dgp.DGPParams(K=2, t_break=t_star, break_in_X=break_in_X, beta_post=0.85, **SMALL)
    Y0, S0, R0 = dgp.simulate_and_filter(3, null, np.random.default_rng(7))
    Y1, S1, R1 = dgp.simulate_and_filter(3, brk, np.random.default_rng(7))
    # out-of-sample days 1..t* (0-based < t*) are pre-break
    np.testing.assert_array_equal(Y0[:, :t_star], Y1[:, :t_star])
    # first post-break day: banks' variance changes; X's eps is unaffected by R (L[0,0] = 1), so Y_X
    # changes only when X's GARCH breaks too
    assert not np.allclose(Y0[:, t_star, 1:], Y1[:, t_star, 1:])
    assert np.allclose(Y0[:, t_star, 0], Y1[:, t_star, 0]) == (not break_in_X)
    # the filter always uses pre-break parameters: forecasts for day t*+1 only use data up to day t*
    np.testing.assert_array_equal(S0[:, :t_star + 1], S1[:, :t_star + 1])


def test_k_subset_exact_with_shared_draws():
    big = dgp.DGPParams(K=10, **SMALL)
    small = dgp.DGPParams(K=2, **SMALL)
    Yb, Sb, Rb = dgp.simulate_and_filter(5, big, np.random.default_rng(3), draw_dim=11)
    Ys, Ss, Rs = dgp.simulate_and_filter(5, small, np.random.default_rng(3), draw_dim=11)
    np.testing.assert_allclose(Ys, Yb[..., :3], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(Ss, Sb[..., :3], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(Rs, Rb[..., :2], rtol=0, atol=1e-12)


def test_k_subset_same_law():
    big = dgp.DGPParams(K=10, **SMALL)
    small = dgp.DGPParams(K=2, **SMALL)
    Yb, _, Rb = dgp.simulate_and_filter(100, big, np.random.default_rng(4))
    Ys, _, Rs = dgp.simulate_and_filter(100, small, np.random.default_rng(5))
    assert (Ys[..., :3] ** 2).mean() == pytest.approx((Yb[..., :3] ** 2).mean(), rel=0.05)
    assert Rs.mean() == pytest.approx(Rb[..., :2].mean(), abs=0.01)


@pytest.mark.parametrize("rho", [-0.6, 0.0, 0.5, 0.9])
@pytest.mark.parametrize("level", [0.9, 0.95])
def test_covar_c_root_and_table(rho, level):
    nu = 5.0
    c = dgp.covar_c(rho, level, level, nu)
    np.testing.assert_allclose(dgp.covar_c_vec(np.array([rho]), level, level, nu), c, rtol=0, atol=1e-9)
    np.testing.assert_allclose(dgp._c_of(np.array([rho + 1.3e-4]), level, level, nu),
                               dgp.covar_c(rho + 1.3e-4, level, level, nu), rtol=0, atol=1e-8)
    # the root really solves the defining equation (exceedance via scipy's bivariate t CDF)
    q, y = stats.t.ppf(level, nu), c * np.sqrt(nu / (nu - 2))
    F = stats.multivariate_t(loc=[0, 0], shape=[[1, rho], [rho, 1]], df=nu)
    p = F.cdf([np.inf, np.inf], lower_limit=[q, y], maxpts=10**7, random_state=0)
    assert p == pytest.approx((1 - level) ** 2, abs=1e-5)


def test_covar_c_vs_monte_carlo():
    nu, rho, lv = 5.0, 0.5, 0.9
    rng = np.random.default_rng(0)
    N = 2_000_000
    L = np.linalg.cholesky([[1, rho], [rho, 1]])
    T = (rng.standard_normal((N, 2)) @ L.T) * np.sqrt(nu / rng.chisquare(nu, N))[:, None]
    sd = np.sqrt(nu / (nu - 2))
    hit = (T[:, 0] > stats.t.ppf(lv, nu)) & (T[:, 1] > dgp.covar_c(rho, lv, lv, nu) * sd)
    se = np.sqrt(0.01 * 0.99 / N)
    assert abs(hit.mean() - 0.01) < 4 * se


def test_null_miss_rates():
    p = dgp.DGPParams(K=2)
    Y, S, R = dgp.simulate_and_filter(100, p, np.random.default_rng(9))
    for lv in (0.9, 0.95):
        ind = dgp.violation_sequences(Y, S, R, np.repeat(5.0, 4), lv, lv)
        N = ind["I_VaR"].size
        for key, pr in (("I_VaR", 1 - lv), ("I_VaR_Rev", 1 - lv),
                        ("I_CoVaR", (1 - lv) ** 2), ("I_CoVaR_Rev", (1 - lv) ** 2)):
            x = ind[key]
            se = np.sqrt(pr * (1 - pr) / x.size)
            assert abs(x.mean() - pr) < 4 * se, (lv, key, x.mean(), pr)
        assert not np.any(ind["I_CoVaR"] > ind["I_VaR"][:, None, :])        # CoVaR hit implies VaR hit
        assert not np.any(ind["I_CoVaR_Rev"] > ind["I_VaR_Rev"])
        assert N == 100 * 1000


def test_simulate_indicators_chunking_deterministic():
    p = dgp.DGPParams(K=2, **SMALL)
    a = dgp.simulate_indicators(7, p, [0.9], seed=1, chunk=3)
    b = dgp.simulate_indicators(7, p, [0.9], seed=1, chunk=3)
    for k in a[0.9]:
        np.testing.assert_array_equal(a[0.9][k], b[0.9][k])
    sub = dgp.subset_K(a[0.9], 1)
    assert sub["I_CoVaR"].shape == (7, 1, SMALL["n"]) and sub["I_VaR"].shape == (7, SMALL["n"])


def test_sims_io_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "SIMS", tmp_path)
    monkeypatch.setattr(paths, "ensure_dirs", lambda: None)
    name = sims_io.config_name(K=2, level=0.95, t_star=1000, beta_post=0.85, break_in_X=False, M=3, seed=1)
    assert name == "K2_a0.95_b0.95_t1000_bp0.85_bx0_M3_n1000_m250_E5001_s1"
    arr = dict(I_VaR=np.zeros((3, 10), np.int8), rl=np.array([[1.0, np.inf]]))
    sims_io.save_sim(name, arr, dict(K=2, level=0.95))
    assert sims_io.exists(name)
    back, meta = sims_io.load_sim(name)
    np.testing.assert_array_equal(back["rl"], arr["rl"])
    assert back["I_VaR"].dtype == np.int8 and meta["K"] == 2
    assert [m["name"] for m in sims_io.list_sims(level=0.95)] == [name]
    assert sims_io.list_sims(level=0.9) == []
