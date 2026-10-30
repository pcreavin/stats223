import numpy as np
import pytest
from scipy import stats

from src import dgp, oracle

NU = 5.0


def _random_abr(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.5, 3.0, n), rng.uniform(0.0, 4.0, n), rng.uniform(-0.8, 0.9, n)


def test_joint_sf_vs_adaptive_quad_and_multivariate_t():
    # CHECK B: P(T1 > a, T2 > b) = P(T1 < -a, T2 < -b) by symmetry, so the scipy CDF has no cancellation
    a, b, r = _random_abr(30)
    q = oracle.joint_sf(a, b, r, NU)
    ref = np.array([dgp.joint_exceedance(ai, bi, ri, NU) for ai, bi, ri in zip(a, b, r)])
    np.testing.assert_allclose(q, ref, rtol=1e-4)
    # scipy's QMC CDF needs ~1e7 points for 1e-4 (default: ~2e-3) and is slow, so only the first 8 cases
    mvt = np.array([stats.multivariate_t(shape=[[1, ri], [ri, 1]], df=NU).cdf(
        [-ai, -bi], maxpts=10_000_000, random_state=1) for ai, bi, ri in zip(a[:8], b[:8], r[:8])])
    np.testing.assert_allclose(q[:8], mvt, rtol=1e-4)


def test_joint_sf_vs_monte_carlo():
    rng = np.random.default_rng(3)
    n = 2_000_000
    for a, b, r in zip(*_random_abr(4, seed=5)):
        T = stats.multivariate_t(shape=[[1, r], [r, 1]], df=NU).rvs(n, random_state=rng)
        hit = (T[:, 0] > a) & (T[:, 1] > b)
        se = np.sqrt(hit.mean() * (1 - hit.mean()) / n)
        assert abs(oracle.joint_sf(a, b, r, NU) - hit.mean()) < 4 * se


@pytest.mark.parametrize("level", [0.9, 0.95])
def test_band_probs_equal_p_at_correct_forecasts(level):
    rho = np.array([-0.3, 0.2, 0.5, 0.8])
    s = np.sqrt((NU - 2) / NU)
    a = np.full_like(rho, dgp.var_unit(level, NU) / s)
    b = dgp._c_of(rho, level, level, NU) / s
    q = oracle.band_probs(a, b, rho, NU)
    np.testing.assert_allclose(q, np.broadcast_to(oracle.p_null(level), q.shape), atol=1e-8)
    assert oracle.kl(q, oracle.p_null(level)).max() < 1e-12
