"""Time-varying oracle for the joint three-band e-detector (K = 1).

Re-runs the Section 3.4 break DGP with exactly the seeds/draws of power_sims (common random numbers: K = 1 with
draw_dim = 11 reproduces the first two series of the saved K = 5 runs), keeps the TRUE conditional sd and
correlation next to the pre-break forecasts, and computes the true band probabilities
q_t = P_true(Z_t = j | F_{t-1}), j = 0 (no sector VaR miss), 1 (sector miss, bank within CoVaR), 2 (joint miss).

With s = sqrt((nu-2)/nu), X_t / (sigma_X,t s) and Y_t / (sigma_Y,t s) are standard bivariate t_nu with
correlation rho_t, so with a_t = VaR_t / (sigma_X,t s) and b_t = CoVaR_t / (sigma_Y,t s):
  q_sector = P(T1 > a_t),  q_joint = P(T1 > a_t, T2 > b_t).
"""
from __future__ import annotations

import numpy as np
from scipy import special

from . import dgp, power_sims

N_NODES = 64
_XI, _W = np.polynomial.legendre.leggauss(N_NODES)


def p_null(level: float) -> np.ndarray:
    """Band probabilities under correct VaR/CoVaR forecasts with alpha = beta = level."""
    a = b = level
    return np.array([b, (1 - b) * a, (1 - b) * (1 - a)])


def joint_sf(a, b, rho, nu: float) -> np.ndarray:
    """P(T1 > a, T2 > b) for a standard bivariate t_nu with correlation rho (broadcast), by Gauss-Legendre in
    u = F_nu(x) over [F_nu(a), 1], written as 1 - u = sf(a) w^nu, w in [0, 1]. In u the integrand behaves like
    (1 - u)^(2/nu) at u = 1 (x ~ (1 - u)^(-1/nu)), so plain Gauss-Legendre converges only algebraically
    (rel. error ~1e-4 at 64 nodes); in w it is smooth and 64 nodes reach ~1e-15. Uses
    T2 | T1 = x ~ rho x + sqrt((nu + x^2)(1 - rho^2)/(nu + 1)) t_{nu+1}."""
    a, b, rho = (np.asarray(z, float)[..., None] for z in np.broadcast_arrays(a, b, rho))
    sf_a = special.stdtr(nu, -a)                                  # P(T1 > a)
    w = (1 + _XI) / 2
    x = -special.stdtrit(nu, sf_a * w ** nu)                      # F^{-1}(1 - sf(a) w^nu)
    z = (b - rho * x) / np.sqrt((nu + x * x) * (1 - rho * rho) / (nu + 1))
    return (sf_a * nu * w ** (nu - 1) / 2 * (_W * special.stdtr(nu + 1, -z))).sum(-1)


def band_probs(a, b, rho, nu: float) -> np.ndarray:
    """q [..., 3] = (1 - q_sector, q_sector - q_joint, q_joint)."""
    qs = special.stdtr(nu, -np.asarray(a, float))
    qj = joint_sf(a, b, rho, nu)
    return np.stack([1 - qs, qs - qj, qj], axis=-1)


def kl(q, p) -> np.ndarray:
    """KL(q || p) along the last axis."""
    q = np.asarray(q)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(q > 0, q * np.log(q / p), 0.0)
    return t.sum(-1)


def params(t_star: int, beta_post: float, K: int = 1) -> dgp.DGPParams:
    p = power_sims.params(t_star, beta_post)
    return dgp.DGPParams(**{**p.__dict__, "K": K})


def truth_chunks(t_star: int, beta_post: float, M: int, seed: int, chunk: int = 500, n_chunks: int | None = None,
                 K: int = 1):
    """Yield (path slice, dict) per seed chunk, as in dgp.simulate_indicators with power_sims' settings.

    dict: Y [m, n, 2], sig/rho (pre-break filter, i.e. forecast inputs), sig_true/rho_true (DGP)."""
    p = params(t_star, beta_post, K)
    children = np.random.SeedSequence(seed).spawn(-(-M // chunk))
    for c, ss in enumerate(children[:n_chunks]):
        m = min(chunk, M - c * chunk)
        Y, S, R, St, Rt = dgp.simulate_and_filter(m, p, np.random.default_rng(ss), draw_dim=power_sims.DRAW_DIM,
                                                  return_true=True)
        yield slice(c * chunk, c * chunk + m), dict(Y=Y[..., :2], sig=S[..., :2], rho=R[..., 0],
                                                     sig_true=St[..., :2], rho_true=Rt[..., 0])


def forecasts(tr: dict, level: float, nu: float) -> tuple[np.ndarray, np.ndarray]:
    """VaR_t and CoVaR_t of the pre-break filter, exactly as in dgp.violation_sequences."""
    var = tr["sig"][..., 0] * dgp.var_unit(level, nu)
    covar = tr["sig"][..., 1] * dgp._c_of(tr["rho"], level, level, nu)
    return var, covar


def indicators(tr: dict, var, covar) -> tuple[np.ndarray, np.ndarray]:
    """I_VaR [m, n], I_CoVaR [m, 1, n] (int8), same comparisons as dgp.violation_sequences."""
    iv = tr["Y"][..., 0] > var
    ic = iv & (tr["Y"][..., 1] > covar)
    return iv.astype(np.int8), ic.astype(np.int8)[:, None, :]


def oracle_q(tr: dict, var, covar, nu: float, block: int = 100) -> np.ndarray:
    """q_t [m, n, 3] in blocks of paths (quadrature arrays [block, n, 64])."""
    s = np.sqrt((nu - 2) / nu)
    a = var / (tr["sig_true"][..., 0] * s)
    b = covar / (tr["sig_true"][..., 1] * s)
    out = np.empty(a.shape + (3,))
    for i in range(0, a.shape[0], block):
        sl = slice(i, i + block)
        out[sl] = band_probs(a[sl], b[sl], tr["rho_true"][sl], nu)
    return out


__all__ = ["p_null", "joint_sf", "band_probs", "kl", "params", "truth_chunks", "forecasts", "indicators",
           "oracle_q", "N_NODES"]
