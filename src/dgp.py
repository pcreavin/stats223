"""Port of R/DGP.R (DCC.GARCH.t.sim, DCC.GARCH.t.Forecast) and R/ViolSeq_tDist.R (VaR/CoVaR only).

Simulation and true-parameter filtering run in one streaming time loop, vectorised over M Monte Carlo
replications, so the full (E + n)-day series is never stored. Conventions reproduced from their code:
  * DGP time i = 1..n_trans+E+n (1-based, as in R); Y[,1], eps[,1] ~ N(0,1), sigma.sq[,1] = rnorm^2,
    Q_1 = Qbar; the first n_trans days are dropped;
  * the coefficient used at time i is the post-break one iff i > n_trans + E + t_break, i.e. the break hits
    out-of-sample day j > t_break; with break_in_X=False only X's GARCH stays at the pre-break values
    (the DCC break still acts on the whole Q, including the X-bank correlations);
  * eps_i = sqrt(nu / chi2_nu) * Z_i, Z_i ~ N(0, (nu-2)/nu * R_i) (mvtnorm::rmvt, one chi2 per day);
  * the forecast filter re-runs GARCH/DCC with the PRE-break parameters from sigma.sq[1,] = rnorm^2,
    eps[1,] = Y[1,]/sqrt(sigma.sq[1,]), Q_1 = Qbar; forecasts for out-of-sample days E+1..E+n are kept.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import integrate, optimize, stats
from scipy.interpolate import CubicSpline

from . import paths


@dataclass(frozen=True)
class DGPParams:
    K: int = 10
    E: int = 5001
    n: int = 1000
    t_break: int = 1000          # break hits out-of-sample days j > t_break (t_break = n: no break)
    beta_post: float = 0.85      # post-break GARCH beta and DCC beta
    break_in_X: bool = False
    nu: float = 5.0
    rho: float = 0.5             # off-diagonal of Qbar
    garch_pre: tuple = (0.1, 0.1, 0.7)
    dcc_pre: tuple = (0.1, 0.7)
    n_trans: int = 100

    @property
    def garch_post(self):
        return (self.garch_pre[0], self.garch_pre[1], self.beta_post)

    @property
    def dcc_post(self):
        return (self.dcc_pre[0], self.beta_post)


def _qbar(d: int, rho: float) -> np.ndarray:
    return np.full((d, d), rho) + (1 - rho) * np.eye(d)


def _normalise(Q: np.ndarray) -> np.ndarray:
    s = 1.0 / np.sqrt(np.diagonal(Q, axis1=-2, axis2=-1))
    return Q * s[..., :, None] * s[..., None, :]


def simulate_and_filter(M: int, p: DGPParams, rng: np.random.Generator, draw_dim: int | None = None):
    """Simulate M paths and filter them with the pre-break parameters.

    Returns Y [M, n, K+1] (out-of-sample losses), sig [M, n, K+1] (filtered conditional sd) and
    rho [M, n, K] (filtered correlation of X with bank k). `draw_dim` >= K+1 draws that many normal columns
    per day and uses the first K+1; since Z = L e with L lower-triangular, the first K+1 series then coincide
    exactly with the first K+1 series of a larger simulation that used the same draws."""
    d = p.K + 1
    dd = d if draw_dim is None else draw_dim
    if dd < d:
        raise ValueError("draw_dim must be >= K+1")
    T = p.n_trans + p.E + p.n
    i_break = p.n_trans + p.E + p.t_break          # last 1-based time index with pre-break coefficients
    Qbar = _qbar(d, p.rho)
    nu, scale = p.nu, np.sqrt((p.nu - 2) / p.nu)

    def norm(shape_m):
        return rng.standard_normal((shape_m, dd))[:, :d]

    garch_pre = np.array(p.garch_pre)[:, None] * np.ones(d)        # [3, d]
    garch_post = np.array(p.garch_post)[:, None] * np.ones(d)
    if not p.break_in_X:
        garch_post[:, 0] = p.garch_pre
    fw, fa, fb = p.garch_pre
    fA, fB = p.dcc_pre

    Y = np.empty((M, p.n, d)); SIG = np.empty((M, p.n, d)); RHO = np.empty((M, p.n, d - 1))

    # i = 1: start values (draw order as in R: Y, eps, sigma.sq)
    y = norm(M); eps = norm(M); s2 = norm(M) ** 2
    Q = np.broadcast_to(Qbar, (M, d, d)).copy()
    fs2 = feps = fQ = y_prev = None
    for i in range(2, T + 1):
        g = garch_pre if i <= i_break else garch_post
        a, b = p.dcc_pre if i <= i_break else p.dcc_post
        s2 = g[0] + g[1] * y ** 2 + g[2] * s2
        Q = (1 - a - b) * Qbar + a * (eps[:, :, None] * eps[:, None, :]) + b * Q
        L = np.linalg.cholesky(_normalise(Q))
        z = scale * np.einsum("mij,mj->mi", L, norm(M))
        eps = z * np.sqrt(nu / rng.chisquare(nu, M))[:, None]
        y = np.sqrt(s2) * eps

        s = i - p.n_trans                          # 1-based index of the returned series
        if s < 1:
            continue
        if s == 1:                                 # DCC.GARCH.t.Forecast start values
            fs2 = norm(M) ** 2
            fQ = np.broadcast_to(Qbar, (M, d, d)).copy()
        else:
            fs2 = fw + fa * y_prev ** 2 + fb * fs2
            fQ = (1 - fA - fB) * Qbar + fA * (feps[:, :, None] * feps[:, None, :]) + fB * fQ
        feps = y / np.sqrt(fs2)
        y_prev = y
        j = s - p.E - 1                            # 0-based out-of-sample day
        if j >= 0:
            Y[:, j] = y
            SIG[:, j] = np.sqrt(fs2)
            RHO[:, j] = _normalise(fQ)[:, 0, 1:]
    return Y, SIG, RHO


# --------------------------------------------------------------------------- CoVaR of a bivariate t
def _sd_wt(nu: float) -> float:
    return np.sqrt(nu / (nu - 2))


def joint_exceedance(q: float, y: float, rho: float, nu: float) -> float:
    """P(T1 > q, T2 > y) for a standard bivariate t (correlation rho, df nu), by 1-D quadrature over T1
    using T2 | T1 = t ~ rho t + sqrt((1 - rho^2)(nu + t^2)/(nu + 1)) * t_{nu+1}."""
    r2 = 1 - rho * rho

    def f(t):
        return stats.t.pdf(t, nu) * stats.t.sf((y - rho * t) / np.sqrt(r2 * (nu + t * t) / (nu + 1)), nu + 1)

    val, _ = integrate.quad(f, q, np.inf, epsabs=1e-13, epsrel=1e-11, limit=200)
    return val


def covar_c(rho: float, alpha: float, beta: float, nu: float) -> float:
    """c(rho) with CoVaR = s2 * c(rho): unit-variance threshold y*/sd_wt solving
    P(T1 > qt(beta, nu), T2 > y*) = (1 - alpha)(1 - beta)."""
    q, target = stats.t.ppf(beta, nu), (1 - alpha) * (1 - beta)
    y = optimize.brentq(lambda y: joint_exceedance(q, y, rho, nu) - target, -60.0, 60.0, xtol=1e-12)
    return y / _sd_wt(nu)


RHO_GRID = np.linspace(-0.9995, 0.9995, 4001)


def covar_c_vec(rho: np.ndarray, alpha: float, beta: float, nu: float, tol: float = 1e-11) -> np.ndarray:
    """Vectorised covar_c: safeguarded Newton (bisection fallback) on all rho at once, with P(y) by quad_vec
    and the exact derivative dP/dy = -f_nu(y) * P(T1 > q | T2 = y) (conditional t_{nu+1})."""
    rho = np.asarray(rho, float)
    q, target, r2 = stats.t.ppf(beta, nu), (1 - alpha) * (1 - beta), 1 - rho * rho

    def P(y):
        f = lambda t: stats.t.pdf(t, nu) * stats.t.sf((y - rho * t) / np.sqrt(r2 * (nu + t * t) / (nu + 1)), nu + 1)
        return integrate.quad_vec(f, q, np.inf, epsabs=1e-14, epsrel=1e-11)[0]

    lo, hi = np.full_like(rho, -60.0), np.full_like(rho, 60.0)     # P(lo) > target > P(hi)
    y = np.full_like(rho, q)
    for _ in range(200):
        g = P(y) - target                                          # decreasing in y
        lo, hi = np.where(g > 0, y, lo), np.where(g > 0, hi, y)
        dP = -stats.t.pdf(y, nu) * stats.t.sf((q - rho * y) / np.sqrt(r2 * (nu + y * y) / (nu + 1)), nu + 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            y_new = y - g / dP
        bad = ~np.isfinite(y_new) | (y_new <= lo) | (y_new >= hi)
        y_new = np.where(bad, 0.5 * (lo + hi), y_new)
        done = np.abs(y_new - y) < tol
        y = y_new
        if done.all():
            return y / _sd_wt(nu)
    raise RuntimeError("covar_c_vec: root-finding did not converge")


@lru_cache(maxsize=None)
def covar_c_table(alpha: float, beta: float, nu: float) -> CubicSpline:
    """Cubic spline of c(rho) on RHO_GRID, cached to data/derived/covar_tables/."""
    d = paths.DERIVED / "covar_tables"
    f = d / f"c_a{alpha:g}_b{beta:g}_nu{nu:g}_g{RHO_GRID.size}_r{RHO_GRID[-1]:g}.npz"
    if f.exists():
        c = np.load(f)["c"]
    else:
        c = covar_c_vec(RHO_GRID, alpha, beta, nu)
        d.mkdir(parents=True, exist_ok=True)
        np.savez(f, rho=RHO_GRID, c=c)
    return CubicSpline(RHO_GRID, c)


def _c_of(rho: np.ndarray, alpha, beta, nu) -> np.ndarray:
    if rho.size and (rho.min() < RHO_GRID[0] or rho.max() > RHO_GRID[-1]):
        raise ValueError(f"rho outside the tabulated grid: [{rho.min()}, {rho.max()}]")
    return covar_c_table(float(alpha), float(beta), float(nu))(rho)


def var_unit(beta: float, nu: float) -> float:
    """VaR of a unit-variance t_nu: qt(beta, nu) / sqrt(nu/(nu-2))."""
    return stats.t.ppf(beta, nu) / _sd_wt(nu)


def violation_sequences(Y, sig, rho, nu, alpha: float, beta: float) -> dict:
    """Port of ViolSeq_tDist (no CoES/MES). Y, sig: [..., n, K+1]; rho: [..., n, K]; nu: scalar or vector
    with nu[0] for X and nu[k] for bank k (their nu = rep(nu, K+2) is accepted).

    Returns int8 arrays I_VaR [..., n], I_CoVaR, I_VaR_Rev, I_CoVaR_Rev [..., K, n]."""
    K = Y.shape[-1] - 1
    nu = np.broadcast_to(np.asarray(nu, float), (K + 2,)) if np.ndim(nu) == 0 else np.asarray(nu, float)
    x, sx = Y[..., 0], sig[..., 0]
    I_VaR = x > sx * var_unit(beta, nu[0])
    out = {k: np.empty(Y.shape[:-2] + (K, Y.shape[-2]), np.int8)
           for k in ("I_CoVaR", "I_VaR_Rev", "I_CoVaR_Rev")}
    for k in range(K):
        yk, sk, rk = Y[..., k + 1], sig[..., k + 1], rho[..., k]
        covar = sk * _c_of(rk, alpha, beta, nu[k + 1])         # CoVaR.true.t(nu = nu[k+1])
        out["I_CoVaR"][..., k, :] = I_VaR & (yk > covar)
        ivr = yk > sk * var_unit(beta, nu[k + 1])
        out["I_VaR_Rev"][..., k, :] = ivr
        rev = sx * _c_of(rk, alpha, beta, nu[0])               # CoVaR.true.t(nu = nu[1]), roles swapped
        out["I_CoVaR_Rev"][..., k, :] = ivr & (x > rev)
    return dict(I_VaR=I_VaR.astype(np.int8), **out)


# --------------------------------------------------------------------------- full simulation pipeline
def simulate_indicators(M: int, p: DGPParams, levels, seed: int, chunk: int = 500, verbose: bool = False,
                        draw_dim: int | None = None):
    """Simulate M paths in fixed chunks (child seeds from SeedSequence(seed), so results do not depend on
    anything but (M, p, seed, chunk, draw_dim)) and return {level: violation_sequences(...)} with
    alpha = beta = level. draw_dim: see simulate_and_filter."""
    import time
    n_chunks = -(-M // chunk)
    children = np.random.SeedSequence(seed).spawn(n_chunks)
    res = {lv: [] for lv in levels}
    for c, ss in enumerate(children):
        t0 = time.time()
        m = min(chunk, M - c * chunk)
        Y, S, R = simulate_and_filter(m, p, np.random.default_rng(ss), draw_dim=draw_dim)
        for lv in levels:
            res[lv].append(violation_sequences(Y, S, R, p.nu, lv, lv))
        if verbose:
            print(f"  chunk {c + 1}/{n_chunks} ({m} paths): {time.time() - t0:.1f}s", flush=True)
    return {lv: {k: np.concatenate([r[k] for r in res[lv]]) for k in res[lv][0]} for lv in levels}


def subset_K(ind: dict, K: int) -> dict:
    """First K banks of a larger simulation (exact in law; see REPLICATION_NOTES.md)."""
    return dict(I_VaR=ind["I_VaR"], **{k: ind[k][..., :K, :] for k in ("I_CoVaR", "I_VaR_Rev", "I_CoVaR_Rev")})


__all__ = ["DGPParams", "simulate_and_filter", "violation_sequences", "covar_c", "covar_c_vec",
           "covar_c_table", "joint_exceedance", "var_unit", "simulate_indicators", "subset_K", "RHO_GRID"]
