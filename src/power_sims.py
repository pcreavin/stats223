"""Break and misspecification simulations for Figures 2-3 (break_in_X = TRUE), Milestone C (reduced scope).

Each scenario (t*, beta_post) is simulated once with K = 5 and draw_dim = 11, so with the same seed the first
6 series coincide exactly with Milestone B's K = 10 null simulation (common random numbers across scenarios);
K = 1 is the first bank. Monitoring uses THEIR stored thresholds/moments (null_sims.run_lengths) and every
configuration is persisted via sims_io in Milestone B's format.
"""
from __future__ import annotations

import time

from . import data, dgp, null_sims, sims_io

K_SET = (1, 5)
K_SIM, DRAW_DIM = 5, 11
LEVELS = null_sims.LEVELS
N, M_WIN, E = null_sims.N, null_sims.M_WIN, null_sims.E
FIG2_LEFT_T = (0, 400, 700, 800, 900, 1000)            # beta_post = 0.85
FIG2_RIGHT_BP = (0.70, 0.75, 0.80, 0.85)               # t* = 0
SCENARIOS = [(t, 0.85) for t in FIG2_LEFT_T] + [(0, bp) for bp in FIG2_RIGHT_BP if bp != 0.85]


def params(t_star: int, beta_post: float) -> dgp.DGPParams:
    return dgp.DGPParams(K=K_SIM, E=E, n=N, t_break=t_star, beta_post=beta_post, break_in_X=True)


def name(K: int, level: float, t_star: int, beta_post: float, M: int, seed: int) -> str:
    return sims_io.config_name(K, level, t_star, beta_post, True, M, seed, n=N, m=M_WIN, E=E)


def get(t_star: int, beta_post: float, M: int, seed: int, force: bool = False, chunk: int = 500) -> dict:
    """{(K, level): (arrays, meta)}; loads the persisted files unless missing or force=True."""
    keys = [(K, lv) for K in K_SET for lv in LEVELS]
    if not force and all(sims_io.exists(name(K, lv, t_star, beta_post, M, seed)) for K, lv in keys):
        return {k: sims_io.load_sim(name(*k, t_star, beta_post, M, seed)) for k in keys}
    p = params(t_star, beta_post)
    t0 = time.time()
    ind_all = dgp.simulate_indicators(M, p, LEVELS, seed, chunk=chunk, draw_dim=DRAW_DIM)
    t_sim = time.time() - t0
    cv = data.load_crit_values()
    out = {}
    for K, lv in keys:
        ind = dgp.subset_K(ind_all[lv], K)
        arrays = dict(ind, **null_sims.run_lengths(ind, cv, K, lv))
        meta = dict(K=K, alpha=lv, beta=lv, level=lv, t_star=t_star, beta_post=beta_post, break_in_X=True,
                    M=M, n=N, m=M_WIN, E=E, seed=seed, iota=null_sims.IOTA, nu=p.nu, rho=p.rho,
                    garch_pre=p.garch_pre, dcc_pre=p.dcc_pre, n_trans=p.n_trans, chunk=chunk, draw_dim=DRAW_DIM,
                    derived_from=None if K == K_SIM else name(K_SIM, lv, t_star, beta_post, M, seed),
                    critical_values="df_CritVal_full_20250805.rds (their stored thresholds and moments)",
                    rl_covar_columns=["VaR"] + [f"CoVaR_{k}" for k in range(1, K + 1)] + ["Joint"],
                    rl_rev_columns=([f"VaR_{k}" for k in range(1, K + 1)] + [f"CoVaR_{k}" for k in range(1, K + 1)]
                                    + ["Joint"]) if K > 1 else None)
        sims_io.save_sim(name(K, lv, t_star, beta_post, M, seed), arrays, meta)
        out[(K, lv)] = (arrays, meta)
    print(f"  t*={t_star}, beta_post={beta_post:g}: simulated in {t_sim:.0f}s, monitored + saved in "
          f"{time.time() - t0:.0f}s total", flush=True)
    return out
