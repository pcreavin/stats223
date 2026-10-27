"""Null simulations for Tables 1-2 (t* = n = 1000, break_in_X = False): simulate once with K = 10, derive
K in {1, 2, 5} as the first K banks, monitor with THEIR stored thresholds/moments and persist everything
via sims_io. Shared by experiments 02 and 03."""
from __future__ import annotations

import time

import numpy as np

from . import data, dgp, sims_io
from .monitoring import monitor_covar, monitor_reverse

K_SET = (1, 2, 5, 10)
LEVELS = (0.9, 0.95)
N, M_WIN, E, T_STAR, BETA_POST, IOTA = 1000, 250, 5001, 1000, 0.85, 0.1
MOMENT_COLS = [f"{s}_{w}_{t}" for s in ("mu", "sd") for w in ("VaR", "SysRisk") for t in ("UC", "IID")]


def null_params() -> dgp.DGPParams:
    return dgp.DGPParams(K=max(K_SET), E=E, n=N, t_break=T_STAR, beta_post=BETA_POST, break_in_X=False)


def name(K: int, level: float, M: int, seed: int) -> str:
    return sims_io.config_name(K, level, T_STAR, BETA_POST, False, M, seed, n=N, m=M_WIN, E=E)


def cv_rows(cv, K: int, level: float):
    """Their CoVaR row and reverse row. SysRisk_Monitoring takes the moments for BOTH modes from the CoVaR
    row (cv_CoVaR[1, ...]); the reverse row contributes only c_VaR / c_SysRisk."""
    row = data.crit_value_row(cv, "CoVaR", level, level, K, n=N, m=M_WIN, iota=IOTA)
    rev = data.crit_value_row(cv, "CoVaR_Reverse", level, level, K, n=N, m=M_WIN, iota=IOTA).copy()
    rev[MOMENT_COLS] = row[MOMENT_COLS]
    return row, rev


def run_lengths(ind: dict, cv, K: int, level: float, chunk: int = 1000) -> dict:
    """rl_covar [M, K+2] = (VaR, CoVaR_1..K, Joint); rl_rev [M, 2K+1] = (VaR_1..K, CoVaR_1..K, Joint), K > 1."""
    row, rev = cv_rows(cv, K, level)
    M = ind["I_VaR"].shape[0]
    rc, rr = [], []
    for s in range(0, M, chunk):
        sl = slice(s, s + chunk)
        r = monitor_covar(ind["I_VaR"][sl], ind["I_CoVaR"][sl], row, M_WIN)
        rc.append(np.column_stack([r.run_length, r.joint]))
        if K > 1:
            r = monitor_reverse(ind["I_VaR_Rev"][sl], ind["I_CoVaR_Rev"][sl], rev, M_WIN)
            rr.append(np.column_stack([r.run_length, r.joint]))
    out = dict(rl_covar=np.concatenate(rc))
    if K > 1:
        out["rl_rev"] = np.concatenate(rr)
    return out


def get(M: int, seed: int, force: bool = False, simulate: bool = True, chunk: int = 500) -> dict:
    """{(K, level): (arrays, meta)}; loads the persisted files unless missing or force=True."""
    keys = [(K, lv) for K in K_SET for lv in LEVELS]
    if not force and all(sims_io.exists(name(K, lv, M, seed)) for K, lv in keys):
        return {k: sims_io.load_sim(name(*k, M, seed)) for k in keys}
    if not simulate:
        raise FileNotFoundError(f"null sims for M={M}, seed={seed} not found in data/derived/sims/; "
                                "run experiments/02_table1_size.py first")
    p = null_params()
    t0 = time.time()
    print(f"simulating M={M} null paths (K={p.K}, E={p.E}, n={p.n}) ...", flush=True)
    ind_all = dgp.simulate_indicators(M, p, LEVELS, seed, chunk=chunk, verbose=True)
    print(f"  simulated + indicators in {time.time() - t0:.0f}s", flush=True)
    cv = data.load_crit_values()
    out = {}
    for K, lv in keys:
        ind = dgp.subset_K(ind_all[lv], K)
        arrays = dict(ind, **run_lengths(ind, cv, K, lv))
        meta = dict(K=K, alpha=lv, beta=lv, level=lv, t_star=T_STAR, beta_post=BETA_POST, break_in_X=False,
                    M=M, n=N, m=M_WIN, E=E, seed=seed, iota=IOTA, nu=p.nu, rho=p.rho, garch_pre=p.garch_pre,
                    dcc_pre=p.dcc_pre, n_trans=p.n_trans, chunk=chunk,
                    derived_from=None if K == p.K else name(p.K, lv, M, seed),
                    critical_values="df_CritVal_full_20250805.rds (their stored thresholds and moments)",
                    rl_covar_columns=["VaR"] + [f"CoVaR_{k}" for k in range(1, K + 1)] + ["Joint"],
                    rl_rev_columns=([f"VaR_{k}" for k in range(1, K + 1)] + [f"CoVaR_{k}" for k in range(1, K + 1)]
                                    + ["Joint"]) if K > 1 else None)
        sims_io.save_sim(name(K, lv, M, seed), arrays, meta)
        out[(K, lv)] = (arrays, meta)
    print(f"  monitored + saved in {time.time() - t0:.0f}s total", flush=True)
    return out
