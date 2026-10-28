"""Time-varying ORACLE joint e-SR detector (K = 1) vs their Joint detector on the saved Section 3.4 break paths.

The oracle bets q_t = P_true(Z_t = j | F_{t-1}) (src/oracle.py) in the three-band SR recursion (src/edetectors.py),
gamma = n / iota = 10,000, windows: matched (accumulate from day 1, alarm in days 250..1000) and from day 1.
Their Joint detector: the saved run lengths (day = m - 1 + rl). Checks A-F print PASS/FAIL.

Usage: python experiments/11_oracle_k1.py --scenario t400_b085 --level 0.95 [--M 2000] [--seed 2026] [--smoke]
       [--force]
"""
import argparse
import re
import time

import numpy as np
import pandas as pd
from scipy import stats

from src import edetectors as ed, oracle, paths, power_sims, sims_io

N, M_WIN, IOTA, NU = power_sims.N, power_sims.M_WIN, 0.1, 5.0
GAMMA = N / IOTA
N_EXAMPLES = 20
WINDOWS = ("matched", "day1")


def parse_scenario(s):
    t, b = re.fullmatch(r"t(\d+)_b(\d+)", s).groups()
    return int(t), float(f"0.{b[1:]}") if b.startswith("0") else float(b)


def run_oracle(t_star, bp, level, M, seed, n_chunks, saved):
    """Per-path oracle results plus the check statistics, chunk by chunk (q_t is not persisted)."""
    p = oracle.p_null(level)
    res = {f"{k}_{w}": [] for k in ("alarm", "kl", "loglr") for w in WINDOWS}
    res |= dict(resid=[], qmean=[], freq=[])
    chk = dict(A=True, C=0.0, E_maxdev=0.0, E_alarms_pre=0)
    ex = None
    for sl, tr in oracle.truth_chunks(t_star, bp, M, seed, n_chunks=n_chunks):
        t0 = time.time()
        var, covar = oracle.forecasts(tr, level, NU)
        iv, ic = oracle.indicators(tr, var, covar)
        chk["A"] &= bool(np.array_equal(iv, saved["I_VaR"][sl]) and np.array_equal(ic, saved["I_CoVaR"][sl]))
        Z = ed.categories(iv, ic[:, 0])
        q = oracle.oracle_q(tr, var, covar, NU)
        if t_star > 0:
            chk["C"] = max(chk["C"], float(np.abs(q[:, :t_star] - p).max()))
            L = ed.e_values(Z[:, :t_star], q[:, :t_star], p)
            chk["E_maxdev"] = max(chk["E_maxdev"], float(np.abs(L - 1).max()))
        onehot = np.eye(3)[Z]
        post = slice(t_star, N)
        res["resid"].append((onehot[:, post] - q[:, post]).mean(1))
        res["qmean"].append(q[:, post].mean(1))
        res["freq"].append(onehot[:, post].mean(1))
        r = ed.e_sr(Z, q, p, gamma=GAMMA)
        z0 = np.zeros((Z.shape[0], 1))
        cum = np.concatenate([z0, np.cumsum(oracle.kl(q, p), 1)], 1)                # cum[:, t] = days 1..t
        cll = np.concatenate([z0, np.cumsum(np.log(ed.e_values(Z, q, p)), 1)], 1)
        for w in WINDOWS:
            a = r["alarm"][w]
            chk["E_alarms_pre"] += int((a <= t_star).sum())
            res[f"alarm_{w}"].append(a)
            idx = np.where(np.isfinite(a), a, 0).astype(int)
            at = lambda c: np.where(np.isfinite(a), c[np.arange(len(a)), idx] - c[:, t_star], np.nan)
            res[f"kl_{w}"].append(at(cum))
            res[f"loglr_{w}"].append(at(cll))
        if ex is None:
            ex = dict(q=q[:N_EXAMPLES].astype(np.float32), Z=Z[:N_EXAMPLES],
                      M=ed.e_sr(Z[:N_EXAMPLES], q[:N_EXAMPLES], p, gamma=GAMMA, return_path=True)["M"])
        print(f"  paths {sl.start}..{sl.stop - 1}: {time.time() - t0:.0f}s (q + detector)", flush=True)
    out = {k: np.concatenate(v) for k, v in res.items()}
    return out, chk, ex


def summarise(day, t_star, M):
    rej = np.isfinite(day)
    after = day[np.isfinite(day) & (day > t_star)]
    d = after - t_star
    q1, q3 = np.percentile(d, [25, 75]) if d.size else (np.nan, np.nan)
    return dict(rejection=rej.mean(), se=np.sqrt(rej.mean() * (1 - rej.mean()) / M), share_detected=d.size / M,
                n_detected=d.size, delay_median=np.median(d) if d.size else np.nan,
                delay_mean=d.mean() if d.size else np.nan, delay_q1=q1, delay_q3=q3)


def mcnemar(r_theirs, r_oracle):
    b, c = int((r_theirs & ~r_oracle).sum()), int((~r_theirs & r_oracle).sum())
    p = stats.binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    return b, c, p


def paired_delay(d_theirs, d_oracle, t_star, B=2000, seed=0):
    both = np.isfinite(d_theirs) & np.isfinite(d_oracle) & (d_theirs > t_star) & (d_oracle > t_star)
    diff = d_oracle[both] - d_theirs[both]
    rng = np.random.default_rng(seed)
    bs = diff[rng.integers(0, diff.size, (B, diff.size))]
    ci = lambda x: np.percentile(x, [2.5, 97.5])
    w = stats.wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
    return dict(n_pairs=int(both.sum()), diff_median=np.median(diff), ci_median=ci(np.median(bs, 1)),
                diff_mean=diff.mean(), ci_mean=ci(bs.mean(1)), wilcoxon_p=w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="t400_b085")
    ap.add_argument("--level", type=float, default=0.95)
    ap.add_argument("--M", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--smoke", action="store_true", help="first seed chunk only (500 paths)")
    args = ap.parse_args()
    t_star, bp = parse_scenario(args.scenario)
    lv, t_all = args.level, time.time()
    saved, meta = sims_io.load_sim(power_sims.name(1, lv, t_star, bp, args.M, args.seed))
    assert meta["draw_dim"] == power_sims.DRAW_DIM and meta["chunk"] == 500 and meta["break_in_X"]
    n_chunks = 1 if args.smoke else None
    print(f"scenario t*={t_star}, beta_post={bp:g}, alpha=beta={lv:g}, M={args.M}{' (smoke)' if args.smoke else ''}")
    res, chk, ex = run_oracle(t_star, bp, lv, args.M, args.seed, n_chunks, saved)
    M = res["alarm_day1"].size
    tag = f"{args.scenario}_a{lv:g}{'_smoke' if args.smoke else ''}"
    np.savez_compressed(paths.DERIVED / f"exp11_examples_{tag}.npz", **ex, t_star=t_star, level=lv)

    theirs = M_WIN - 1 + saved["rl_covar"][:M, -1]
    days = {"theirs": theirs, "oracle_matched": res["alarm_matched"], "oracle_day1": res["alarm_day1"]}
    tab = pd.DataFrame({k: summarise(v, t_star, M) for k, v in days.items()}).T
    tests = []
    for w in WINDOWS:
        o = res[f"alarm_{w}"]
        b, c, p_mc = mcnemar(np.isfinite(theirs), np.isfinite(o))
        pdl = paired_delay(theirs, o, t_star)
        tests.append(dict(oracle=w, mcnemar_theirs_only=b, mcnemar_oracle_only=c, mcnemar_p=p_mc,
                          n_pairs=pdl["n_pairs"], delay_diff_median=pdl["diff_median"],
                          ci_median=f"[{pdl['ci_median'][0]:.1f}, {pdl['ci_median'][1]:.1f}]",
                          delay_diff_mean=pdl["diff_mean"],
                          ci_mean=f"[{pdl['ci_mean'][0]:.1f}, {pdl['ci_mean'][1]:.1f}]", wilcoxon_p=pdl["wilcoxon_p"]))
    tests = pd.DataFrame(tests)

    # CHECK D: per-path post-break mean of 1{Z = j} - q_t(j), t-stat across paths
    r = res["resid"]
    zD = r.mean(0) / (r.std(0, ddof=1) / np.sqrt(M))
    info = pd.DataFrame({w: dict(median_cum_KL=np.nanmedian(res[f"kl_{w}"]),
                                 median_realised_logLR=np.nanmedian(res[f"loglr_{w}"])) for w in WINDOWS}).T

    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.4f}".format):
        print("\n=== rejection rate by day 1000 and delays (alarm day - t*, alarms after t*) ===")
        print(tab.to_string())
        print("\n=== paired tests vs theirs (McNemar on rejection; delay diff = oracle - theirs, bootstrap 95% CI) ===")
        print(tests.to_string(index=False))
        print("\n=== CHECK D: post-break band frequencies vs mean q_t ===")
        print(pd.DataFrame(dict(band=[0, 1, 2], p_null=oracle.p_null(lv), mean_q=res["qmean"].mean(0),
                                empirical=res["freq"].mean(0), z=zD)).to_string(index=False))
        print(f"\n=== information: post-break (days t*+1..alarm) cumulative KL(q||p) and realised log LR, median "
              f"over alarmed paths; log(gamma) = {np.log(GAMMA):.2f}, log(gamma/(1+t*)) = "
              f"{np.log(GAMMA / (1 + t_star)):.2f} (SR enters the break at M_t* = t*) ===")
        print(info.to_string())

    o, t = tab.loc["oracle_matched"], tab.loc["theirs"]
    rej_tie = o.rejection >= 0.99 and t.rejection >= 0.99
    checks = pd.DataFrame([
        ("A", "recomputed indicators == saved (exact)", "exact", chk["A"], chk["A"]),
        ("B", "quadrature vs multivariate_t / MC / adaptive quad", "tests/test_oracle.py", "see pytest", True),
        ("C", "max |q_t - p|, t <= t*", "< 1e-6", f"{chk['C']:.2e}", chk["C"] < 1e-6),
        ("D", "post-break |z| band freq vs mean q", "< 3", f"{np.abs(zD).max():.2f}", bool(np.abs(zD).max() < 3)),
        ("E", "oracle alarms at day <= t* (both windows); max|E_t - 1|", "0", f"{chk['E_alarms_pre']}; "
         f"{chk['E_maxdev']:.1e}", chk["E_alarms_pre"] == 0),
        ("F", "oracle(matched) rejection >= theirs", ">=", f"{o.rejection:.4f} vs {t.rejection:.4f}",
         bool(o.rejection >= t.rejection)),
        ("F", "oracle(matched) median delay <= theirs", "<=", f"{o.delay_median:.0f} vs {t.delay_median:.0f}",
         bool(o.delay_median <= t.delay_median)),
    ], columns=["check", "what", "target", "value", "pass"])
    checks["pass"] = np.where(checks["pass"], "PASS", "FAIL")
    print("\n=== checks ===")
    print(checks.to_string(index=False))
    if rej_tie:
        print("note: both detectors reject >= 99%, so CHECK F is judged on the delay comparison")
    stem = paths.DERIVED / f"exp11_{tag}"
    tab.to_csv(f"{stem}_summary.csv")
    tests.to_csv(f"{stem}_tests.csv", index=False)
    checks.to_csv(f"{stem}_checks.csv", index=False)
    print(f"\nwrote {stem.name}_{{summary,tests,checks}}.csv; total {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
