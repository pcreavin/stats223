"""Extension 2, M2: tune and calibrate the learned e-detectors (K = 1).

--tune   hyperparameter grid for `plugin` and `cluster`, scored on SEPARATE paths (seed 7, M = 500, t* = 400,
         beta_post = 0.85, alpha = beta = 0.95) after calibrating each candidate on its own null (B = 5000, seed 7).
         Score: mean alarm day after the break, with no alarm or a pre-break alarm counted as day n + 1 (lower is
         better). The winner is then frozen by hand in src/edetectors.HP.
default  gamma_hat for every detector, level and window (B = 20,000 iid null paths, seed 11), Ville size, and the
         out-of-sample size check on the DGP null paths (PASS if within 3 binomial SE of iota).

Usage: python experiments/12a_calibrate.py --tune
       python experiments/12a_calibrate.py [--B 20000] [--force]
"""
import argparse
import itertools
import time

import numpy as np
import pandas as pd

from src import ecalibrate as ec, edetectors as ed, null_sims, oracle, paths, power_sims, sims_io

N, IOTA = ec.N, ec.IOTA
TUNE = dict(seed=7, M=500, B=5000, t_star=400, beta_post=0.85, level=0.95)
GRID = dict(plugin=[dict(lam=l, kappa=k) for l, k in itertools.product((0.97, 0.98, 0.99), (50.0, 200.0))],
            cluster=[dict(h=h, lam=l, kappa=k) for h, l, k in itertools.product((1, 5), (0.97, 0.98, 0.99),
                                                                               (50.0, 200.0))])


def Z_of(arrays):
    return ed.categories(arrays["I_VaR"], arrays["I_CoVaR"][:, 0])


def tune():
    s = TUNE
    sims = power_sims.get(s["t_star"], s["beta_post"], s["M"], s["seed"])
    Z = Z_of(sims[(1, s["level"])][0])
    p = oracle.p_null(s["level"])
    rows = []
    for name, cands in GRID.items():
        for hp in cands:
            t0 = time.time()
            g = ec.gamma_hat(ec.null_max(name, s["level"], s["B"], s["seed"], {name: hp})["matched"])
            day = ed.alarms(ed.learned_path(name, Z, p, {name: hp}), g)["matched"]
            ok = np.isfinite(day) & (day > s["t_star"])
            score = np.where(ok, day, N + 1).mean()
            rows.append(dict(detector=name, **hp, gamma_hat=g, detected=ok.mean(),
                             median_delay=np.median(day[ok] - s["t_star"]) if ok.any() else np.nan, score=score))
            print(f"  {name} {hp}: {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(df.to_string(index=False))
        print("\nbest per detector (lowest score):")
        print(df.loc[df.groupby("detector").score.idxmin()].to_string(index=False))
    out = paths.DERIVED / "exp12_tuning.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out.relative_to(paths.ROOT)}; now freeze the winners in src/edetectors.HP")


def null_datasets(level):
    """Out-of-sample null paths from the DGP (correct forecasts): {label: Z}."""
    a, _ = sims_io.load_sim(null_sims.name(1, level, 5000, 2026))
    out = {"Table 1 null (M=5000)": Z_of(a)}
    for t, bp, lab in ((1000, 0.85, "t*=1000 (M=2000)"), (0, 0.70, "beta_post=0.70 (M=2000)")):
        out[lab] = Z_of(sims_io.load_sim(power_sims.name(1, level, t, bp, 2000, 2026))[0])
    return out


def calibrate(B, force):
    t0 = time.time()
    cal = ec.calibrate(B=B, force=force)
    print(f"calibration (B={B}, seed={ec.CAL_SEED}) ready in {time.time() - t0:.0f}s")
    with pd.option_context("display.width", 200, "display.float_format", "{:.4g}".format):
        print(cal.pivot_table(index=["level", "detector"], columns="window",
                              values=["gamma_hat", "se", "size_ville"]).to_string())
    rows = []
    for lv in (0.9, 0.95):
        p = oracle.p_null(lv)
        for lab, Z in null_datasets(lv).items():
            for name in ed.LEARNED:
                M = ed.learned_path(name, Z, p)
                for w in ed.WINDOWS:
                    for kind, g in (("calibrated", ec.lookup(cal, name, lv, w)), ("Ville", N / IOTA)):
                        size = float(np.isfinite(ed.alarms(M, g, {w: ed.WINDOWS[w]})[w]).mean())
                        se = np.sqrt(IOTA * (1 - IOTA) / Z.shape[0])
                        ok = abs(size - IOTA) < 3 * se if kind == "calibrated" else size <= IOTA + 3 * se
                        rows.append(dict(level=lv, data=lab, detector=name, window=w, threshold=kind, size=size,
                                         z=(size - IOTA) / se, result="PASS" if ok else "FAIL"))
    df = pd.DataFrame(rows)
    out = paths.DERIVED / f"exp12_size_B{B}.csv"
    df.to_csv(out, index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 500, "display.float_format", "{:.4f}".format):
        print("\n=== out-of-sample size on DGP null paths (calibrated: |z| < 3; Ville: size <= iota + 3 SE) ===")
        print(df.pivot_table(index=["level", "data", "detector"], columns=["threshold", "window"], values="size")
              .to_string())
    bad = df[df.result == "FAIL"]
    print(f"\n{len(df) - len(bad)}/{len(df)} size checks PASS")
    if len(bad):
        print(bad.to_string(index=False))
    print(f"wrote {out.relative_to(paths.ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--B", type=int, default=ec.CAL_B)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    tune() if a.tune else calibrate(a.B, a.force)


if __name__ == "__main__":
    main()
