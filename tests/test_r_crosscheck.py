"""Exact cross-checks against their unchanged R code (skipped if Rscript is unavailable)."""
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from src import critical_values as CV, paths
from src.detectors import Moments, covar_detector

pytestmark = pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript not found")
XDIR = paths.DERIVED / "xcheck"


def _run(script):
    subprocess.run(["Rscript", script], cwd=paths.ROOT, check=True, capture_output=True)


def test_detectors_exact_vs_R():
    XDIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    n, m, p = 600, 100, 0.05
    cols = {f"I_{j + 1}": (rng.random(n) < q).astype(int)
            for j, q in enumerate([0.0025, 0.01, 0.05, 0.1, 0.3])}
    edge = np.zeros(n, int); edge[[0, 150, 151, 599]] = 1
    cols["I_6"] = edge
    full = np.zeros(n, int); full[100:250] = 1                 # windows with S == m
    cols["I_7"] = full
    pd.DataFrame(cols).to_csv(XDIR / "indicators.csv", index=False)
    ms = Moments(0.011, 0.0083, 0.443, 0.0796)
    pd.DataFrame([dict(m=m, p=p, mu_UC=ms.mu_UC, sd_UC=ms.sd_UC, mu_IID=ms.mu_IID,
                       sd_IID=ms.sd_IID)]).to_csv(XDIR / "params.csv", index=False, float_format="%.17g")
    _run("r/xcheck_detectors.R")
    r = pd.read_csv(XDIR / "detectors_R.csv")
    for j, I in enumerate(cols.values(), start=1):
        uc, iid = covar_detector(I, m, p, ms)
        np.testing.assert_allclose(uc, r[f"UC_{j}"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(iid, r[f"IID_{j}"], rtol=0, atol=1e-12)


def test_cv_selection_exact_vs_R():
    XDIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(12)
    B = 800
    sv = rng.normal(size=B)
    sc = 0.5 * sv + rng.normal(size=B)
    sc[:5] = sc[5]                                            # some ties
    pd.DataFrame(dict(sup_VaR=sv, sup_CoVaR=sc)).to_csv(XDIR / "sups.csv", index=False, float_format="%.17g")
    cases = pd.DataFrame(dict(K=[1, 2, 5, 10], iota=[0.1] * 4))
    cases.to_csv(XDIR / "cv_params.csv", index=False)
    _run("r/xcheck_cv_selection.R")
    r = pd.read_csv(XDIR / "cv_selection_R.csv")
    for _, row in r.iterrows():
        th = CV.select_thresholds(sv, sc, int(row.K), row.iota)
        for k in ("c_VaR", "c_SysRisk", "c_VaR_rev", "c_SysRisk_rev"):
            assert th[k] == pytest.approx(row[k], abs=1e-14), (row.K, k)
