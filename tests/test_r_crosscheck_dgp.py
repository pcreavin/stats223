"""Cross-check of c(rho, nu) against their unchanged CoVaR.true.t (skipped if Rscript is unavailable)."""
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from src import dgp, paths

pytestmark = pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript not found")
XDIR = paths.DERIVED / "xcheck"


def test_covar_true_t_vs_R():
    XDIR.mkdir(parents=True, exist_ok=True)
    pts = pd.DataFrame([dict(rho=r, s1=s1, s2=s2, nu=5.0, level=lv)
                        for r, s1, s2 in [(-0.6, 1.0, 1.0), (0.0, 0.7, 1.3), (0.3, 1.0, 0.5),
                                          (0.5, 1.0, 1.0), (0.8, 2.0, 0.9), (0.95, 0.6, 1.1)]
                        for lv in (0.9, 0.95)])
    pts.to_csv(XDIR / "covar_points.csv", index=False, float_format="%.17g")
    subprocess.run(["Rscript", "r/xcheck_covar_true_t.R"], cwd=paths.ROOT, check=True, capture_output=True)
    r = pd.read_csv(XDIR / "covar_R.csv")
    g = r.groupby("point").covar
    theirs, spread = g.mean().to_numpy(), (g.max() - g.min()).to_numpy()
    ours = np.array([p.s2 * dgp._c_of(np.array([p.rho]), p.level, p.level, p.nu)[0] for p in pts.itertuples()])
    rel = np.abs(ours / theirs - 1)
    # in 2-D pmvt turned out seed-independent; the remaining gap is uniroot's tolerance (~1.2e-4 absolute)
    assert spread.max() < 1e-12
    assert rel.max() < 1e-3, rel
