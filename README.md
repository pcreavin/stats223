# Systemic Risk Surveillance: replication and extension

STATS 223 project. We replicate Dimitriadis & Hoga (2026), *Systemic Risk Surveillance* (arXiv:2601.08598), in Python, covering their VaR, CoVaR and reverse CoVaR monitoring. We then extend it with e-detectors. The authors' R code (CC0) is included as a read-only submodule in `external/dh_replication`.

**Status:** the replication is complete. See [REPLICATION.md](REPLICATION.md). The e-detector extension is in progress: see [EXTENSION.md](EXTENSION.md) (time-varying oracle, K = 1).

## Setup

```bash
git clone --recurse-submodules https://github.com/pcreavin/stats223.git && cd stats223
python3.13 -m venv .venv                       # numba/arch wheels; 3.14 not yet supported
.venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
cd r && Rscript -e 'renv::restore()' && cd ..  # R 4.3.1; only for the application fit and R cross-checks
```

## Layout

| path | content |
|---|---|
| `src/` | Python port: detectors, critical values, DGP, monitoring, loaders for their stored results |
| `experiments/` | replication scripts; each prints *ours vs theirs* with PASS/FAIL |
| `tests/` | unit tests and exact cross-checks against their R code |
| `r/` | renv project: multivariate GARCH fit for the application, `xcheck_*.R` cross-checks |
| `data/targets/` | numbers extracted from their stored results (Figures 2–6, delays) |
| `docs/img/` | figures used in REPLICATION.md |
| `external/dh_replication` | their replication code (submodule, read-only) |

Outputs go to `data/derived/` and `figures/`. Both are git-ignored.

## Running

```bash
.venv/bin/pytest -q                                                   # R cross-checks skip if Rscript is missing
.venv/bin/python experiments/01_moments_and_thresholds.py --B 500    # quick smoke test
```

Run R scripts from the repo root (`Rscript r/<script>.R`). The command for every result is in [REPLICATION.md § 8](REPLICATION.md#8-how-to-reproduce).
