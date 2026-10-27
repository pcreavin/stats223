# Systemic Risk Surveillance: Python replication

STATS 223 project. This is a Python replication of Dimitriadis & Hoga (2026), *Systemic Risk Surveillance* (arXiv:2601.08598). It covers VaR, CoVaR and reverse CoVaR monitoring only (no CoES/MES). Extensions (e-detectors) come later and are **not** part of this repo yet.

The authors' R replication code (CC0) is a read-only git submodule:

- `external/dh_replication` → https://github.com/TimoDimi/replication_CoVaR_Monitoring @ `f025ef2ea1209548f4547705a99facf658da78a4`

## Setup

```bash
git submodule update --init
/usr/local/bin/python3.13 -m venv .venv          # numba/arch wheels; 3.14 not yet supported
.venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
# R (only for multivariate GARCH fitting + exact cross-checks); R 4.3.1, binaries from CRAN
cd r && Rscript -e 'renv::restore()' && cd ..
```

R scripts are run from the repo root (`Rscript r/<script>.R`). They activate `r/renv` themselves via `r/utils_io.R`. R writes CSVs to `data/derived/`, and Python reads them (no rpy2).

## Layout

| path | content |
|---|---|
| `src/data.py` | loaders for their `.rds` data and stored results |
| `src/detectors.py` | UC + Gini-of-durations detectors, null moments (ports `R/Detectors.R`, `R/mu.R`) |
| `src/critical_values.py` | Algorithm 1 thresholds (port of `R/CriticalValues.R`) |
| `src/their_moments.py` | thresholds from our sups standardised with their stored moments (01b, 01c) |
| `src/monitoring.py` | generic `monitor(paths, thresholds)` + CoVaR / reverse wrappers (port of `R/SysRisk_Monitoring.R`) |
| `r/xcheck_*.R` | exact cross-checks of the Python port against their unchanged R functions |
| `experiments/0X_*.py` | replication scripts; each prints *ours vs theirs* with PASS/FAIL |
| `tests/` | `pytest -q` |

## Running

```bash
.venv/bin/python -m pytest -q
.venv/bin/python experiments/01_moments_and_thresholds.py --B 5000   # ~1 min
.venv/bin/python experiments/01b_threshold_seeds.py                  # seed study of the c_VaR tilt, ~1 min (needs 01)
.venv/bin/python experiments/01c_reduced_table.py                    # their reduced (n, m) table, ~12 min
.venv/bin/python experiments/01d_bootstrap_se_check.py               # bootstrap SE vs across-seed sd, ~35 min (needs 01c)
```

See `REPLICATION_NOTES.md` for every replicated number, deviations and known issues in the reference code.
