# Onboarding (for a coding agent)

STATS 223 project: a Python replication of Dimitriadis & Hoga (2026), *Systemic Risk Surveillance* (arXiv:2601.08598), covering VaR, CoVaR and reverse CoVaR monitoring, plus an extension with e-detectors. Read `README.md` (setup, layout) and `REPLICATION.md` (results, deviations, commands) first.

## Status
| part | state | where |
|---|---|---|
| Replication (critical values, Tables 1–2, Figures 2–3, 5–6) | done, all checks PASS | `paul` = `origin/main` + Figure 2 polish; `REPLICATION.md` |
| Extension 1: time-varying oracle e-SR detector (K = 1) | done, written up | branch `edet-oracle`; `EXTENSION.md` |
| Extension 2+: realistic (non-oracle) e-detectors | next | not started |

New extension work should branch from `edet-oracle`, which contains everything in `paul` plus `src/oracle.py`, `src/edetectors.py`, `experiments/11_oracle_k1.py` and their tests.

## Setup
```bash
git clone --recurse-submodules https://github.com/pcreavin/stats223.git && cd stats223
git checkout edet-oracle
python3.13 -m venv .venv          # 3.14 not supported by numba/arch wheels
.venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
.venv/bin/pytest -q               # R cross-checks skip if Rscript is missing
```
R (4.3.1, `cd r && Rscript -e 'renv::restore()'`) is only needed for the application GARCH fit and the R cross-checks. Run R scripts from the repo root.

## Ground rules
- `external/dh_replication` is the authors' code as a git submodule. **Read-only**: never edit it; call their functions from `r/xcheck_*.R` instead.
- All paths go through `src/paths.py`. Outputs go to `data/derived/` and `figures/` (git-ignored). Figures meant for docs are copied to `docs/img/`.
- Simulations are cached: `src/power_sims.get(t_star, beta_post, M, seed)` and `src/null_sims` persist to `data/derived/sims/`. Reuse them rather than re-simulating. The default seed is 2026 (NumPy PCG64).
- Experiments print *ours vs theirs* tables with PASS/FAIL. Keep that pattern, and smoke-test with small `--M`/`--B` before a full run.
- Work in milestones and stop for review between them. Keep scope tight, with no unrequested refactors.
- Commit messages: short imperative subject. No `Co-Authored-By` or AI attribution lines.

## Key modules
| file | what |
|---|---|
| `src/detectors.py` | UC and Gini detectors (numba) |
| `src/critical_values.py` | Algorithm 1 threshold selection |
| `src/dgp.py` | DCC-GARCH-t DGP; can return true conditional sd/correlation (used by the oracle) |
| `src/monitoring.py` | monitoring loop / Joint detector |
| `src/their_*.py` | loaders for their stored `.rds` results |
| `src/oracle.py` | true band probabilities q_t (Gauss–Legendre quadrature) |
| `src/edetectors.py` | joint three-band e-Shiryaev–Roberts detector |

## Open questions from Extension 1 (see `EXTENSION.md` § Findings)
- The oracle's advantage at late breaks is partly a head start (M_t = t before the break). A fair comparison needs a detector that has to learn q_t.
- The threshold γ = n/ι is uncalibrated (size 0), which costs power for small breaks (β_post = 0.75). Calibrating γ, or a different e-process, is a natural next step.
