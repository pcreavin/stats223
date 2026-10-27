# Replication status

Dimitriadis & Hoga (2026), VaR / CoVaR / reverse CoVaR monitoring. Branch `paul` after merging `milestone-b` and `targets` (2026-10-27). Details and every number are in `REPLICATION_NOTES.md`.

| # | Target | Status | Checked by | Result |
|---|---|---|---|---|
| **Milestone A: detectors and critical values** |||||
| A1 | `CoVaR_Detector` (UC + IID) vs their R code | done | `tests/test_r_crosscheck.py` (`r/xcheck_detectors.R`) | identical, max abs diff 5.8e-15 |
| A2 | Threshold selection (Steps 4–5) vs their R code | done | `tests/test_r_crosscheck.py` (`r/xcheck_cv_selection.R`) | identical to 1e-14 |
| A3 | Null moments, main CV table (n=1000, m=250) | done | exp 01 | 24/24 PASS, worst 0.83% |
| A4 | Critical values, main CV table (B=5000) | done | exp 01, 01b | 48/48 PASS. c_VaR tilt (+0.021) is ~60% moment MC noise (01b) |
| A5 | Reduced CV table, 24 (n, m) pairs (B=2000) | done | exp 01c, 01d | 536/576 PASS under the specified rule. All FAILs explained: moment tolerance too tight for a heavy-tailed sd; bootstrap SE breaks down at α=β=0.95, m ≤ 100 (01d); achieved sizes agree |
| **Milestone B: Tables 1–2 (size under the null)** |||||
| B1 | c(ρ, ν) vs their `CoVaR.true.t` | done | `tests/test_r_crosscheck_dgp.py` (`r/xcheck_covar_true_t.R`) | within 1.2e-4 relative (their `uniroot` tolerance) |
| B2 | Table 1, CoVaR Joint size | done | exp 02 | 8/8 PASS (±1.55pp), all within ±1.1pp; first-detector shares 44/44; iid sanity 2/2 |
| B3 | Table 2, reverse CoVaR Joint size | done | exp 03 | 6/6 PASS (±1.55pp), all within ±1.1pp; first-detector shares 68/68 |
| **Targets from their stored results** |||||
| T1 | Our aggregation vs their processed files | done | exp 08 | rates and rep counts identical, ARL max diff 4.8e-13 |
| T2 | Their size tables (`simulations/output/size_*.txt`) | done | exp 08 (also exp 02/03 for Table 1/2) | 310/310 cells match at 2 dp |
| T3 | Figure 2 targets (power by break point; by misspecification) | done | exp 08 | `data/derived/targets_fig2_fig3_fig4.csv`, `figures/targets_fig2.png` |
| T4 | Figure 3 targets (first detectors by break point, K=5) | done | exp 08 | same CSV, `figures/targets_fig3.png` |
| T5 | Figure 4 targets (estimation effect, E=1000) | done | exp 08 | same CSV, `figures/targets_fig4.png`. Uses all 6 EstWindow files (4849 reps). Null size 0.19–0.29 vs ι=0.1 |
| T6 | Detection delays (not in the paper) | done | exp 08 | `data/derived/targets_delays.csv` (bootstrap SEs, B=500) |
| **Our simulations vs the Figure 2–4 targets** |||||
| C1 | Figure 2 points from our DGP (t* ∈ {0, 700, 800, 1000}; β_post grid) | pending | exp 04 (Milestone C) | — |
| C2 | Figure 3 first-detector shares from our DGP | pending | not yet planned; can reuse the exp 04 simulations | — |
| C3 | Figure 4 estimation effect from our DGP | pending | not in the plan: needs GARCH re-estimation with E=1000 | — |
| **Application (Figures 5–6)** |||||
| D1 | Application targets from `appl_res` (first alarms, event offsets, claims) | pending | exp 09 (targets plan) | — |
| D2 | Date check: are `appl_res` date labels shifted by NA rows? | pending | exp 10 (targets plan) | — |
| D3 | Our application fit and monitoring, Figures 5–6 | pending | `r/fit_forecasts_application.R` + exp 05 (Milestone D) | — |

## Merge verification (2026-10-27)
| check | result |
|---|---|
| `requirements.txt` covers every import of all branches | yes (numba; pyarrow added by `targets`); reinstalled, `pip install -e .` |
| `pytest -q` | 42 passed, including the R cross-checks |
| exp 01 `--B 500` | 24/24 + 48/48 PASS; outputs byte-identical to pre-merge `paul` |
| exp 02 / 03 `--M 500` | 7/8 and 3/6 PASS. At M=500 the ±1.55pp band (sized for M=5000) is too tight. Outputs byte-identical to pre-merge `milestone-b`, sims equal array for array |
| exp 08 `--n-boot 50` | all PASS; PASS/FAIL lines identical to the committed log; targets CSV and figures byte-identical; delays differ only in bootstrap-SE columns |
