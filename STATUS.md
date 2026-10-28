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
| T3 | Figure 2 targets (power by break point; by misspecification) | done | exp 08 | `data/targets/targets_fig2_fig3_fig4.csv`, `figures/targets_fig2.png` |
| T4 | Figure 3 targets (first detectors by break point, K=5) | done | exp 08 | same CSV, `figures/targets_fig3.png` |
| T5 | Figure 4 targets (estimation effect, E=1000) | done | exp 08 | same CSV, `figures/targets_fig4.png`. Uses all 6 EstWindow files (4849 reps). Null size 0.19–0.29 vs ι=0.1 |
| T6 | Detection delays (not in the paper) | done | exp 08 | `data/targets/targets_delays.csv` (bootstrap SEs, B=500) |
| **Milestone C: our simulations vs the Figure 2–4 targets** |||||
| C1 | Figure 2 from our simulations (break point t* ∈ {0, 700, 800, 1000}; misspecification β_post grid) | pending | exp 04 | — |
| C2 | Figure 3 from our simulations: same break simulations as Figure 2, recording which detector fires first | pending | exp 04 | — |
| C3 | Figure 4 from our simulations (estimation effect, E=1000) | optional | needs GARCH re-estimation in every replication | — |
| **Application (Figures 5–6)** |||||
| D1 | Application targets from `appl_res` (first alarms, event offsets, claims) | pending | exp 09 (targets plan) | — |
| D2 | Date check: are `appl_res` date labels shifted by NA rows? | pending | exp 10 (targets plan) | — |
| D3 | Our application fit and monitoring, Figures 5–6 (CCC_norm, DCC_std × 4 windows, α=β=0.95) | done | `r/fit_forecasts_application.R` + exp 05 | vs their unchanged `SysRisk_Monitoring` 16/16 PASS (max diff 6.7e-16); vs stored `appl_res` max diff 6.7e-16, first alarms identical 16/16; recomputed indicators 100% agreement; qualitative checks E1, E2(i), E3, E4 PASS. E2(ii) "WFC first in 2008" FAILs for ours and theirs alike: BAC/WFC tie on 2007-11-07 (a finding about the claim) |
| D3b | CCC_std and DCC_norm application fits | not replicated (scope) | — | — |

## Persisted simulations (input for the e-detector comparison)
Milestone B's null simulations are saved, and the e-detectors will run on exactly these paths.

- **Location:** `data/derived/sims/` (git-ignored, local only, 21 MB). They can be regenerated deterministically with `python experiments/02_table1_size.py --M 5000` (seed 2026, about 8.5 min).
- **Format:** each configuration has `{name}.npz` (compressed arrays) and `{name}.json` (every setting, the source simulation of K-subsets, the git commit and the creation time). Load with `src.sims_io.load_sim(name)` → `(arrays, meta)`, and list with `src.sims_io.list_sims(**filters)`.
- **Configurations (8):** K ∈ {1, 2, 5, 10} × α=β ∈ {0.9, 0.95}, all at the null t* = n = 1000 (no break), break_in_X = FALSE, M = 5000, n = 1000, m = 250, E = 5001 (true parameters), ν = 5, seed 2026. Name pattern: `K{K}_a{lv}_b{lv}_t1000_bp0.85_bx0_M5000_n1000_m250_E5001_s2026`. K = 1, 2, 5 are the first K banks of the K = 10 simulation, which is exact in law.

| array | shape | dtype | content |
|---|---|---|---|
| `I_VaR` | [M, n] | int8 | system VaR violations |
| `I_CoVaR` | [M, K, n] | int8 | CoVaR violations per bank |
| `I_VaR_Rev` | [M, K, n] | int8 | bank VaR violations (reverse mode) |
| `I_CoVaR_Rev` | [M, K, n] | int8 | reverse CoVaR violations per bank |
| `rl_covar` | [M, K+2] | float64 | run lengths of VaR, CoVaR_1..K and Joint (inf = no alarm) |
| `rl_rev` | [M, 2K+1] | float64 | run lengths of VaR_1..K, CoVaR_1..K and Joint (K > 1 only) |

The break and misspecification simulations for Figures 2–3 don't exist yet. Milestone C (exp 04) will save them in the same format via `sims_io.save_sim`.

## Merge verification (2026-10-27)
| check | result |
|---|---|
| `requirements.txt` covers every import of all branches | yes (numba; pyarrow added by `targets`); reinstalled, `pip install -e .` |
| `pytest -q` | 42 passed, including the R cross-checks |
| exp 01 `--B 500` | 24/24 + 48/48 PASS; outputs byte-identical to pre-merge `paul` |
| exp 02 / 03 `--M 500` | 7/8 and 3/6 PASS. At M=500 the ±1.55pp band (sized for M=5000) is too tight. Outputs byte-identical to pre-merge `milestone-b`, sims equal array for array |
| exp 08 `--n-boot 50` | all PASS; PASS/FAIL lines identical to the committed log; targets CSV and figures byte-identical; delays differ only in bootstrap-SE columns |
