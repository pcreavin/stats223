# Replication notes

Reference code: `external/dh_replication` @ `f025ef2` (read-only). Scope: VaR, CoVaR and reverse CoVaR only.

## Milestone A: detectors, null moments, critical values

### Exact cross-checks vs their unchanged R code
| check | result |
|---|---|
| `CoVaR_Detector` (UC + IID, 7 indicator sequences incl. S=0, S=1, S=m windows), `r/xcheck_detectors.R` | max abs diff 5.8e-15 (floating-point only) |
| Steps 4–5 threshold selection (type-7 quantiles, `>=`, `which.min`), K ∈ {1,2,5,10}, with ties, `r/xcheck_cv_selection.R` | identical to 1e-14 |

### Null moments (10^5 windows, m = 250), experiment 01
All 24 moments (mu/sd × UC/IID × VaR/CoVaR × α=β ∈ {0.75, 0.9, 0.95}) agree within 2% relative.
The largest deviation is 0.83% (CoVaR IID mean at 0.95, where MC noise is largest). The full table is `data/derived/exp01_moments_B5000.csv`.

### Critical values (n=1000, m=250, ι=0.1, B=5000), experiment 01
48/48 thresholds (c_VaR, c_SysRisk for CoVaR and reverse; 3 levels × 4 K) lie within 3·sqrt(SE_ours² + SE_theirs²). The SEs are bootstrap SEs of the whole selection procedure (30 resamples). The largest |z| is 2.2 (reverse c_VaR, 0.95, K=5). Examples:

| α=β | K | type | ours c_VaR | theirs | ours c_SysRisk | theirs |
|---|---|---|---|---|---|---|
| 0.95 | 1 | CoVaR | 2.386 | 2.350 | 4.856 | 4.819 |
| 0.95 | 10 | CoVaR | 2.912 | 2.880 | 6.103 | 6.258 |
| 0.90 | 5 | CoVaR | 2.700 | 2.663 | 3.356 | 3.431 |
| 0.95 | 10 | Reverse | 2.995 | 3.008 | 6.497 | 6.584 |

Our c_VaR is slightly above theirs in 10 of 12 CoVaR cases (mean z ≈ 0.8). The K's at one level share one set of simulated sups, so these are effectively 3 correlated draws, not 12. This is consistent with noise.

### Deviations (milestone A)
- RNG: NumPy PCG64 with fixed seeds, not R's Mersenne Twister. Results agree in distribution only.
- We simulate moments and sup samples **once per (α, β)** and reuse them for all K. They re-run `Critical_values` per K, so their moments differ slightly across K rows (spread printed as `their_spread_K`). The estimator is the same in distribution.
- The Gini sum Σ_{i<j}|d_i − d_j| is computed via the sorted identity Σ(2k−S−1)d_(k) in integer arithmetic. This is algebraically identical to `dist()`, and floating point agrees to 1e-15.

## Known issues in their code (replicated where they affect results)
- `simulations/sim_CriticalValues.R` sources the missing `simulations/DGP.R` and `R/PIT_tDist.R`, and `applications/apply_SysRisk_Monitoring.R` sources the missing `simulations/DGP.R` (DGP.R actually lives in `R/`).
- `SysRisk_Monitoring.R` calls `error()` (not an R function) instead of `stop()` for a bad `I_VaR` dimension.
- Threshold selection uses `which.min(|target − ι|)`, i.e. the closest ν, which may give a size slightly *above* ι, rather than the largest ν with target ≤ ι. Replicated.
- The UC detector uses `abs(mean(I) − p)` (two-sided), which differs from the paper's eq. (10). Replicated.
- The IID (Gini) detector ignores the censored duration after the last violation. Replicated.
- The stored CV table labels reverse CoVaR as `"CoVaR_Exposure"`, and their scripts recode it to `"CoVaR_Reverse"`. We do the same in `src/data.py`.

## Targets (part 1): simulation targets from their stored results, experiment 08

Code: `src/their_results.py` (loaders, parquet cache in `data/derived/their_results/`, their aggregation), `experiments/08_targets_sims.py`. Log: `data/derived/exp08_log.txt`.
Aggregation copies `simulations/eval_sim_process_files.R`:
- A run length (RL) is the first window 1..751 with a normalised detector > 1, and Inf if none. So the alarm day is m − 1 + RL.
- Rows where a measure does not apply are NA, and every mean uses na.rm.
- First detection keeps every detector with RL == min(RL) within (i_MC, design). Ties all count, and Joint is always first.
- Figure 2 right is `sim_ParamMisspec_20251006.rds` (t* = 0, x = `DCC_coef_post`).

### Exact checks
| check | result |
|---|---|
| Rates, ARLs and rep counts vs `sim_eval_processed_20251117.rds` (BreakPoint, ParamMisspec, EstWindow; CoVaR and reverse) | PASS: rates and counts identical, ARL max diff 4.8e-13 |
| First-detection rates vs `sim_eval_firstdetection_processed_20251117.rds` | PASS: identical (the file only contains BreakPoint) |
| Size tables `size_accum_rate.txt` (Table 1), `size_accumrate_individual.txt`, `size_individual.txt`, `size_first_individual.txt` (Table 2), `size_first_individual_{CoVaR,RevCoVaR}.txt`: 310 cells at 2 dp | PASS: all match |

Outputs:
- `data/derived/targets_fig2_fig3_fig4.csv`: Table 1, Figs 2–4 rates with binomial MC SEs (max SE 0.0072).
- `data/derived/targets_delays.csv`: false-alarm, post-break and censored shares; delay median, mean and IQR, with bootstrap SEs (B = 500). The delay is alarm_day − t*, counting only alarms after t*.
- Re-plots in `figures/targets_fig{2,3,4}.png`.

### Discrepancies found
1. **Figure 4 uses all six EstWindow files**, not just `_1` and `_2`. `eval_sim_process_files.R` binds `sim_EstWindow_20251107_1..6.rds` with i_MC offsets 0/1000/3000/4000/5000/6000. That gives 4849 runs that did not fail, which matches `sim_replications` in their processed file. We follow their script.
2. **No α = β = 0.75 level is stored.** All simulation files contain only 0.9 and 0.95, and so do Tables 1–2.
3. **Label typo in `eval_sim_AnyDetection.R`.** It filters on `"Reversed CoVaR"`, but the level is `"Reverse CoVaR"`, so the reverse row of `size_accumrate_individual.txt` is blank apart from Joint (reproduced in our check). The same typo drops reverse CoVaR from `power_by_*_CoVaRonly.pdf`. The correct reverse row (any detection, K = 5, α = 0.9, %) is Joint 6.96; VaR_1..5 0.74, 0.98, 0.68, 0.98, 1.10; CoVaR_1..5 0.90, 0.90, 0.68, 0.84, 0.76.
4. **Estimation noise inflates the size in Figure 4.** At Break_point = 1000 (no break), α = 0.9, the Joint rejection rate with E = 1000 is 0.19–0.29 across K and both measures, against ι = 0.1. With the true parameters (E = 5001) it is 0.07–0.09.
