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

Our c_VaR is slightly above theirs in 11 of 12 CoVaR cases (mean z ≈ 0.8; an earlier version of these notes said 10, which was a miscount). The seed study below checks whether this is a bias.

### Seed study of the c_VaR tilt (experiment 01b)
Our whole procedure was re-run for 7 seeds (2026 plus 101, 202, ..., 606; seed s uses streams s, s+1, s+2 for the three levels, so seed 2026 reproduces exp01). Files: `data/derived/exp01b_seeds_ours.csv` and `exp01b_seeds_theirmoments.csv`.

| CoVaR c_VaR | 2026 | 101 | 202 | 303 | 404 | 505 | 606 |
|---|---|---|---|---|---|---|---|
| cells (of 12) with ours > theirs | 11 | 11 | 8 | 8 | 8 | 7 | 7 |
| mean diff | +0.035 | +0.039 | +0.006 | +0.027 | +0.023 | +0.006 | +0.012 |

- The 11/12 was partly luck, but the mean difference is positive for all 7 seeds. Averaged over seeds, 8 of 12 cells are above theirs. The mean gap is +0.021, about 1% of a threshold around 2.5.
- Averaging over seeds removes our noise but not the noise in their single table. Pooled over the 12 CoVaR c_VaR cells, z = +1.84 with SE_theirs = bootstrap SE, and +2.42 with SE_theirs = across-seed sd. Reverse c_VaR (9 cells, K>1): +1.74 / +1.89. c_SysRisk tilts the other way and is not significant (CoVaR −1.33 / −0.87; reverse +0.13 / +0.16).
- **About 60% of the tilt is moment Monte Carlo noise.** Our VaR-UC null sd is 0.25–0.5% below theirs at all three levels, which inflates the standardised detector. With our sups standardised by *their* stored per-K moments (8 seeds, `src/their_moments.py`), the mean c_VaR gap falls from +0.021 to +0.008 (7/12 cells above theirs), and the pooled z is +1.77. What is left is consistent with noise.
- In the reduced table (experiment 01c below), our c_VaR is slightly *below* theirs on average (mean −0.005, 45% of cells above). Across both tables there is no consistent direction.

### Reduced critical-value table over (n, m) (experiment 01c)
Their `df_CritVal_reduced_20250805_mn.rds` covers 24 (n, m) pairs × α=β ∈ {0.75, 0.9, 0.95} × K ∈ {1, 2, 5, 10} × {CoVaR, reverse}, with B = 2000. The pairs are n ∈ {500, 1000, 1500} and m ∈ {50, ..., 999}, with m ≤ n, including (500, 500), which has a single window. That makes 576 cells. Its design columns are stored as string categoricals; `data.load_crit_values_reduced()` casts them and recodes the reverse label. We ran seed 2026 with B = 2000 for selection, 100 bootstrap resamples and B = 20,000 fresh null sups for the achieved size. Files: `data/derived/exp01c_reduced.csv` (one row per cell) and `exp01c_reduced_moments.csv`. Figure: `figures/exp01c_tilt_vs_m.png`.

| check | result |
|---|---|
| 1. null moments vs their K=1 row, tol max(2% rel, 4·sd/√1e5) | 572/576 PASS |
| 2. thresholds, tol 3·√2·bootstrap SE (non-discrete cells) | 543/571 PASS |
| 3. achieved size, 99% bands incl. B=2000 selection noise (non-discrete) | 555/571 PASS |
| 3. achieved size (discrete cells, judged by this check only) | 5/5 PASS |
| overall verdict as specified | 536/576 PASS, 40 FAIL |
| mean achieved size over all cells | ours 0.1017, theirs 0.1023 (nominal 0.10), mean paired diff −0.0006 |

**Moments (4 FAILs).** All four are the CoVaR IID sd at α=β=0.95 with m ≤ 100, where systemic violations are rare (p = 0.0025). Three of them are the same m=100 draw, reused for n = 500, 1000, 1500. The IID statistic is 0 unless a window has two or more violations, so it is extremely heavy-tailed (kurtosis 251 at m=50, 66 at m=100). The SE of a 10^5-window sd is therefore ≈ 0.00059 in both cases, but the specified tolerance 4·sd/√1e5 is only 0.0003–0.0006, i.e. 0.5–1 real SE. A 10^7-window reference run (printed by 01c) gives a true sd of 0.02358 at m=50 and 0.04643 at m=100. Ours is +1.3 and +1.8 SE away; their 12 (n, K) rows average 0.02372 and 0.04611. These FAILs are noise measured against too tight a tolerance. They are not a convention difference: their `R/mu.R` simulates exactly what we do. The m=999 moments all pass. The m=50 case also has the largest relative moment difference: the CoVaR IID mean is +13% (0.00184 vs 0.00163), within 4 SE.

**Thresholds (28 FAILs in non-discrete cells, plus 4 in discrete cells).**
- c_VaR behaves as expected: its z-scores have sd 1.07, median |z| 0.70 (0.67 expected), and 6 of 500 cells have |z| > 3.
- The excess is in c_SysRisk (sd(z) 3.3, with heavy tails at α=β=0.95 and m = 50).
- Experiment 01d re-ran our whole procedure for 20 seeds in every group with a CoVaR check-2 FAIL, plus 3 controls, and compared the across-seed sd with the bootstrap SE. Elsewhere the ratio is about 1 (median 1.03–1.06, controls included).
- **At α=β=0.95, m ∈ {50, 100}, K ∈ {1, 2}** (and n=500, m=50, K=5), the across-seed sd of c_SysRisk is **3.7–53× the bootstrap SE**: the true sd is 0.09–0.30, the bootstrap SE 0.004–0.08. There, 99.3% (m=50) of windows have at most one systemic violation. The CoVaR sup distribution is clustered, so within one B=2000 sample the bootstrap quantile stays in one cluster, while a new sample can land in another.
- These 9 cells (18 rows with reverse) hold 17 of the 32 check-2 FAILs. Re-judged with the across-seed sd, all have |z| ≤ 2.0, and all 18 pass check 3. For reverse rows this uses the sd of the CoVaR threshold, which is an approximation.
- The remaining 15 FAILs are marginal (|z| 3.0–4.3), against ~3 expected at 3σ. 11 of them pass check 3. The bootstrap SE is still mildly optimistic at K ≥ 5 (ratios up to ~2).

**Achieved size (16 FAILs in non-discrete cells).** About as many as three correlated 99% conditions per cell produce by chance. All 16 are borderline: 15 exceed their band by at most 17%, one by 38%. 7 are ours off nominal, 5 theirs off nominal, 5 the paired difference. No cell is `SHARED`, i.e. there is no cell where closest-ν selection keeps both procedures away from 0.1. The 1e-9 `>=` tolerance changes the achieved size in 3 of our cells and none of theirs.

**Discrete cells.** The pre-specified criterion (< 50 distinct sup values, or ≥ 0.5% mass at a threshold, or zero bootstrap SE) flags only 5 cells: (1000, 50) and (1500, 50) at 0.95, where atom masses reach 0.9–1.4% and the CoVaR sup takes 266–306 distinct values. All 5 pass check 3. The criterion misses the clustered cells above. A local-spacing diagnostic (`coarse`: the gap in the support around the threshold is wider than its tolerance) flags no cell, because values within a cluster are distinct but close together. The across-seed comparison in 01d is what identifies them.

**Tilt.** Pooled over 504 cells (reverse K=1 excluded as a duplicate of CoVaR K=1), mean diff ours − theirs:

| moments | threshold | share ours > theirs | mean diff | pooled z (Σz/√N) | group t (72 groups) |
|---|---|---|---|---|---|
| ours | c_VaR | 0.45 | −0.0045 | −2.51 | −1.32 |
| ours | c_SysRisk | 0.52 | −0.0167 | −11.8 | −1.52 |
| theirs | c_VaR | 0.45 | −0.0050 | −2.50 | −1.34 |
| theirs | c_SysRisk | 0.54 | +0.0068 | +1.64 | +0.80 |

The 7 cells of an (n, m, level) group share our simulated sups, so their z's are correlated and Σz/√N overstates the evidence. The group t averages z within each group and tests across the 72 groups; it is the honest statistic, and no tilt is significant by it. The c_SysRisk Σz/√N of −11.8 comes from the cells where the bootstrap SE breaks down. The mean c_VaR z stays within ±0.4 across m and across n (figure), so the comparison does not depend on window size.

### CoVaR-threshold granularity at α=β=0.95
At 0.95 the systemic violation probability is 0.0025, so for small m most windows hold 0 or 1 violations and the CoVaR detector takes few distinct values. Consequences:
- In the main table (m=250), K=2 c_SysRisk comes out 0.06–0.16 below theirs in every one of the 7 seeds, and in all 8 seeds with their moments (mean −0.078, across-seed sd only 0.013). Their single draw sits in a different cluster of the sup distribution from ours.
- In the reduced table, the bootstrap SE of c_SysRisk is useless at m ≤ 100 (see above), and five cells are flagged discrete by the specified rule.
- Achieved sizes still agree in all of these cells, which is the comparison that matters for monitoring.

### Deviations (milestone A)
- RNG: NumPy PCG64 with fixed seeds, not R's Mersenne Twister. Results agree in distribution only.
- We simulate moments and sup samples **once per (α, β)** and reuse them for all K. They re-run `Critical_values` per K, so their moments differ slightly across K rows (spread printed as `their_spread_K`). The estimator is the same in distribution.
- Achieved-size checks (01c) use `sup >= c − 1e-9`, not exact `>=`. Their R-computed threshold can sit on an atom whose value we compute a few ulps differently (5.8e-15 in the detector cross-check). Exact `>=` gives different sizes in 3 of our cells and none of theirs.
- The Gini sum Σ_{i<j}|d_i − d_j| is computed via the sorted identity Σ(2k−S−1)d_(k) in integer arithmetic. This is algebraically identical to `dist()`, and floating point agrees to 1e-15.

## Known issues in their code (replicated where they affect results)
- `simulations/sim_CriticalValues.R` sources the missing `simulations/DGP.R` and `R/PIT_tDist.R`, and `applications/apply_SysRisk_Monitoring.R` sources the missing `simulations/DGP.R` (DGP.R actually lives in `R/`).
- `SysRisk_Monitoring.R` calls `error()` (not an R function) instead of `stop()` for a bad `I_VaR` dimension.
- Threshold selection uses `which.min(|target − ι|)`, i.e. the closest ν, which may give a size slightly *above* ι, rather than the largest ν with target ≤ ι. Replicated.
- The UC detector uses `abs(mean(I) − p)` (two-sided), which differs from the paper's eq. (10). Replicated.
- The IID (Gini) detector ignores the censored duration after the last violation. Replicated.
- The stored CV table labels reverse CoVaR as `"CoVaR_Exposure"`, and their scripts recode it to `"CoVaR_Reverse"`. We do the same in `src/data.py`.

## Milestone B: DGP, Tables 1–2 (size under the null)

Reference: `R/DGP.R`, `R/ViolSeq_tDist.R`, `code_UDE/sim_monitoring_function.R`, `code_UDE/sim_szenarios.R`, `simulations/eval_sim_*.R`. Code: `src/dgp.py`, `src/null_sims.py`, `src/sims_io.py`, `src/their_size.py`, `experiments/02_table1_size.py`, `experiments/03_table2_size_reverse.py`.

### DGP settings (from their code)
| item | value |
|---|---|
| series | X (system) + K banks, W_t = D_t ε_t |
| innovations | ε_t = √(ν/χ²_ν)·Z_t, Z_t ~ N(0, (ν−2)/ν·R_t), one χ² per day (`mvtnorm::rmvt`), ν = 5 |
| GARCH(1,1) | (ω, α, β) = (0.1, 0.1, 0.7) pre, (0.1, 0.1, β_post) post |
| DCC | (a, b) = (0.1, 0.7) pre, (0.1, β_post) post; Q̄ = 1 on diag, 0.5 off |
| start values | Y[,1], ε[,1] ~ N(0,1), σ²[,1] = rnorm², Q_1 = Q̄; 100 burn-in steps dropped |
| sample | E + n = 5001 + 1000 days after burn-in; t.ast passed as E + t* |
| break | coefficient at time i is post iff i > 100 + E + t*, i.e. out-of-sample day j > t* |
| break_in_X = FALSE | X's GARCH stays pre-break; the DCC break still acts on all of Q (incl. X–bank correlations) |
| forecasts (E = 5001 ⇒ true parameters) | `DCC.GARCH.t.Forecast` re-filters all 6001 days with the PRE-break parameters from σ²[1,] = rnorm², ε[1,] = Y[1,]/σ[1,], Q_1 = Q̄; days E+1..E+n kept; ν = rep(5, K+2) |
| VaR / CoVaR | VaR = σ_X·qt(β,ν)/√(ν/(ν−2)); CoVaR_k = σ_k·c(ρ_{Xk,t}); reverse: VaR_k = σ_k·qt(β,ν)/√(ν/(ν−2)), RevCoVaR_k = σ_X·c(ρ_{Xk,t}) |
| null for Tables 1–2 | `sim_BreakPoint`, Break_point = 1000 (= n, so no break), break_in_X = FALSE, M = 5000, m = 250, ι = 0.1 |
| monitoring | their stored `df_CritVal_full_20250805.rds`; moments for both modes come from the CoVaR row (`SysRisk_Monitoring` uses `cv_CoVaR[1,]`; the reverse rows hold identical moments anyway) |

### Their stored results are internally consistent
Recomputing from `sim_BreakPoint_20251006.rds` (5000 reps in every null cell) reproduces their `.txt` outputs **exactly**: 14/14 Joint cells (`size_accum_rate.txt`), 44/44 CoVaR first-detector shares and 68/68 reverse shares (`size_first_individual_{CoVaR,RevCoVaR}.txt`, ties counted for every tied detector).

### c(ρ, ν) vs their `CoVaR.true.t`
At 12 points (ρ ∈ [−0.6, 0.95], both levels, different σ's), our tabulated c agrees with their unchanged R function within 1.2e-4 relative (`r/xcheck_covar_true_t.R`). Their value is identical across 5 seeds: in 2-D `pmvt` turned out seed-independent, so the remaining gap is `uniroot`'s default tolerance (≈1.2e-4 absolute). Our c matches an independent `brentq` root to 1e-13, `scipy.stats.multivariate_t` to 1e-5 in probability, and a 2·10⁶-draw Monte Carlo within 4 SE.

### Table 1: Joint rejection rate (%), CoVaR monitoring (M = 5000, seed 2026)
| α=β | K | ours | theirs | diff | ±1.55pp | ±1.1pp |
|---|---|---|---|---|---|---|
| 0.90 | 1 | 9.90 | 10.00 | −0.10 | PASS | yes |
| 0.90 | 2 | 9.38 | 10.28 | −0.90 | PASS | yes |
| 0.90 | 5 | 7.60 | 7.58 | +0.02 | PASS | yes |
| 0.90 | 10 | 8.14 | 8.36 | −0.22 | PASS | yes |
| 0.95 | 1 | 9.96 | 9.62 | +0.34 | PASS | yes |
| 0.95 | 2 | 8.72 | 9.26 | −0.54 | PASS | yes |
| 0.95 | 5 | 8.76 | 8.24 | +0.52 | PASS | yes |
| 0.95 | 10 | 6.30 | 6.38 | −0.08 | PASS | yes |

First-detector shares: 44/44 within 3 SD of theirs (band 3·√(2p(1−p)/M)), largest gap 0.54pp. Full tables: `data/derived/exp02_{joint,first}_M5000_s2026.csv`.

### Table 2: Joint rejection rate (%), reverse CoVaR monitoring
| α=β | K | ours | theirs | diff | ±1.55pp | ±1.1pp |
|---|---|---|---|---|---|---|
| 0.90 | 2 | 9.50 | 9.32 | +0.18 | PASS | yes |
| 0.90 | 5 | 7.70 | 6.96 | +0.74 | PASS | yes |
| 0.90 | 10 | 8.04 | 7.70 | +0.34 | PASS | yes |
| 0.95 | 2 | 9.28 | 9.64 | −0.36 | PASS | yes |
| 0.95 | 5 | 8.62 | 8.12 | +0.50 | PASS | yes |
| 0.95 | 10 | 7.04 | 6.70 | +0.34 | PASS | yes |

First-detector shares: 68/68 within 3 SD, largest gap 0.44pp. Our reverse Joint is above theirs in 5/6 cells, but the cells share paths (see the K-subset deviation), so this is not 6 independent draws.

### Sanity checks and diagnostics
- iid indicators (Algorithm 1 null), K = 1, their thresholds: Joint 11.24% (0.90) and 9.80% (0.95) against ι = 10%. Both within ±1.55pp.
- Null miss rates (K = 10, 5·10⁷ obs): I_VaR 0.0999 / 0.0501, I_CoVaR 0.01000 / 0.00249, I_VaR_Rev 0.1000 / 0.0500, I_CoVaR_Rev 0.0101 / 0.0025 (nominal 0.10/0.05 and 0.01/0.0025). There are no CoVaR violations without a VaR violation.
- Joint < ι for K ≥ 5 is expected. The bank CoVaR indicators share the system's VaR exceedance and are correlated across banks (mean pairwise corr 0.35 / 0.31; sup-detectors 0.26 / 0.23). The threshold calibration K·P(C) − K·P(V∧C) treats the K CoVaR detectors as disjoint events, so it is conservative under positive dependence.

### Persisted simulations
`data/derived/sims/K{K}_a{lv}_b{lv}_t1000_bp0.85_bx0_M5000_n1000_m250_E5001_s2026.{npz,json}` for K ∈ {1,2,5,10} × α=β ∈ {0.9, 0.95}, 24 MB in total. Each holds `I_VaR [M,n]`, `I_CoVaR`, `I_VaR_Rev`, `I_CoVaR_Rev [M,K,n]` (int8), `rl_covar [M,K+2]` (VaR, CoVaR_1..K, Joint) and `rl_rev [M,2K+1]` (K>1). The JSON sidecar has every setting, the source simulation for K-subsets and the git commit. Load with `src.sims_io.load_sim(name)`. Experiment 03 re-derives the stored reverse run lengths from the stored indicators exactly. Runtime: about 2.5 min to simulate and about 6 min to monitor and save.

### Deviations (milestone B)
- **RNG streams**: NumPy PCG64 via `SeedSequence(2026).spawn` (chunks of 500 paths), not R's `set.seed(i_MC)` with sequential draws. Results agree in distribution only. Their draw order inside a replication (simulation, then filter start values) is not reproduced.
- **K-subset optimisation**: we simulate K = 10 once and use banks 1..K for K ∈ {1, 2, 5}. This is exact in law. GARCH and scalar DCC recursions are elementwise. A sub-vector of a multivariate t (shared χ²) is multivariate t with the sub-correlation. The pairwise (X, k) forecasts don't involve the other banks. With the same draws the first K series even coincide exactly, because Z = L·e with L lower-triangular (tested). As a result our cells are correlated across K, whereas theirs are separate runs.
- **One null simulation** serves both levels and both monitoring modes, as in their code. It also serves both break_in_X values: at t* = n they are the same law, and their table uses FALSE only.
- **CoVaR**: tabulated c(ρ, ν) (4001-point ρ grid on [−0.9995, 0.9995], cubic spline, 1-D quadrature with the conditional t_{ν+1}, safeguarded Newton) instead of `pmvt` plus `uniroot` per day. The difference is ≤ 1.2e-4 relative (above). Tables are cached in `data/derived/covar_tables/` per (α, β, ν) and will be reused for the application's ν values.

### Known issues in their code (milestone B)
- `DGP.R` builds the coefficient matrices with m + 1 columns (one too many); `suppressWarnings` hides R's length warning. This is harmless because only columns 1..m are used.
- `break_in_X = FALSE` switches off X's GARCH break only; X's correlations with the banks still break through the DCC.
- `nu = rep(nu_sim, K+2)` has one entry too many (harmless).
- `CoVaR.true.t` computes its inner VaR with the ν it is passed (nu[k+1] for CoVaR), while `I_VaR` uses nu[1]. This is harmless when all ν are equal and is reproduced (`dgp.violation_sequences`).
- `sim_monitoring_function.R` uses `.errorhandling = "remove"`, so failed replications would silently drop out. None did in the null cells (5000/5000).

## Targets (part 1): simulation targets from their stored results, experiment 08

Code: `src/their_results.py` (loaders, parquet cache in `data/derived/their_results/`, their aggregation), `experiments/08_targets_sims.py`. Log: `data/derived/exp08_log.txt` (not tracked; written with `| tee`). Targets are tracked in `data/targets/`.
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
- `data/targets/targets_fig2_fig3_fig4.csv`: Table 1, Figs 2–4 rates with binomial MC SEs (max SE 0.0072).
- `data/targets/targets_delays.csv`: false-alarm, post-break and censored shares; delay median, mean and IQR, with bootstrap SEs (B = 500). The delay is alarm_day − t*, counting only alarms after t*.
- Re-plots in `figures/targets_fig{2,3,4}.png`.

### Discrepancies found
1. **Figure 4 uses all six EstWindow files**, not just `_1` and `_2`. `eval_sim_process_files.R` binds `sim_EstWindow_20251107_1..6.rds` with i_MC offsets 0/1000/3000/4000/5000/6000. That gives 4849 runs that did not fail, which matches `sim_replications` in their processed file. We follow their script.
2. **No α = β = 0.75 level is stored.** All simulation files contain only 0.9 and 0.95, and so do Tables 1–2.
3. **Label typo in `eval_sim_AnyDetection.R`.** It filters on `"Reversed CoVaR"`, but the level is `"Reverse CoVaR"`, so the reverse row of `size_accumrate_individual.txt` is blank apart from Joint (reproduced in our check). The same typo drops reverse CoVaR from `power_by_*_CoVaRonly.pdf`. The correct reverse row (any detection, K = 5, α = 0.9, %) is Joint 6.96; VaR_1..5 0.74, 0.98, 0.68, 0.98, 1.10; CoVaR_1..5 0.90, 0.90, 0.68, 0.84, 0.76.
4. **Estimation noise inflates the size in Figure 4.** At Break_point = 1000 (no break), α = 0.9, the Joint rejection rate with E = 1000 is 0.19–0.29 across K and both measures, against ι = 0.1. With the true parameters (E = 5001) it is 0.07–0.09.

## Milestone D: real-data application (Figures 5–6)

Code: `r/fit_forecasts_application.R` (an adapted copy of `applications/apply_SysRisk_Monitoring.R`) and `experiments/05_application_fig5_fig6.py`. Outputs go to `data/derived/appl/` (git-ignored) and `data/derived/exp05_*.csv`. The log is `data/derived/exp05_log.txt`, and the figures are `figures/fig5_covar.png` and `figures/fig6_revcovar.png`. Scope is CCC_norm and DCC_std × 4 windows at α = β = 0.95. CCC_std and DCC_norm are not replicated (scope), although the R script runs them if they are passed as arguments.

### Changes to their script
These are also listed in the script header. Windows, fit specs, solver settings and the ν extraction are verbatim.
1. Paths are relative to the git root via `r/utils_io.R`.
2. The missing `simulations/DGP.R` source is dropped.
3. tidyverse/copula/patchwork (not in renv) are replaced by dplyr/tidyr/tibble/lubridate.
4. Only α = β = 0.95 is run.
5. `set.seed(1)` is called before each `ViolSeq_tDist`.
6. The loop over windows uses `mclapply`.
7. Plotting and the reshaping are removed.
8. An output block is added: losses, D_t, upper-triangle R_t, all indicators, ν, and their unchanged `SysRisk_Monitoring` paths and run lengths.

Every window has 2500 rows after na.omit, so E = 1500, n = 1000 and m = 250. The out-of-sample periods are FinCrisis 2005-09-23..2009-09-14, CalmTime 2013-01-11..2016-12-29, COVID 2017-03-17..2021-03-12 and TrumpTariff 2021-10-28..2025-10-22.

**Run time:** 35 s for CCC_norm and 29 s for DCC_std (4 windows in parallel each). The fits take 2–4 s and `ViolSeq_tDist` about 20–30 s per window. All fits converged (flag 0).

Fitted ν (their order: SPF, BAC, C, GS, JPM, WFC, then the Joint mshape for DCC; CCC_norm uses 10000 × 6):

| DCC_std | SPF | BAC | C | GS | JPM | WFC | Joint |
|---|---|---|---|---|---|---|---|
| FinCrisis | 10 | 7 | 9 | 11 | 9 | 11 | 8 |
| CalmTime | 6 | 6 | 5 | 5 | 6 | 5 | 5 |
| COVID | 7 | 6 | 5 | 7 | 5 | 6 | 5 |
| TrumpTariff | 6 | 5 | 5 | 5 | 5 | 5 | 5 |

The ν quirk is reproduced. I_VaR and the reverse CoVaR use their `nu[1]`, the SPF univariate shape. Bank k's CoVaR and reverse VaR use `nu[k+1]`. The joint mshape is never used.

### Results
| check | result |
|---|---|
| (a) our monitoring on R's indicators (their stored K=5 thresholds; moments from the CoVaR row in both modes, as in `SysRisk_Monitoring`) vs their unchanged `SysRisk_Monitoring` | 16/16 PASS (2 specs × 4 windows × 2 modes); max abs diff 6.7e-16; run lengths identical |
| (b) our normalised paths vs stored `appl_res_20251024.rds` (aligned by window index) | max abs diff 6.7e-16 in all 16 cells; first-alarm index identical for every detector |
| (c) indicators recomputed in Python from R's losses, D_t, ρ_{X,k} and ν with our c(ρ, ν) tables | 100% agreement, 0 disagreeing days, for all 16 indicator sequences × 8 cells |

The match with `appl_res` is exact, not only within MC noise. With integer ν, 2-D `pmvt` is deterministic (`CoVaR.true.t` is identical under seeds 1 and 2), so the unseeded pmvt calls in their script didn't matter. Our rmgarch/rugarch versions also reproduce their fits.

First alarms (CoVaR mode; date = correct date, label = their date label):

| spec | window | first alarms |
|---|---|---|
| CCC_norm | FinCrisis | CoVaR_BAC and CoVaR_WFC tie on 2007-11-07; CoVaR_C 2008-12-02 |
| CCC_norm | COVID | CoVaR_GS 2018-12-21 (label 2018-12-28); CoVaR_C 2020-02-24; CoVaR_BAC 2020-05-06; VaR 2020-07-28 |
| CCC_norm | TrumpTariff | CoVaR_WFC 2025-07-15 |
| DCC_std | COVID | VaR 2020-03-18 (3 trading days after 2020-03-13) |
| both | CalmTime, and DCC_std FinCrisis/TrumpTariff | none |

Reverse mode: in COVID, both specs fire all five reverse CoVaRs on 2019-01-25 (label 2019-01-31). The full table is `data/derived/exp05_first_alarms.csv`.

### Qualitative checks (e)
| check | ours | theirs |
|---|---|---|
| E1 CCC_norm, crisis windows: the first alarm is a CoVaR detector | PASS | PASS |
| E2(i) CCC_norm FinCrisis: WFC is among the first CoVaR alarms | PASS | PASS |
| E2(ii) CCC_norm FinCrisis: WFC's alarm falls in 2008 | FAIL | FAIL |
| E3 DCC_std: the only alarm is VaR_SPF, in COVID, within 20 days of 2020-03-13 | PASS | PASS |
| E4 CalmTime: no alarms (both specs, both modes) | PASS | PASS |

E2(ii) is a finding about the claim, not a replication failure. WFC is first, but tied with BAC, and both fire on 2007-11-07, before 2008. That holds both in their stored results and in ours.

### Date labels
In COVID, 326 of the 751 window dates in `appl_res` are shifted relative to the data rows: every date up to 2019-06-24 is labelled 4 trading days late (575 of the 1000 out-of-sample days). Their labels come from `tail(data_Assets_wide, n)`, which is taken before na.omit, and the COVID window contains 4 NA rows. The other windows are unaffected. Our CSVs carry both `date` (correct) and `date_theirs`. The figures use the correct dates. The diagnosis belongs to D2 / exp 10.

### Deviations and notes (milestone D)
- Figures 5–6 are re-plots from our Python paths with `src/ggstyle`. They match their `Detectors_GARCH_CoVaR.pdf` / `..._RevCoVaR.pdf` visually, except that the COVID x-axis uses the correct dates.
- Building the c(ρ, ν) table at ν = 10000 emits a harmless overflow warning in the Newton step. Those points fall back to bisection.
