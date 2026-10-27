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
