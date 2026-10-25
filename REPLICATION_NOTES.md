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
