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
