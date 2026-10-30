# Findings: systemic risk surveillance with e-detectors

These are working notes for the STATS 223 project. They cover the paper, the repo, Extension 1 (the oracle), the
concepts behind the comparison, and the plan and progress for Extension 2 (learned e-detectors). Last updated
2026-10-30, on branch `edet-learned`.

---

## 1. The paper: Dimitriadis & Hoga (2026), *Systemic Risk Surveillance* (arXiv:2601.08598)

### Question
Banks and regulators forecast systemic risk every day. A standard backtest is a one-off test, so re-running it daily
piles up false alarms. The paper builds an **online monitor** that checks VaR, CoVaR and reverse CoVaR forecasts every
day and keeps **P(any false alarm over the whole monitoring period) ≤ ι = 10%**. It monitors K banks jointly and
**attributes an alarm to a specific bank**.

### Risk measures
X_t is the market loss (S&P 500 Financials, SPF) and Y_kt is the loss of bank k.
- **VaR_t,β**: the β-quantile of X_t given the past.
- **CoVaR_kt,α|β**: the α-quantile of Y_kt given X_t ≥ VaR_t,β (bank k's exposure to market distress).
- **Reverse CoVaR**: the same with the roles swapped (bank k's contribution to the market).
- CoES and MES are in the appendix. They are not replicated here.

### The key result: misses have a known null distribution
- I_t = 1{X_t > VaR̂_t} is a market VaR miss.
- I_kt = 1{X_t > VaR̂_t, Y_kt > CoVaR̂_kt} is a joint miss.
- **Proposition 1:** if the forecasts are correct, (I_t, I_kt) has the same law as (1{U₁ > β}, 1{U₁ > β, U₂ > α}),
  with U₁ and U₂ iid uniform. So I_t ~ iid Bern(1 − β) and I_kt ~ iid Bern((1 − α)(1 − β)).
- This is exact in finite samples and holds for any data-generating model.
- At α = β = 0.95 the expected rates are 5% market misses (12.5 per 250 days) and 0.25% joint misses (0.6 per 250
  days).
- Remark 1: VaR must be monitored as well. Otherwise a wrong pair (α′, β′) with the same product (1 − α)(1 − β) goes
  undetected.

### The paper's detector
Each detector is computed every day T on the **last m = 250 days** of misses:
- **UC part:** |window miss rate − expected rate|, a two-sided check on how often misses happen.
- **IID part:** the **Gini coefficient of the gaps between misses**. It is 0 for evenly spaced misses, about 0.5 under
  the null, and high when misses cluster. Only high values count. In the code, Gini = 0 when there are fewer than 2
  misses in the window.
- Both parts are z-scored with null moments, which come from simulating 100,000 iid windows. The detector is
  0.5·z_UC + 0.5·z_Gini. The weight a = 0.5 and the window m = 250 are fixed by the authors, not tuned.
- There is one VaR detector and one CoVaR detector per bank. **Joint alarm** is the first day any detector exceeds its
  threshold.

### Thresholds (Algorithm 1)
1. Simulate B = 5000 null histories of n = 1000 days from uniforms.
2. Record the **maximum** of each detector over each history.
3. Take the same upper quantile ν of the VaR and CoVaR maxima, and choose ν so that
   P(VaR alarm) + K·P(CoVaR alarm) − K·P(both) ≈ ι. This is a Bonferroni-type bound with an overlap correction.

At α = β = 0.95 and K = 1 the thresholds are c_VaR ≈ 2.35 and c_CoVaR ≈ 4.82. CoVaR's threshold is higher because
joint misses are rare and lumpy. **Theorem 1** gives size ≤ ι; it is exact for K = 1 and conservative for K > 1.
Estimation error in the forecast model is deliberately excluded (Remark 3).

### Simulations
- **DGP:** DCC-GARCH with t₅ innovations, equicorrelation 0.5, GARCH (ω, α_G, β_G) = (0.1, 0.1, 0.7), and DCC
  (a, b) = (0.1, 0.7).
- **Break (Section 3.4):** at day t*, the **GARCH and DCC persistence jump from 0.7 to β_post** (baseline 0.85). The
  forecasts keep the pre-break parameters.
- **Size, Tables 1–2:** about 6–10%.
- **Power, Figure 2:** higher for earlier breaks and larger β_post. Reverse CoVaR is the most powerful.
- **Figure 3:** the VaR detector usually fires first.
- **Figure 4:** small estimation windows inflate the false-alarm rate by up to about 20%.

### Application
- **Data:** SPF plus BAC, C, GS, JPM, WFC (K = 5), with α = β = 0.95 and estimation sample E = 1500.
- **Four 1000-day windows:** financial crisis, calm, COVID, Trump tariffs.
- **CCC-Gaussian**, a deliberately poor model, fails in three of the four windows through the CoVaR detectors. In the
  financial crisis the failure is driven by Wells Fargo.
- **DCC-t** fails only at the COVID crash, through the VaR detector. The calm window raises no alarms for either model.

### Notes on reading the paper
- **m = 250 is the monitoring window and E is the estimation sample.** They are easy to confuse.
- The printed VaR̂ formula (D̂⁻¹ times the quantile) looks like a typo. The code uses σ̂ times the quantile.

---

## 2. The repo (`pcreavin/stats223`)

### Status
| part | state |
|---|---|
| Python replication: critical values, Tables 1–2, Figures 2–3, 5–6 | done, all checks PASS (`REPLICATION.md`) |
| Extension 1: oracle e-SR detector, K = 1 | done (`EXTENSION.md`) |
| Extension 2: learned e-detectors | in progress (§ 6 below) |

### Replication results
- **Exact checks:** the detectors and threshold selection match the authors' unchanged R functions to about 6e-15.
- **Critical values:** 48/48 within 3σ of theirs; the maximum |z| is 2.2.
- **Size (Tables 1–2):** 14 cells, all within ±1.1pp.
- **Power (Figure 2, Figure 3 shares, delays):** 192/192 cells within 3σ.
- **Application:** detector paths match to 7e-16, and the first alarm dates are identical. The exception is a 4-day
  label shift in COVID windows before July 2019, which is a quirk in their code (labels are taken before 4 SPF NA rows
  are dropped).
- **Issues found in their code:** a "Reversed CoVaR" label typo blanks a Table 2 row; some R files are sourced but
  missing; the ν index differs between the market and the banks.

### Key modules
| file | what |
|---|---|
| `src/detectors.py` | UC and Gini detectors (numba) |
| `src/critical_values.py` | Algorithm 1 |
| `src/dgp.py` | DCC-GARCH-t DGP and pre-break forecast filter; can return the true σ and ρ |
| `src/monitoring.py` | normalised paths, run lengths, Joint detector |
| `src/null_sims.py`, `src/power_sims.py`, `src/sims_io.py` | cached simulations, seed 2026, common random numbers |
| `src/oracle.py` | true band probabilities (Gauss–Legendre quadrature) |
| `src/edetectors.py` | e-SR detectors (oracle plus the learned ones from Extension 2) |

### Ground rules (`ONBOARDING.md`)
- `external/dh_replication` is read-only.
- Reuse the cached simulations rather than re-simulating.
- Experiments print *ours vs theirs* tables with PASS/FAIL.
- Work in milestones.
- No AI attribution lines in commit messages.

---

## 3. Concepts: the paper's detector vs an e-detector

### What the oracle knows that the paper's detector doesn't
| | paper's detector | oracle |
|---|---|---|
| sees daily outcomes (calm / market miss / joint miss) | yes | yes |
| knows the null miss rates | yes | yes |
| knows the **true** daily volatility and correlation | no | **yes** (from the simulation) |

The oracle is told the true chance of a miss each day. In real data nobody knows this, so the oracle is an **upper
bound on performance** and not a usable method.

### Window statistic vs running bet
| | paper's detector | e-detector |
|---|---|---|
| computes | how unusual the last 250 days look | accumulated betting wealth against H₀ |
| memory | fixed 250-day window | all history; SR restarts the bet at every past day |
| earliest alarm | day 250 | day 1 |
| needs an alternative | **no explicit one**: it measures distance from the null (implicit choices are a, m, UC + Gini) | **yes**: the e-value q_t(Z_t)/p(Z_t) needs a bet q_t |
| threshold | simulated for a fixed n | formula γ = n/ι (valid for any horizon), or simulated |
| size | ≈ 10% (calibrated) | ≤ 10% guaranteed, often far below |

- **Validity doesn't depend on the bet:** under H₀ any predictable q gives E[L_t] = 1.
- **Power depends entirely on the bet.** If q = p, the detector never alarms. A wrong-direction bet makes wealth
  shrink.
- **Whether an e-detector picks up clustering depends on the bet.** A constant bet only responds to how often misses
  happen; the SR sum over start days does catch bursts, but it can't tell clustering apart from a higher rate.
  Because H₀ is iid, **any bet built from past data is valid, including bets on "misses follow misses."** That lets an
  e-detector target clustering directly, the same job the Gini does.

---

## 4. Extension 1: the time-varying oracle (K = 1)

### Construction
- **Bands:** Z_t ∈ {0 calm, 1 market miss with the bank within its CoVaR, 2 joint miss}, with null
  p = (β, (1−β)α, (1−β)(1−α)).
- **True band probabilities:** a_t = VaR̂_t/(σ_X,t s) and b_t = CoVaR̂_t/(σ_Y,t s), with s = √((ν−2)/ν). Then
  q_t = (1 − P(T₁ > a_t), P(T₁ > a_t) − P(T₁ > a_t, T₂ > b_t), P(T₁ > a_t, T₂ > b_t)), using the true ρ_t.
- **Statistic:** L_t = q_t(Z_t)/p(Z_t) and M_t = (1 + M_{t−1})·L_t, which equals the sum over k of
  Π_{s=k..t} L_s (the sum over all possible change points). It alarms at γ = n/ι = 10,000.
- **Null before, alternative after happens automatically.** The oracle doesn't know t*. Before the break the true and
  forecast parameters coincide, so q_t = p (to 6e-13). After it, σ_t > σ̂_t, and the bet shifts toward misses.

### Results (α = β = 0.95, M = 2000, rejection by day 1000)
| t* | β_post | theirs (pre-break false alarms) | oracle, matched window |
|---|---|---|---|
| 0 | 0.85 | 1.000 | 1.000 |
| 400 | 0.85 | 0.995 (0.029) | 1.000 |
| 700 | 0.85 | 0.893 (0.073) | 0.973 |
| 800 | 0.85 | 0.632 (0.083) | 0.921 |
| 850 | 0.85 | 0.441 (0.088) | 0.840 |
| 900 | 0.85 | 0.266 (0.095) | 0.707 |
| 950 | 0.85 | 0.141 (0.102) | 0.351 |
| 1000 | (null) | 0.104 | 0 |
| 0 | 0.75 | 0.380 | 0.329 |
| 0 | 0.80 | 0.898 | 0.977 |

### Findings
- **Head start.** Before the break M_t = t exactly, so the oracle needs only log(γ/(1+t*)) of post-break evidence
  instead of log γ. That makes its advantage largest for late breaks, and it is partly structural.
- **Uncalibrated threshold.** The oracle's size is 0. On a small break (β_post = 0.75) it loses to their calibrated
  detector (0.33 vs 0.38). That break carries only about 1.8 nats of total evidence, against log γ ≈ 9.2.
- **Delays** are 40–70 days shorter after mid-sample breaks (e.g. a median of 94 vs 174 at t* = 400).

---

## 5. Ways to break the model

- **Paper's break (used now):** GARCH and DCC persistence go from 0.7 to β_post. This raises the miss rate, the joint
  misses and the clustering all at once. It is needed for comparability, but it can't isolate which detector feature
  does the work.
- **Possible later additions:**
  - **Correlation-only break**, e.g. ρ̄ 0.5 → 0.8: market misses stay at exactly 5% and only joint misses rise. This
    is a pure systemic failure.
  - **Static forecaster:** the correct unconditional quantile while the truth is GARCH. The miss rate is right on
    average but the misses cluster. This is the cleanest test of clustering detection.
  - **Gradual drift**, and **temporary break then recovery**.

---

## 6. Extension 2: learned e-detectors vs the paper (their break, K = 1)

The design follows the Extension 2 plan: M0 setup, M1 bets, M2 calibration, M3 power comparison, M4 write-up.

### Detectors (`src/edetectors.py`)
These use **only the violation indicators** (the same information as the paper's detector). Each bet q_t uses only
Z_1..Z_{t−1}.

| id | name | bet | paper analogue |
|---|---|---|---|
| E0 | oracle | true band probabilities | ceiling |
| E1 | `rate` | average of 12 SR processes; each bets on a fixed inflated miss rate (market ×1.5/2/3, joint ×1.5/2/3/5) | UC part |
| E2 | `plugin` | EWMA band frequencies shrunk toward p: (ñ + κp)/(Ñ + κ), with λ = 0.98 and κ = 50; one-sided (miss bands ≥ p) | UC + some clustering |
| E3 | `cluster` | separate shrunk frequencies for "miss in the last h days" vs not, with h = 1 | Gini part |
| E4 | `combined` | ½ M^rate + ½ M^cluster | 0.5·UC + 0.5·Gini |

An average of SR processes is again an SR-type process with E[M_t] = t under H₀, so E1 and E4 are valid.

**Change from the plan:** the clustering lookback is h ∈ {1, 5}, not {10, 20}. With a 5% miss rate, about 64% of days
have a miss in the previous 20 days, which would remove the contrast between the two states.

### Thresholds
1. **Guaranteed γ_Ville = n/ι = 10,000.** M_t is a nonnegative submartingale with E[M_n] = n, so Doob's inequality
   gives P(max_{t≤n} M_t ≥ γ) ≤ n/γ = ι.
2. **Calibrated γ̂ (the headline, matching how the paper sets its thresholds).** The K = 1 null of Z_t is exactly iid
   categorical(p). Simulate B = 20,000 null paths (seed 11), take each path's maximum of M_t over the alarm window, and
   set γ̂ to the 90% quantile. Report a bootstrap SE.
3. **Out-of-sample check.** Apply γ̂ to the GARCH-based null sims; the size should be 10% ± 3 SE.

Hyperparameters are tuned before evaluation, on a separate seed (7), the scenario t* = 400 and β_post = 0.85, and the
null. They are then frozen.

### Early results (rough: 2000 null paths at α = β = 0.95, not yet the M2 calibration)
| detector | size with γ = 10,000 | γ̂ for 10% (matched) | γ̂ for 10% (day 1) |
|---|---|---|---|
| rate | 5.2% | ≈ 5,200 | ≈ 5,600 |
| plugin | 2.2% | ≈ 2,800 | ≈ 2,900 |
| cluster | 1.8% | ≈ 2,900 | ≈ 2,900 |
| combined | 3.6% | ≈ 4,000 | ≈ 4,200 |

**Finding:** the guaranteed threshold is 2–5× too conservative for realistic bets. Calibration lowers γ to roughly
2,800–5,600, which should recover power, especially for small and late breaks.

### Progress
| milestone | state |
|---|---|
| M0: environment, tests, regenerating sims | **done**: 51/51 tests pass; Table 1 at 8/8 PASS; Exp 11 reproduces `EXTENSION.md` |
| M1: learned bets plus tests | **done** |
| M2: calibration and tuning | **done**: tuned on seed 7 (plug-in and cluster λ = 0.99, κ = 50, h = 1); γ̂ from 20,000 null paths; 96/96 out-of-sample size checks PASS |
| M3: power comparison on the paper's break grid | **done**; full results, tables and figures in **`EXTENSION2_RESULTS.md`** |
| M4: write-up into `EXTENSION.md` | not done (`EXTENSION2_RESULTS.md` is the stand-alone version) |

**Headline (α = β = 0.95; e-detectors may alarm from day 1, theirs from day 250):**
- At the same 10% false-alarm rate, learned e-detectors beat their Joint detector: power 0.67–0.70 vs 0.44 at
  t* = 850, and 0.52 vs 0.38 on the small break (β_post = 0.75).
- On the same paths they detect about 110 days sooner for a break at day 0, and 23–35 days sooner for breaks at days
  400–850.
- They still win when held to the same day-250 start.
- The oracle is not a strict ceiling: its threshold γ = 10,000 is uncalibrated, so it loses on small breaks.
- The simple rate-mixture bet is best; the clustering bet doesn't help on this break.

### Open questions for M3
- Do learned bets close the gap to the oracle once there is no head start?
- Does the clustering bet (E3) add power over the rate bet (E1) on the paper's break?
- With a calibrated γ̂, do the e-detectors beat their detector on small breaks (β_post = 0.75), where the oracle lost?
