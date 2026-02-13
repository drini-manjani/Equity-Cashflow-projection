# Equity Cashflow Model: Detailed Methodology

This document describes the current production logic in this repository (`src/ecf_model`) end to end: data contract, fitting, recalibration, simulation, scenario generation, and neutral tuning.

## 1. Scope and Design

The engine is a quarter-based Monte Carlo model for private-market fund cashflows and NAV.

Each quarter, for each fund, it models:

- Drawdown event probability and drawdown size ratio
- Repayment event probability and repayment size ratio
- Recallable event probability and recallable ratio conditional on repayment
- NAV growth/shrinkage (`omega`) linked to MSCI and lifecycle state

The model combines:

- Hierarchical empirical fitting (strategy, grade, age, size buckets)
- Holdout-based recalibration and shrinkage
- Lifecycle/end-of-life behavior controls
- One-factor Gaussian copula dependence within each fund
- Scenario-based MSCI path generation (neutral, bullish, bearish)

## 2. Data Contract and Canonicalization

### Required canonical columns

The core required schema is defined in `src/ecf_model/schema.py`:

- `FundID`
- `Adj Strategy`
- `Grade`
- `Year`
- `Quarter`
- `First Closing Date`
- `Planned End Date`
- `Commitment EUR`
- `Adj Drawdown EUR`
- `Adj Repayment EUR`
- `Recallable`
- `NAV Adjusted EUR`

The loader (`src/ecf_model/data_io.py`) performs:

- Column canonicalization via `CANONICAL_RENAME`
- Type normalization
- `quarter_end` construction from `Year` and `Quarter`
- Sorting and dropping invalid fund/quarter rows

### Recallable interpretation

`Recallable` can be cumulative or per-quarter flow. The helper `recallable_to_flow`:

- Uses first differences (floored at 0) if the series is non-decreasing
- Otherwise treats values as per-period flow (absolute value)

## 3. Pipeline Stages

Main runtime entry points:

- Fit: `run_fit_pipeline(...)`
- Projection: `run_projection_pipeline(...)`
- Neutral tuning: `run_neutral_tuning_pipeline(...)`
- End-to-end quarter runner: `scripts/run_quarter_pipeline.py`

The run structure is:

- `runs_v2/<run_tag>/calibration/` for model artifacts
- `runs_v2/<run_tag>/projection/` for simulation outputs

## 4. Fit Stage (Historical up to Cutoff)

Fit runs on `hist_to_cutoff = history[quarter_end <= cutoff]`.

### 4.1 Feature engineering

In `prepare_fit_base` (`src/ecf_model/fitting.py`), the model creates:

- Event flags:
  - `draw_event = 1(draw_abs > 0)`
  - `rep_event = 1(rep_abs > 0 and nav_prev > nav_gate)`
  - `rc_event = 1(recall_flow > 0 and rep_abs > 0)`
- Ratios:
  - `draw_ratio = draw_abs / commitment_max`
  - `rep_ratio = rep_abs / nav_prev` (only when `nav_prev > nav_gate`)
  - `rc_ratio_given_rep = recall_flow / rep_abs` (only when repayment > 0)
- NAV residual growth:
  - `omega_obs = nav_curr / (nav_prev + draw - rep) - 1`
  - Clipped to `omega_fit_clip`

### 4.2 Timing probability model

`fit_timing_probs` estimates `p_draw`, `p_rep`, `p_rc` across hierarchical levels:

- `global`
- `strategy`
- `strategy_age`
- `strategy_grade_age`
- `strategy_grade_age_size`

It applies Bayesian-style smoothing to each probability:

```text
p_smooth = (events + prior_strength * p_parent) / (n + prior_strength)
```

with Wilson confidence intervals saved for diagnostics.

### 4.3 Ratio marginals

`fit_ratio_models` fits positive ratios (`draw_ratio`, `rep_ratio`, `rc_ratio_given_rep`) by the same hierarchy.

For each bucket:

- Winsorize at configurable quantiles
- Fit candidate distributions:
  - Lognormal (loc fixed to 0)
  - Gamma (loc fixed to 0)
- Select by lowest AIC
- Fallback to empirical mean when sparse/failed

### 4.4 NAV (`omega`) regression

`fit_omega_models` regresses `omega_obs` on:

- Current MSCI return `msci_ret_q`
- Lagged MSCI return `msci_ret_q_lag1`
- Fund age (`age_y`, `age_y2`)
- Time-to-end (`q_to_end_y`)
- Endgame/post-end indicators (`is_pre_end`, `is_post_end`)

Fitted with robust ridge regression (Tukey bisquare reweighting + L2 penalty), hierarchical by:

- `strategy_grade_age_size`
- `strategy_grade_age`
- `strategy`
- `global`

### 4.5 Fund lifetime model

`compute_fund_end_dates` estimates `fund_end_qe` using:

- Planned end date if available
- Strategy/grade overrun estimates (shrunken medians/means)
- Fallback planned life estimates if planned end is missing
- Hard minimum life floor (`min_life_q = 32`)

`compute_invest_end_by_fund` estimates investment period end from first close plus inferred life of 6-7 years depending on repayment onset.

### 4.6 MSCI model fit

`fit_msci_model` (`src/ecf_model/msci.py`) builds:

- AR(1) return structure (`phi`, clipped to `[-0.75, 0.75]`)
- Regimes from return terciles (`bear/neutral/bull`)
- Regime-specific drift and volatility
- Regime transition matrix with Laplace smoothing

### 4.7 Copula rho calibration

`_calibrate_copula_from_history` computes weighted lag-1 autocorrelation across funds for:

- Any-event indicator
- Total flow ratio

Then blends:

```text
rho_raw = 0.60 * max(rho_event, 0) + 0.40 * max(rho_flow, 0)
rho_calibrated = clip(rho_raw, 0.05, 0.85)
```

Saved to `copula_calibration.json`.

### 4.8 Calibration bucket map (explicit)

| Component | Bucket(s) |
| --- | --- |
| Timing probabilities | `global -> strategy -> strategy_age -> strategy_grade_age -> strategy_grade_age_size` |
| Ratio marginals | `strategy_grade_age_size -> strategy_grade_age -> strategy_age -> strategy -> global` |
| Omega regression | `strategy_grade_age_size -> strategy_grade_age -> strategy -> global` |
| Holdout group deltas | `strategy + grade` |
| Pre-end repayment multipliers | `strategy + pre_end_phase` (`far`, `mid`, `near`) |
| Endgame ramp | `strategy` |
| Post-invest draw multipliers | `strategy` |
| Tail floors/caps/omega drag | `strategy + tail_bucket` (`0-3`, `4-7`, `8-11`, `12+`) |
| Lifecycle multipliers + NAV floor | `strategy + grade + life_phase` |
| Neutral tuning | `strategy + grade` targets + pooled portfolio objective |

Fallbacks used at runtime:

- Timing/ratio: fall back in hierarchy order down to `global`.
- Omega: fall back to `strategy`, then `global`.
- Lifecycle: `strategy+grade+phase -> strategy+ALL+phase -> ALL+ALL+phase`.

## 5. Holdout Recalibration Stage

`calibrate_from_history` (`src/ecf_model/calibration.py`) uses a holdout window (default last 8 quarters) to adjust fit outputs.

### 5.1 Train/holdout split

- Train: quarters before holdout window
- Holdout: final `holdout_quarters` up to cutoff

### 5.2 Global deltas

`_compute_deltas` estimates multiplicative/additive adjustments for:

- Draw event probability
- Repayment event probability
- Repayment ratio
- Draw ratio
- Recallable propensity placeholders
- Omega bump
- Repayment timing shift (in quarters)

Mechanics:

- Significance gating (`alpha`) on each signal
- Shrinkage weight `n / (n + shrink_n)`
- Configurable clipping ranges

### 5.3 Stability attenuation

`_compute_stability_attenuation` performs rolling pseudo-out-of-sample checks and fits attenuation coefficients per metric so unstable signals are damped.

### 5.4 Group-level deltas (strategy x grade)

`_compute_group_deltas` computes local deltas per `(Adj Strategy, Grade)` and shrinks them toward global deltas based on group sample size.

It also calibrates `rc_propensity` by group (fraction of funds historically showing recallables).

### 5.5 Specialized lifecycle tables

Additional tables are fit and saved:

- `tail_repayment_floors.csv`
  - Post-end repayment floors, draw caps, omega drag by tail bucket (`0-3`, `4-7`, `8-11`, `12+` quarters after end)
- `endgame_repayment_ramp.csv`
  - Near-end repayment acceleration (`1-12` quarters to end) vs mid pre-end baseline (`13-24`)
- `lifecycle_phase_multipliers.csv`
  - Multipliers for repay/draw probabilities and ratios, omega bump, and NAV floor ratio across lifecycle phases
- `post_invest_draw_multipliers.csv`
  - Post-investment-period draw attenuation relative to investment period
- `pre_end_repayment_multipliers.csv`
  - Far/mid pre-end attenuation (near phase left neutral to avoid overlap with endgame ramp)

## 6. Simulation Stage

`simulate_portfolio` (`src/ecf_model/simulator.py`) runs Monte Carlo paths by fund and quarter.

### 6.1 State initialization

Per fund initial state from latest snapshot at projection start:

- Commitment, NAV, cumulative draw/repay/recallable, age, investment end, fund end
- Recallable balance:

```text
recall_bal0 = max(recall_cum0 - max(draw_cum0 - commitment0, 0), 0)
```

### 6.2 Copula dependence

Within each simulation path:

- Draw one latent factor per fund: `Z_common ~ N(0,1)`
- For each random draw:

```text
Z = sqrt(rho) * Z_common + sqrt(1-rho) * eps
U = Phi(Z)
```

- Use `U` for event Bernoulli thresholds and inverse-CDF sampling for ratio marginals

`rho` resolution order:

1. User-provided `copula_rho` (if set)
2. `copula_calibration.json`
3. Fallback `0.35`

### 6.3 Quarterly event logic

For each fund-quarter:

- Build base timing probabilities from hierarchical lookup
- Apply global/group delta multipliers
- Apply lifecycle multipliers
- Apply pre-end repayment attenuation
- Apply endgame ramp when `1 <= q_to_end <= 12`
- Apply post-invest draw adjustments
- Apply post-end tail floors/caps

Gates:

- Pre-first-close: no activity, no aging
- Draws disallowed after fund end if `post_end_no_draws = True`
- Repayments gated by NAV threshold (`nav > nav_gate`)

### 6.4 Amount sampling and caps

Draw amount:

- Event from Bernoulli(`p_draw`)
- Ratio sampled from fitted marginal (`draw_ratio`)
- Scaled by multipliers
- Capped by available drawable capital:

```text
draw_cap = max(commitment + recall_bal - draw_cum, 0)
draw_amt = min(draw_cap, ratio * (commitment + recall_bal))
```

Repayment amount:

- Event from Bernoulli(`p_rep`)
- Ratio sampled from `rep_ratio`
- Includes lifecycle/endgame/tail multipliers/floors
- Capped at available NAV after same-quarter draw:

```text
rep_amt = min(rr * nav, nav + draw_amt)
```

Recallable amount:

- Triggered only if repayment occurred
- Sample `rc_ratio_given_rep`
- Enforce cumulative recallables <= cumulative repayments

### 6.5 NAV update

After cashflows:

```text
nav_after_flow = max(nav + draw_amt - rep_amt, 0)
omega = intercept + alpha + b0*msci_now + b1*msci_lag + age terms + time-to-end terms + phase dummies + bumps + noise
nav_next = max(nav_after_flow * (1 + omega), 0)
```

Post-end tiny NAV is hard-zeroed when below 1.

Residual NAV floor logic near/after end-of-life:

- Uses lifecycle `nav_floor_ratio * commitment`
- Applies configurable floor strength by phase
- Pulls NAV upward toward historical residual floor behavior

### 6.6 Outputs

Saved in `projection/sim_outputs/`:

- `sim_portfolio_series.csv`
  - Mean and quantiles by quarter (`draw`, `rep`, `nav`, `recall`, DPI p05/p50/p95)
- `sim_fund_quarterly_mean.csv`
  - Fund-by-quarter means
- `sim_fund_end_summary.csv`
  - Fund terminal NAV means
- `msci_paths.csv`
  - Full MSCI paths by `sim_id`

Also:

- `projection/portfolio_observation_summary.csv` with cumulative metrics, DPI/RVPI/TVPI, rolling IRR
- `projection/projection_run_config.json`

Note: full per-path fund cashflow trajectories are not currently persisted (only aggregated cashflow outputs; MSCI full paths are saved).

## 7. Scenario Mechanics

In `project_msci_paths`:

- `neutral`: drift shift `0.0`
- `bullish`: drift shift `+0.0075` per quarter and transition tilt toward bullish regime
- `bearish`: drift shift `-0.0075` per quarter and transition tilt toward bearish regime

`volatility_scale` scales regime sigma.

## 8. Neutral Tuning Logic

`tune_neutral_calibration_to_history` adjusts calibration for better historical alignment on the target portfolio mix.

### 8.1 Targets

Built from historical data for the strategy/grade mix of the test portfolio.

IRR target modes:

- `pooled` (recommended): pooled cashflow IRR and TVPI
- `mix_median`: weighted median-style group targets

Additional alignment targets:

- Recallables as share of commitment
- Drawdowns as share of commitment for mature funds

### 8.2 Objective

The tuning objective is:

```text
objective = 8.0 * |IRR error| + 0.6 * |TVPI error| + 1.0 * strategy-grade TVPI shape error
```

### 8.3 Iterative updates

Per iteration:

- Simulate neutral portfolio
- Compare predicted vs target by strategy-grade
- Update repayment ratio, repayment probability, omega bump, and optional timing shift
- Apply portfolio-level IRR correction

Post-iteration global alignments:

- Repayment intensity scale search for IRR
- Recallable intensity scale search
- Draw intensity scale search for mature drawdown ratio
- Final repayment-only IRR nudge

Outputs saved in tuned calibration directory:

- `neutral_tuning_targets_by_strategy_grade.csv`
- `neutral_tuning_diagnostics.csv`
- `irr_alignment_search.csv`

## 9. Quarter Runner (`scripts/run_quarter_pipeline.py`)

The cross-platform quarter runner does:

1. Fit base calibration on `anonymized.csv` + `msci.xlsx`
2. Tune neutral calibration on primary portfolio
3. Apply a fixed timing profile overlay (`TIMING_PROFILE_V2`) for PE/VC repayment timing
4. Run primary portfolio projection
5. Run Sentral projection

If portfolio input is a snapshot template, it auto-converts to canonical CSV under `inputs/<cutoff>/`.

## 10. Important Assumptions and Limits

- Quarter-level model; no intra-quarter timing
- Distributional fit limited to lognormal/gamma/empirical mean for ratios
- Copula dependence is within fund, not an explicit cross-fund factor for cashflows
- Cross-fund co-movement primarily comes through shared MSCI scenario paths
- Calibration relies on holdout stability and significance gates; sparse groups shrink heavily
- Full per-path cashflow export is not enabled by default

## 11. Reproducibility

Reproducibility controls:

- `seed` for Monte Carlo and MSCI paths
- `n_sims`
- Fixed calibration artifacts under `runs_v2/<tag>/calibration`
- Projection config snapshot in `projection_run_config.json`

When reusing calibration, keep:

- Same input canonicalization rules
- Same cutoff quarter
- Same scenario and simulation settings
