# Equity Cashflow Model Deep Dive

This document is the full implementation-level explanation of the model in this repository.

It is intentionally long and detailed. The goal is that a quant, data scientist, or model reviewer can read this and understand:

- What each part of the model does
- How each part is calibrated from historical data
- Where business overlays are applied
- How the final simulation is produced quarter by quarter

This document reflects the current code in:

- `src/ecf_model/`
- `scripts/run_quarter_pipeline.py`

---

## 1. Purpose and Design Philosophy

The model projects private-market fund cashflows and NAV in quarterly steps.

For each fund and quarter it models:

- Drawdowns
- Repayments
- Recallables
- NAV evolution

It is a **semi-parametric hybrid**:

- Empirical and hierarchical where data is rich
- Parametric where useful (ratio distributions, NAV regression, MSCI process)
- Shrinkage and clipping for stability
- Lifecycle overlays to reproduce observed end-of-life behavior

The business objective is not only point prediction. It is realistic portfolio-level path behavior that is aligned with historical evidence and operational assumptions.

---

## 2. End-to-End Pipeline

At a high level:

1. Ingest historical data and canonicalize schema.
2. Fit base model components up to a cutoff quarter.
3. Recalibrate from holdout behavior (recent historical window).
4. Build fund states from portfolio snapshot.
5. Simulate forward with MSCI-driven NAV process and cashflow mechanics.
6. Optionally run neutral tuning to align with historical targets.
7. Save projections and diagnostics.

Main entry points:

- `run_fit_pipeline` in `src/ecf_model/pipeline.py`
- `run_projection_pipeline` in `src/ecf_model/pipeline.py`
- `run_neutral_tuning_pipeline` in `src/ecf_model/tuning.py`
- Quarterly orchestration in `scripts/run_quarter_pipeline.py`

---

## 3. Data Contract and Canonicalization

### 3.1 Required canonical fields

Model-required columns are defined in `src/ecf_model/schema.py`:

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

### 3.2 Canonicalization behavior

The loader in `src/ecf_model/data_io.py`:

- Renames known alternate headers to canonical names.
- Parses quarter-end timestamp from `Year` + `Quarter`.
- Normalizes numeric fields.
- Normalizes strategy/grade text.
- Creates age bucket labels.
- Sorts rows by fund and quarter.

### 3.3 Recallable interpretation

`Recallable` can arrive in two styles:

- Already a period flow
- A cumulative stock-like series

The helper `recallable_to_flow` (`src/ecf_model/utils.py`) detects monotonic cumulative behavior and converts by first-difference. Otherwise it treats values as flow magnitudes.

This is important because recallables are used both in fitting and simulation state transitions.

---

## 4. Time, Cutoff, and Snapshot Logic

The model is quarter-driven.

- Fit uses only rows with `quarter_end <= cutoff`.
- Projection starts from latest quarter in the provided portfolio file.
- Horizon extends to the furthest modeled fund end date, plus an additional buffer.

Cutoff handling:

- `slice_to_cutoff` in `src/ecf_model/data_io.py`
- Quarter parsing in `parse_quarter_text` (`src/ecf_model/utils.py`)

---

## 5. Fund Lifecycle Dates

Lifecycle dates are calculated in `src/ecf_model/features.py`.

## 5.1 Investment period end (`invest_end`)

`compute_invest_end_by_fund`:

- Starts from first closing date.
- Looks at first repayment timing.
- Uses a 6-7 year investment period heuristic depending on early repayment behavior.

## 5.2 Fund end date (`fund_end_qe`)

`compute_fund_end_dates` blends:

- Planned end date
- Historical overrun by strategy and grade
- Shrinkage toward strategy/global overrun
- Fallback planned life if planned end is missing
- Hard minimum total life floor (`min_life_q = 32` quarters)

This gives each fund a modeled end-of-life anchor, which is central for phase multipliers and tail behavior.

---

## 6. Base Feature Engineering for Fit

Implemented in `prepare_fit_base` (`src/ecf_model/fitting.py`).

For each fund-quarter:

- Draw event flag: draw amount > 0
- Repayment event flag: repayment > 0 and prior NAV above gate
- Recallable event flag: recallable flow > 0 and repayment > 0

Ratios:

- Draw ratio over max commitment.
- Repayment ratio over previous NAV (conditional on NAV gate).
- Recallable ratio conditional on repayment.

NAV residual growth (`omega_obs`) is estimated as:

```text
omega_obs = NAV_t / (NAV_{t-1} + Draw_t - Repay_t) - 1
```

and clipped to configured bounds.

---

## 7. Base Fitting Components

Base fitting outputs:

- `timing_probs_selected.csv`
- `ratio_fit_selected.csv`
- `omega_selected.csv`

These are saved by `FitArtifacts.save` in `src/ecf_model/fitting.py`.

## 7.1 Timing probability model

Function: `fit_timing_probs`.

Hierarchical levels:

- `global`
- `strategy`
- `strategy_age`
- `strategy_grade_age`
- `strategy_grade_age_size`

For each metric (`draw`, `rep`, `rc`) and bucket:

```text
p_smooth = (events + prior_strength * p_parent) / (n + prior_strength)
```

Parent prior comes from less granular hierarchy (or global rate).

Outputs include:

- Smoothed probabilities
- Raw probabilities
- Event counts
- Wilson confidence intervals

## 7.2 Ratio marginal model

Function: `fit_ratio_models`.

For each ratio metric and bucket:

1. Keep positive observations.
2. Winsorize tails.
3. Fit candidate distributions:
   - Lognormal with fixed location 0
   - Gamma with fixed location 0
4. Select by minimum AIC.
5. Store distribution name and parameters.
6. Fall back to empirical mean when data is too sparse or fit is unstable.

This gives bucket-specific marginals that will be sampled in simulation.

## 7.3 NAV omega regression model

Function: `fit_omega_models`.

Regression target:

- `omega_obs`

Regressors:

- Intercept
- Current MSCI return
- Lagged MSCI return
- Age (linear and quadratic)
- Time to end-of-life
- Pre-end indicator
- Post-end indicator

The fit uses robust ridge:

- L2 regularization (penalty varies by hierarchy depth)
- Tukey bisquare reweighting to reduce outlier leverage

Coefficients and residual sigma are stored per hierarchy level.

Important note:

- `alpha` column exists in output but is currently set to zero in fitting.
- Intercept absorbs baseline drift not explained by explicit regressors.

## 7.4 Exact calibration buckets and fallbacks

This is the code-accurate bucket definition for all calibration layers.

| Layer | Buckets used | Fallback |
| --- | --- | --- |
| Timing probabilities | `global -> strategy -> strategy_age -> strategy_grade_age -> strategy_grade_age_size` | Parent smoothing and runtime fallback in same order |
| Ratio marginals | `strategy_grade_age_size`, `strategy_grade_age`, `strategy_age`, `strategy`, `global` | Runtime fallback to less granular level; empirical mean if sparse |
| Omega regression | `strategy_grade_age_size`, `strategy_grade_age`, `strategy`, `global` | Runtime fallback to less granular level |
| Holdout global recalibration | Pooled (no bucket split) | Neutral defaults |
| Holdout group recalibration | `strategy + grade` | Shrink toward global; drop groups below min obs |
| Pre-end repayment multipliers | `strategy + pre_end_phase` (`far`, `mid`, `near`) | `ALL` strategy per same phase |
| Endgame ramp | `strategy` | `ALL` strategy |
| Post-invest draw multipliers | `strategy` | `ALL` strategy |
| Tail floors/caps/drag | `strategy + tail_bucket` (`0-3`, `4-7`, `8-11`, `12+`) | Same tail bucket at `ALL` strategy |
| Lifecycle multipliers + NAV floor | `strategy + grade + life_phase` | `strategy + ALL + phase`, then `ALL + ALL + phase` |
| Neutral tuning | `strategy + grade` plus pooled portfolio objective | Global scales applied after group updates |

---

## 8. MSCI Model

Implemented in `src/ecf_model/msci.py`.

### 8.1 Fit

On quarterly MSCI data up to cutoff:

- Compute quarterly return series.
- Fit AR(1) coefficient for persistence (`phi`), clipped for stability.
- Split returns into 3 regimes using terciles:
  - Bearish
  - Neutral
  - Bullish
- Estimate regime-specific means and volatilities.
- Estimate regime transition matrix with Laplace smoothing.

### 8.2 Projection

For each simulation path and quarter:

1. Transition regime according to scenario-adjusted transition matrix.
2. Draw return shock from regime volatility.
3. Generate return:

```text
ret_t = regime_mu + drift_shift + phi*(ret_{t-1} - overall_mu) + shock
```

4. Update level.
5. Save both current and lagged return for omega model use.

Scenario mechanics:

- Neutral: no drift shift.
- Bullish: positive drift shift and transition tilt toward bullish regime.
- Bearish: negative drift shift and transition tilt toward bearish regime.

---

## 9. Copula Calibration and Dependence

### 9.1 What is calibrated

`_calibrate_copula_from_history` in `src/ecf_model/pipeline.py` computes lag-1 correlation signatures by fund for:

- Any cashflow-event indicator
- Total flow ratio to commitment

Then blends:

```text
rho_raw = 0.60 * max(rho_event, 0) + 0.40 * max(rho_flow, 0)
rho_calibrated = clip(rho_raw, 0.05, 0.85)
```

Saved to `copula_calibration.json`.

### 9.2 How rho is chosen at projection time

Order:

1. User input `--copula-rho` if provided
2. `copula_calibration.json` from calibration directory
3. Fallback `0.35`

Clipped to `[0.0, 0.95]` at resolution and `[0.0, 0.999]` in simulator.

### 9.3 Copula implementation

Within each fund and simulation path:

- Draw one latent factor `Z_common`.
- For each random draw:

```text
Z = sqrt(rho)*Z_common + sqrt(1-rho)*epsilon
U = Phi(Z)
```

Use `U` for:

- Event Bernoulli draws (`U < p`)
- Inverse-CDF ratio sampling (lognormal/gamma)

So dependence is imposed across all stochastic components within fund.

---

## 10. Holdout Recalibration Framework

Core function: `calibrate_from_history` (`src/ecf_model/calibration.py`).

Purpose:

- Adjust fitted behavior to match the most recent holdout window.
- Control overfitting via significance, shrinkage, and stability attenuation.

### 10.1 Train and holdout split

- Holdout = last `holdout_quarters` before cutoff.
- Train = all earlier quarters.

### 10.2 Statistical primitives used

- Two-proportion z-test for event-rate differences.
- Welch t-test for mean differences (often on `log1p` transformed ratios).

If not significant at configured alpha, effects are neutralized.

### 10.3 Global deltas

`_compute_deltas` estimates adjustments:

- Draw event multiplier
- Repayment event multiplier
- Repayment ratio scale
- Draw ratio scale
- Omega bump
- Repayment timing shift in quarters

Timing shift is chosen by minimizing holdout log-loss across candidate shifts (default -2 to +2 quarters), then suppressed unless improvement is material.

Each effect is then shrunk:

```text
weight = n / (n + shrink_n)
adjusted = 1 + weight*(raw - 1)  (or weight*raw for additive effects)
```

and clipped.

### 10.4 Stability attenuation

`_compute_stability_attenuation` runs rolling pseudo-out-of-sample checks:

- Compare deltas that worked in one period against next period realization.
- Fit slope-through-origin from historical delta to future delta.
- Use slope as attenuation coefficient in `[0,1]`.

`_apply_stability_attenuation` dampens unstable signals before simulation.

### 10.5 Group-level deltas (strategy x grade)

`_compute_group_deltas` computes local deltas per `(Adj Strategy, Grade)` and shrinks them toward global deltas.

This gives grade-aware behavior while preventing small-group overreaction.

Also computes recallable propensity by group (share of funds ever showing recallables).

---

## 11. Specialized Phase and Tail Tables

These are the key "phase multipliers" and related controls.

All are estimated from historical data in calibration and applied in simulation.

---

## 11.1 Pre-end repayment multipliers

File: `pre_end_repayment_multipliers.csv`  
Function: `_compute_pre_end_repayment_multipliers`

Phases:

- Far: more than 24 quarters to end
- Mid: 13 to 24 quarters to end
- Near: 1 to 12 quarters to end

For each strategy and phase:

1. Compare phase repayment event frequency and conditional repayment ratio to strategy pre-end baseline.
2. Test significance.
3. If not significant, set raw effect to neutral.
4. Shrink by sample size.
5. Clip to configured bounds.

Important design:

- Near phase is forced neutral here (`1.0`) so near-end acceleration is handled by dedicated endgame ramp only.

Application in simulation:

- Applied only when `q_to_end >= 13` and in post-investment pre-end period.

---

## 11.2 Endgame repayment ramp

File: `endgame_repayment_ramp.csv`  
Function: `_compute_endgame_ramp`

Compares:

- Near-end window (`1..12` quarters to end)
- Mid pre-end window (`13..24`)

For each strategy:

1. Estimate near/mid repayment event ratio.
2. Estimate near/mid conditional repayment-size ratio.
3. Keep uplift only if significant and greater than 1.
4. Shrink by sample size.
5. Clip.

In simulation this is not a hard step. It ramps gradually:

```text
ramp = (12 - q_to_end)/11
effective_multiplier = 1 + (multiplier - 1) * ramp
```

So acceleration grows as fund approaches the final quarter before end.

---

## 11.3 Post-invest draw multipliers

File: `post_invest_draw_multipliers.csv`  
Function: `_compute_post_invest_draw_multipliers`

Compares draw behavior:

- During investment period
- After investment period but before fund end

For each strategy:

1. Compare draw event frequency and draw ratio.
2. Significance gate.
3. Shrink.
4. Clip with max = 1.0 (attenuation-only).

Meaning:

- Late drawdowns are reduced relative to investment period.

---

## 11.4 Tail repayment floors and post-end controls

File: `tail_repayment_floors.csv`  
Function: `_compute_tail_floors`

Buckets by quarters after fund end:

- `0-3`
- `4-7`
- `8-11`
- `12+`

For each strategy x bucket, compute:

- Minimum repayment probability
- Minimum repayment ratio
- Draw probability cap
- Omega drag

Then:

- Add strategy-agnostic fallback rows (`ALL`)
- Fill missing combinations via fallback logic

Use in simulation (post-end):

- Repayment probability floored.
- Draw probability capped.
- Repayment ratio floored.
- Omega reduced by drag.

This creates realistic runoff instead of immediate liquidation.

---

## 11.5 Lifecycle phase multipliers

File: `lifecycle_phase_multipliers.csv`  
Function: `_compute_lifecycle_multipliers`

Phases:

- `far`
- `mid`
- `late`
- `final`
- `post_0_4`
- `post_5_8`
- `post_9_12`
- `post_12p`

For each strategy x grade x phase:

1. Compute phase metrics:
   - repayment event frequency
   - conditional repayment ratio
   - draw event frequency
   - draw ratio
   - omega mean
   - NAV-to-commit quantile
2. Compute baseline metrics for that strategy x grade.
3. Raw effects:
   - ratios for event/rate metrics
   - difference for omega
4. Significance gate on each metric.
5. If observations are below minimum threshold:
   - most multipliers neutralize to 1.0
   - omega bump neutralizes to 0.0
   - NAV floor can still use anchored fallback for late/post phases
6. Else shrink by sample size and clip to configured bounds.

Fallback layers are then added:

- Strategy-level (`Grade = ALL`)
- Global (`Adj Strategy = ALL, Grade = ALL`)
- Completed phase grid with hierarchical fallback fill

---

## 11.6 NAV floor ratio in lifecycle table

`nav_floor_ratio` is a key late-life control.

Concept:

- Historically, many funds keep residual NAV near and after end date.
- This ratio encodes a target residual NAV as a fraction of commitment.

During simulation, when fund is old enough and within 24 quarters of end:

1. Compute target floor `target = nav_floor_ratio * commitment`.
2. If current projected NAV is below target:
   - Pull NAV upward toward target with configurable strength.
3. Strength varies by phase (mid, late, final, post-end) using model config.

This is not hard clipping. It is a smooth pull toward a historical floor.

---

## 12. Simulation Engine Details

Function: `simulate_portfolio` (`src/ecf_model/simulator.py`)

Simulation loops:

- Over simulation paths
- Over projection quarters
- Over funds

### 12.1 Fund state variables

Maintained each quarter:

- Current NAV
- Cumulative drawdowns
- Cumulative repayments
- Cumulative recallables
- Recallable balance available for redrawing
- Age in quarters

### 12.2 Event probabilities

Start from timing table probabilities and apply:

1. Global/group recalibration deltas
2. Lifecycle multipliers (scaled by lifecycle strengths)
3. Pre-end phase multipliers
4. Endgame ramp multipliers
5. Post-invest draw multipliers
6. Post-end tail floors/caps

### 12.3 Drawdown mechanics

Eligibility:

- Allowed through investment end
- Optionally allowed post-invest pre-end
- Blocked post-end if configured

Amount:

- Event sampled from probability.
- Ratio sampled from fitted marginal.
- Scaled by deltas and lifecycle phase multipliers.
- Capped by available drawable capacity.
- If draw exceeds uncalled commitment, excess consumes recallable balance.

### 12.4 Repayment mechanics

Eligibility:

- Requires NAV above gate.

Amount:

- Event sampled from probability.
- Ratio sampled from fitted repayment marginal.
- Scaled by deltas, lifecycle, pre-end, endgame, and tail floors.
- Capped not to exceed available NAV after same-quarter flows.

### 12.5 Recallable mechanics

Eligibility:

- Only if repayment occurs.

Amount:

- Event and ratio sampled from recallable conditional model.
- Cumulative recallables capped by cumulative repayments.
- Added to recallable balance for potential future redraw.

### 12.6 NAV mechanics

After flows:

1. Compute NAV after flow.
2. Compute omega from:
   - omega regression coefficients
   - MSCI current and lagged returns
   - age and age-squared
   - time-to-end
   - pre/post-end indicators
   - global/group omega bump
   - lifecycle omega bump (scaled)
   - random residual with fitted sigma
   - post-end omega drag from tail table
3. Clip omega.
4. Update NAV multiplicatively.
5. Post-end tiny NAV threshold can force hard zero.
6. Apply lifecycle NAV floor pull if conditions are met.

### 12.7 Aggregation outputs

Saved outputs include:

- Portfolio series means and quantiles by quarter.
- Fund-level quarterly means.
- Fund terminal NAV means.
- DPI distribution percentiles by quarter.

Important:

- Full path-level cashflow outputs are not currently persisted (except full MSCI paths).

---

## 13. Scenario Handling

Scenario affects projected MSCI paths via:

- Drift shift
- Regime transition tilt
- Volatility scaling

Cashflow timing tables themselves are not scenario-specific. Scenario enters mainly through NAV via omega’s market betas.

---

## 14. Neutral Tuning: Detailed Mechanics

Core function:

- `tune_neutral_calibration_to_history`

Purpose:

- Align neutral projection for a target portfolio mix with historical outcomes.

### 14.1 Target construction

Targets are computed from historical funds matching strategy/grade mix of the test portfolio.

Target modes:

- `pooled`: pooled cashflow IRR and pooled TVPI
- `mix_median`: weighted group medians

Also computes:

- Historical recallable-to-commit ratio
- Mature-fund drawdown-to-commit ratio

### 14.2 Objective

```text
objective = 8.0 * |IRR error|
          + 0.6 * |TVPI error|
          + 1.0 * strategy-grade TVPI shape error
```

IRR is intentionally weighted highest.

### 14.3 Iterative local updates

Per iteration:

1. Simulate with current calibration.
2. Compare predicted vs target by strategy-grade.
3. Update group deltas:
   - repayment ratio scale
   - repayment probability multiplier
   - omega bump
   - occasional timing shift tweak
4. Apply portfolio-level IRR correction.

Learning rate decays with iterations.

### 14.4 Post-iteration global alignment searches

After local iterations:

- Search global repayment intensity scale to close IRR gap.
- Search recallable intensity scale to match historical recall ratio.
- Search draw intensity scale to match mature draw ratio while balancing IRR.
- Final repayment-only IRR nudge.

These searches are coarse-to-refined and objective-guided.

### 14.5 Saved tuning diagnostics

In tuned calibration directory:

- `neutral_tuning_targets_by_strategy_grade.csv`
- `neutral_tuning_diagnostics.csv`
- `irr_alignment_search.csv`

---

## 15. Manual Timing Overlay in Quarter Runner

The quarterly orchestrator (`scripts/run_quarter_pipeline.py`) applies an additional timing overlay (`TIMING_PROFILE_V2`) after tuning:

- Slight global repayment intensity scale-up.
- Partial relaxation of far pre-end attenuation.
- Partial flattening of endgame repayment-ratio uplift.
- Strong downscaling of tail repayment floors.
- Compression of late/final lifecycle repayment multipliers toward neutral.

This is a business-policy layer, not directly estimated in the same run.

If strict pure data calibration is desired, this overlay can be disabled in runner arguments.

---

## 16. Calibration Artifacts: What Each File Means

Typical calibration folder includes:

- `timing_probs_selected.csv`: base event probabilities by hierarchy.
- `ratio_fit_selected.csv`: chosen marginal distribution and params by bucket.
- `omega_selected.csv`: omega regression coefficients and diagnostics.
- `holdout_recalibration.json`: global deltas.
- `holdout_group_recalibration.csv`: strategy-grade deltas.
- `stability_attenuation.csv`: attenuation factors for unstable signals.
- `tail_repayment_floors.csv`: post-end floors/caps/drag.
- `endgame_repayment_ramp.csv`: near-end repayment acceleration.
- `pre_end_repayment_multipliers.csv`: far/mid pre-end attenuation.
- `post_invest_draw_multipliers.csv`: post-invest draw attenuation.
- `lifecycle_phase_multipliers.csv`: phase multipliers and nav floor ratios.
- `copula_calibration.json`: calibrated one-factor dependence loading.
- `msci_model.json`: fitted MSCI process.
- `avg_overrun_by_strategy.csv`: strategy-level overrun used in fund-end estimation.
- `timing_adjustment_applied.json`: only when runner applies manual timing profile.

---

## 17. Projection Outputs: What They Represent

In projection folder:

- `sim_outputs/sim_portfolio_series.csv`
  - Portfolio means and quantiles by quarter.
- `sim_outputs/sim_fund_quarterly_mean.csv`
  - Fund-level means by quarter.
- `sim_outputs/sim_fund_end_summary.csv`
  - Mean terminal NAV by fund.
- `sim_outputs/msci_paths.csv`
  - Full MSCI paths by simulation id.
- `portfolio_observation_summary.csv`
  - Cumulative draw/repay/recall, DPI/RVPI/TVPI, and rolling IRR.
- `projection_run_config.json`
  - Full serialized runtime configuration.

---

## 18. Defensibility and Statistical Guardrails

What is learned from history:

- Timing probabilities
- Ratio marginal distributions
- Omega betas and residual volatility
- Holdout recalibration deltas
- Phase multipliers and tail controls
- Copula loading
- Overrun/lifetime behavior

What is constrained by policy:

- Clip ranges
- Prior strengths
- Minimum observation thresholds
- Scenario drift shifts
- Optional manual timing overlay

Defensibility controls in code:

- Significance gating
- Shrinkage by sample size
- Stability attenuation from rolling tests
- Hierarchical fallback when granular buckets are sparse
- Hard clips on multipliers and omega

---

## 19. Known Limitations

- Quarter-level granularity only.
- Full per-path cashflow outputs are not persisted (aggregates are).
- Some granular omega buckets have low explanatory power and high variance.
- Cross-fund dependence in cashflow shocks is not explicitly modeled (dependence is within-fund via copula; cross-fund co-movement mostly enters through MSCI).
- Manual overlay can improve practical fit but introduces non-estimated policy influence.

---

## 20. Practical Audit Checklist

When validating a run, check:

1. `projection_run_config.json`
   - cutoff, scenario, simulation seed/count, copula settings
2. `copula_calibration.json`
   - calibrated rho and diagnostics
3. `omega_selected.csv`
   - strategy/global signs for market betas, sigma ranges, R2
4. Phase tables
   - pre-end, endgame, lifecycle, tail, post-invest
5. `timing_adjustment_applied.json`
   - confirm whether manual overlay was applied
6. `portfolio_observation_summary.csv`
   - final cumulative draw, repay, end NAV, DPI, TVPI, IRR
7. Historical comparison diagnostics
   - tuning target files and alignment search outputs

---

## 21. Plain-Language Summary of Phase Multipliers

In simple terms:

- We first learn "normal" behavior for similar funds.
- Then we check if behavior changes in specific life stages.
- If a stage truly behaves differently in history, we apply a stage adjustment.
- If data is weak or noisy, we reduce the adjustment toward neutral.
- We cap extreme adjustments to keep projections stable.

So the model does not blindly force stage effects. It applies them only when history supports them and only as strongly as evidence allows.

---

## 22. Where to Modify Behavior Safely

If you want to change behavior with controlled risk:

- Adjust clipping and shrinkage in `src/ecf_model/config.py`
- Modify tuning weights in `src/ecf_model/tuning.py`
- Enable or disable manual timing overlay in `scripts/run_quarter_pipeline.py`

Avoid changing:

- Event gating logic
- NAV accounting identity
- Recallable conservation constraints

unless you also update diagnostics and backtests.

---

## 23. Suggested Future Enhancements

- Persist full path-level cashflow outputs (optional flag).
- Add explicit cross-fund factor for cashflow co-movement.
- Add Bayesian hierarchical shrinkage for omega betas.
- Add richer distribution families for ratio tails.
- Add automated model governance report per run (parameter drift, p-values, bucket coverage).

---

## 24. Reference Map (Code Locations)

Core modules:

- Data IO and schema: `src/ecf_model/data_io.py`, `src/ecf_model/schema.py`
- Utility helpers: `src/ecf_model/utils.py`
- Lifecycle state build: `src/ecf_model/features.py`
- Fit stage: `src/ecf_model/fitting.py`
- Calibration stage: `src/ecf_model/calibration.py`
- MSCI process: `src/ecf_model/msci.py`
- Simulation engine: `src/ecf_model/simulator.py`
- Fit/projection pipeline: `src/ecf_model/pipeline.py`
- Neutral tuning: `src/ecf_model/tuning.py`
- Backtesting: `src/ecf_model/backtest.py`
- CLI: `src/ecf_model/cli.py`
- Quarter orchestrator and timing overlay: `scripts/run_quarter_pipeline.py`

---

## 25. Final Note

This model is intentionally built as a practical forecasting engine with statistical discipline.

It balances three competing needs:

- Statistical learning from historical behavior
- Stability under sparse/noisy private-market data
- Business realism across full fund lifecycles, especially near and after end-of-life

That balance is why you see both learned components and controlled overlays.
