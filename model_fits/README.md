# Model Fits

Scripts to fit candidate distributions for drawdown/repayment/recallable ratios
using the anonymized dataset.

## Quick start

```bash
python3 model_fits/fit_ratios.py --input anonymized.csv --out-dir model_fits/outputs
```

## Additional scripts

```bash
python3 model_fits/fit_grade_transitions.py --input anonymized.csv --out-dir model_fits/outputs/transitions
python3 model_fits/fit_timing_curves.py --input anonymized.csv --out-dir model_fits/outputs
python3 model_fits/fit_copula.py --input anonymized.csv --out model_fits/outputs/copula_params.json
python3 model_fits/simulate_cashflows.py --input anonymized.csv --fit-dir model_fits/outputs --trans-dir model_fits/outputs/transitions --start-year 2018 --start-quarter Q1
```

## Notebook

Open `model_fits/Distribution_Fits.ipynb` and run all cells.
Open and run these notebooks for the rest of the pipeline:
- `model_fits/Omega_Calibration.ipynb`
- `model_fits/Grade_Transitions.ipynb`
- `model_fits/Timing_Curves.ipynb`
- `model_fits/Copula_Fit.ipynb`
- `model_fits/Simulate_Cashflows.ipynb`

Outputs CSV summaries in `model_fits/outputs/`:
- `ratio_fit_global.csv`: global fits by ratio/distribution
- `ratio_fit_by_group.csv`: best-fit distribution per group/level
- `ratio_fit_selected.csv`: per Strategy×Grade×AgeBucket with fallback (S×G×A → S×G → S → global)
- `timing_probs_by_group.csv`: timing probabilities for draw/rep/recallable
- `timing_probs_selected.csv`: timing probs with fallback
- `grade_transition_1y_all.csv` (+ per-strategy) in `outputs/transitions/`
- `copula_params.json`: suggested rho_event/rho_size

## Notes
- Requires: `pandas`, `numpy`, `scipy`.
- Zero-inflation is handled: zeros are treated as a mass at 0, and the positive
  tail is fit to candidate continuous distributions.
- Uses `Adj strategy`, `Grade`, and `AgeBucket` for grouping by default.
- If `Grade_Current` (or `Current Grade`) exists, it is used. Otherwise a per‑fund forward‑fill
  of `Grade` by quarter is computed as `Grade_Current`.
- Fallback thresholds can be tuned with `--min-obs-age`, `--min-obs-sg`, `--min-obs-s`.
