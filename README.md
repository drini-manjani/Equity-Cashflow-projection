# Equity Cashflow Projection

Cross-platform project for private-equity cashflow projection with quarter-based runs.

## Data Update Flow

1. Run `data.ipynb` (reads SQL from `cashflows.txt`, `fund.txt`, `grades.txt`, `kmp.txt`).
2. Run `data_preparation.ipynb`.
3. Run your anonymization notebook to produce `anonymized.csv`.
4. Keep `msci.xlsx` updated.

## One Command Workflow (Mac + Windows)

Use the quarter runner. It will:

1. Fit calibration on `anonymized.csv` + `msci.xlsx` for the selected cutoff quarter.
2. Tune neutral calibration on the primary test portfolio.
3. Apply the latest timing profile.
4. Project both portfolios:
   - `test_portfolio.xlsx`
   - `tests/test_portfolio_Sentral.xlsx`

It also auto-converts those portfolio templates into canonical model input CSVs.

### macOS/Linux

```bash
./scripts/run_quarter_pipeline.sh --cutoff 2025Q3
```

### Windows (PowerShell)

```powershell
.\scripts\run_quarter_pipeline.ps1 --cutoff 2025Q3
```

### Direct Python (any OS)

```bash
python scripts/run_quarter_pipeline.py --cutoff 2025Q3
```

By default, simulations use a one-factor Gaussian copula within each fund across drawdowns, repayments, and recallables. If `--copula-rho` is not provided, rho is calibrated from historical data during `fit` and loaded from `calibration/copula_calibration.json`. You can override it, for example:

```bash
python scripts/run_quarter_pipeline.py --cutoff 2025Q3 --copula-rho 0.45
```

## Run Structure

Outputs are organized by quarter:

- `runs_v2/<quarter>/test_portfolio/`
- `runs_v2/<quarter>/test_portfolio_Sentral/`

Key files in each projection run:

- `projection/portfolio_observation_summary.csv`
- `projection/sim_outputs/sim_portfolio_series.csv`
- `projection/sim_outputs/sim_fund_quarterly_mean.csv`
- `projection/sim_outputs/sim_fund_end_summary.csv`
- `calibration/` (fitted+tuned artifacts used for the run)
