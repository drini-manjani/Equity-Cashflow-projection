#!/usr/bin/env python3
"""
Standalone script to anonymize fund names and keep only selected columns.
Usage:
  python3 anonymize_data.py --input /path/to/data.csv --output /path/to/anonymized.csv
  python3 anonymize_data.py --input /path/to/data.parquet --output /path/to/anonymized.csv --map-out /path/to/fund_map.csv
"""

import argparse
import os
import sys
from typing import List

import pandas as pd


TARGET_COLUMNS: List[str] = [
    "VC Fund Name",
    "Year of Transaction Date",
    "Quarter",
    "Adj strategy",
    "VC Fund Status",
    "Fund Workflow",
    "First Closing Date",
    "Planned End Date",
    "Transaction Quarter",
    "Commitment EUR",
    "Signed Amount EUR",
    "Adj Drawdown EUR",
    "Adj Repayment EUR",
    "Recallable",
    "NAV Adjusted EUR",
    "Grade",
    "Recallable_Percentage_Decimal",
    "Expiration_Quarters",
    "Fund_Age_Quarters",
    "draw_cum_prev",
    "Capacity",
    "Drawdown_Ratio",
    "Repayment_Ratio",
]


def _norm_key(s: str) -> str:
    return " ".join(s.strip().lower().replace("_", " ").split())


def _read_any(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError("Unsupported input type. Use .csv or .parquet")


def _write_any(df: pd.DataFrame, path: str) -> None:
    if path.lower().endswith(".parquet"):
        df.to_parquet(path, index=False)
        return
    if path.lower().endswith(".csv"):
        df.to_csv(path, index=False)
        return
    raise ValueError("Unsupported output type. Use .csv or .parquet")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data.csv", help="Input data file (.csv or .parquet). Default: data.csv")
    ap.add_argument("--output", default="data_anonymized.csv", help="Output anonymized file (.csv or .parquet)")
    ap.add_argument("--map-out", default="fund_map.csv", help="Fund name mapping output (.csv or .parquet)")
    # In notebooks/IDEs, extra args may be injected (e.g., -f). Ignore unknowns.
    args, unknown = ap.parse_known_args()
    if unknown:
        print("Warning: ignoring unknown arguments:", " ".join(unknown))

    df = _read_any(args.input)
    # normalize column names for matching
    col_map = {_norm_key(c): c for c in df.columns}

    # resolve requested columns by normalized name
    resolved = {}
    missing = []
    for c in TARGET_COLUMNS:
        key = _norm_key(c)
        if key in col_map:
            resolved[c] = col_map[key]
        else:
            missing.append(c)

    if missing:
        msg = "Missing required columns: " + ", ".join(missing)
        raise KeyError(msg)

    # subset and rename to target column names
    out = df[[resolved[c] for c in TARGET_COLUMNS]].copy()
    out.columns = TARGET_COLUMNS

    # anonymize fund names
    fund_col = "VC Fund Name"
    if fund_col not in out.columns:
        raise KeyError("VC Fund Name column not found after normalization.")

    # preserve order of appearance (deterministic for a given file order)
    seen = []
    seen_set = set()
    for v in out[fund_col].astype(str).tolist():
        if v not in seen_set:
            seen.append(v)
            seen_set.add(v)
    name_map = {orig: f"Fund{idx+1}" for idx, orig in enumerate(seen)}
    out[fund_col] = out[fund_col].astype(str).map(name_map)

    _write_any(out, args.output)

    map_df = pd.DataFrame({"VC Fund Name": list(name_map.keys()),
                           "Fund_Anon": list(name_map.values())})
    _write_any(map_df, args.map_out)

    print(f"Anonymized rows: {len(out)}")
    print(f"Unique funds: {len(name_map)}")
    print(f"Output: {args.output}")
    print(f"Mapping file: {args.map_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
