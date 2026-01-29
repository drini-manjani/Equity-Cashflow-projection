#!/usr/bin/env python3
"""
Standalone script to anonymize fund names and keep only selected columns.
Usage:
  python3 anonymize_data.py --input /path/to/data.csv --output /path/to/anonymized.csv
  python3 anonymize_data.py --input /path/to/data.parquet --output /path/to/anonymized.csv --map-out /path/to/fund_map.csv
"""

import argparse
import os
from typing import List

import pandas as pd


TARGET_COLUMNS: List[str] = [
    "FundID",
    "Year of Transaction Date",
    "Quarter of Transaction Date",
    "Adj strategy",
    "VC Fund Status",
    "Fund Workflow Stage",
    "First Closing Date",
    "Planned end date with add. years as per legal doc",
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


def _read_any(path: str, sep: str = "", encoding: str = "utf-8", bad_lines: str = "warn") -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    if path.lower().endswith(".csv"):
        # Try robust CSV parsing (auto-detect delimiter, tolerant to bad lines)
        if not sep:
            try:
                return pd.read_csv(path, sep=None, engine="python", encoding=encoding, on_bad_lines=bad_lines)
            except Exception:
                pass
            # Fallback delimiters
            for s in [",", ";", "\t", "|"]:
                try:
                    return pd.read_csv(path, sep=s, engine="python", encoding=encoding, on_bad_lines=bad_lines)
                except Exception:
                    continue
        return pd.read_csv(path, sep=sep, engine="python", encoding=encoding, on_bad_lines=bad_lines)
    raise ValueError("Unsupported input type. Use .csv or .parquet")


def _write_any(df: pd.DataFrame, path: str) -> None:
    if path.lower().endswith(".parquet"):
        df.to_parquet(path, index=False)
        return
    if path.lower().endswith(".csv"):
        df.to_csv(path, index=False)
        return
    raise ValueError("Unsupported output type. Use .csv or .parquet")


def _find_default_input() -> str:
    # 1) current working directory
    cwd_path = os.path.abspath("data.csv")
    if os.path.exists(cwd_path):
        return cwd_path

    # 2) Windows-style Documents/Equity/*/data/data.csv (pick most recent)
    home = os.environ.get("USERPROFILE") or os.environ.get("HOME") or ""
    if home:
        base = os.path.join(home, "Documents", "Equity")
        if os.path.isdir(base):
            candidates = []
            for root, _, files in os.walk(base):
                if "data.csv" in files and os.path.basename(root).lower() == "data":
                    candidates.append(os.path.join(root, "data.csv"))
            if candidates:
                candidates.sort(key=os.path.getmtime, reverse=True)
                return candidates[0]

    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="", help="Input data file (.csv or .parquet). Default: auto-detect")
    ap.add_argument("--output", default="", help="Output anonymized file (.csv or .parquet). Default: alongside input")
    ap.add_argument("--map-out", default="", help="Fund name mapping output (.csv or .parquet). Default: alongside input")
    ap.add_argument("--sep", default="", help="CSV separator (leave blank to auto-detect)")
    ap.add_argument("--encoding", default="utf-8", help="CSV encoding (default: utf-8)")
    ap.add_argument("--bad-lines", default="warn", choices=["error", "warn", "skip"],
                    help="How to handle bad CSV lines (default: warn)")
    # In notebooks/IDEs, extra args may be injected (e.g., -f). Ignore unknowns.
    args, unknown = ap.parse_known_args()
    if unknown:
        print("Warning: ignoring unknown arguments:", " ".join(unknown))

    input_path = args.input or _find_default_input()
    if not input_path:
        raise FileNotFoundError(
            "Could not auto-detect data.csv. Please pass --input with the full path."
        )
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(os.path.dirname(input_path), "data_anonymized.csv")

    if args.map_out:
        map_path = args.map_out
    else:
        map_path = os.path.join(os.path.dirname(input_path), "fund_map.csv")

    df = _read_any(input_path, sep=args.sep, encoding=args.encoding, bad_lines=args.bad_lines)
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
    fund_col = "FundID"
    if fund_col not in out.columns:
        raise KeyError("FundID column not found after normalization.")

    # preserve order of appearance (deterministic for a given file order)
    seen = []
    seen_set = set()
    for v in out[fund_col].astype(str).tolist():
        if v not in seen_set:
            seen.append(v)
            seen_set.add(v)
    name_map = {orig: f"Fund{idx+1}" for idx, orig in enumerate(seen)}
    out[fund_col] = out[fund_col].astype(str).map(name_map)

    _write_any(out, output_path)

    map_df = pd.DataFrame({"FundID": list(name_map.keys()),
                           "Fund_Anon": list(name_map.values())})
    _write_any(map_df, map_path)

    print(f"Anonymized rows: {len(out)}")
    print(f"Unique funds: {len(name_map)}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Mapping file: {map_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
