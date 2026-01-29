#!/usr/bin/env python3
"""
Fit 1-year grade transition matrices from anonymized data.

Outputs:
  model_fits/outputs/transitions/grade_transition_1y_all.csv
  model_fits/outputs/transitions/grade_transition_1y_<strategy>.csv
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


GRADE_ORDER = ["A", "B", "C", "D"]


def _norm_key(s: str) -> str:
    return " ".join(s.strip().lower().replace("_", " ").split())


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {_norm_key(c): c for c in df.columns}
    def _get(name: str) -> str:
        k = _norm_key(name)
        return col_map.get(k, name)

    rename = {}
    rename[_get("Adj strategy")] = "Adj Strategy"
    rename[_get("Adj Strategy")] = "Adj Strategy"
    rename[_get("Quarter of Transaction Date")] = "Quarter"
    rename[_get("Year of Transaction Date")] = "Year"
    rename[_get("FundID")] = "FundID"
    rename[_get("Grade")] = "Grade"
    rename[_get("Current Grade")] = "Grade_Current"
    rename[_get("CurrentGrade")] = "Grade_Current"
    rename[_get("Grade Current")] = "Grade_Current"
    rename[_get("Grade_Current")] = "Grade_Current"
    return df.rename(columns=rename)


def apply_current_grade(df: pd.DataFrame, context: str = "") -> pd.DataFrame:
    df = df.copy()
    if "Grade_Current" in df.columns:
        df["Grade"] = df["Grade_Current"]
        if context:
            print(f"Using Grade_Current for {context}.")
        return df
    if all(c in df.columns for c in ["Grade", "FundID", "quarter_end"]):
        df["Grade"] = df["Grade"].astype(str).str.strip()
        df.loc[df["Grade"].isin(["", "nan", "None", "NaN", "<NA>"]), "Grade"] = np.nan
        df = df.sort_values(["FundID", "quarter_end"])
        df["Grade_Current"] = df.groupby("FundID")["Grade"].ffill()
        df["Grade"] = df["Grade_Current"]
        if context:
            print(f"Computed Grade_Current (forward fill) for {context}.")
    return df

def parse_quarter(q) -> float:
    if pd.isna(q):
        return np.nan
    if isinstance(q, (int, np.integer, float, np.floating)):
        return float(q)
    s = str(q).strip().upper()
    if s.startswith("Q"):
        s = s[1:]
    try:
        return float(s)
    except Exception:
        return np.nan


def add_quarter_end(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Quarter"] = df["Quarter"].apply(parse_quarter)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    m = df["Year"].notna() & df["Quarter"].notna()
    years = df.loc[m, "Year"].astype(int)
    quarters = df.loc[m, "Quarter"].astype(int)
    df.loc[m, "quarter_end"] = pd.PeriodIndex(year=years, quarter=quarters, freq="Q").to_timestamp("Q")
    return df


def transition_matrix(df: pd.DataFrame, out_path: str) -> None:
    counts = (df.groupby(["Grade", "Grade_next"])
                .size()
                .unstack(fill_value=0)
                .reindex(index=GRADE_ORDER, columns=GRADE_ORDER, fill_value=0))
    probs = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    probs.to_csv(out_path)


def slug(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(s)).strip("_")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="anonymized.csv", help="Input anonymized CSV")
    ap.add_argument("--out-dir", default="model_fits/outputs/transitions", help="Output directory")
    args = ap.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        candidates = list(Path.cwd().glob("**/anonymized.csv"))
        if candidates:
            input_path = str(candidates[0])
            print("Using INPUT_PATH:", input_path)
        else:
            raise FileNotFoundError("anonymized.csv not found. Pass --input with full path.")

    df = pd.read_csv(input_path, engine="python")
    df = normalize_columns(df)
    df = add_quarter_end(df)
    df = apply_current_grade(df, context="transitions")
    df = df.dropna(subset=["FundID", "Grade", "quarter_end"])
    df["Grade"] = df["Grade"].astype(str).str.strip()
    df = df[df["Grade"].isin(GRADE_ORDER)]

    # Map grade at t and t+4 quarters
    df = df.sort_values(["FundID", "quarter_end"])
    base = df[["FundID", "quarter_end", "Grade", "Adj Strategy"]].drop_duplicates()
    base["next_qe"] = (base["quarter_end"].dt.to_period("Q") + 4).dt.to_timestamp("Q")
    nxt = base[["FundID", "quarter_end", "Grade"]].rename(columns={
        "quarter_end": "next_qe",
        "Grade": "Grade_next",
    })
    merged = base.merge(nxt, on=["FundID", "next_qe"], how="left").dropna(subset=["Grade_next"])

    os.makedirs(args.out_dir, exist_ok=True)
    transition_matrix(merged, os.path.join(args.out_dir, "grade_transition_1y_all.csv"))

    if "Adj Strategy" in merged.columns:
        for strat, g in merged.groupby("Adj Strategy", dropna=False):
            out = os.path.join(args.out_dir, f"grade_transition_1y_{slug(strat)}.csv")
            transition_matrix(g, out)

    print("Wrote transitions to:", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
