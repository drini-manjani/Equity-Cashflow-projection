#!/usr/bin/env python3
"""
Fit timing probabilities for draw/rep/recallable events by Strategy×Grade×AgeBucket.
Produces a selected table with fallback: S×G×A → S×G → S → global.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


AGE_BINS_Q = [-1, 3, 7, 11, 15, 19, 1000]
AGE_LABELS = ["0-3", "4-7", "8-11", "12-15", "16-19", "20+"]


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
    rename[_get("Adj Drawdown EUR")] = "Adj Drawdown EUR"
    rename[_get("Adj Repayment EUR")] = "Adj Repayment EUR"
    rename[_get("Recallable")] = "Recallable"
    rename[_get("Fund_Age_Quarters")] = "Fund_Age_Quarters"
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="anonymized.csv", help="Input anonymized CSV")
    ap.add_argument("--out-dir", default="model_fits/outputs", help="Output directory")
    ap.add_argument("--min-obs-age", type=int, default=150)
    ap.add_argument("--min-obs-sg", type=int, default=200)
    ap.add_argument("--min-obs-s", type=int, default=300)
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
    df = apply_current_grade(df, context="timing curves")

    # Age buckets
    if "Fund_Age_Quarters" in df.columns:
        df["AgeBucket"] = pd.cut(pd.to_numeric(df["Fund_Age_Quarters"], errors="coerce"),
                                 bins=AGE_BINS_Q, labels=AGE_LABELS)
    else:
        df["AgeBucket"] = "ALL"

    # Events
    df["draw_event"] = pd.to_numeric(df["Adj Drawdown EUR"], errors="coerce").fillna(0.0) > 0
    df["rep_event"] = pd.to_numeric(df["Adj Repayment EUR"], errors="coerce").fillna(0.0) > 0
    df["rc_event"] = pd.to_numeric(df["Recallable"], errors="coerce").fillna(0.0) > 0

    os.makedirs(args.out_dir, exist_ok=True)

    def _fit_level(level_name: str, cols: list, min_obs: int):
        rows = []
        for gkey, g in df.groupby(cols):
            if not isinstance(gkey, tuple):
                gkey = (gkey,)
            if len(g) < min_obs:
                continue
            n_obs = len(g)
            n_draw = int(g["draw_event"].sum())
            n_rep = int(g["rep_event"].sum())
            n_rc = int(g["rc_event"].sum())
            p_draw = n_draw / n_obs if n_obs else 0.0
            p_rep = n_rep / n_obs if n_obs else 0.0
            p_rc = n_rc / n_rep if n_rep else 0.0
            row = {
                "level": level_name,
                "n_obs": n_obs,
                "n_draw": n_draw,
                "n_rep": n_rep,
                "n_rc": n_rc,
                "p_draw": p_draw,
                "p_rep": p_rep,
                "p_rc_given_rep": p_rc,
            }
            for idx, col in enumerate(cols):
                row[col] = gkey[idx]
            rows.append(row)
        return rows

    lvl_age = _fit_level("strategy_grade_age", ["Adj Strategy", "Grade", "AgeBucket"], args.min_obs_age)
    lvl_sg = _fit_level("strategy_grade", ["Adj Strategy", "Grade"], args.min_obs_sg)
    lvl_s = _fit_level("strategy", ["Adj Strategy"], args.min_obs_s)

    by_group = pd.DataFrame(lvl_age + lvl_sg + lvl_s)
    by_group.to_csv(os.path.join(args.out_dir, "timing_probs_by_group.csv"), index=False)

    # Fallback table
    base_groups = df[["Adj Strategy", "Grade", "AgeBucket"]].dropna().drop_duplicates()
    selected_rows = []
    global_row = _fit_level("global", [], 1)
    global_row = global_row[0] if global_row else {}

    for _, r in base_groups.iterrows():
        s, g, a = r["Adj Strategy"], r["Grade"], r["AgeBucket"]
        row = None
        if not by_group.empty:
            m = (by_group["level"] == "strategy_grade_age") & \
                (by_group["Adj Strategy"] == s) & (by_group["Grade"] == g) & (by_group["AgeBucket"] == a)
            if m.any():
                row = by_group.loc[m].iloc[0].to_dict()
        if row is None and not by_group.empty:
            m = (by_group["level"] == "strategy_grade") & \
                (by_group["Adj Strategy"] == s) & (by_group["Grade"] == g)
            if m.any():
                row = by_group.loc[m].iloc[0].to_dict()
        if row is None and not by_group.empty:
            m = (by_group["level"] == "strategy") & (by_group["Adj Strategy"] == s)
            if m.any():
                row = by_group.loc[m].iloc[0].to_dict()
        if row is None and global_row:
            row = dict(global_row)
            row["level"] = "global"

        if row:
            row["Adj Strategy"] = s
            row["Grade"] = g
            row["AgeBucket"] = a
            selected_rows.append(row)

    selected = pd.DataFrame(selected_rows)
    selected.to_csv(os.path.join(args.out_dir, "timing_probs_selected.csv"), index=False)

    print("Wrote:")
    print(os.path.join(args.out_dir, "timing_probs_by_group.csv"))
    print(os.path.join(args.out_dir, "timing_probs_selected.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
