#!/usr/bin/env python3
"""
Estimate copula correlation parameters (rho_event, rho_size) from anonymized data.
Uses rank-normalized series across funds to approximate Gaussian copula correlation.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


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
    rename[_get("Adj Drawdown EUR")] = "Adj Drawdown EUR"
    rename[_get("Adj Repayment EUR")] = "Adj Repayment EUR"
    rename[_get("Recallable")] = "Recallable"
    rename[_get("Capacity")] = "Capacity"
    rename[_get("NAV Adjusted EUR")] = "NAV Adjusted EUR"
    return df.rename(columns=rename)


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


def rank_to_normal(x: pd.Series) -> pd.Series:
    x = x.copy()
    n = x.notna().sum()
    if n < 3:
        return pd.Series(index=x.index, dtype=float)
    ranks = x.rank(method="average", na_option="keep")
    u = (ranks - 0.5) / n
    u = u.clip(1e-6, 1 - 1e-6)
    return pd.Series(norm.ppf(u), index=x.index)


def avg_offdiag_corr(mat: pd.DataFrame) -> float:
    if mat.shape[1] < 2:
        return float("nan")
    corr = mat.corr()
    vals = corr.values
    n = vals.shape[0]
    if n <= 1:
        return float("nan")
    return float((vals.sum() - np.trace(vals)) / (n * (n - 1)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="anonymized.csv", help="Input anonymized CSV")
    ap.add_argument("--out", default="model_fits/outputs/copula_params.json", help="Output JSON path")
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
    df = df.dropna(subset=["FundID", "quarter_end"])
    df = df.sort_values(["FundID", "quarter_end"])

    # Event indicators
    df["draw_event"] = pd.to_numeric(df["Adj Drawdown EUR"], errors="coerce").fillna(0.0) > 0
    df["rep_event"] = pd.to_numeric(df["Adj Repayment EUR"], errors="coerce").fillna(0.0) > 0

    # Ratios for size
    df["nav_prev"] = df.groupby("FundID")["NAV Adjusted EUR"].shift(1)
    cap = pd.to_numeric(df["Capacity"], errors="coerce")
    draw = pd.to_numeric(df["Adj Drawdown EUR"], errors="coerce")
    rep = pd.to_numeric(df["Adj Repayment EUR"], errors="coerce")
    nav_prev = pd.to_numeric(df["nav_prev"], errors="coerce")

    df["draw_ratio"] = np.where(cap > 0, draw / cap, 0.0)
    df["rep_ratio"] = np.where(nav_prev.abs() > 1.0, rep / nav_prev.abs(), 0.0)

    # Wide matrices: quarter_end x fund
    draw_event_w = df.pivot_table(index="quarter_end", columns="FundID", values="draw_event", aggfunc="last")
    rep_event_w = df.pivot_table(index="quarter_end", columns="FundID", values="rep_event", aggfunc="last")
    draw_ratio_w = df.pivot_table(index="quarter_end", columns="FundID", values="draw_ratio", aggfunc="last").fillna(0.0)
    rep_ratio_w = df.pivot_table(index="quarter_end", columns="FundID", values="rep_ratio", aggfunc="last").fillna(0.0)

    # Rank-normalize per fund (column-wise)
    draw_event_z = draw_event_w.apply(rank_to_normal, axis=0)
    rep_event_z = rep_event_w.apply(rank_to_normal, axis=0)
    draw_ratio_z = draw_ratio_w.apply(rank_to_normal, axis=0)
    rep_ratio_z = rep_ratio_w.apply(rank_to_normal, axis=0)

    avg_corr_draw_event = avg_offdiag_corr(draw_event_z)
    avg_corr_rep_event = avg_offdiag_corr(rep_event_z)
    avg_corr_draw_size = avg_offdiag_corr(draw_ratio_z)
    avg_corr_rep_size = avg_offdiag_corr(rep_ratio_z)

    # Suggested rho for one-factor Gaussian: corr ≈ rho^2
    def rho_from_corr(c):
        if not np.isfinite(c) or c <= 0:
            return 0.0
        return float(np.sqrt(min(c, 0.99)))

    rho_event = rho_from_corr(np.nanmean([avg_corr_draw_event, avg_corr_rep_event]))
    rho_size = rho_from_corr(np.nanmean([avg_corr_draw_size, avg_corr_rep_size]))

    out = {
        "avg_corr_draw_event": avg_corr_draw_event,
        "avg_corr_rep_event": avg_corr_rep_event,
        "avg_corr_draw_size": avg_corr_draw_size,
        "avg_corr_rep_size": avg_corr_rep_size,
        "rho_event": rho_event,
        "rho_size": rho_size,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print("Wrote:", args.out)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
