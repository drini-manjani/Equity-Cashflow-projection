#!/usr/bin/env python3
"""
Fit candidate distributions for drawdown/repayment/recallable ratios.

Usage:
  python3 model_fits/fit_ratios.py --input anonymized.csv --out-dir model_fits/outputs
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception as e:
    raise ImportError("scipy is required: pip install scipy") from e


RATIO_CONFIG = {
    "draw_ratio": {
        "label": "draw_ratio",
        "src": "Drawdown_Ratio",
        "compute": True,
    },
    "rep_ratio": {
        "label": "rep_ratio",
        "src": "Repayment_Ratio",
        "compute": True,
    },
    "rc_ratio_given_rep": {
        "label": "rc_ratio_given_rep",
        "src": None,
        "compute": True,
    },
}


AGE_BINS_Q = [-1, 3, 7, 11, 15, 19, 1000]
AGE_LABELS = ["0-3", "4-7", "8-11", "12-15", "16-19", "20+"]


def _norm_key(s: str) -> str:
    return " ".join(s.strip().lower().replace("_", " ").split())


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # map normalized column names to actual
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
    # Optional current-grade column (preferred if present)
    rename[_get("Current Grade")] = "Grade_Current"
    rename[_get("CurrentGrade")] = "Grade_Current"
    rename[_get("Grade Current")] = "Grade_Current"
    rename[_get("Grade_Current")] = "Grade_Current"
    rename[_get("Adj Drawdown EUR")] = "Adj Drawdown EUR"
    rename[_get("Adj Repayment EUR")] = "Adj Repayment EUR"
    rename[_get("NAV Adjusted EUR")] = "NAV Adjusted EUR"
    rename[_get("Capacity")] = "Capacity"
    rename[_get("Recallable")] = "Recallable"
    rename[_get("Fund_Age_Quarters")] = "Fund_Age_Quarters"
    rename[_get("Drawdown_Ratio")] = "Drawdown_Ratio"
    rename[_get("Repayment_Ratio")] = "Repayment_Ratio"

    df = df.rename(columns=rename)
    return df


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


def build_quarter_end(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Quarter"] = df["Quarter"].apply(parse_quarter)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    m = df["Year"].notna() & df["Quarter"].notna()
    years = df.loc[m, "Year"].astype(int)
    quarters = df.loc[m, "Quarter"].astype(int)
    df.loc[m, "quarter_end"] = pd.PeriodIndex(year=years, quarter=quarters, freq="Q").to_timestamp("Q")
    return df


def compute_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["FundID", "quarter_end"])
    df["nav_prev"] = df.groupby("FundID")["NAV Adjusted EUR"].shift(1)

    # Drawdown ratio (prefer computed)
    cap = pd.to_numeric(df["Capacity"], errors="coerce")
    draw = pd.to_numeric(df["Adj Drawdown EUR"], errors="coerce")
    df["draw_ratio_calc"] = np.where(cap > 0, draw / cap, np.nan)

    # Repayment ratio (prefer computed)
    nav_prev = pd.to_numeric(df["nav_prev"], errors="coerce")
    rep = pd.to_numeric(df["Adj Repayment EUR"], errors="coerce")
    df["rep_ratio_calc"] = np.where(nav_prev.abs() > 1.0, rep / nav_prev.abs(), np.nan)

    # Recallable ratio given repayment
    rc = pd.to_numeric(df["Recallable"], errors="coerce")
    df["rc_ratio_given_rep"] = np.where(rep > 0, rc / rep, np.nan)

    # Age buckets
    if "Fund_Age_Quarters" in df.columns:
        df["AgeBucket"] = pd.cut(df["Fund_Age_Quarters"], bins=AGE_BINS_Q, labels=AGE_LABELS)
    else:
        df["AgeBucket"] = "ALL"

    return df


def _fit_dist(x: np.ndarray, dist_name: str) -> Tuple[Dict, float, float]:
    dist = getattr(stats, dist_name)
    # Choose fit constraints for stability
    if dist_name in ("beta", "logitnorm"):
        params = dist.fit(x, floc=0, fscale=1)
    elif dist_name in ("lognorm", "gamma", "weibull_min", "fisk"):
        params = dist.fit(x, floc=0)
    else:
        params = dist.fit(x)
    ll = float(np.sum(dist.logpdf(x, *params)))
    k = len(params)
    return {"params": params, "k": k}, ll, k


def _ks_pvalue(x: np.ndarray, dist_name: str, params: Tuple) -> float:
    dist = getattr(stats, dist_name)
    try:
        d, p = stats.kstest(x, dist_name, args=params)
        return float(p)
    except Exception:
        return float("nan")


def fit_candidates(x: np.ndarray, dist_names: List[str], eps: float = 1e-9) -> List[Dict]:
    rows = []
    for name in dist_names:
        try:
            if name in ("beta", "logitnorm"):
                x_fit = x[(x > eps) & (x < 1.0 - eps)]
            else:
                x_fit = x
            n_fit = len(x_fit)
            if n_fit == 0:
                rows.append({
                    "dist": name,
                    "ll": np.nan,
                    "aic": np.nan,
                    "bic": np.nan,
                    "ks_p": np.nan,
                    "k": np.nan,
                    "n_fit": 0,
                    "params": "no_data",
                })
                continue
            fit_info, ll, k = _fit_dist(x_fit, name)
            n = n_fit
            aic = 2 * k - 2 * ll
            bic = np.log(n) * k - 2 * ll
            ks_p = _ks_pvalue(x_fit, name, fit_info["params"])
            rows.append({
                "dist": name,
                "ll": ll,
                "aic": aic,
                "bic": bic,
                "ks_p": ks_p,
                "k": k,
                "n_fit": n_fit,
                "params": repr(tuple(float(p) for p in fit_info["params"])),
            })
        except Exception as e:
            rows.append({
                "dist": name,
                "ll": np.nan,
                "aic": np.nan,
                "bic": np.nan,
                "ks_p": np.nan,
                "k": np.nan,
                "n_fit": 0,
                "params": f"error: {e}",
            })
    return rows


def fit_ratio_series(x: pd.Series, dist_names: List[str], eps: float = 1e-9) -> Dict:
    x = pd.to_numeric(x, errors="coerce").dropna()
    n = len(x)
    if n == 0:
        return {"n": 0, "n_pos": 0, "zero_share": np.nan, "share_gt_1": np.nan, "fits": []}
    x = x.clip(lower=0.0)
    n_zero = int((x <= eps).sum())
    x_pos = x[x > eps].to_numpy(dtype=float)
    zero_share = n_zero / float(n)
    share_gt_1 = float((x > 1.0).mean()) if n else np.nan
    fits = fit_candidates(x_pos, dist_names, eps=eps)
    return {"n": n, "n_pos": len(x_pos), "zero_share": zero_share, "share_gt_1": share_gt_1, "fits": fits}


def _best_fit(res: Dict) -> Dict:
    if not res.get("fits"):
        return {}
    return sorted(res["fits"], key=lambda r: (np.nan_to_num(r["aic"], nan=np.inf)))[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="anonymized.csv", help="Input anonymized CSV/Parquet (auto-detect if missing)")
    ap.add_argument("--out-dir", default="model_fits/outputs", help="Output directory")
    ap.add_argument("--group-cols", default="Adj Strategy,Grade,AgeBucket", help="Comma-separated group columns")
    ap.add_argument("--min-obs", type=int, default=None, help="(Deprecated) Minimum observations for all groups")
    ap.add_argument("--min-obs-age", type=int, default=150, help="Minimum observations for Strategy×Grade×AgeBucket")
    ap.add_argument("--min-obs-sg", type=int, default=200, help="Minimum observations for Strategy×Grade")
    ap.add_argument("--min-obs-s", type=int, default=300, help="Minimum observations for Strategy")
    ap.add_argument("--eps", type=float, default=1e-9, help="Zero threshold for ratios")
    args = ap.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        candidates = list(Path.cwd().glob("**/anonymized.csv"))
        if candidates:
            input_path = str(candidates[0])
            print("Using INPUT_PATH:", input_path)
        else:
            raise FileNotFoundError("anonymized.csv not found. Pass --input with the full path.")

    df = pd.read_csv(input_path, engine="python")
    df = normalize_columns(df)
    df = build_quarter_end(df)
    df = apply_current_grade(df, context="fitting")
    if "Adj Strategy" not in df.columns or "Grade" not in df.columns:
        raise KeyError("Missing required columns: Adj Strategy, Grade")

    df = compute_ratios(df)

    # choose series
    draw_series = df["draw_ratio_calc"] if "draw_ratio_calc" in df.columns else df.get("Drawdown_Ratio")
    rep_series = df["rep_ratio_calc"] if "rep_ratio_calc" in df.columns else df.get("Repayment_Ratio")
    rc_series = df["rc_ratio_given_rep"]

    dist_names = ["beta", "logitnorm", "lognorm", "gamma", "weibull_min", "fisk"]

    global_rows = []
    for name, series in [("draw_ratio", draw_series), ("rep_ratio", rep_series), ("rc_ratio_given_rep", rc_series)]:
        res = fit_ratio_series(series, dist_names, eps=args.eps)
        for f in res["fits"]:
            global_rows.append({
                "ratio": name,
                "n": res["n"],
                "n_pos": res["n_pos"],
                "zero_share": res["zero_share"],
                "share_gt_1": res["share_gt_1"],
                **f,
            })

    os.makedirs(args.out_dir, exist_ok=True)
    pd.DataFrame(global_rows).sort_values(["ratio", "aic"]).to_csv(
        os.path.join(args.out_dir, "ratio_fit_global.csv"), index=False
    )

    # Grouped fits at multiple levels with fallback
    if args.min_obs is not None:
        args.min_obs_age = args.min_obs
        args.min_obs_sg = args.min_obs
        args.min_obs_s = args.min_obs

    group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()]
    if "AgeBucket" not in group_cols:
        group_cols.append("AgeBucket")

    def _fit_level(level_name: str, cols: List[str], min_obs: int) -> List[Dict]:
        rows = []
        for gkey, g in df.groupby(cols):
            if not isinstance(gkey, tuple):
                gkey = (gkey,)
            if len(g) < min_obs:
                continue
            for name, series in [("draw_ratio", g["draw_ratio_calc"]),
                                 ("rep_ratio", g["rep_ratio_calc"]),
                                 ("rc_ratio_given_rep", g["rc_ratio_given_rep"])]:
                res = fit_ratio_series(series, dist_names, eps=args.eps)
                best = _best_fit(res)
                if not best:
                    continue
                row = {
                    "ratio": name,
                    "level": level_name,
                    "group_n": len(g),
                    "n": res["n"],
                    "n_pos": res["n_pos"],
                    "zero_share": res["zero_share"],
                    "share_gt_1": res["share_gt_1"],
                    **best,
                }
                for idx, col in enumerate(cols):
                    row[col] = gkey[idx]
                rows.append(row)
        return rows

    lvl_age = _fit_level("strategy_grade_age", ["Adj Strategy", "Grade", "AgeBucket"], args.min_obs_age)
    lvl_sg = _fit_level("strategy_grade", ["Adj Strategy", "Grade"], args.min_obs_sg)
    lvl_s = _fit_level("strategy", ["Adj Strategy"], args.min_obs_s)

    by_group = pd.DataFrame(lvl_age + lvl_sg + lvl_s)
    by_group.to_csv(os.path.join(args.out_dir, "ratio_fit_by_group.csv"), index=False)

    # Build a selected table with fallback: SxGxA -> SxG -> S -> global
    global_df = pd.DataFrame(global_rows)
    best_global = (global_df.sort_values(["ratio", "aic"])
                   .groupby("ratio").head(1)
                   .set_index("ratio"))

    base_groups = df[["Adj Strategy", "Grade", "AgeBucket"]].dropna().drop_duplicates()
    selected_rows = []
    for _, r in base_groups.iterrows():
        s, g, a = r["Adj Strategy"], r["Grade"], r["AgeBucket"]
        for ratio in ["draw_ratio", "rep_ratio", "rc_ratio_given_rep"]:
            row = None
            if not by_group.empty:
                m = (by_group["ratio"] == ratio) & (by_group["level"] == "strategy_grade_age") & \
                    (by_group["Adj Strategy"] == s) & (by_group["Grade"] == g) & (by_group["AgeBucket"] == a)
                if m.any():
                    row = by_group.loc[m].iloc[0].to_dict()
            if row is None and not by_group.empty:
                m = (by_group["ratio"] == ratio) & (by_group["level"] == "strategy_grade") & \
                    (by_group["Adj Strategy"] == s) & (by_group["Grade"] == g)
                if m.any():
                    row = by_group.loc[m].iloc[0].to_dict()
            if row is None and not by_group.empty:
                m = (by_group["ratio"] == ratio) & (by_group["level"] == "strategy") & \
                    (by_group["Adj Strategy"] == s)
                if m.any():
                    row = by_group.loc[m].iloc[0].to_dict()
            if row is None and ratio in best_global.index:
                row = best_global.loc[ratio].to_dict()
                row["level"] = "global"
            if row is None:
                continue
            row["Adj Strategy"] = s
            row["Grade"] = g
            row["AgeBucket"] = a
            selected_rows.append(row)

    selected = pd.DataFrame(selected_rows)
    selected.to_csv(os.path.join(args.out_dir, "ratio_fit_selected.csv"), index=False)

    print("Wrote:")
    print(os.path.join(args.out_dir, "ratio_fit_global.csv"))
    print(os.path.join(args.out_dir, "ratio_fit_by_group.csv"))
    print(os.path.join(args.out_dir, "ratio_fit_selected.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
