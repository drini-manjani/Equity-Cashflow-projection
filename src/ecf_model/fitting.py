from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from .config import FitConfig
from .features import add_size_bucket, compute_fund_end_dates
from .utils import clip01, ensure_dir, make_age_bucket, recallable_to_flow, winsorize_series


TIMING_METRICS = ("draw", "rep", "rc")
SIZE_METRICS = ("draw_ratio", "rep_ratio", "rc_ratio_given_rep")
GROUP_COLS = ["Adj Strategy", "Grade", "AgeBucket", "size_bucket"]


@dataclass
class FitArtifacts:
    timing_probs: pd.DataFrame
    ratio_fits: pd.DataFrame
    omega_fits: pd.DataFrame
    target_size_bins: dict

    def save(self, calibration_dir: Path) -> None:
        ensure_dir(calibration_dir)
        self.timing_probs.to_csv(calibration_dir / "timing_probs_selected.csv", index=False)
        self.ratio_fits.to_csv(calibration_dir / "ratio_fit_selected.csv", index=False)
        self.omega_fits.to_csv(calibration_dir / "omega_selected.csv", index=False)
        with (calibration_dir / "target_fund_size_bins.json").open("w", encoding="utf-8") as f:
            json.dump(self.target_size_bins, f, indent=2)


@dataclass
class FitBase:
    data: pd.DataFrame
    global_rates: dict[str, float]


def _wilson_ci(events: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = events / n
    denom = 1 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    spread = (z * np.sqrt((p * (1 - p) / n) + (z**2) / (4 * n**2))) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


def _level_parent(level: str, row: pd.Series) -> dict:
    strat = row["Adj Strategy"]
    grade = row["Grade"]
    age = row["AgeBucket"]
    if level == "strategy":
        return {"level": "global", "Adj Strategy": "ALL", "Grade": "ALL", "AgeBucket": "ALL", "size_bucket": "ALL"}
    if level == "strategy_age":
        return {"level": "strategy", "Adj Strategy": strat, "Grade": "ALL", "AgeBucket": "ALL", "size_bucket": "ALL"}
    if level == "strategy_grade_age":
        return {"level": "strategy_age", "Adj Strategy": strat, "Grade": "ALL", "AgeBucket": age, "size_bucket": "ALL"}
    if level == "strategy_grade_age_size":
        return {"level": "strategy_grade_age", "Adj Strategy": strat, "Grade": grade, "AgeBucket": age, "size_bucket": "ALL"}
    return {"level": "global", "Adj Strategy": "ALL", "Grade": "ALL", "AgeBucket": "ALL", "size_bucket": "ALL"}


def _attach_group_level(df: pd.DataFrame, level: str) -> pd.DataFrame:
    out = df.copy()
    out["level"] = level
    if level == "global":
        out["Adj Strategy"] = "ALL"
        out["Grade"] = "ALL"
        out["AgeBucket"] = "ALL"
        out["size_bucket"] = "ALL"
    elif level == "strategy":
        out["Grade"] = "ALL"
        out["AgeBucket"] = "ALL"
        out["size_bucket"] = "ALL"
    elif level == "strategy_age":
        out["Grade"] = "ALL"
        out["size_bucket"] = "ALL"
    elif level == "strategy_grade_age":
        out["size_bucket"] = "ALL"
    return out


def prepare_fit_base(hist: pd.DataFrame, cfg: FitConfig) -> FitBase:
    df = add_size_bucket(hist, list(cfg.size_bins), list(cfg.size_labels))
    df = df.sort_values(["FundID", "quarter_end"]).copy()
    df["AgeBucket"] = df["Fund_Age_Quarters"].apply(make_age_bucket)

    draw_abs = df["Adj Drawdown EUR"].abs()
    rep_abs = df["Adj Repayment EUR"].abs()
    rc_abs = (
        df.sort_values(["FundID", "quarter_end"])
        .groupby("FundID")["Recallable"]
        .apply(recallable_to_flow)
        .reset_index(level=0, drop=True)
        .reindex(df.index)
        .abs()
    )
    nav_curr = df["NAV Adjusted EUR"].clip(lower=0)
    nav_prev = nav_curr.groupby(df["FundID"]).shift(1)

    df["draw_event"] = (draw_abs > 0).astype(int)
    df["rep_event"] = ((rep_abs > 0) & (nav_prev > cfg.nav_gate)).astype(int)
    df["rc_event"] = ((rc_abs > 0) & (rep_abs > 0)).astype(int)

    denom_draw = df.groupby("FundID")["Commitment EUR"].transform("max").replace(0, np.nan)
    df["draw_ratio"] = np.where(denom_draw > 0, draw_abs / denom_draw, np.nan)
    df["rep_ratio"] = np.where(nav_prev > cfg.nav_gate, rep_abs / nav_prev, np.nan)
    df["rc_ratio_given_rep"] = np.where(rep_abs > 0, rc_abs / rep_abs, np.nan)

    flow = draw_abs - rep_abs
    base = nav_prev + flow
    omega_raw = np.where(base > cfg.nav_gate, nav_curr / base - 1.0, np.nan)
    omega_s = pd.Series(omega_raw, index=df.index, dtype=float)
    df["omega_obs"] = omega_s.clip(lower=cfg.omega_fit_clip[0], upper=cfg.omega_fit_clip[1])

    global_rates = {
        "draw": float(df["draw_event"].mean()),
        "rep": float(df["rep_event"].mean()),
        "rc": float(df["rc_event"].mean()),
    }
    return FitBase(data=df, global_rates=global_rates)


def fit_timing_probs(base: FitBase, cfg: FitConfig) -> pd.DataFrame:
    df = base.data
    levels = [
        ("global", []),
        ("strategy", ["Adj Strategy"]),
        ("strategy_age", ["Adj Strategy", "AgeBucket"]),
        ("strategy_grade_age", ["Adj Strategy", "Grade", "AgeBucket"]),
        ("strategy_grade_age_size", ["Adj Strategy", "Grade", "AgeBucket", "size_bucket"]),
    ]

    rows = []
    ref: dict[tuple, dict[str, float]] = {}

    for level, gcols in levels:
        if gcols:
            grouped = df.groupby(gcols, dropna=False)
        else:
            grouped = [((), df)]

        for key, g in grouped:
            key_tuple = key if isinstance(key, tuple) else (key,)
            base_row = {
                "Adj Strategy": key_tuple[0] if "Adj Strategy" in gcols else "ALL",
                "Grade": key_tuple[gcols.index("Grade")] if "Grade" in gcols else "ALL",
                "AgeBucket": key_tuple[gcols.index("AgeBucket")] if "AgeBucket" in gcols else "ALL",
                "size_bucket": key_tuple[gcols.index("size_bucket")] if "size_bucket" in gcols else "ALL",
                "level": level,
                "n_obs": int(len(g)),
            }

            parent_key = None
            if level != "global":
                parent = _level_parent(level, pd.Series(base_row))
                parent_key = (
                    parent["level"],
                    parent["Adj Strategy"],
                    parent["Grade"],
                    parent["AgeBucket"],
                    parent["size_bucket"],
                )

            row = dict(base_row)
            for metric in TIMING_METRICS:
                event_col = f"{metric}_event"
                events = int(g[event_col].sum())
                n = int(len(g))
                raw = events / n if n > 0 else 0.0
                if parent_key is None or parent_key not in ref:
                    prior_p = base.global_rates[metric]
                else:
                    prior_p = ref[parent_key][f"p_{metric}"]
                p_smooth = (events + cfg.prior_strength * prior_p) / (n + cfg.prior_strength)
                lo, hi = _wilson_ci(events, n)

                row[f"events_{metric}"] = events
                row[f"p_{metric}_raw"] = raw
                row[f"p_{metric}"] = clip01(float(p_smooth))
                row[f"p_{metric}_ci_low"] = lo
                row[f"p_{metric}_ci_high"] = hi

            ref[(level, row["Adj Strategy"], row["Grade"], row["AgeBucket"], row["size_bucket"])] = {
                f"p_{m}": row[f"p_{m}"] for m in TIMING_METRICS
            }
            rows.append(row)

    out = pd.DataFrame(rows)
    # Keep only granular + useful fallback levels for projection.
    keep_levels = {"strategy_grade_age_size", "strategy_grade_age", "strategy_age", "strategy", "global"}
    out = out[out["level"].isin(keep_levels)].reset_index(drop=True)
    return out


def _fit_candidate_distributions(values: np.ndarray) -> tuple[str, tuple[float, ...], float, float]:
    # Returns dist_name, params, aic, ks_pvalue
    vals = values[np.isfinite(values) & (values > 0)]
    if len(vals) < 8:
        return ("empirical", (float(np.mean(vals)) if len(vals) else 0.0,), np.nan, np.nan)

    fits: list[tuple[str, tuple[float, ...], float, float]] = []

    # Lognormal with loc fixed at 0 for stability.
    try:
        shape, loc, scale = stats.lognorm.fit(vals, floc=0)
        ll = np.sum(stats.lognorm.logpdf(vals, shape, loc=loc, scale=scale))
        aic = 2 * 2 - 2 * ll
        ks_p = stats.kstest(vals, "lognorm", args=(shape, loc, scale)).pvalue
        fits.append(("lognorm", (shape, loc, scale), aic, ks_p))
    except Exception:
        pass

    # Gamma with loc fixed 0.
    try:
        a, loc, scale = stats.gamma.fit(vals, floc=0)
        ll = np.sum(stats.gamma.logpdf(vals, a, loc=loc, scale=scale))
        aic = 2 * 2 - 2 * ll
        ks_p = stats.kstest(vals, "gamma", args=(a, loc, scale)).pvalue
        fits.append(("gamma", (a, loc, scale), aic, ks_p))
    except Exception:
        pass

    if not fits:
        return ("empirical", (float(np.mean(vals)),), np.nan, np.nan)
    fits.sort(key=lambda x: x[2])
    return fits[0]


def fit_ratio_models(base: FitBase, cfg: FitConfig) -> pd.DataFrame:
    df = base.data
    levels = [
        ("strategy_grade_age_size", ["Adj Strategy", "Grade", "AgeBucket", "size_bucket"]),
        ("strategy_grade_age", ["Adj Strategy", "Grade", "AgeBucket"]),
        ("strategy_age", ["Adj Strategy", "AgeBucket"]),
        ("strategy", ["Adj Strategy"]),
        ("global", []),
    ]

    rows = []
    for metric in SIZE_METRICS:
        for level, gcols in levels:
            if gcols:
                grouped = df.groupby(gcols, dropna=False)
            else:
                grouped = [((), df)]

            for key, g in grouped:
                key_tuple = key if isinstance(key, tuple) else (key,)

                if metric == "rc_ratio_given_rep":
                    vals = g.loc[g[metric] > 0, metric]
                else:
                    vals = g.loc[g[metric] > 0, metric]

                vals = vals[np.isfinite(vals)]
                vals = winsorize_series(vals, cfg.ratio_winsor_lower_q, cfg.ratio_winsor_upper_q)

                if len(vals) < 4:
                    continue

                dist, params, aic, ks_p = _fit_candidate_distributions(vals.to_numpy(dtype=float))

                row = {
                    "metric": metric,
                    "level": level,
                    "Adj Strategy": key_tuple[0] if "Adj Strategy" in gcols else "ALL",
                    "Grade": key_tuple[gcols.index("Grade")] if "Grade" in gcols else "ALL",
                    "AgeBucket": key_tuple[gcols.index("AgeBucket")] if "AgeBucket" in gcols else "ALL",
                    "size_bucket": key_tuple[gcols.index("size_bucket")] if "size_bucket" in gcols else "ALL",
                    "n_pos": int(len(vals)),
                    "mean": float(vals.mean()),
                    "p50": float(vals.quantile(0.5)),
                    "p90": float(vals.quantile(0.9)),
                    "dist": dist,
                    "params": json.dumps([float(x) for x in params]),
                    "aic": float(aic) if np.isfinite(aic) else np.nan,
                    "ks_pvalue": float(ks_p) if np.isfinite(ks_p) else np.nan,
                }
                rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values(["metric", "level", "Adj Strategy", "Grade", "AgeBucket", "size_bucket"]).reset_index(drop=True)
    return out


def _ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = 1.0, robust_iters: int = 2) -> tuple[np.ndarray, float]:
    if len(y) == 0:
        return np.zeros(X.shape[1], dtype=float), 0.0

    beta = np.zeros(X.shape[1], dtype=float)
    w = np.ones(len(y), dtype=float)
    n_iters = max(int(robust_iters), 0) + 1

    for _ in range(n_iters):
        sw = np.sqrt(np.clip(w, 1e-6, None))
        Xw = X * sw[:, None]
        yw = y * sw

        xtx = Xw.T @ Xw
        penalty = np.eye(xtx.shape[0]) * lam
        penalty[0, 0] = 0.0  # Do not penalize intercept.
        rhs = Xw.T @ yw
        try:
            beta = np.linalg.solve(xtx + penalty, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(xtx + penalty) @ rhs

        resid = y - X @ beta
        med = float(np.median(resid))
        mad = float(np.median(np.abs(resid - med))) + 1e-6
        u = (resid - med) / (4.685 * mad)
        # Tukey bisquare to reduce leverage from extreme omega outliers.
        w = np.where(np.abs(u) < 1.0, (1.0 - u**2) ** 2, 0.05)

    resid = y - X @ beta
    sigma = float(np.sqrt(np.mean(resid**2))) if len(resid) else 0.0
    return beta, sigma


def fit_omega_models(base: FitBase, msci_quarterly: pd.DataFrame, cfg: FitConfig) -> pd.DataFrame:
    df = base.data.copy()
    msci = msci_quarterly[["quarter_end", "msci_ret_q", "msci_ret_q_lag1"]].copy()
    df = df.merge(msci, on="quarter_end", how="left")
    end_info = compute_fund_end_dates(df)
    df["fund_end_qe"] = df["FundID"].map(end_info.fund_end_qe)
    df["q_ord"] = df["quarter_end"].dt.to_period("Q").map(lambda p: p.ordinal if pd.notna(p) else np.nan)
    df["end_ord"] = df["fund_end_qe"].dt.to_period("Q").map(lambda p: p.ordinal if pd.notna(p) else np.nan)
    df["q_to_end_q"] = df["end_ord"] - df["q_ord"]
    df["age_y"] = pd.to_numeric(df["Fund_Age_Quarters"], errors="coerce").fillna(0.0).clip(lower=0.0) / 4.0
    df["age_y2"] = df["age_y"] ** 2
    df["q_to_end_y"] = np.clip(pd.to_numeric(df["q_to_end_q"], errors="coerce").fillna(40.0) / 4.0, -4.0, 12.0)
    df["is_pre_end"] = ((df["q_to_end_q"] >= 1) & (df["q_to_end_q"] <= 12)).astype(float)
    df["is_post_end"] = (df["q_to_end_q"] < 0).astype(float)

    valid = df[np.isfinite(df["omega_obs"]) & np.isfinite(df["msci_ret_q"]) & np.isfinite(df["msci_ret_q_lag1"])].copy()

    levels = [
        ("strategy_grade_age_size", ["Adj Strategy", "Grade", "AgeBucket", "size_bucket"]),
        ("strategy_grade_age", ["Adj Strategy", "Grade", "AgeBucket"]),
        ("strategy", ["Adj Strategy"]),
        ("global", []),
    ]

    rows = []
    for level, gcols in levels:
        grouped = valid.groupby(gcols, dropna=False) if gcols else [((), valid)]
        for key, g in grouped:
            if len(g) < cfg.min_obs_group:
                continue

            key_tuple = key if isinstance(key, tuple) else (key,)
            y = g["omega_obs"].to_numpy(dtype=float)
            X = np.column_stack(
                [
                    np.ones(len(g), dtype=float),
                    g["msci_ret_q"].to_numpy(dtype=float),
                    g["msci_ret_q_lag1"].to_numpy(dtype=float),
                    g["age_y"].to_numpy(dtype=float),
                    g["age_y2"].to_numpy(dtype=float),
                    g["q_to_end_y"].to_numpy(dtype=float),
                    g["is_pre_end"].to_numpy(dtype=float),
                    g["is_post_end"].to_numpy(dtype=float),
                ]
            )
            if level == "strategy_grade_age_size":
                lam = 4.0
            elif level == "strategy_grade_age":
                lam = 3.0
            elif level == "strategy":
                lam = 2.0
            else:
                lam = 1.5
            beta, sigma = _ridge_fit(X, y, lam=lam, robust_iters=2)
            resid = y - X @ beta
            sst = float(np.sum((y - np.mean(y)) ** 2))
            sse = float(np.sum(resid**2))
            r2 = 1.0 - sse / sst if sst > 1e-12 else np.nan

            row = {
                "level": level,
                "Adj Strategy": key_tuple[0] if "Adj Strategy" in gcols else "ALL",
                "Grade": key_tuple[gcols.index("Grade")] if "Grade" in gcols else "ALL",
                "AgeBucket": key_tuple[gcols.index("AgeBucket")] if "AgeBucket" in gcols else "ALL",
                "size_bucket": key_tuple[gcols.index("size_bucket")] if "size_bucket" in gcols else "ALL",
                "n_obs": int(len(g)),
                "a_intercept": float(beta[0]),
                "b0": float(beta[1]),
                "b1": float(beta[2]),
                "b_age": float(beta[3]),
                "b_age2": float(beta[4]),
                "b_q_to_end": float(beta[5]),
                "b_pre_end": float(beta[6]),
                "b_post_end": float(beta[7]),
                "alpha": 0.0,
                "sigma": float(max(sigma, 1e-4)),
                "rmse_in_sample": float(np.sqrt(np.mean(resid**2))),
                "r2_in_sample": float(r2) if np.isfinite(r2) else np.nan,
            }
            rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values(["level", "Adj Strategy", "Grade", "AgeBucket", "size_bucket"]).reset_index(drop=True)
    return out


def fit_all(hist_to_cutoff: pd.DataFrame, msci_quarterly: pd.DataFrame, cfg: FitConfig) -> FitArtifacts:
    base = prepare_fit_base(hist_to_cutoff, cfg)
    timing = fit_timing_probs(base, cfg)
    ratios = fit_ratio_models(base, cfg)
    omega = fit_omega_models(base, msci_quarterly, cfg)

    return FitArtifacts(
        timing_probs=timing,
        ratio_fits=ratios,
        omega_fits=omega,
        target_size_bins={
            "bins": list(cfg.size_bins),
            "labels": list(cfg.size_labels),
        },
    )


def load_artifacts(calibration_dir: Path) -> FitArtifacts:
    timing = pd.read_csv(calibration_dir / "timing_probs_selected.csv")
    ratios = pd.read_csv(calibration_dir / "ratio_fit_selected.csv")
    omega = pd.read_csv(calibration_dir / "omega_selected.csv")
    with (calibration_dir / "target_fund_size_bins.json").open("r", encoding="utf-8") as f:
        bins = json.load(f)
    return FitArtifacts(timing_probs=timing, ratio_fits=ratios, omega_fits=omega, target_size_bins=bins)
