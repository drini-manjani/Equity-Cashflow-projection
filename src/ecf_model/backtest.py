from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .data_io import load_cashflow_table, slice_to_cutoff
from .pipeline import run_fit_pipeline, run_projection_pipeline
from .utils import ensure_dir, parse_quarter_text


def _xnpv(rate: float, cfs: np.ndarray, dts: np.ndarray) -> float:
    t0 = dts[0]
    years = ((dts - t0) / np.timedelta64(1, "D")) / 365.0
    return float(np.sum(cfs / ((1.0 + rate) ** years)))


def _xirr(cfs: np.ndarray, dts: np.ndarray) -> float:
    if not (np.any(cfs < 0) and np.any(cfs > 0)):
        return math.nan
    lo, hi = -0.9999, 10.0
    f_lo, f_hi = _xnpv(lo, cfs, dts), _xnpv(hi, cfs, dts)
    k = 0
    while np.isfinite(f_lo) and np.isfinite(f_hi) and f_lo * f_hi > 0 and k < 80:
        hi *= 2.0
        f_hi = _xnpv(hi, cfs, dts)
        k += 1
    if (not np.isfinite(f_lo)) or (not np.isfinite(f_hi)) or f_lo * f_hi > 0:
        return math.nan
    for _ in range(240):
        mid = (lo + hi) / 2.0
        f_mid = _xnpv(mid, cfs, dts)
        if not np.isfinite(f_mid):
            return math.nan
        if abs(f_mid) < 1e-8:
            return mid
        if f_lo * f_mid <= 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return (lo + hi) / 2.0


def _safe_wape(pred: pd.Series, actual: pd.Series) -> float:
    den = float(actual.abs().sum())
    if den <= 1e-12:
        return math.nan
    return float((pred - actual).abs().sum() / den)


def _active_funds_at_cutoff(snapshot: pd.DataFrame) -> set[str]:
    last = snapshot.sort_values(["FundID", "quarter_end"]).groupby("FundID", as_index=False).tail(1)
    active = last[~last.get("Fund Workflow Stage", "").astype(str).str.lower().str.contains("terminated", na=False)]
    return set(active["FundID"].astype(str))


def evaluate_backtest_window(
    hist_path: str | Path,
    msci_path: str | Path,
    cutoff_quarter: str,
    eval_to_quarter: str,
    cfg: PipelineConfig,
) -> dict[str, Any]:
    hist = load_cashflow_table(hist_path)
    cutoff_qe = parse_quarter_text(cutoff_quarter)
    eval_to_qe = parse_quarter_text(eval_to_quarter)

    snapshot = slice_to_cutoff(hist, cutoff_quarter)
    if snapshot.empty:
        raise ValueError(f"No history up to cutoff {cutoff_quarter}")

    window_cfg = replace(cfg, cutoff_quarter=cutoff_quarter)
    run_fit_pipeline(hist_path=hist_path, msci_path=msci_path, cfg=window_cfg)

    snap_path = window_cfg.run_root / "backtest_snapshot.csv"
    ensure_dir(snap_path.parent)
    snapshot.to_csv(snap_path, index=False)

    run_projection_pipeline(
        portfolio_path=snap_path,
        calibration_dir=window_cfg.calibration_dir,
        cfg=window_cfg,
    )

    active_ids = _active_funds_at_cutoff(snapshot)
    actual = hist[(hist["FundID"].isin(active_ids)) & (hist["quarter_end"] > cutoff_qe) & (hist["quarter_end"] <= eval_to_qe)].copy()

    actual_q = (
        actual.groupby("quarter_end", as_index=False)
        .agg(
            actual_draw=("Adj Drawdown EUR", lambda s: float(pd.to_numeric(s, errors="coerce").abs().sum())),
            actual_rep=("Adj Repayment EUR", lambda s: float(pd.to_numeric(s, errors="coerce").abs().sum())),
            actual_nav=("NAV Adjusted EUR", lambda s: float(pd.to_numeric(s, errors="coerce").sum())),
        )
        .sort_values("quarter_end")
    )

    proj = pd.read_csv(window_cfg.projection_dir / "sim_outputs" / "sim_portfolio_series.csv")
    proj["quarter_end"] = pd.to_datetime(proj["quarter_end"], errors="coerce")
    proj = proj[(proj["quarter_end"] > cutoff_qe) & (proj["quarter_end"] <= eval_to_qe)].copy()

    merged = actual_q.merge(
        proj[["quarter_end", "sim_draw_mean", "sim_rep_mean", "sim_nav_mean"]],
        on="quarter_end",
        how="outer",
    ).fillna(0.0).sort_values("quarter_end")

    if merged.empty:
        raise ValueError("No overlapping quarters in backtest window")

    act_draw = float(merged["actual_draw"].sum())
    act_rep = float(merged["actual_rep"].sum())
    prj_draw = float(merged["sim_draw_mean"].sum())
    prj_rep = float(merged["sim_rep_mean"].sum())

    nav_row = merged[merged["quarter_end"] == eval_to_qe]
    if len(nav_row):
        act_nav = float(nav_row["actual_nav"].iloc[0])
        prj_nav = float(nav_row["sim_nav_mean"].iloc[0])
    else:
        act_nav = float(merged["actual_nav"].iloc[-1])
        prj_nav = float(merged["sim_nav_mean"].iloc[-1])

    act_tvpi = (act_rep + act_nav) / act_draw if act_draw > 1e-12 else math.nan
    prj_tvpi = (prj_rep + prj_nav) / prj_draw if prj_draw > 1e-12 else math.nan

    dts = merged["quarter_end"].to_numpy(dtype="datetime64[ns]")
    act_cfs = (merged["actual_rep"] - merged["actual_draw"]).to_numpy(dtype=float)
    prj_cfs = (merged["sim_rep_mean"] - merged["sim_draw_mean"]).to_numpy(dtype=float)
    act_irr = _xirr(np.append(act_cfs, act_nav), np.append(dts, np.datetime64(eval_to_qe.to_datetime64())))
    prj_irr = _xirr(np.append(prj_cfs, prj_nav), np.append(dts, np.datetime64(eval_to_qe.to_datetime64())))
    if np.isfinite(act_irr) and (act_irr > 2.0 or act_irr < -0.95):
        act_irr = math.nan
    if np.isfinite(prj_irr) and (prj_irr > 2.0 or prj_irr < -0.95):
        prj_irr = math.nan

    out = {
        "cutoff_quarter": cutoff_quarter,
        "eval_to_quarter": eval_to_quarter,
        "n_active_funds": int(len(active_ids)),
        "actual_draw": act_draw,
        "actual_rep": act_rep,
        "actual_nav_end": act_nav,
        "actual_tvpi": float(act_tvpi) if np.isfinite(act_tvpi) else math.nan,
        "actual_irr": float(act_irr) if np.isfinite(act_irr) else math.nan,
        "projected_draw": prj_draw,
        "projected_rep": prj_rep,
        "projected_nav_end": prj_nav,
        "projected_tvpi": float(prj_tvpi) if np.isfinite(prj_tvpi) else math.nan,
        "projected_irr": float(prj_irr) if np.isfinite(prj_irr) else math.nan,
        "draw_bias_pct": float(prj_draw / act_draw - 1.0) if act_draw > 1e-12 else math.nan,
        "rep_bias_pct": float(prj_rep / act_rep - 1.0) if act_rep > 1e-12 else math.nan,
        "nav_bias_pct": float(prj_nav / act_nav - 1.0) if abs(act_nav) > 1e-12 else math.nan,
        "wape_draw": _safe_wape(merged["sim_draw_mean"], merged["actual_draw"]),
        "wape_rep": _safe_wape(merged["sim_rep_mean"], merged["actual_rep"]),
        "wape_nav": _safe_wape(merged["sim_nav_mean"], merged["actual_nav"]),
    }

    merged.to_csv(window_cfg.run_root / "backtest_compare_quarterly.csv", index=False)
    pd.DataFrame([out]).to_csv(window_cfg.run_root / "backtest_summary.csv", index=False)
    return out


def run_backtest_suite(
    hist_path: str | Path,
    msci_path: str | Path,
    cutoffs: list[str],
    eval_to_quarter: str,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    rows = []
    for cutoff in cutoffs:
        tag = f"{cfg.run_tag}_{cutoff}"
        c = replace(cfg, run_tag=tag, cutoff_quarter=cutoff)
        c.simulation = cfg.simulation
        c.fit = cfg.fit
        c.calibration = cfg.calibration
        c.scenarios = cfg.scenarios
        rows.append(evaluate_backtest_window(hist_path, msci_path, cutoff, eval_to_quarter, c))

    out = pd.DataFrame(rows).sort_values("cutoff_quarter").reset_index(drop=True)
    suite_dir = Path("runs_v2") / cfg.run_tag
    ensure_dir(suite_dir)
    out.to_csv(suite_dir / "backtest_suite_summary.csv", index=False)
    return out


__all__ = ["evaluate_backtest_window", "run_backtest_suite"]
