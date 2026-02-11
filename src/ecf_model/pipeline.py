from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .calibration import CalibrationArtifacts, calibrate_from_history, load_calibration_artifacts
from .config import PipelineConfig
from .data_io import load_cashflow_table, slice_to_cutoff
from .features import build_fund_states, compute_fund_end_dates
from .fitting import FitArtifacts, fit_all, load_artifacts
from .msci import fit_msci_model, load_msci_model, load_msci_quarterly, project_msci_paths, save_msci_model
from .simulator import SimulationOutputs, simulate_portfolio
from .utils import ensure_dir, format_quarter, parse_quarter_text, recallable_to_flow


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def _serialize_cfg(cfg: PipelineConfig) -> dict[str, Any]:
    return asdict(cfg)


def _calc_horizon(start_qe: pd.Timestamp, states: dict, min_horizon_q: int) -> int:
    if not states:
        return int(min_horizon_q)
    ord_start = pd.Period(start_qe, freq="Q").ordinal
    ord_ends = []
    for st in states.values():
        if pd.notna(st.fund_end_qe):
            ord_ends.append(pd.Period(st.fund_end_qe, freq="Q").ordinal)
    if not ord_ends:
        return int(min_horizon_q)
    max_end = max(ord_ends)
    return int(max(min_horizon_q, max_end - ord_start + 1 + 8))


def _weighted_lag1_corr_by_fund(df: pd.DataFrame, value_col: str, min_obs: int = 6) -> tuple[float, int]:
    num = 0.0
    den = 0.0
    n_funds = 0
    for _, g in df.groupby("FundID"):
        v = pd.to_numeric(g[value_col], errors="coerce").to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        n = len(v)
        if n < int(min_obs):
            continue
        x = v[:-1]
        y = v[1:]
        if len(x) < 2 or float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
            continue
        c = float(np.corrcoef(x, y)[0, 1])
        if not np.isfinite(c):
            continue
        w = float(max(n - 1, 1))
        num += w * c
        den += w
        n_funds += 1
    if den <= 0.0:
        return math.nan, 0
    return float(num / den), int(n_funds)


def _calibrate_copula_from_history(hist_to_cutoff: pd.DataFrame) -> dict[str, Any]:
    h = hist_to_cutoff.sort_values(["FundID", "quarter_end"]).copy()
    if h.empty:
        return {
            "method": "weighted_lag1_corr_blend",
            "rho_calibrated": 0.35,
            "rho_event_lag1": math.nan,
            "rho_flow_lag1": math.nan,
            "n_funds_event": 0,
            "n_funds_flow": 0,
        }

    draw_abs = pd.to_numeric(h["Adj Drawdown EUR"], errors="coerce").abs().fillna(0.0)
    rep_abs = pd.to_numeric(h["Adj Repayment EUR"], errors="coerce").abs().fillna(0.0)
    recall_flow = (
        h.groupby("FundID")["Recallable"]
        .apply(recallable_to_flow)
        .reset_index(level=0, drop=True)
        .reindex(h.index)
    )
    recall_flow = pd.to_numeric(recall_flow, errors="coerce").fillna(0.0)
    commit = pd.to_numeric(h.groupby("FundID")["Commitment EUR"].transform("max"), errors="coerce").fillna(0.0)

    h["any_event"] = ((draw_abs > 0.0) | (rep_abs > 0.0) | (recall_flow > 0.0)).astype(float)
    h["flow_ratio"] = (draw_abs + rep_abs + recall_flow) / np.maximum(commit, 1.0)

    rho_event, n_event = _weighted_lag1_corr_by_fund(h, "any_event", min_obs=6)
    rho_flow, n_flow = _weighted_lag1_corr_by_fund(h, "flow_ratio", min_obs=6)

    event_pos = max(float(rho_event), 0.0) if np.isfinite(rho_event) else math.nan
    flow_pos = max(float(rho_flow), 0.0) if np.isfinite(rho_flow) else math.nan

    if np.isfinite(event_pos) and np.isfinite(flow_pos):
        rho_raw = 0.60 * event_pos + 0.40 * flow_pos
    elif np.isfinite(event_pos):
        rho_raw = event_pos
    elif np.isfinite(flow_pos):
        rho_raw = flow_pos
    else:
        rho_raw = 0.35

    rho_cal = float(np.clip(rho_raw, 0.05, 0.85))
    return {
        "method": "weighted_lag1_corr_blend",
        "rho_calibrated": rho_cal,
        "rho_event_lag1": float(rho_event) if np.isfinite(rho_event) else math.nan,
        "rho_flow_lag1": float(rho_flow) if np.isfinite(rho_flow) else math.nan,
        "n_funds_event": int(n_event),
        "n_funds_flow": int(n_flow),
    }


def _resolve_copula_rho(calibration_dir: Path, cfg: PipelineConfig) -> None:
    raw = cfg.simulation.copula_rho
    rho = float(raw) if raw is not None and np.isfinite(raw) else math.nan
    if not np.isfinite(rho):
        cp = calibration_dir / "copula_calibration.json"
        if cp.exists():
            try:
                payload = json.loads(cp.read_text(encoding="utf-8"))
                rho = float(payload.get("rho_calibrated", math.nan))
            except Exception:
                rho = math.nan
    if not np.isfinite(rho):
        rho = 0.35
    cfg.simulation.copula_rho = float(np.clip(rho, 0.0, 0.95))


def _xnpv(rate: float, cfs: np.ndarray, dts: np.ndarray) -> float:
    if len(cfs) == 0:
        return math.nan
    t0 = dts[0]
    years = ((dts - t0) / np.timedelta64(1, "D")) / 365.0
    base = 1.0 + float(rate)
    if base <= 0.0:
        return math.inf
    disc = np.exp(np.clip(years * np.log(base), -700.0, 700.0))
    return float(np.sum(cfs / disc))


def _xirr(cfs: np.ndarray, dts: np.ndarray) -> float:
    if len(cfs) < 2:
        return math.nan
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


def _build_portfolio_observation_summary(port_series: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "quarter_end",
        "quarter_text",
        "quarter_index",
        "years_from_start",
        "draw_q_mean",
        "rep_q_mean",
        "recall_q_mean",
        "nav_mean",
        "cum_draw_mean",
        "cum_rep_mean",
        "cum_recall_mean",
        "dpi_to_date",
        "rvpi_to_date",
        "tvpi_to_date",
        "nav_share_of_total_value",
        "irr_to_date",
        "is_final",
    ]
    if port_series is None or port_series.empty:
        return pd.DataFrame(columns=cols)

    d = port_series.sort_values("quarter_end").copy()
    d["quarter_end"] = pd.to_datetime(d["quarter_end"], errors="coerce")
    d = d[d["quarter_end"].notna()].reset_index(drop=True)
    if d.empty:
        return pd.DataFrame(columns=cols)

    draw = pd.to_numeric(d["sim_draw_mean"], errors="coerce").fillna(0.0)
    rep = pd.to_numeric(d["sim_rep_mean"], errors="coerce").fillna(0.0)
    nav = pd.to_numeric(d["sim_nav_mean"], errors="coerce").fillna(0.0)
    rc = pd.to_numeric(d.get("sim_rc_mean", 0.0), errors="coerce").fillna(0.0)

    cum_draw = draw.cumsum()
    cum_rep = rep.cumsum()
    cum_rc = rc.cumsum()
    total_value = cum_rep + nav
    q_index = np.arange(len(d), dtype=int)

    irr_vals: list[float] = []
    cfs_q = (rep - draw).to_numpy(dtype=float)
    dts = d["quarter_end"].to_numpy(dtype="datetime64[ns]")
    for i in range(len(d)):
        cfs_i = np.append(cfs_q[: i + 1], float(nav.iloc[i]))
        dts_i = np.append(dts[: i + 1], dts[i])
        irr = _xirr(cfs_i, dts_i)
        if np.isfinite(irr) and (-0.95 <= irr <= 2.0):
            irr_vals.append(float(irr))
        else:
            irr_vals.append(math.nan)

    out = pd.DataFrame(
        {
            "quarter_end": d["quarter_end"],
            "quarter_text": d["quarter_end"].map(lambda x: format_quarter(x) if pd.notna(x) else None),
            "quarter_index": q_index,
            "years_from_start": q_index / 4.0,
            "draw_q_mean": draw,
            "rep_q_mean": rep,
            "recall_q_mean": rc,
            "nav_mean": nav,
            "cum_draw_mean": cum_draw,
            "cum_rep_mean": cum_rep,
            "cum_recall_mean": cum_rc,
            "dpi_to_date": np.where(cum_draw > 1e-12, cum_rep / cum_draw, np.nan),
            "rvpi_to_date": np.where(cum_draw > 1e-12, nav / cum_draw, np.nan),
            "tvpi_to_date": np.where(cum_draw > 1e-12, total_value / cum_draw, np.nan),
            "nav_share_of_total_value": np.where(total_value > 1e-12, nav / total_value, np.nan),
            "irr_to_date": irr_vals,
            "is_final": False,
        }
    )
    out.loc[out.index[-1], "is_final"] = True
    return out


def _historical_end_of_life_diagnostics(hist_to_cutoff: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    h = hist_to_cutoff.sort_values(["FundID", "quarter_end"]).copy()
    end_info = compute_fund_end_dates(h)

    rows = []
    for fid, g in h.groupby("FundID"):
        g = g.sort_values("quarter_end").copy()
        strategy = str(g["Adj Strategy"].mode().iat[0] if len(g["Adj Strategy"].mode()) else g["Adj Strategy"].iloc[0])
        fund_end = end_info.fund_end_qe.get(fid, pd.NaT)

        g["q_ord"] = g["quarter_end"].dt.to_period("Q").map(lambda p: p.ordinal if pd.notna(p) else np.nan)
        end_ord = pd.Period(fund_end, freq="Q").ordinal if pd.notna(fund_end) else np.nan

        nav_at_end = np.nan
        if np.isfinite(end_ord):
            ge = g[g["q_ord"] == end_ord]
            if len(ge):
                nav_at_end = float(pd.to_numeric(ge["NAV Adjusted EUR"], errors="coerce").fillna(0.0).iloc[-1])
            else:
                gp = g[g["q_ord"] <= end_ord]
                if len(gp):
                    nav_at_end = float(pd.to_numeric(gp["NAV Adjusted EUR"], errors="coerce").fillna(0.0).iloc[-1])

        def _window_rep(lo: int, hi: int) -> float:
            if not np.isfinite(end_ord):
                return np.nan
            ww = g[(g["q_ord"] > end_ord + lo) & (g["q_ord"] <= end_ord + hi)]
            if ww.empty:
                return 0.0
            return float(pd.to_numeric(ww["Adj Repayment EUR"], errors="coerce").abs().fillna(0.0).sum())

        nav_4q = np.nan
        nav_8q = np.nan
        if np.isfinite(end_ord):
            n4 = g[g["q_ord"] == end_ord + 4]
            n8 = g[g["q_ord"] == end_ord + 8]
            if len(n4):
                nav_4q = float(pd.to_numeric(n4["NAV Adjusted EUR"], errors="coerce").fillna(0.0).iloc[-1])
            if len(n8):
                nav_8q = float(pd.to_numeric(n8["NAV Adjusted EUR"], errors="coerce").fillna(0.0).iloc[-1])

        rows.append(
            {
                "FundID": fid,
                "Adj Strategy": strategy,
                "fund_end_qe": fund_end,
                "last_obs_qe": g["quarter_end"].max(),
                "nav_at_fund_end": nav_at_end,
                "nav_4q_after": nav_4q,
                "nav_8q_after": nav_8q,
                "rep_0_4q_after": _window_rep(0, 4),
                "rep_4_8q_after": _window_rep(4, 8),
                "rep_8_12q_after": _window_rep(8, 12),
            }
        )

    fund_summary = pd.DataFrame(rows)
    strategy_summary = (
        fund_summary.groupby("Adj Strategy", as_index=False)
        .agg(
            n_funds=("FundID", "count"),
            nav_end_mean=("nav_at_fund_end", "mean"),
            nav_end_median=("nav_at_fund_end", "median"),
            nav_end_p90=("nav_at_fund_end", lambda x: float(np.nanquantile(pd.to_numeric(x, errors="coerce"), 0.9))),
            rep_0_4q_mean=("rep_0_4q_after", "mean"),
            rep_4_8q_mean=("rep_4_8q_after", "mean"),
            rep_8_12q_mean=("rep_8_12q_after", "mean"),
        )
        .sort_values("n_funds", ascending=False)
        .reset_index(drop=True)
    )
    return fund_summary, strategy_summary


def run_fit_pipeline(
    hist_path: str | Path,
    msci_path: str | Path,
    cfg: PipelineConfig,
) -> dict[str, Any]:
    ensure_dir(cfg.run_root)
    ensure_dir(cfg.calibration_dir)

    hist = load_cashflow_table(hist_path)
    hist_to_cutoff = slice_to_cutoff(hist, cfg.cutoff_quarter)
    if hist_to_cutoff.empty:
        raise ValueError("No history available up to cutoff quarter")

    msci_q = load_msci_quarterly(msci_path)
    msci_model = fit_msci_model(msci_q, cutoff_quarter=cfg.cutoff_quarter)

    fits = fit_all(hist_to_cutoff, msci_q, cfg.fit)
    fits.save(cfg.calibration_dir)

    calib = calibrate_from_history(
        hist_to_cutoff,
        cutoff_quarter=cfg.cutoff_quarter,
        fit_cfg=cfg.fit,
        cal_cfg=cfg.calibration,
    )
    calib.save(cfg.calibration_dir)

    end_info = compute_fund_end_dates(hist_to_cutoff)
    (
        end_info.avg_overrun_by_strategy.rename("avg_overrun_q")
        .reset_index()
        .rename(columns={"index": "Adj Strategy"})
        .to_csv(cfg.calibration_dir / "avg_overrun_by_strategy.csv", index=False)
    )

    fund_eol, strat_eol = _historical_end_of_life_diagnostics(hist_to_cutoff)
    fund_eol.to_csv(cfg.calibration_dir / "historical_end_of_life_fund_summary.csv", index=False)
    strat_eol.to_csv(cfg.calibration_dir / "historical_end_of_life_strategy_summary.csv", index=False)

    save_msci_model(cfg.calibration_dir / "msci_model.json", msci_model)
    msci_q.to_csv(cfg.calibration_dir / "msci_quarterly_history.csv", index=False)
    copula_diag = _calibrate_copula_from_history(hist_to_cutoff)
    _json_dump(cfg.calibration_dir / "copula_calibration.json", copula_diag)
    _json_dump(cfg.calibration_dir / "fit_run_config.json", _serialize_cfg(cfg))

    return {
        "run_root": str(cfg.run_root),
        "calibration_dir": str(cfg.calibration_dir),
        "n_hist_rows": int(len(hist_to_cutoff)),
        "n_funds": int(hist_to_cutoff["FundID"].nunique()),
        "cutoff_quarter": cfg.cutoff_quarter,
    }


def _load_fits_and_calibration(calibration_dir: Path) -> tuple[FitArtifacts, CalibrationArtifacts, Any]:
    fits = load_artifacts(calibration_dir)
    calibs = load_calibration_artifacts(calibration_dir)
    msci_model = load_msci_model(calibration_dir / "msci_model.json")
    return fits, calibs, msci_model


def run_projection_pipeline(
    portfolio_path: str | Path,
    calibration_dir: str | Path,
    cfg: PipelineConfig,
) -> dict[str, Any]:
    calibration_dir = Path(calibration_dir)
    if not calibration_dir.exists():
        raise FileNotFoundError(calibration_dir)

    ensure_dir(cfg.run_root)
    ensure_dir(cfg.projection_dir)

    fits, calibs, msci_model = _load_fits_and_calibration(calibration_dir)
    _resolve_copula_rho(calibration_dir, cfg)

    df = load_cashflow_table(portfolio_path)
    start_qe = pd.to_datetime(df["quarter_end"], errors="coerce").max()
    if pd.isna(start_qe):
        start_qe = parse_quarter_text(cfg.cutoff_quarter)

    # Build states from all rows up to projection start.
    portfolio_hist = df[df["quarter_end"] <= start_qe].copy()
    bins = fits.target_size_bins.get("bins", list(cfg.fit.size_bins))
    labels = fits.target_size_bins.get("labels", list(cfg.fit.size_labels))
    avg_overrun = None
    overrun_path = calibration_dir / "avg_overrun_by_strategy.csv"
    if overrun_path.exists():
        o = pd.read_csv(overrun_path)
        if {"Adj Strategy", "avg_overrun_q"}.issubset(set(o.columns)):
            avg_overrun = pd.Series(o["avg_overrun_q"].values, index=o["Adj Strategy"].astype(str).values)

    states = build_fund_states(
        portfolio_hist,
        size_bins=list(bins),
        size_labels=list(labels),
        avg_overrun_by_strategy=avg_overrun,
    )
    if not states:
        raise ValueError("No active funds found in the projection input")

    horizon = _calc_horizon(start_qe, states, cfg.simulation.horizon_q)

    if cfg.scenario.lower() == "bullish":
        drift_shift_q = cfg.scenarios.bullish_drift_shift_q
    elif cfg.scenario.lower() == "bearish":
        drift_shift_q = cfg.scenarios.bearish_drift_shift_q
    else:
        drift_shift_q = cfg.scenarios.neutral_drift_shift_q

    msci_paths = project_msci_paths(
        model=msci_model,
        start_quarter=format_quarter(start_qe),
        horizon_q=horizon,
        n_sims=cfg.simulation.n_sims,
        seed=cfg.simulation.seed,
        scenario=cfg.scenario,
        drift_shift_q=drift_shift_q,
        volatility_scale=cfg.scenarios.volatility_scale,
    )

    sim = simulate_portfolio(
        states=states,
        fit_artifacts=fits,
        calibration=calibs,
        msci_paths=msci_paths,
        sim_cfg=cfg.simulation,
        fit_cfg=cfg.fit,
    )
    sim.save(cfg.projection_dir)
    obs_summary = _build_portfolio_observation_summary(sim.portfolio_series)
    obs_summary_path = cfg.projection_dir / "portfolio_observation_summary.csv"
    obs_summary.to_csv(obs_summary_path, index=False)

    msci_paths.to_csv(cfg.projection_dir / "sim_outputs" / "msci_paths.csv", index=False)
    _json_dump(cfg.projection_dir / "projection_run_config.json", _serialize_cfg(cfg))

    end_nav = float(sim.portfolio_series["sim_nav_mean"].iloc[-1]) if len(sim.portfolio_series) else 0.0
    total_draw = float(sim.portfolio_series["sim_draw_mean"].sum()) if len(sim.portfolio_series) else 0.0
    total_rep = float(sim.portfolio_series["sim_rep_mean"].sum()) if len(sim.portfolio_series) else 0.0

    return {
        "run_root": str(cfg.run_root),
        "projection_dir": str(cfg.projection_dir),
        "scenario": cfg.scenario,
        "n_input_funds": int(df["FundID"].nunique()),
        "n_active_funds": int(len(states)),
        "start_quarter": format_quarter(start_qe),
        "projection_end": str(sim.portfolio_series["quarter_end"].iloc[-1]) if len(sim.portfolio_series) else None,
        "total_draw": total_draw,
        "total_rep": total_rep,
        "end_nav": end_nav,
        "copula_enabled": bool(cfg.simulation.copula_enabled),
        "copula_rho": float(cfg.simulation.copula_rho) if cfg.simulation.copula_rho is not None else None,
        "observation_summary_path": str(obs_summary_path),
    }


__all__ = ["run_fit_pipeline", "run_projection_pipeline"]
