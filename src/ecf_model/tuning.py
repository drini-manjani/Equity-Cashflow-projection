from __future__ import annotations

import copy
import math
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .calibration import CalibrationArtifacts, load_calibration_artifacts
from .config import PipelineConfig
from .data_io import load_cashflow_table
from .features import build_fund_states, compute_invest_end_by_fund
from .fitting import FitArtifacts, load_artifacts
from .msci import load_msci_model, project_msci_paths
from .simulator import simulate_portfolio
from .utils import ensure_dir, format_quarter, parse_quarter_text, recallable_to_flow

IRR_TARGET_MODES = {"mix_median", "pooled"}


@dataclass
class NeutralTuneResult:
    target_irr: float
    target_tvpi: float
    achieved_irr: float
    achieved_tvpi: float
    objective: float
    iterations: int
    tuned_calibration_dir: str
    neutral_run_root: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_irr": self.target_irr,
            "target_tvpi": self.target_tvpi,
            "achieved_irr": self.achieved_irr,
            "achieved_tvpi": self.achieved_tvpi,
            "objective": self.objective,
            "iterations": self.iterations,
            "tuned_calibration_dir": self.tuned_calibration_dir,
            "neutral_run_root": self.neutral_run_root,
        }


def _resolve_copula_rho(base_calibration_dir: Path, cfg: PipelineConfig) -> None:
    raw = cfg.simulation.copula_rho
    rho = float(raw) if raw is not None and np.isfinite(raw) else math.nan
    if not np.isfinite(rho):
        cp = base_calibration_dir / "copula_calibration.json"
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
    t0 = dts[0]
    years = ((dts - t0) / np.timedelta64(1, "D")) / 365.0
    base = 1.0 + float(rate)
    if base <= 0.0:
        return math.inf
    # Compute discount factors in log-space to avoid overflow at large bracket rates.
    disc = np.exp(np.clip(years * np.log(base), -700.0, 700.0))
    return float(np.sum(cfs / disc))


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


def _calc_horizon(start_qe: pd.Timestamp, states: dict[str, Any], min_horizon_q: int) -> int:
    if not states:
        return int(min_horizon_q)
    ord_start = pd.Period(start_qe, freq="Q").ordinal
    ord_ends: list[int] = []
    for st in states.values():
        if pd.notna(st.fund_end_qe):
            ord_ends.append(pd.Period(st.fund_end_qe, freq="Q").ordinal)
    if not ord_ends:
        return int(min_horizon_q)
    max_end = max(ord_ends)
    return int(max(min_horizon_q, max_end - ord_start + 1 + 8))


def _calc_portfolio_metrics(port_series: pd.DataFrame) -> dict[str, float]:
    d = port_series.copy()
    d["quarter_end"] = pd.to_datetime(d["quarter_end"], errors="coerce")
    draw = float(pd.to_numeric(d["sim_draw_mean"], errors="coerce").fillna(0.0).sum())
    rep = float(pd.to_numeric(d["sim_rep_mean"], errors="coerce").fillna(0.0).sum())
    recall = float(pd.to_numeric(d.get("sim_rc_mean", 0.0), errors="coerce").fillna(0.0).sum())
    nav = float(pd.to_numeric(d["sim_nav_mean"], errors="coerce").fillna(0.0).iloc[-1]) if len(d) else 0.0
    tvpi = (rep + nav) / draw if draw > 1e-12 else math.nan
    cfs = (pd.to_numeric(d["sim_rep_mean"], errors="coerce").fillna(0.0) - pd.to_numeric(d["sim_draw_mean"], errors="coerce").fillna(0.0)).to_numpy(dtype=float)
    dts = d["quarter_end"].to_numpy(dtype="datetime64[ns]")
    irr = _xirr(np.append(cfs, nav), np.append(dts, np.datetime64(d["quarter_end"].iloc[-1])))
    if np.isfinite(irr) and (irr > 2.0 or irr < -0.95):
        irr = math.nan
    return {"draw": draw, "rep": rep, "recall": recall, "end_nav": nav, "tvpi": tvpi, "irr": irr}


def _fund_terminal_metrics_from_sim(
    fund_q: pd.DataFrame,
    state_map: dict[str, Any],
) -> pd.DataFrame:
    d = fund_q.copy()
    d["quarter_end"] = pd.to_datetime(d["quarter_end"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for fid, g in d.groupby("FundID"):
        g = g.sort_values("quarter_end")
        draw = float(pd.to_numeric(g["sim_draw_mean"], errors="coerce").fillna(0.0).sum())
        rep = float(pd.to_numeric(g["sim_rep_mean"], errors="coerce").fillna(0.0).sum())
        nav = float(pd.to_numeric(g["sim_nav_mean"], errors="coerce").fillna(0.0).iloc[-1])
        if draw <= 1e-12:
            continue
        st = state_map.get(str(fid))
        strategy = str(st.strategy) if st is not None else "Unknown"
        grade = str(st.grade) if st is not None else "D"
        rows.append(
            {
                "FundID": str(fid),
                "Adj Strategy": strategy,
                "Grade": grade,
                "sim_draw": draw,
                "sim_rep": rep,
                "sim_nav_end": nav,
                "sim_tvpi": (rep + nav) / draw,
            }
        )
    return pd.DataFrame(rows)


def _historical_fund_metrics(hist: pd.DataFrame) -> pd.DataFrame:
    h = hist.sort_values(["FundID", "quarter_end"]).copy()
    rows: list[dict[str, Any]] = []
    for fid, g in h.groupby("FundID"):
        g = g.sort_values("quarter_end")
        draw = pd.to_numeric(g["Adj Drawdown EUR"], errors="coerce").abs().fillna(0.0)
        rep = pd.to_numeric(g["Adj Repayment EUR"], errors="coerce").abs().fillna(0.0)
        nav = float(pd.to_numeric(g["NAV Adjusted EUR"], errors="coerce").fillna(0.0).iloc[-1])
        draw_total = float(draw.sum())
        if draw_total <= 1e-12:
            continue
        cfs = (rep - draw).to_numpy(dtype=float)
        dts = g["quarter_end"].to_numpy(dtype="datetime64[ns]")
        irr = _xirr(np.append(cfs, nav), np.append(dts, np.datetime64(g["quarter_end"].iloc[-1])))
        if np.isfinite(irr) and (irr > 2.0 or irr < -0.95):
            irr = math.nan
        strategy = str(g["Adj Strategy"].mode().iat[0] if len(g["Adj Strategy"].mode()) else g["Adj Strategy"].iloc[-1])
        grade = str(g["Grade"].mode().iat[0] if len(g["Grade"].mode()) else g["Grade"].iloc[-1])
        rows.append(
            {
                "FundID": str(fid),
                "Adj Strategy": strategy,
                "Grade": grade,
                "hist_draw": draw_total,
                "hist_rep": float(rep.sum()),
                "hist_nav_end": nav,
                "hist_tvpi": (float(rep.sum()) + nav) / draw_total,
                "hist_irr": irr,
            }
        )
    return pd.DataFrame(rows)


def _pooled_metrics_from_history(hist: pd.DataFrame) -> dict[str, float]:
    h = hist.sort_values(["FundID", "quarter_end"]).copy()
    if h.empty:
        return {"pooled_irr": math.nan, "pooled_tvpi": math.nan, "draw": 0.0, "rep": 0.0, "nav": 0.0, "n_funds": 0}

    all_cfs: list[float] = []
    all_dts: list[np.datetime64] = []
    draw_total = 0.0
    rep_total = 0.0
    nav_total = 0.0
    n_funds = 0

    for _, g in h.groupby("FundID"):
        g = g.sort_values("quarter_end")
        draw = pd.to_numeric(g["Adj Drawdown EUR"], errors="coerce").abs().fillna(0.0)
        rep = pd.to_numeric(g["Adj Repayment EUR"], errors="coerce").abs().fillna(0.0)
        nav = float(pd.to_numeric(g["NAV Adjusted EUR"], errors="coerce").fillna(0.0).iloc[-1])
        draw_f = float(draw.sum())
        rep_f = float(rep.sum())
        if draw_f <= 1e-12:
            continue

        cfs = (rep - draw).to_numpy(dtype=float)
        dts = g["quarter_end"].to_numpy(dtype="datetime64[ns]")

        all_cfs.extend(cfs.tolist())
        all_dts.extend(dts.tolist())
        all_cfs.append(nav)
        all_dts.append(np.datetime64(g["quarter_end"].iloc[-1]))

        draw_total += draw_f
        rep_total += rep_f
        nav_total += nav
        n_funds += 1

    if n_funds == 0:
        return {"pooled_irr": math.nan, "pooled_tvpi": math.nan, "draw": 0.0, "rep": 0.0, "nav": 0.0, "n_funds": 0}

    cf = np.asarray(all_cfs, dtype=float)
    dt = np.asarray(all_dts, dtype="datetime64[ns]")
    ord_idx = np.argsort(dt)
    pooled_irr = _xirr(cf[ord_idx], dt[ord_idx])
    if np.isfinite(pooled_irr) and (pooled_irr > 2.0 or pooled_irr < -0.95):
        pooled_irr = math.nan
    pooled_tvpi = (rep_total + nav_total) / draw_total if draw_total > 1e-12 else math.nan
    return {
        "pooled_irr": float(pooled_irr) if np.isfinite(pooled_irr) else math.nan,
        "pooled_tvpi": float(pooled_tvpi) if np.isfinite(pooled_tvpi) else math.nan,
        "draw": float(draw_total),
        "rep": float(rep_total),
        "nav": float(nav_total),
        "n_funds": int(n_funds),
    }


def _recall_ratio_from_history(hist: pd.DataFrame) -> dict[str, float]:
    h = hist.sort_values(["FundID", "quarter_end"]).copy()
    if h.empty:
        return {"recall": 0.0, "commit": 0.0, "rep": 0.0, "recall_to_commit": math.nan, "recall_to_rep": math.nan}
    h["recall_flow"] = (
        h.groupby("FundID")["Recallable"]
        .apply(recallable_to_flow)
        .reset_index(level=0, drop=True)
        .reindex(h.index)
    )
    recall = float(pd.to_numeric(h["recall_flow"], errors="coerce").fillna(0.0).sum())
    commit = float(pd.to_numeric(h.groupby("FundID")["Commitment EUR"].max(), errors="coerce").fillna(0.0).sum())
    rep = float(pd.to_numeric(h["Adj Repayment EUR"], errors="coerce").abs().fillna(0.0).sum())
    recall_to_commit = recall / commit if commit > 1e-12 else math.nan
    recall_to_rep = recall / rep if rep > 1e-12 else math.nan
    return {
        "recall": recall,
        "commit": commit,
        "rep": rep,
        "recall_to_commit": float(recall_to_commit) if np.isfinite(recall_to_commit) else math.nan,
        "recall_to_rep": float(recall_to_rep) if np.isfinite(recall_to_rep) else math.nan,
    }


def _mature_draw_ratio_from_history(hist: pd.DataFrame) -> dict[str, float]:
    h = hist.sort_values(["FundID", "quarter_end"]).copy()
    if h.empty:
        return {"draw": 0.0, "commit": 0.0, "draw_to_commit": math.nan, "n_mature_funds": 0}

    invest_end = compute_invest_end_by_fund(h)
    last_obs = h.groupby("FundID")["quarter_end"].max()
    commit = pd.to_numeric(h.groupby("FundID")["Commitment EUR"].max(), errors="coerce").fillna(0.0)
    draw = pd.to_numeric(h.groupby("FundID")["Adj Drawdown EUR"].sum(), errors="coerce").fillna(0.0)

    per = pd.DataFrame({"invest_end": invest_end, "last_obs": last_obs, "commit": commit, "draw": draw})
    per = per[per["commit"] > 0.0].copy()
    if per.empty:
        return {"draw": 0.0, "commit": 0.0, "draw_to_commit": math.nan, "n_mature_funds": 0}

    mature_mask = per["invest_end"].notna() & per["last_obs"].notna() & (per["last_obs"] >= per["invest_end"])
    mature = per[mature_mask].copy()
    if mature.empty:
        mature = per

    draw_total = float(mature["draw"].sum())
    commit_total = float(mature["commit"].sum())
    ratio = draw_total / commit_total if commit_total > 1e-12 else math.nan
    return {
        "draw": draw_total,
        "commit": commit_total,
        "draw_to_commit": float(ratio) if np.isfinite(ratio) else math.nan,
        "n_mature_funds": int(len(mature)),
    }


def _pooled_metrics_by_group(hist: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if hist.empty:
        return pd.DataFrame(columns=[*group_cols, "n_hist", "target_irr", "target_tvpi"])
    for key, g in hist.groupby(group_cols, dropna=False):
        key_t = key if isinstance(key, tuple) else (key,)
        pm = _pooled_metrics_from_history(g)
        row: dict[str, Any] = {group_cols[i]: key_t[i] for i in range(len(group_cols))}
        row["n_hist"] = int(pm["n_funds"])
        row["target_irr"] = float(pm["pooled_irr"]) if np.isfinite(pm["pooled_irr"]) else math.nan
        row["target_tvpi"] = float(pm["pooled_tvpi"]) if np.isfinite(pm["pooled_tvpi"]) else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _build_sg_targets(
    hist: pd.DataFrame,
    hist_fund: pd.DataFrame,
    states: dict[str, Any],
    *,
    irr_target_mode: str = "mix_median",
) -> tuple[pd.DataFrame, float, float]:
    mode = str(irr_target_mode).strip().lower().replace("-", "_")
    if mode not in IRR_TARGET_MODES:
        raise ValueError(f"Unsupported irr_target_mode: {irr_target_mode}")

    state_rows = pd.DataFrame(
        [{"Adj Strategy": str(st.strategy), "Grade": str(st.grade)} for st in states.values()]
    )
    mix = state_rows.groupby(["Adj Strategy", "Grade"], as_index=False).size().rename(columns={"size": "n_test"})
    mix["w"] = mix["n_test"] / max(float(mix["n_test"].sum()), 1.0)

    h = hist_fund.copy()
    h = h[h["Adj Strategy"].isin(mix["Adj Strategy"].unique()) & h["Grade"].isin(mix["Grade"].unique())].copy()

    if mode == "pooled":
        hq = hist.copy()
        hq = hq[hq["Adj Strategy"].isin(mix["Adj Strategy"].unique()) & hq["Grade"].isin(mix["Grade"].unique())].copy()
        sg = _pooled_metrics_by_group(hq, ["Adj Strategy", "Grade"])
        s = _pooled_metrics_by_group(hq, ["Adj Strategy"]).rename(columns={"target_irr": "s_target_irr", "target_tvpi": "s_target_tvpi"})
        if "n_hist" in s.columns:
            s = s.drop(columns=["n_hist"])
        p_all = _pooled_metrics_from_history(hq)
        g_target_irr = float(p_all["pooled_irr"]) if np.isfinite(p_all["pooled_irr"]) else math.nan
        g_target_tvpi = float(p_all["pooled_tvpi"]) if np.isfinite(p_all["pooled_tvpi"]) else math.nan
    else:
        sg = (
            h.groupby(["Adj Strategy", "Grade"], as_index=False)
            .agg(
                n_hist=("FundID", "count"),
                target_irr=("hist_irr", "median"),
                target_tvpi=("hist_tvpi", "median"),
            )
        )
        s = h.groupby("Adj Strategy", as_index=False).agg(s_target_irr=("hist_irr", "median"), s_target_tvpi=("hist_tvpi", "median"))
        g_target_irr = float(h["hist_irr"].median()) if len(h) else math.nan
        g_target_tvpi = float(h["hist_tvpi"].median()) if len(h) else math.nan

    out = mix.merge(sg, on=["Adj Strategy", "Grade"], how="left").merge(s, on="Adj Strategy", how="left")
    out["target_irr"] = out["target_irr"].fillna(out["s_target_irr"]).fillna(g_target_irr)
    out["target_tvpi"] = out["target_tvpi"].fillna(out["s_target_tvpi"]).fillna(g_target_tvpi)
    out["n_hist"] = out["n_hist"].fillna(0).astype(int)
    out = out[["Adj Strategy", "Grade", "n_test", "w", "n_hist", "target_irr", "target_tvpi"]]

    weighted_irr = float(np.sum(out["w"] * out["target_irr"]))
    weighted_tvpi = float(np.sum(out["w"] * out["target_tvpi"]))
    if mode == "pooled":
        target_irr = float(g_target_irr) if np.isfinite(g_target_irr) else weighted_irr
        target_tvpi = float(g_target_tvpi) if np.isfinite(g_target_tvpi) else weighted_tvpi
    else:
        target_irr = weighted_irr
        target_tvpi = weighted_tvpi
    return out, target_irr, target_tvpi


def _objective(pred: dict[str, float], target_irr: float, target_tvpi: float, sg_pred: pd.DataFrame, sg_target: pd.DataFrame) -> float:
    irr_err = abs(float(pred.get("irr", math.nan)) - target_irr) if np.isfinite(pred.get("irr", np.nan)) else 10.0
    tvpi_err = abs(float(pred.get("tvpi", math.nan)) - target_tvpi) if np.isfinite(pred.get("tvpi", np.nan)) else 10.0
    m = sg_target.merge(sg_pred, on=["Adj Strategy", "Grade"], how="left")
    m["sim_tvpi"] = m["sim_tvpi"].replace(0.0, np.nan).fillna(m["target_tvpi"])
    sg_err = np.average(np.abs(np.log(np.clip(m["sim_tvpi"] / m["target_tvpi"], 1e-6, 1e6))), weights=np.maximum(m["w"], 1e-9))
    # Prioritize IRR alignment; keep TVPI/shape as soft constraints.
    return float(8.0 * irr_err + 0.6 * tvpi_err + 1.0 * sg_err)


def _group_col(df: pd.DataFrame, col: str, default: float) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _scale_repayment_intensity(calibs: CalibrationArtifacts, scale: float) -> CalibrationArtifacts:
    out = copy.deepcopy(calibs)
    gd = out.group_deltas.copy()
    if gd.empty:
        out.group_deltas = gd
        return out

    rr = _group_col(gd, "delta_rep_ratio_scale", 1.0)
    pp = _group_col(gd, "delta_p_rep_mult", 1.0)
    om = _group_col(gd, "delta_omega_bump", 0.0)

    gd["delta_rep_ratio_scale"] = np.clip(1.0 + (rr - 1.0) * float(scale), 0.70, 1.90)
    gd["delta_p_rep_mult"] = np.clip(1.0 + (pp - 1.0) * float(scale), 0.75, 1.40)
    gd["delta_omega_bump"] = np.clip(om * float(scale), -0.10, 0.10)
    out.group_deltas = gd
    return out


def _scale_draw_intensity(calibs: CalibrationArtifacts, scale: float) -> CalibrationArtifacts:
    out = copy.deepcopy(calibs)
    gd = out.group_deltas.copy()

    if gd.empty:
        out.global_deltas["delta_p_draw_mult"] = float(
            np.clip(float(out.global_deltas.get("delta_p_draw_mult", 1.0)) * float(scale), 0.40, 2.20)
        )
        out.global_deltas["delta_draw_ratio_scale"] = float(
            np.clip(float(out.global_deltas.get("delta_draw_ratio_scale", 1.0)) * float(scale), 0.50, 2.20)
        )
        out.group_deltas = gd
        return out

    pdm = _group_col(gd, "delta_p_draw_mult", 1.0)
    drs = _group_col(gd, "delta_draw_ratio_scale", 1.0)
    gd["delta_p_draw_mult"] = np.clip(pdm * float(scale), 0.40, 2.20)
    gd["delta_draw_ratio_scale"] = np.clip(drs * float(scale), 0.50, 2.20)
    out.group_deltas = gd
    out.global_deltas["delta_p_draw_mult"] = float(
        np.clip(float(out.global_deltas.get("delta_p_draw_mult", 1.0)) * float(scale), 0.40, 2.20)
    )
    out.global_deltas["delta_draw_ratio_scale"] = float(
        np.clip(float(out.global_deltas.get("delta_draw_ratio_scale", 1.0)) * float(scale), 0.50, 2.20)
    )
    return out


def _scale_recallable_intensity(calibs: CalibrationArtifacts, scale: float) -> CalibrationArtifacts:
    out = copy.deepcopy(calibs)
    gd = out.group_deltas.copy()
    if gd.empty:
        out.group_deltas = gd
        out.global_deltas["delta_p_rc_mult"] = float(out.global_deltas.get("delta_p_rc_mult", 1.0)) * float(scale)
        out.global_deltas["delta_rc_ratio_scale"] = float(out.global_deltas.get("delta_rc_ratio_scale", 1.0)) * float(scale)
        return out

    prc = _group_col(gd, "delta_p_rc_mult", 1.0)
    rrc = _group_col(gd, "delta_rc_ratio_scale", 1.0)

    gd["delta_p_rc_mult"] = np.clip(prc * float(scale), 0.10, 1.80)
    gd["delta_rc_ratio_scale"] = np.clip(rrc * float(scale), 0.20, 2.50)
    out.group_deltas = gd
    out.global_deltas["delta_p_rc_mult"] = float(out.global_deltas.get("delta_p_rc_mult", 1.0)) * float(scale)
    out.global_deltas["delta_rc_ratio_scale"] = float(out.global_deltas.get("delta_rc_ratio_scale", 1.0)) * float(scale)
    return out


def tune_neutral_calibration_to_history(
    *,
    hist_path: str | Path,
    portfolio_path: str | Path,
    base_calibration_dir: str | Path,
    cfg: PipelineConfig,
    n_sims_tune: int = 120,
    n_iters: int = 10,
    irr_target_mode: str = "pooled",
) -> tuple[CalibrationArtifacts, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    base_calibration_dir = Path(base_calibration_dir)
    _resolve_copula_rho(base_calibration_dir, cfg)
    fits: FitArtifacts = load_artifacts(base_calibration_dir)
    base_calibs = load_calibration_artifacts(base_calibration_dir)
    msci_model = load_msci_model(base_calibration_dir / "msci_model.json")

    portfolio = load_cashflow_table(portfolio_path)
    start_qe = pd.to_datetime(portfolio["quarter_end"], errors="coerce").max()
    if pd.isna(start_qe):
        start_qe = parse_quarter_text(cfg.cutoff_quarter)

    avg_overrun = None
    overrun_path = base_calibration_dir / "avg_overrun_by_strategy.csv"
    if overrun_path.exists():
        o = pd.read_csv(overrun_path)
        if {"Adj Strategy", "avg_overrun_q"}.issubset(set(o.columns)):
            avg_overrun = pd.Series(o["avg_overrun_q"].values, index=o["Adj Strategy"].astype(str).values)

    states = build_fund_states(
        portfolio,
        size_bins=list(fits.target_size_bins.get("bins", list(cfg.fit.size_bins))),
        size_labels=list(fits.target_size_bins.get("labels", list(cfg.fit.size_labels))),
        avg_overrun_by_strategy=avg_overrun,
    )
    if not states:
        raise ValueError("No active funds in portfolio for tuning")

    horizon = _calc_horizon(start_qe, states, cfg.simulation.horizon_q)
    msci_paths = project_msci_paths(
        model=msci_model,
        start_quarter=format_quarter(start_qe),
        horizon_q=horizon,
        n_sims=n_sims_tune,
        seed=cfg.simulation.seed,
        scenario="neutral",
        drift_shift_q=cfg.scenarios.neutral_drift_shift_q,
        volatility_scale=cfg.scenarios.volatility_scale,
    )

    hist = load_cashflow_table(hist_path)
    hist_to_cutoff = hist[hist["quarter_end"] <= parse_quarter_text(cfg.cutoff_quarter)].copy()
    hist_fund = _historical_fund_metrics(hist_to_cutoff)
    sg_target, target_irr, target_tvpi = _build_sg_targets(
        hist_to_cutoff,
        hist_fund,
        states,
        irr_target_mode=irr_target_mode,
    )
    # Recallable target for the same strategy/grade mix as the test portfolio.
    strat_set = {str(st.strategy) for st in states.values()}
    grade_set = {str(st.grade) for st in states.values()}
    hist_sub = hist_to_cutoff[
        hist_to_cutoff["Adj Strategy"].isin(strat_set) & hist_to_cutoff["Grade"].isin(grade_set)
    ].copy()
    recall_target = _recall_ratio_from_history(hist_sub)
    draw_target = _mature_draw_ratio_from_history(hist_sub)
    commit_total = float(sum(float(st.commitment) for st in states.values()))

    sim_cfg = copy.deepcopy(cfg.simulation)
    sim_cfg.n_sims = int(n_sims_tune)
    sim_cfg.seed = int(cfg.simulation.seed)

    best_obj = float("inf")
    best_calibs = copy.deepcopy(base_calibs)
    best_pred: dict[str, float] = {"irr": math.nan, "tvpi": math.nan}
    best_sg_pred = pd.DataFrame()

    # Work on a mutable calibration copy.
    calibs = copy.deepcopy(base_calibs)
    gd = calibs.group_deltas.copy()
    if gd.empty:
        raise ValueError("Group deltas are required for grade-aware tuning")

    target_idx = sg_target.set_index(["Adj Strategy", "Grade"])

    def _evaluate(calibs_eval: CalibrationArtifacts) -> tuple[dict[str, float], pd.DataFrame, float]:
        sim = simulate_portfolio(
            states=states,
            fit_artifacts=fits,
            calibration=calibs_eval,
            msci_paths=msci_paths,
            sim_cfg=sim_cfg,
            fit_cfg=cfg.fit,
        )
        pred = _calc_portfolio_metrics(sim.portfolio_series)
        fund_pred = _fund_terminal_metrics_from_sim(sim.fund_quarterly_mean, states)
        sg_pred = (
            fund_pred.groupby(["Adj Strategy", "Grade"], as_index=False)
            .agg(
                sim_draw=("sim_draw", "sum"),
                sim_rep=("sim_rep", "sum"),
                sim_nav_end=("sim_nav_end", "sum"),
            )
        )
        sg_pred["sim_tvpi"] = np.where(sg_pred["sim_draw"] > 1e-12, (sg_pred["sim_rep"] + sg_pred["sim_nav_end"]) / sg_pred["sim_draw"], np.nan)
        obj = _objective(pred, target_irr, target_tvpi, sg_pred, sg_target)
        return pred, sg_pred, obj

    for it in range(int(max(n_iters, 1))):
        calibs.group_deltas = gd
        pred, sg_pred, obj = _evaluate(calibs)
        if obj < best_obj:
            best_obj = float(obj)
            best_calibs = copy.deepcopy(calibs)
            best_pred = {"irr": float(pred.get("irr", math.nan)), "tvpi": float(pred.get("tvpi", math.nan))}
            best_sg_pred = sg_pred.copy()

        # Last iteration only evaluates.
        if it >= int(n_iters) - 1:
            break

        sg_pred_idx = sg_pred.set_index(["Adj Strategy", "Grade"])
        lr = float(max(0.20, 0.70 - 0.05 * it))

        upd_rows: list[dict[str, Any]] = []
        for _, row in gd.iterrows():
            s = str(row.get("Adj Strategy", "ALL"))
            g = str(row.get("Grade", "ALL"))
            if (s, g) not in target_idx.index:
                upd_rows.append(row.to_dict())
                continue

            t_row = target_idx.loc[(s, g)]
            target_tvpi_sg = float(t_row["target_tvpi"])
            pred_tvpi_sg = float(sg_pred_idx.loc[(s, g), "sim_tvpi"]) if (s, g) in sg_pred_idx.index else target_tvpi_sg
            if not np.isfinite(pred_tvpi_sg) or pred_tvpi_sg <= 1e-9:
                pred_tvpi_sg = target_tvpi_sg

            ratio = float(np.clip(target_tvpi_sg / pred_tvpi_sg, 0.6, 1.8))
            # Repayment ratio has strongest direct impact on terminal multiple.
            rr_adj = float(np.clip(ratio ** (0.55 * lr), 0.92, 1.12))
            # Probability adjustment is milder to avoid destabilizing event frequencies.
            p_adj = float(np.clip(ratio ** (0.25 * lr), 0.95, 1.06))
            # Small omega nudge in log space for smoother changes in NAV growth.
            om_adj = float(np.clip(0.020 * lr * np.log(max(ratio, 1e-6)), -0.006, 0.006))

            r = row.to_dict()
            r["delta_rep_ratio_scale"] = float(np.clip(float(r.get("delta_rep_ratio_scale", 1.0)) * rr_adj, 0.75, 1.60))
            r["delta_p_rep_mult"] = float(np.clip(float(r.get("delta_p_rep_mult", 1.0)) * p_adj, 0.80, 1.20))
            r["delta_omega_bump"] = float(np.clip(float(r.get("delta_omega_bump", 0.0)) + om_adj, -0.06, 0.06))

            # Optional timing shift tweak for severe miss.
            shift = int(r.get("delta_rep_timing_shift_q", 0))
            if ratio > 1.20:
                shift = max(shift - 1, -2)
            elif ratio < 0.85:
                shift = min(shift + 1, 2)
            r["delta_rep_timing_shift_q"] = int(shift)
            upd_rows.append(r)

        gd = pd.DataFrame(upd_rows)

        # Portfolio-level IRR correction:
        # if neutral IRR is below historical target, nudge repayment/NAV intensity up
        # through the same strategy-grade deltas (grade-aware via target_irr_sg).
        pred_irr = float(pred.get("irr", math.nan))
        if np.isfinite(pred_irr) and np.isfinite(target_irr):
            irr_gap = float(target_irr - pred_irr)
            if abs(irr_gap) > 1e-6 and len(gd):
                # Learning-rate-aware corrections.
                om_step = float(np.clip(0.20 * irr_gap * lr, -0.006, 0.010))
                rr_step = float(np.clip(1.0 + 2.20 * irr_gap * lr, 0.94, 1.10))
                p_step = float(np.clip(1.0 + 1.00 * irr_gap * lr, 0.96, 1.06))

                rows2: list[dict[str, Any]] = []
                for _, row in gd.iterrows():
                    s = str(row.get("Adj Strategy", "ALL"))
                    g = str(row.get("Grade", "ALL"))
                    if (s, g) in target_idx.index and np.isfinite(target_irr):
                        irr_sg = float(target_idx.loc[(s, g), "target_irr"])
                        grade_factor = float(np.clip(irr_sg / max(target_irr, 1e-6), 0.7, 1.5))
                    else:
                        grade_factor = 1.0

                    r = row.to_dict()
                    r["delta_rep_ratio_scale"] = float(np.clip(float(r.get("delta_rep_ratio_scale", 1.0)) * (rr_step ** grade_factor), 0.75, 1.80))
                    r["delta_p_rep_mult"] = float(np.clip(float(r.get("delta_p_rep_mult", 1.0)) * (p_step ** grade_factor), 0.80, 1.30))
                    r["delta_omega_bump"] = float(np.clip(float(r.get("delta_omega_bump", 0.0)) + om_step * grade_factor, -0.08, 0.08))
                    rows2.append(r)

                gd = pd.DataFrame(rows2)

    # Post-tune global IRR alignment:
    # keep strategy-grade pattern intact and only scale repayment intensity globally.
    align_rows: list[dict[str, Any]] = []
    coarse_scales = [0.60, 0.75, 0.90, 1.00, 1.10, 1.25, 1.40, 1.60]
    for sc in coarse_scales:
        c_try = _scale_repayment_intensity(best_calibs, sc)
        p_try, sg_try, o_try = _evaluate(c_try)
        irr_try = float(p_try.get("irr", math.nan))
        tvpi_try = float(p_try.get("tvpi", math.nan))
        align_rows.append(
            {
                "phase": "coarse",
                "scale": float(sc),
                "pred_irr": irr_try,
                "pred_tvpi": tvpi_try,
                "irr_gap": abs(irr_try - target_irr) if np.isfinite(irr_try) else np.nan,
                "objective": float(o_try),
            }
        )

    coarse_df = pd.DataFrame(align_rows)
    valid = coarse_df[np.isfinite(pd.to_numeric(coarse_df["pred_irr"], errors="coerce"))].copy()
    align_scale = 1.0
    if not valid.empty:
        cand = valid.iloc[(valid["pred_irr"] - target_irr).abs().argmin()]
        align_scale = float(cand["scale"])

        below = valid[valid["pred_irr"] <= target_irr].sort_values("pred_irr")
        above = valid[valid["pred_irr"] >= target_irr].sort_values("pred_irr")
        if not below.empty and not above.empty:
            s_below = float(below.iloc[-1]["scale"])
            s_above = float(above.iloc[0]["scale"])
            irr_below = float(below.iloc[-1]["pred_irr"])
            irr_above = float(above.iloc[0]["pred_irr"])

            lo_scale = float(min(s_below, s_above))
            hi_scale = float(max(s_below, s_above))
            if s_below <= s_above:
                irr_lo, irr_hi = irr_below, irr_above
            else:
                irr_lo, irr_hi = irr_above, irr_below

            gap_lo = irr_lo - target_irr
            gap_hi = irr_hi - target_irr
            if (
                hi_scale > lo_scale
                and np.isfinite(irr_lo)
                and np.isfinite(irr_hi)
                and gap_lo * gap_hi <= 0.0
            ):
                for _ in range(5):
                    mid_scale = 0.5 * (lo_scale + hi_scale)
                    c_mid = _scale_repayment_intensity(best_calibs, mid_scale)
                    p_mid, sg_mid, o_mid = _evaluate(c_mid)
                    irr_mid = float(p_mid.get("irr", math.nan))
                    tvpi_mid = float(p_mid.get("tvpi", math.nan))
                    align_rows.append(
                        {
                            "phase": "refine",
                            "scale": float(mid_scale),
                            "pred_irr": irr_mid,
                            "pred_tvpi": tvpi_mid,
                            "irr_gap": abs(irr_mid - target_irr) if np.isfinite(irr_mid) else np.nan,
                            "objective": float(o_mid),
                        }
                    )
                    if not np.isfinite(irr_mid):
                        break
                    gap_mid = irr_mid - target_irr
                    if gap_lo * gap_mid <= 0.0:
                        hi_scale = mid_scale
                        irr_hi = irr_mid
                        gap_hi = gap_mid
                    else:
                        lo_scale = mid_scale
                        irr_lo = irr_mid
                        gap_lo = gap_mid
                valid2 = pd.DataFrame(align_rows)
                valid2 = valid2[np.isfinite(pd.to_numeric(valid2["pred_irr"], errors="coerce"))].copy()
                if not valid2.empty:
                    cand2 = valid2.iloc[(valid2["pred_irr"] - target_irr).abs().argmin()]
                    align_scale = float(cand2["scale"])

    aligned_calibs = _scale_repayment_intensity(best_calibs, align_scale)
    aligned_pred, aligned_sg_pred, aligned_obj = _evaluate(aligned_calibs)
    aligned_irr = float(aligned_pred.get("irr", math.nan))
    aligned_tvpi = float(aligned_pred.get("tvpi", math.nan))
    if np.isfinite(aligned_irr):
        best_calibs = aligned_calibs
        best_sg_pred = aligned_sg_pred
        best_pred = {"irr": aligned_irr, "tvpi": aligned_tvpi}
        best_obj = float(aligned_obj)

    # Recallable alignment (optional): scale recallable intensity to match historical recall/commitment.
    recall_align_scale = 1.0
    recall_align_rows: list[dict[str, Any]] = []
    target_recall_commit = float(recall_target.get("recall_to_commit", math.nan))
    if np.isfinite(target_recall_commit) and commit_total > 1e-9:
        for sc in [0.15, 0.25, 0.40, 0.60, 0.80, 1.00, 1.20]:
            c_try = _scale_recallable_intensity(best_calibs, sc)
            p_try, sg_try, o_try = _evaluate(c_try)
            recall_ratio = float(p_try.get("recall", 0.0)) / commit_total
            recall_align_rows.append(
                {
                    "scale": float(sc),
                    "pred_recall_ratio": recall_ratio,
                    "target_recall_ratio": target_recall_commit,
                    "gap": abs(recall_ratio - target_recall_commit),
                }
            )
        r_df = pd.DataFrame(recall_align_rows)
        if not r_df.empty and np.isfinite(target_recall_commit):
            best_row = r_df.iloc[(r_df["gap"]).argmin()]
            recall_align_scale = float(best_row["scale"])
            best_calibs = _scale_recallable_intensity(best_calibs, recall_align_scale)
            p_post, sg_post, o_post = _evaluate(best_calibs)
            best_sg_pred = sg_post
            best_pred = {"irr": float(p_post.get("irr", math.nan)), "tvpi": float(p_post.get("tvpi", math.nan))}
            best_obj = float(o_post)

    # Mature drawdown alignment: target draw/commit from historically mature funds.
    draw_align_scale = 1.0
    draw_align_rows: list[dict[str, Any]] = []
    target_draw_commit = float(draw_target.get("draw_to_commit", math.nan))
    if np.isfinite(target_draw_commit) and commit_total > 1e-9:
        for sc in [0.70, 0.80, 0.90, 1.00, 1.10, 1.25, 1.40, 1.60]:
            c_try = _scale_draw_intensity(best_calibs, sc)
            p_try, sg_try, o_try = _evaluate(c_try)
            draw_ratio = float(p_try.get("draw", 0.0)) / commit_total
            irr_gap = abs(float(p_try.get("irr", math.nan)) - target_irr) if np.isfinite(float(p_try.get("irr", math.nan))) else 1.0
            score = 6.0 * abs(draw_ratio - target_draw_commit) + 1.5 * irr_gap
            draw_align_rows.append(
                {
                    "scale": float(sc),
                    "pred_draw_to_commit": draw_ratio,
                    "target_draw_to_commit": target_draw_commit,
                    "irr_gap": irr_gap,
                    "score": score,
                }
            )
        d_df = pd.DataFrame(draw_align_rows)
        if not d_df.empty:
            best_row = d_df.iloc[d_df["score"].argmin()]
            draw_align_scale = float(best_row["scale"])
            best_calibs = _scale_draw_intensity(best_calibs, draw_align_scale)
            p_post, sg_post, o_post = _evaluate(best_calibs)
            best_sg_pred = sg_post
            best_pred = {"irr": float(p_post.get("irr", math.nan)), "tvpi": float(p_post.get("tvpi", math.nan))}
            best_obj = float(o_post)

    # Final IRR nudge after draw/recall alignment (repayment-only, preserves draw level).
    irr_post_align_scale = 1.0
    if np.isfinite(target_irr):
        irr_rows: list[dict[str, Any]] = []
        for sc in [0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20]:
            c_try = _scale_repayment_intensity(best_calibs, sc)
            p_try, sg_try, o_try = _evaluate(c_try)
            irr_try = float(p_try.get("irr", math.nan))
            irr_rows.append(
                {
                    "scale": float(sc),
                    "pred_irr": irr_try,
                    "irr_gap": abs(irr_try - target_irr) if np.isfinite(irr_try) else np.nan,
                }
            )
        irr_df = pd.DataFrame(irr_rows)
        irr_df = irr_df[np.isfinite(pd.to_numeric(irr_df["pred_irr"], errors="coerce"))].copy()
        if not irr_df.empty:
            best_row = irr_df.iloc[(irr_df["pred_irr"] - target_irr).abs().argmin()]
            irr_post_align_scale = float(best_row["scale"])
            best_calibs = _scale_repayment_intensity(best_calibs, irr_post_align_scale)
            p_post, sg_post, o_post = _evaluate(best_calibs)
            best_sg_pred = sg_post
            best_pred = {"irr": float(p_post.get("irr", math.nan)), "tvpi": float(p_post.get("tvpi", math.nan))}
            best_obj = float(o_post)

    # Return best iteration artifacts + diagnostics.
    diag = {
        "irr_target_mode": str(irr_target_mode),
        "target_irr": float(target_irr),
        "target_tvpi": float(target_tvpi),
        "best_objective": float(best_obj),
        "best_pred_irr": float(best_pred.get("irr", math.nan)),
        "best_pred_tvpi": float(best_pred.get("tvpi", math.nan)),
        "irr_alignment_scale": float(align_scale),
        "irr_alignment_pred_irr": float(aligned_irr) if np.isfinite(aligned_irr) else math.nan,
        "irr_alignment_pred_tvpi": float(aligned_tvpi) if np.isfinite(aligned_tvpi) else math.nan,
        "target_recall_to_commit": float(recall_target.get("recall_to_commit", math.nan)),
        "target_recall_to_rep": float(recall_target.get("recall_to_rep", math.nan)),
        "recall_alignment_scale": float(recall_align_scale),
        "target_draw_to_commit_mature": float(draw_target.get("draw_to_commit", math.nan)),
        "target_draw_n_mature_funds": int(draw_target.get("n_mature_funds", 0)),
        "draw_alignment_scale": float(draw_align_scale),
        "irr_post_alignment_scale": float(irr_post_align_scale),
    }
    align_df = pd.DataFrame(align_rows)
    return best_calibs, sg_target, diag, align_df


def run_neutral_tuning_pipeline(
    *,
    hist_path: str | Path,
    portfolio_path: str | Path,
    base_calibration_dir: str | Path,
    cfg: PipelineConfig,
    n_sims_tune: int = 120,
    n_iters: int = 10,
    n_sims_final: int | None = None,
    irr_target_mode: str = "pooled",
) -> NeutralTuneResult:
    tuned_calibs, sg_target, diag, irr_align = tune_neutral_calibration_to_history(
        hist_path=hist_path,
        portfolio_path=portfolio_path,
        base_calibration_dir=base_calibration_dir,
        cfg=cfg,
        n_sims_tune=n_sims_tune,
        n_iters=n_iters,
        irr_target_mode=irr_target_mode,
    )

    # Persist tuned calibration artifacts.
    tuned_dir = cfg.calibration_dir
    ensure_dir(tuned_dir)
    # Preserve existing fit artifacts/MSCI/overrun inputs from base calibration.
    base = Path(base_calibration_dir)
    for fn in [
        "timing_probs_selected.csv",
        "ratio_fit_selected.csv",
        "omega_selected.csv",
        "target_fund_size_bins.json",
        "msci_model.json",
        "avg_overrun_by_strategy.csv",
        "msci_quarterly_history.csv",
    ]:
        src = base / fn
        dst = tuned_dir / fn
        if src.exists():
            dst.write_bytes(src.read_bytes())

    tuned_calibs.save(tuned_dir)
    sg_target.to_csv(tuned_dir / "neutral_tuning_targets_by_strategy_grade.csv", index=False)
    pd.DataFrame([diag]).to_csv(tuned_dir / "neutral_tuning_diagnostics.csv", index=False)
    if len(irr_align):
        irr_align.to_csv(tuned_dir / "irr_alignment_search.csv", index=False)

    # Final neutral projection with tuned calibration.
    if n_sims_final is not None:
        cfg.simulation.n_sims = int(n_sims_final)
    from .pipeline import run_projection_pipeline  # local import to avoid cycle

    out = run_projection_pipeline(
        portfolio_path=portfolio_path,
        calibration_dir=tuned_dir,
        cfg=cfg,
    )
    port = pd.read_csv(Path(out["projection_dir"]) / "sim_outputs" / "sim_portfolio_series.csv")
    achieved = _calc_portfolio_metrics(port)

    return NeutralTuneResult(
        target_irr=float(diag["target_irr"]),
        target_tvpi=float(diag["target_tvpi"]),
        achieved_irr=float(achieved["irr"]) if np.isfinite(achieved["irr"]) else math.nan,
        achieved_tvpi=float(achieved["tvpi"]) if np.isfinite(achieved["tvpi"]) else math.nan,
        objective=float(diag["best_objective"]),
        iterations=int(n_iters),
        tuned_calibration_dir=str(tuned_dir),
        neutral_run_root=str(cfg.run_root),
    )


__all__ = ["NeutralTuneResult", "run_neutral_tuning_pipeline", "tune_neutral_calibration_to_history"]
