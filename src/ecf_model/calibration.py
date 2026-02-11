from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats

from .config import CalibrationConfig, FitConfig
from .features import add_size_bucket, compute_fund_end_dates, compute_invest_end_by_fund
from .utils import ensure_dir, make_age_bucket, parse_quarter_text, recallable_to_flow


AGE_BINS_Q = [-1, 3, 7, 11, 15, 19, 1000]
AGE_LABELS = ["0-3", "4-7", "8-11", "12-15", "16-19", "20+"]
TAIL_BINS = [-1, 3, 7, 11, 1000]
TAIL_LABELS = ["0-3", "4-7", "8-11", "12+"]
LIFECYCLE_PHASES = ["far", "mid", "late", "final", "post_0_4", "post_5_8", "post_9_12", "post_12p"]
LIFECYCLE_NAV_FLOOR_PHASES = {"late", "final", "post_0_4", "post_5_8", "post_9_12", "post_12p"}


@dataclass
class CalibrationArtifacts:
    global_deltas: dict[str, Any]
    group_deltas: pd.DataFrame
    tail_floors: pd.DataFrame
    endgame_ramp: pd.DataFrame
    lifecycle_phase: pd.DataFrame
    post_invest_draw: pd.DataFrame
    pre_end_repayment: pd.DataFrame
    stability_attenuation: pd.DataFrame
    summary: pd.DataFrame

    def save(self, calibration_dir: Path) -> None:
        ensure_dir(calibration_dir)
        with (calibration_dir / "holdout_recalibration.json").open("w", encoding="utf-8") as f:
            json.dump(self.global_deltas, f, indent=2)
        self.group_deltas.to_csv(calibration_dir / "holdout_group_recalibration.csv", index=False)
        self.tail_floors.to_csv(calibration_dir / "tail_repayment_floors.csv", index=False)
        self.endgame_ramp.to_csv(calibration_dir / "endgame_repayment_ramp.csv", index=False)
        self.lifecycle_phase.to_csv(calibration_dir / "lifecycle_phase_multipliers.csv", index=False)
        self.post_invest_draw.to_csv(calibration_dir / "post_invest_draw_multipliers.csv", index=False)
        self.pre_end_repayment.to_csv(calibration_dir / "pre_end_repayment_multipliers.csv", index=False)
        self.stability_attenuation.to_csv(calibration_dir / "stability_attenuation.csv", index=False)
        self.summary.to_csv(calibration_dir / "holdout_recalibration_summary.csv", index=False)


@dataclass
class _PreparedHistory:
    data: pd.DataFrame
    cutoff_qe: pd.Timestamp


def _finite(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def _trimmed_mean(x: pd.Series, lo: float = 0.01, hi: float = 0.99) -> float:
    y = _finite(x)
    if y.empty:
        return np.nan
    ql = float(y.quantile(lo))
    qh = float(y.quantile(hi))
    return float(y.clip(ql, qh).mean())


def _safe_ratio(a: float, b: float, default: float = 1.0) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or abs(float(b)) < 1e-12:
        return float(default)
    return float(a / b)


def _two_prop_pvalue(events_a: int, n_a: int, events_b: int, n_b: int) -> float:
    if min(n_a, n_b) <= 0:
        return 1.0
    p_a = events_a / n_a
    p_b = events_b / n_b
    p_pool = (events_a + events_b) / (n_a + n_b)
    denom = np.sqrt(max(p_pool * (1.0 - p_pool) * (1.0 / n_a + 1.0 / n_b), 1e-16))
    z = (p_b - p_a) / denom
    return float(2.0 * (1.0 - stats.norm.cdf(abs(z))))


def _mean_diff_pvalue(train: pd.Series, holdout: pd.Series, log_transform: bool = True) -> float:
    tr = _finite(train)
    ho = _finite(holdout)
    if len(tr) < 8 or len(ho) < 8:
        return 1.0
    if log_transform:
        tr = np.log1p(np.clip(tr, 0.0, None))
        ho = np.log1p(np.clip(ho, 0.0, None))
    _, p = stats.ttest_ind(tr, ho, equal_var=False, nan_policy="omit")
    return float(p) if np.isfinite(p) else 1.0


def _best_timing_shift(
    tr_rep: pd.DataFrame,
    ho_rep: pd.DataFrame,
    shift_min: int,
    shift_max: int,
    eps: float,
) -> tuple[int, dict[int, float]]:
    if tr_rep.empty or ho_rep.empty:
        return 0, {}

    p_by_age = tr_rep.groupby("AgeBucket", observed=False)["rep_event"].mean().to_dict()
    p_global = float(np.clip(tr_rep["rep_event"].mean() if len(tr_rep) else 0.0, eps, 1.0 - eps))

    y = pd.to_numeric(ho_rep["rep_event"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    age = pd.to_numeric(ho_rep["age_q"], errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(age)
    if not m.any():
        return 0, {}

    y = y[m]
    age = age[m]
    losses: dict[int, float] = {}

    for shift in range(int(shift_min), int(shift_max) + 1):
        shifted = np.maximum(age + shift, 0.0)
        buckets = pd.cut(shifted, bins=AGE_BINS_Q, labels=AGE_LABELS)
        p = pd.Series(buckets).map(p_by_age).astype(float).fillna(p_global).to_numpy(dtype=float)
        p = np.clip(p, eps, 1.0 - eps)
        losses[int(shift)] = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    if not losses:
        return 0, {}
    return int(min(losses, key=losses.get)), losses


def _prepare_history(hist_to_cutoff: pd.DataFrame, cutoff_qe: pd.Timestamp, fit_cfg: FitConfig) -> _PreparedHistory:
    h = add_size_bucket(hist_to_cutoff, list(fit_cfg.size_bins), list(fit_cfg.size_labels)).copy()
    h = h.sort_values(["FundID", "quarter_end"]).copy()
    end_info = compute_fund_end_dates(h)

    draw_abs = pd.to_numeric(h["Adj Drawdown EUR"], errors="coerce").abs().fillna(0.0)
    rep_abs = pd.to_numeric(h["Adj Repayment EUR"], errors="coerce").abs().fillna(0.0)
    nav_curr = pd.to_numeric(h["NAV Adjusted EUR"], errors="coerce").clip(lower=0.0)
    nav_prev = nav_curr.groupby(h["FundID"]).shift(1)

    recall_abs = (
        h.sort_values(["FundID", "quarter_end"])
        .groupby("FundID")["Recallable"]
        .apply(recallable_to_flow)
        .reset_index(level=0, drop=True)
        .reindex(h.index)
    )
    recall_abs = pd.to_numeric(recall_abs, errors="coerce").abs().fillna(0.0)
    h["recall_abs"] = recall_abs
    recall_cum = recall_abs.groupby(h["FundID"]).cumsum()

    commit = pd.to_numeric(h["Commitment EUR"], errors="coerce").clip(lower=0.0)
    commit_max = commit.groupby(h["FundID"]).transform("max").fillna(0.0)
    draw_denom = (commit_max + recall_cum).replace(0.0, np.nan)

    h["draw_abs"] = draw_abs
    h["rep_abs"] = rep_abs
    h["nav_curr"] = nav_curr
    h["nav_prev"] = nav_prev
    h["draw_event"] = (draw_abs > 0.0).astype(int)
    h["rep_event"] = ((rep_abs > 0.0) & (nav_prev > fit_cfg.nav_gate)).astype(int)
    h["draw_flow_ratio"] = np.where(draw_denom > 0, draw_abs / draw_denom, np.nan)
    h["rep_ratio_cond"] = np.where(nav_prev > fit_cfg.nav_gate, rep_abs / nav_prev, np.nan)
    h["commit_max"] = commit_max
    h["nav_to_commit"] = np.where(commit_max > 0, nav_curr / commit_max, np.nan)

    flow_net = rep_abs - draw_abs
    omega = np.where(nav_prev > fit_cfg.nav_gate, ((nav_curr - nav_prev) - flow_net) / nav_prev, np.nan)
    h["omega"] = pd.Series(omega, index=h.index, dtype=float).clip(
        lower=fit_cfg.omega_fit_clip[0],
        upper=fit_cfg.omega_fit_clip[1],
    )

    age_q = pd.to_numeric(h.get("Fund_Age_Quarters"), errors="coerce")
    h["age_q"] = age_q.fillna(h.groupby("FundID").cumcount() + 1)
    h["AgeBucket"] = h["age_q"].apply(make_age_bucket)

    q_ord = h["quarter_end"].dt.to_period("Q").map(lambda p: p.ordinal if pd.notna(p) else np.nan)
    h["q_ord"] = q_ord
    h["fund_end_qe"] = h["FundID"].map(end_info.fund_end_qe)
    h["end_ord"] = h["fund_end_qe"].dt.to_period("Q").map(lambda p: p.ordinal if pd.notna(p) else np.nan)
    h["q_to_end"] = h["end_ord"] - h["q_ord"]
    h["rep_gate"] = h["nav_prev"] > fit_cfg.nav_gate
    h["cutoff_qe"] = cutoff_qe
    return _PreparedHistory(data=h, cutoff_qe=cutoff_qe)


def _compute_deltas(
    tr: pd.DataFrame,
    ho: pd.DataFrame,
    cfg: CalibrationConfig,
    fit_cfg: FitConfig,
    *,
    shift_min: int = -2,
    shift_max: int = 2,
    apply_significance_gate: bool = True,
) -> dict[str, Any]:
    default = {
        "delta_p_draw_mult": 1.0,
        "delta_p_rep_mult": 1.0,
        "delta_p_rc_mult": 1.0,
        "delta_rep_ratio_scale": 1.0,
        "delta_draw_ratio_scale": 1.0,
        "delta_rc_ratio_scale": 1.0,
        "delta_omega_bump": 0.0,
        "delta_rep_timing_shift_q": 0,
        "raw": {},
        "weights": {},
        "pvalues": {},
        "counts": {},
    }
    if tr.empty or ho.empty:
        return default

    tr_rep = tr[tr["rep_gate"]].copy()
    ho_rep = ho[ho["rep_gate"]].copy()
    tr_rep_pos = tr_rep[tr_rep["rep_event"] == 1].copy()
    ho_rep_pos = ho_rep[ho_rep["rep_event"] == 1].copy()

    p_draw_tr = float(tr["draw_event"].mean()) if len(tr) else np.nan
    p_draw_ho = float(ho["draw_event"].mean()) if len(ho) else np.nan
    raw_p_draw_mult = _safe_ratio(p_draw_ho, p_draw_tr, default=1.0)

    tr_draw_pos = tr[tr["draw_event"] == 1].copy()
    ho_draw_pos = ho[ho["draw_event"] == 1].copy()

    p_tr = float(tr_rep["rep_event"].mean()) if len(tr_rep) else np.nan
    p_ho = float(ho_rep["rep_event"].mean()) if len(ho_rep) else np.nan
    raw_p_mult = _safe_ratio(p_ho, p_tr, default=1.0)

    rr_tr = _trimmed_mean(tr_rep_pos["rep_ratio_cond"], lo=0.01, hi=0.99)
    rr_ho = _trimmed_mean(ho_rep_pos["rep_ratio_cond"], lo=0.01, hi=0.99)
    raw_rr_scale = _safe_ratio(rr_ho, rr_tr, default=1.0)

    dr_tr = _trimmed_mean(tr_draw_pos["draw_flow_ratio"], lo=0.01, hi=0.99)
    dr_ho = _trimmed_mean(ho_draw_pos["draw_flow_ratio"], lo=0.01, hi=0.99)
    raw_dr_scale = _safe_ratio(dr_ho, dr_tr, default=1.0)

    om_tr = _trimmed_mean(tr["omega"], lo=0.05, hi=0.95)
    om_ho = _trimmed_mean(ho["omega"], lo=0.05, hi=0.95)
    raw_om_bump = float(om_ho - om_tr) if np.isfinite(om_tr) and np.isfinite(om_ho) else 0.0

    shift_raw, shift_losses = _best_timing_shift(
        tr_rep=tr_rep,
        ho_rep=ho_rep,
        shift_min=shift_min,
        shift_max=shift_max,
        eps=1e-6,
    )

    p_p = _two_prop_pvalue(int(tr_rep["rep_event"].sum()), int(len(tr_rep)), int(ho_rep["rep_event"].sum()), int(len(ho_rep)))
    p_draw_p = _two_prop_pvalue(int(tr["draw_event"].sum()), int(len(tr)), int(ho["draw_event"].sum()), int(len(ho)))
    p_rr = _mean_diff_pvalue(tr_rep_pos["rep_ratio_cond"], ho_rep_pos["rep_ratio_cond"], log_transform=True)
    p_dr = _mean_diff_pvalue(tr_draw_pos["draw_flow_ratio"], ho_draw_pos["draw_flow_ratio"], log_transform=True)
    p_om = _mean_diff_pvalue(tr["omega"], ho["omega"], log_transform=False)

    sig_draw_p = p_draw_p < cfg.significance_alpha
    sig_p = p_p < cfg.significance_alpha
    sig_rr = p_rr < cfg.significance_alpha
    sig_dr = p_dr < cfg.significance_alpha
    sig_om = p_om < cfg.significance_alpha

    if apply_significance_gate:
        if not sig_draw_p:
            raw_p_draw_mult = 1.0
        if not sig_p:
            raw_p_mult = 1.0
        if not sig_rr:
            raw_rr_scale = 1.0
        if not sig_dr:
            raw_dr_scale = 1.0
        if not sig_om:
            raw_om_bump = 0.0

    w_p_draw = float(len(ho) / (len(ho) + cfg.shrink_n)) if len(ho) else 0.0
    w_p = float(len(ho_rep) / (len(ho_rep) + cfg.shrink_n)) if len(ho_rep) else 0.0
    w_rr = float(len(ho_rep_pos) / (len(ho_rep_pos) + cfg.shrink_n)) if len(ho_rep_pos) else 0.0
    w_dr = float(len(ho_draw_pos) / (len(ho_draw_pos) + cfg.shrink_n)) if len(ho_draw_pos) else 0.0
    w_om = float(ho["omega"].notna().sum() / (ho["omega"].notna().sum() + cfg.shrink_n)) if len(ho) else 0.0
    w_shift = float(len(ho_rep) / (len(ho_rep) + 2.0 * cfg.shrink_n)) if len(ho_rep) else 0.0

    # Timing shift is only applied if best shift improves holdout logloss enough.
    loss_0 = shift_losses.get(0)
    loss_best = shift_losses.get(shift_raw)
    if loss_0 is None or loss_best is None:
        shift_raw = 0
    else:
        improve = (loss_0 - loss_best) / max(abs(loss_0), 1e-9)
        if improve < 0.005:
            shift_raw = 0

    out = {
        "delta_p_draw_mult": float(np.clip(1.0 + w_p_draw * (raw_p_draw_mult - 1.0), *cfg.p_draw_mult_clip)),
        "delta_p_rep_mult": float(np.clip(1.0 + w_p * (raw_p_mult - 1.0), *cfg.p_rep_mult_clip)),
        "delta_p_rc_mult": 1.0,
        "delta_rep_ratio_scale": float(np.clip(1.0 + w_rr * (raw_rr_scale - 1.0), *cfg.rep_ratio_scale_clip)),
        "delta_draw_ratio_scale": float(np.clip(1.0 + w_dr * (raw_dr_scale - 1.0), *cfg.draw_ratio_scale_clip)),
        "delta_rc_ratio_scale": 1.0,
        "delta_omega_bump": float(np.clip(w_om * raw_om_bump, *cfg.omega_bump_clip)),
        "delta_rep_timing_shift_q": int(np.clip(int(round(w_shift * shift_raw)), shift_min, shift_max)),
        "raw": {
            "p_draw_train": p_draw_tr,
            "p_draw_holdout": p_draw_ho,
            "p_draw_mult": raw_p_draw_mult,
            "p_rep_train": p_tr,
            "p_rep_holdout": p_ho,
            "p_rep_mult": raw_p_mult,
            "rep_ratio_train": rr_tr,
            "rep_ratio_holdout": rr_ho,
            "rep_ratio_scale": raw_rr_scale,
            "draw_ratio_train": dr_tr,
            "draw_ratio_holdout": dr_ho,
            "draw_ratio_scale": raw_dr_scale,
            "omega_train": om_tr,
            "omega_holdout": om_ho,
            "omega_bump": raw_om_bump,
            "timing_shift_q": int(shift_raw),
        },
        "weights": {
            "p_draw": w_p_draw,
            "p_rep": w_p,
            "rep_ratio": w_rr,
            "draw_ratio": w_dr,
            "omega": w_om,
            "timing_shift": w_shift,
        },
        "pvalues": {
            "p_draw": p_draw_p,
            "p_rep": p_p,
            "rep_ratio": p_rr,
            "draw_ratio": p_dr,
            "omega": p_om,
        },
        "counts": {
            "n_train": int(len(tr)),
            "n_holdout": int(len(ho)),
            "n_train_rep_gate": int(len(tr_rep)),
            "n_holdout_rep_gate": int(len(ho_rep)),
            "n_train_rep_pos": int(len(tr_rep_pos)),
            "n_holdout_rep_pos": int(len(ho_rep_pos)),
            "n_train_draw_pos": int(len(tr_draw_pos)),
            "n_holdout_draw_pos": int(len(ho_draw_pos)),
            "n_holdout_omega": int(ho["omega"].notna().sum()),
        },
    }
    return out


def _split_train_holdout(prepared: _PreparedHistory, holdout_q: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    h = prepared.data
    end_ord = pd.Period(prepared.cutoff_qe, freq="Q").ordinal
    start_ord = int(end_ord - int(holdout_q) + 1)
    tr = h[h["q_ord"] < start_ord].copy()
    ho = h[(h["q_ord"] >= start_ord) & (h["q_ord"] <= end_ord)].copy()
    return tr, ho


def _ensure_group_cols(df: pd.DataFrame, group_cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in group_cols:
        if c not in out.columns:
            out[c] = "ALL"
        out[c] = out[c].astype("string").fillna("ALL")
    return out


def _recall_propensity_by_group(
    prepared: _PreparedHistory, group_cols: Iterable[str]
) -> tuple[pd.Series, float]:
    df = prepared.data.copy()
    if "recall_abs" not in df.columns or df.empty:
        return pd.Series(dtype=float), 1.0

    gcols = list(group_cols)
    fund_groups = (
        df.groupby("FundID")[gcols]
        .agg(lambda s: s.mode().iat[0] if len(s.mode()) else s.iloc[0])
        .reset_index()
    )
    recall_any = df.groupby("FundID")["recall_abs"].sum().gt(0.0)
    fund_groups = fund_groups.join(recall_any.rename("recall_any"), on="FundID")

    global_prop = float(fund_groups["recall_any"].mean()) if len(fund_groups) else 1.0
    prop = fund_groups.groupby(gcols)["recall_any"].mean()
    return prop, global_prop


def _compute_group_deltas(
    prepared: _PreparedHistory,
    cfg: CalibrationConfig,
    fit_cfg: FitConfig,
    global_deltas: dict[str, Any],
    group_cols: Iterable[str] = ("Adj Strategy", "Grade"),
) -> pd.DataFrame:
    tr, ho = _split_train_holdout(prepared, cfg.holdout_quarters)
    if tr.empty or ho.empty:
        return pd.DataFrame()

    tr = _ensure_group_cols(tr, group_cols)
    ho = _ensure_group_cols(ho, group_cols)

    keys = pd.concat([tr[list(group_cols)], ho[list(group_cols)]], axis=0).drop_duplicates().reset_index(drop=True)
    rows: list[dict[str, Any]] = []

    g_p_draw = float(global_deltas.get("delta_p_draw_mult", 1.0))
    g_p = float(global_deltas.get("delta_p_rep_mult", 1.0))
    g_p_rc = float(global_deltas.get("delta_p_rc_mult", 1.0))
    g_rr = float(global_deltas.get("delta_rep_ratio_scale", 1.0))
    g_dr = float(global_deltas.get("delta_draw_ratio_scale", 1.0))
    g_rc = float(global_deltas.get("delta_rc_ratio_scale", 1.0))
    g_om = float(global_deltas.get("delta_omega_bump", 0.0))
    g_shift = int(global_deltas.get("delta_rep_timing_shift_q", 0))
    rc_prop_by_group, rc_prop_global = _recall_propensity_by_group(prepared, group_cols)

    for r in keys.itertuples(index=False):
        key_vals = [str(v) if pd.notna(v) else "ALL" for v in r]
        m_tr = np.logical_and.reduce([tr[c].astype(str).eq(key_vals[i]).to_numpy() for i, c in enumerate(group_cols)])
        m_ho = np.logical_and.reduce([ho[c].astype(str).eq(key_vals[i]).to_numpy() for i, c in enumerate(group_cols)])
        tr_g = tr.loc[m_tr].copy()
        ho_g = ho.loc[m_ho].copy()

        if len(ho_g) < cfg.min_obs_for_adjustment:
            continue

        local = _compute_deltas(tr_g, ho_g, cfg, fit_cfg)

        n_ho = int(local["counts"].get("n_holdout", 0))
        n_ho_rep = int(local["counts"].get("n_holdout_rep_gate", 0))
        n_ho_rep_pos = int(local["counts"].get("n_holdout_rep_pos", 0))
        n_ho_draw_pos = int(local["counts"].get("n_holdout_draw_pos", 0))
        n_ho_omega = int(local["counts"].get("n_holdout_omega", 0))

        lam_p_draw = n_ho / (n_ho + cfg.shrink_n) if n_ho else 0.0
        lam_p = n_ho_rep / (n_ho_rep + cfg.shrink_n) if n_ho_rep else 0.0
        lam_rr = n_ho_rep_pos / (n_ho_rep_pos + cfg.shrink_n) if n_ho_rep_pos else 0.0
        lam_dr = n_ho_draw_pos / (n_ho_draw_pos + cfg.shrink_n) if n_ho_draw_pos else 0.0
        lam_om = n_ho_omega / (n_ho_omega + cfg.shrink_n) if n_ho_omega else 0.0
        lam_shift = n_ho / (n_ho + cfg.shrink_n) if n_ho else 0.0

        d_p_draw = float(np.clip(g_p_draw + lam_p_draw * (float(local["delta_p_draw_mult"]) - g_p_draw), *cfg.p_draw_mult_clip))
        d_p = float(np.clip(g_p + lam_p * (float(local["delta_p_rep_mult"]) - g_p), *cfg.p_rep_mult_clip))
        d_rr = float(np.clip(g_rr + lam_rr * (float(local["delta_rep_ratio_scale"]) - g_rr), *cfg.rep_ratio_scale_clip))
        d_dr = float(np.clip(g_dr + lam_dr * (float(local["delta_draw_ratio_scale"]) - g_dr), *cfg.draw_ratio_scale_clip))
        d_om = float(np.clip(g_om + lam_om * (float(local["delta_omega_bump"]) - g_om), *cfg.omega_bump_clip))
        d_shift = int(np.clip(int(round(g_shift + lam_shift * (int(local["delta_rep_timing_shift_q"]) - g_shift))), -2, 2))

        rc_key = tuple(key_vals) if len(key_vals) > 1 else key_vals[0]
        row = {
            "delta_p_draw_mult": d_p_draw,
            "delta_p_rep_mult": d_p,
            "delta_p_rc_mult": g_p_rc,
            "delta_rep_ratio_scale": d_rr,
            "delta_draw_ratio_scale": d_dr,
            "delta_rc_ratio_scale": g_rc,
            "delta_omega_bump": d_om,
            "delta_rep_timing_shift_q": d_shift,
            "rc_propensity": float(rc_prop_by_group.get(rc_key, rc_prop_global)),
            "n_holdout": n_ho,
            "n_holdout_rep_gate": n_ho_rep,
            "n_holdout_rep_pos": n_ho_rep_pos,
            "n_holdout_draw_pos": n_ho_draw_pos,
            "n_holdout_omega": n_ho_omega,
            "lambda_p_draw": lam_p_draw,
            "lambda_p_rep": lam_p,
            "lambda_rep_ratio": lam_rr,
            "lambda_draw_ratio": lam_dr,
            "lambda_omega": lam_om,
            "lambda_shift": lam_shift,
        }
        for i, c in enumerate(group_cols):
            row[c] = key_vals[i]
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    order = list(group_cols) + [c for c in out.columns if c not in group_cols]
    return out[order].sort_values(list(group_cols)).reset_index(drop=True)


def _compute_tail_floors(hist_to_cutoff: pd.DataFrame, fit_cfg: FitConfig) -> pd.DataFrame:
    h = add_size_bucket(hist_to_cutoff, list(fit_cfg.size_bins), list(fit_cfg.size_labels)).copy()
    h = h.sort_values(["FundID", "quarter_end"]).copy()

    end_info = compute_fund_end_dates(h)
    fund_end = end_info.fund_end_qe.rename("fund_end_qe")
    h = h.join(fund_end, on="FundID")

    h["q_ord"] = h["quarter_end"].dt.to_period("Q").map(lambda p: p.ordinal if pd.notna(p) else np.nan)
    h["end_ord"] = h["fund_end_qe"].dt.to_period("Q").map(lambda p: p.ordinal if pd.notna(p) else np.nan)
    h["after_end_q"] = h["q_ord"] - h["end_ord"]

    nav_curr = pd.to_numeric(h["NAV Adjusted EUR"], errors="coerce").clip(lower=0)
    nav_prev = nav_curr.groupby(h["FundID"]).shift(1)
    rep_abs = pd.to_numeric(h["Adj Repayment EUR"], errors="coerce").abs().fillna(0)
    draw_abs = pd.to_numeric(h["Adj Drawdown EUR"], errors="coerce").abs().fillna(0)

    h["rep_event"] = ((rep_abs > 0) & (nav_prev > fit_cfg.nav_gate)).astype(int)
    h["rep_ratio_cond"] = np.where(nav_prev > fit_cfg.nav_gate, rep_abs / nav_prev, np.nan)
    h["draw_event"] = (draw_abs > 0).astype(int)
    flow_net = rep_abs - draw_abs
    h["omega"] = np.where(nav_prev > fit_cfg.nav_gate, ((nav_curr - nav_prev) - flow_net) / nav_prev, np.nan)

    commit_max = pd.to_numeric(h["Commitment EUR"], errors="coerce").groupby(h["FundID"]).transform("max")
    h["nav_to_commit"] = np.where(commit_max > 0, nav_curr / commit_max, np.nan)

    post = h[(h["after_end_q"] >= 0) & np.isfinite(h["after_end_q"])].copy()
    post["tail_bucket"] = pd.cut(post["after_end_q"], bins=TAIL_BINS, labels=TAIL_LABELS)

    rows: list[dict[str, Any]] = []
    grouped = post.groupby(["Adj Strategy", "tail_bucket"], observed=False)
    for (strategy, tail_bucket), g in grouped:
        if len(g) == 0:
            continue
        rep_pos = g[g["rep_event"] == 1]
        rows.append(
            {
                "Adj Strategy": str(strategy),
                "tail_bucket": str(tail_bucket),
                "n_obs": int(len(g)),
                "p_rep_floor": float(np.clip(g["rep_event"].mean(), 0.0, 1.0)),
                "rep_ratio_floor": float(max(_trimmed_mean(rep_pos["rep_ratio_cond"], lo=0.01, hi=0.99), 0.0))
                if len(rep_pos)
                else np.nan,
                "draw_prob_cap": float(np.clip(g["draw_event"].mean(), 0.0, 1.0)),
                "nav_to_commit_p50": float(_finite(g["nav_to_commit"]).quantile(0.5)) if _finite(g["nav_to_commit"]).size else np.nan,
                "omega_drag": float(_trimmed_mean(g["omega"], lo=0.05, hi=0.95)) if _finite(g["omega"]).size else 0.0,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    g_rows: list[dict[str, Any]] = []
    for tb, g in out.groupby("tail_bucket", dropna=False):
        w = np.maximum(g["n_obs"].to_numpy(dtype=float), 1.0)
        g_rows.append(
            {
                "Adj Strategy": "ALL",
                "tail_bucket": tb,
                "n_obs": int(np.sum(w)),
                "p_rep_floor": float(np.average(g["p_rep_floor"].to_numpy(dtype=float), weights=w)),
                "rep_ratio_floor": float(np.average(np.nan_to_num(g["rep_ratio_floor"].to_numpy(dtype=float), nan=0.0), weights=w)),
                "draw_prob_cap": float(np.average(g["draw_prob_cap"].to_numpy(dtype=float), weights=w)),
                "nav_to_commit_p50": float(np.average(np.nan_to_num(g["nav_to_commit_p50"].to_numpy(dtype=float), nan=0.0), weights=w)),
                "omega_drag": float(np.average(np.nan_to_num(g["omega_drag"].to_numpy(dtype=float), nan=0.0), weights=w)),
            }
        )
    out = pd.concat([out, pd.DataFrame(g_rows)], ignore_index=True)

    # Fill missing strategy-tail combinations from strategy aggregate then global.
    strategies = sorted([s for s in out["Adj Strategy"].unique() if s != "ALL"])
    full_rows: list[dict[str, Any]] = []
    for s in strategies + ["ALL"]:
        for tb in TAIL_LABELS:
            sub = out[(out["Adj Strategy"] == s) & (out["tail_bucket"] == tb)]
            if len(sub):
                full_rows.append(sub.iloc[0].to_dict())
                continue
            strat_any = out[(out["Adj Strategy"] == s)]
            glob = out[(out["Adj Strategy"] == "ALL") & (out["tail_bucket"] == tb)]
            if len(strat_any):
                base = strat_any.iloc[0].to_dict()
            elif len(glob):
                base = glob.iloc[0].to_dict()
            else:
                base = {
                    "Adj Strategy": s,
                    "tail_bucket": tb,
                    "n_obs": 0,
                    "p_rep_floor": 0.1,
                    "rep_ratio_floor": 0.02,
                    "draw_prob_cap": 0.05,
                    "nav_to_commit_p50": np.nan,
                    "omega_drag": -0.005,
                }
            base["Adj Strategy"] = s
            base["tail_bucket"] = tb
            full_rows.append(base)

    return pd.DataFrame(full_rows).sort_values(["Adj Strategy", "tail_bucket"]).reset_index(drop=True)


def _compute_endgame_ramp(
    hist_to_cutoff: pd.DataFrame,
    fit_cfg: FitConfig,
    cfg: CalibrationConfig,
) -> pd.DataFrame:
    h = hist_to_cutoff.sort_values(["FundID", "quarter_end"]).copy()
    end_info = compute_fund_end_dates(h)

    nav_curr = pd.to_numeric(h["NAV Adjusted EUR"], errors="coerce").clip(lower=0)
    nav_prev = nav_curr.groupby(h["FundID"]).shift(1)
    rep_abs = pd.to_numeric(h["Adj Repayment EUR"], errors="coerce").abs().fillna(0)

    h["nav_prev"] = nav_prev
    h["rep_abs"] = rep_abs
    h["rep_event"] = ((rep_abs > 0) & (nav_prev > fit_cfg.nav_gate)).astype(int)
    h["rep_ratio_cond"] = np.where(nav_prev > fit_cfg.nav_gate, rep_abs / nav_prev, np.nan)

    h["q_ord"] = h["quarter_end"].dt.to_period("Q").map(lambda p: p.ordinal if pd.notna(p) else np.nan)
    h["fund_end_qe"] = h["FundID"].map(end_info.fund_end_qe)
    h["end_ord"] = h["fund_end_qe"].dt.to_period("Q").map(lambda p: p.ordinal if pd.notna(p) else np.nan)
    h["q_to_end"] = h["end_ord"] - h["q_ord"]

    rows: list[dict[str, Any]] = []
    for strategy, g in h.groupby("Adj Strategy"):
        g = g[g["nav_prev"] > fit_cfg.nav_gate].copy()
        pre = g[(g["q_to_end"] >= 1) & (g["q_to_end"] <= 12)].copy()
        mid = g[(g["q_to_end"] >= 13) & (g["q_to_end"] <= 24)].copy()
        if len(pre) < 30 or len(mid) < 30:
            rows.append(
                {
                    "Adj Strategy": str(strategy),
                    "n_pre": int(len(pre)),
                    "n_mid": int(len(mid)),
                    "p_rep_mult": 1.0,
                    "rep_ratio_mult": 1.0,
                    "p_value_p_rep": np.nan,
                    "p_value_rep_ratio": np.nan,
                }
            )
            continue

        p_pre = float(pre["rep_event"].mean())
        p_mid = float(mid["rep_event"].mean())
        raw_p = _safe_ratio(p_pre, p_mid, default=1.0)
        pval_p = _two_prop_pvalue(int(mid["rep_event"].sum()), int(len(mid)), int(pre["rep_event"].sum()), int(len(pre)))
        if pval_p >= cfg.significance_alpha or raw_p <= 1.0:
            raw_p = 1.0

        pre_pos = pre[pre["rep_event"] == 1]["rep_ratio_cond"]
        mid_pos = mid[mid["rep_event"] == 1]["rep_ratio_cond"]
        rr_pre = _trimmed_mean(pre_pos, lo=0.01, hi=0.99)
        rr_mid = _trimmed_mean(mid_pos, lo=0.01, hi=0.99)
        raw_rr = _safe_ratio(rr_pre, rr_mid, default=1.0)
        pval_rr = _mean_diff_pvalue(mid_pos, pre_pos, log_transform=True)
        if pval_rr >= cfg.significance_alpha or raw_rr <= 1.0:
            raw_rr = 1.0

        n_scale = len(pre) / (len(pre) + cfg.shrink_n)
        p_mult = float(np.clip(1.0 + n_scale * (raw_p - 1.0), 1.0, 2.0))
        rr_mult = float(np.clip(1.0 + n_scale * (raw_rr - 1.0), 1.0, 2.5))

        rows.append(
            {
                "Adj Strategy": str(strategy),
                "n_pre": int(len(pre)),
                "n_mid": int(len(mid)),
                "p_rep_mult": p_mult,
                "rep_ratio_mult": rr_mult,
                "p_value_p_rep": float(pval_p),
                "p_value_rep_ratio": float(pval_rr),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    w = np.maximum(out["n_pre"].to_numpy(dtype=float), 1.0)
    g_row = pd.DataFrame(
        [
            {
                "Adj Strategy": "ALL",
                "n_pre": int(np.sum(w)),
                "n_mid": int(np.sum(out["n_mid"].to_numpy(dtype=float))),
                "p_rep_mult": float(np.average(out["p_rep_mult"].to_numpy(dtype=float), weights=w)),
                "rep_ratio_mult": float(np.average(out["rep_ratio_mult"].to_numpy(dtype=float), weights=w)),
                "p_value_p_rep": np.nan,
                "p_value_rep_ratio": np.nan,
            }
        ]
    )
    out = pd.concat([out, g_row], ignore_index=True)
    return out.sort_values("Adj Strategy").reset_index(drop=True)


def _lifecycle_phase_label(q_to_end: float) -> str:
    if not np.isfinite(q_to_end):
        return "far"
    q = int(q_to_end)
    if q > 24:
        return "far"
    if q >= 13:
        return "mid"
    if q >= 5:
        return "late"
    if q >= 1:
        return "final"
    if q >= -4:
        return "post_0_4"
    if q >= -8:
        return "post_5_8"
    if q >= -12:
        return "post_9_12"
    return "post_12p"


def _compute_lifecycle_multipliers(
    prepared: _PreparedHistory,
    fit_cfg: FitConfig,
    cfg: CalibrationConfig,
) -> pd.DataFrame:
    h = prepared.data.copy()
    if h.empty:
        return pd.DataFrame()

    h["life_phase"] = pd.to_numeric(h["q_to_end"], errors="coerce").apply(_lifecycle_phase_label)
    h["Adj Strategy"] = h["Adj Strategy"].astype(str)
    h["Grade"] = h["Grade"].astype(str)

    rows: list[dict[str, Any]] = []
    keys = h[["Adj Strategy", "Grade"]].drop_duplicates().reset_index(drop=True)
    for k in keys.itertuples(index=False):
        strat = str(k[0])
        grade = str(k[1])
        g = h[(h["Adj Strategy"] == strat) & (h["Grade"] == grade)].copy()
        if g.empty:
            continue

        g_rep = g[g["rep_gate"]].copy()
        g_rep_pos = g_rep[g_rep["rep_event"] == 1].copy()
        g_draw_pos = g[g["draw_event"] == 1].copy()

        p_rep_base = float(g_rep["rep_event"].mean()) if len(g_rep) else np.nan
        rr_base = _trimmed_mean(g_rep_pos["rep_ratio_cond"], lo=0.01, hi=0.99)
        p_draw_base = float(g["draw_event"].mean()) if len(g) else np.nan
        dr_base = _trimmed_mean(g_draw_pos["draw_flow_ratio"], lo=0.01, hi=0.99)
        om_base = _trimmed_mean(g["omega"], lo=0.05, hi=0.95)
        g_nav_near = _finite(g[pd.to_numeric(g["q_to_end"], errors="coerce") <= 24]["nav_to_commit"])
        nav_base = (
            float(g_nav_near.quantile(float(cfg.lifecycle_nav_floor_quantile)))
            if g_nav_near.size
            else np.nan
        )

        for phase in LIFECYCLE_PHASES:
            gp = g[g["life_phase"] == phase].copy()
            if gp.empty:
                continue
            gp_rep = gp[gp["rep_gate"]].copy()
            gp_rep_pos = gp_rep[gp_rep["rep_event"] == 1].copy()
            gp_draw_pos = gp[gp["draw_event"] == 1].copy()

            raw_rep_p = _safe_ratio(float(gp_rep["rep_event"].mean()) if len(gp_rep) else np.nan, p_rep_base, default=1.0)
            raw_rep_r = _safe_ratio(_trimmed_mean(gp_rep_pos["rep_ratio_cond"], lo=0.01, hi=0.99), rr_base, default=1.0)
            raw_draw_p = _safe_ratio(float(gp["draw_event"].mean()), p_draw_base, default=1.0)
            raw_draw_r = _safe_ratio(_trimmed_mean(gp_draw_pos["draw_flow_ratio"], lo=0.01, hi=0.99), dr_base, default=1.0)
            raw_om = (
                _trimmed_mean(gp["omega"], lo=0.05, hi=0.95) - om_base
                if np.isfinite(_trimmed_mean(gp["omega"], lo=0.05, hi=0.95)) and np.isfinite(om_base)
                else 0.0
            )
            gp_nav = _finite(gp["nav_to_commit"])
            raw_nav_floor = (
                float(gp_nav.quantile(float(cfg.lifecycle_nav_floor_quantile)))
                if gp_nav.size
                else np.nan
            )

            pval_rep_p = _two_prop_pvalue(
                int(g_rep["rep_event"].sum()),
                int(len(g_rep)),
                int(gp_rep["rep_event"].sum()),
                int(len(gp_rep)),
            )
            pval_rep_r = _mean_diff_pvalue(g_rep_pos["rep_ratio_cond"], gp_rep_pos["rep_ratio_cond"], log_transform=True)
            pval_draw_p = _two_prop_pvalue(
                int(g["draw_event"].sum()),
                int(len(g)),
                int(gp["draw_event"].sum()),
                int(len(gp)),
            )
            pval_draw_r = _mean_diff_pvalue(g_draw_pos["draw_flow_ratio"], gp_draw_pos["draw_flow_ratio"], log_transform=True)
            pval_om = _mean_diff_pvalue(g["omega"], gp["omega"], log_transform=False)

            if not np.isfinite(pval_rep_p) or pval_rep_p >= cfg.significance_alpha:
                raw_rep_p = 1.0
            if not np.isfinite(pval_rep_r) or pval_rep_r >= cfg.significance_alpha:
                raw_rep_r = 1.0
            if not np.isfinite(pval_draw_p) or pval_draw_p >= cfg.significance_alpha:
                raw_draw_p = 1.0
            if not np.isfinite(pval_draw_r) or pval_draw_r >= cfg.significance_alpha:
                raw_draw_r = 1.0
            if not np.isfinite(pval_om) or pval_om >= cfg.significance_alpha:
                raw_om = 0.0

            if int(len(gp)) < int(cfg.lifecycle_min_obs):
                rep_p_mult = 1.0
                rep_ratio_mult = 1.0
                draw_p_mult = 1.0
                draw_ratio_mult = 1.0
                omega_bump = 0.0
                nav_floor_ratio = (
                    float(np.clip(nav_base, cfg.lifecycle_nav_floor_clip[0], cfg.lifecycle_nav_floor_clip[1]))
                    if (phase in LIFECYCLE_NAV_FLOOR_PHASES) and np.isfinite(nav_base)
                    else 0.0
                )
            else:
                w_rep = float(len(gp_rep) / (len(gp_rep) + cfg.shrink_n)) if len(gp_rep) else 0.0
                w_rep_r = float(len(gp_rep_pos) / (len(gp_rep_pos) + cfg.shrink_n)) if len(gp_rep_pos) else 0.0
                w_draw = float(len(gp) / (len(gp) + cfg.shrink_n))
                w_draw_r = float(len(gp_draw_pos) / (len(gp_draw_pos) + cfg.shrink_n)) if len(gp_draw_pos) else 0.0
                w_om = float(gp["omega"].notna().sum() / (gp["omega"].notna().sum() + cfg.shrink_n))

                rep_p_mult = float(
                    np.clip(
                        1.0 + w_rep * (raw_rep_p - 1.0),
                        cfg.lifecycle_p_mult_clip[0],
                        cfg.lifecycle_p_mult_clip[1],
                    )
                )
                rep_ratio_mult = float(
                    np.clip(
                        1.0 + w_rep_r * (raw_rep_r - 1.0),
                        cfg.lifecycle_ratio_mult_clip[0],
                        cfg.lifecycle_ratio_mult_clip[1],
                    )
                )
                draw_p_mult = float(
                    np.clip(
                        1.0 + w_draw * (raw_draw_p - 1.0),
                        cfg.lifecycle_draw_p_mult_clip[0],
                        cfg.lifecycle_draw_p_mult_clip[1],
                    )
                )
                draw_ratio_mult = float(
                    np.clip(
                        1.0 + w_draw_r * (raw_draw_r - 1.0),
                        cfg.lifecycle_draw_ratio_mult_clip[0],
                        cfg.lifecycle_draw_ratio_mult_clip[1],
                    )
                )
                omega_bump = float(np.clip(w_om * raw_om, cfg.lifecycle_omega_bump_clip[0], cfg.lifecycle_omega_bump_clip[1]))
                if phase in LIFECYCLE_NAV_FLOOR_PHASES and (np.isfinite(raw_nav_floor) or np.isfinite(nav_base)):
                    nav_ref = raw_nav_floor if np.isfinite(raw_nav_floor) else nav_base
                    nav_anchor = nav_base if np.isfinite(nav_base) else nav_ref
                    w_nav = float(gp_nav.size / (gp_nav.size + cfg.shrink_n)) if gp_nav.size else 0.0
                    nav_floor_ratio = float(
                        np.clip(
                            nav_anchor + w_nav * (nav_ref - nav_anchor),
                            cfg.lifecycle_nav_floor_clip[0],
                            cfg.lifecycle_nav_floor_clip[1],
                        )
                    )
                else:
                    nav_floor_ratio = 0.0

            rows.append(
                {
                    "Adj Strategy": strat,
                    "Grade": grade,
                    "life_phase": phase,
                    "n_obs": int(len(gp)),
                    "n_rep_gate": int(len(gp_rep)),
                    "n_rep_pos": int(len(gp_rep_pos)),
                    "n_draw_pos": int(len(gp_draw_pos)),
                    "rep_p_mult": rep_p_mult,
                    "rep_ratio_mult": rep_ratio_mult,
                    "draw_p_mult": draw_p_mult,
                    "draw_ratio_mult": draw_ratio_mult,
                    "omega_bump": omega_bump,
                    "nav_floor_ratio": nav_floor_ratio,
                    "p_value_rep_prob": float(pval_rep_p) if np.isfinite(pval_rep_p) else np.nan,
                    "p_value_rep_ratio": float(pval_rep_r) if np.isfinite(pval_rep_r) else np.nan,
                    "p_value_draw_prob": float(pval_draw_p) if np.isfinite(pval_draw_p) else np.nan,
                    "p_value_draw_ratio": float(pval_draw_r) if np.isfinite(pval_draw_r) else np.nan,
                    "p_value_omega": float(pval_om) if np.isfinite(pval_om) else np.nan,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Add strategy-level and global fallback rows.
    s_rows: list[dict[str, Any]] = []
    for (strat, phase), g in out.groupby(["Adj Strategy", "life_phase"], dropna=False):
        w = np.maximum(g["n_obs"].to_numpy(dtype=float), 1.0)
        s_rows.append(
            {
                "Adj Strategy": str(strat),
                "Grade": "ALL",
                "life_phase": str(phase),
                "n_obs": int(np.sum(w)),
                "n_rep_gate": int(np.sum(g["n_rep_gate"])),
                "n_rep_pos": int(np.sum(g["n_rep_pos"])),
                "n_draw_pos": int(np.sum(g["n_draw_pos"])),
                "rep_p_mult": float(np.average(g["rep_p_mult"].to_numpy(dtype=float), weights=w)),
                "rep_ratio_mult": float(np.average(g["rep_ratio_mult"].to_numpy(dtype=float), weights=w)),
                "draw_p_mult": float(np.average(g["draw_p_mult"].to_numpy(dtype=float), weights=w)),
                "draw_ratio_mult": float(np.average(g["draw_ratio_mult"].to_numpy(dtype=float), weights=w)),
                "omega_bump": float(np.average(g["omega_bump"].to_numpy(dtype=float), weights=w)),
                "nav_floor_ratio": float(np.average(g["nav_floor_ratio"].to_numpy(dtype=float), weights=w)),
                "p_value_rep_prob": np.nan,
                "p_value_rep_ratio": np.nan,
                "p_value_draw_prob": np.nan,
                "p_value_draw_ratio": np.nan,
                "p_value_omega": np.nan,
            }
        )

    g_rows: list[dict[str, Any]] = []
    s_df = pd.DataFrame(s_rows)
    for phase, g in s_df.groupby("life_phase", dropna=False):
        w = np.maximum(g["n_obs"].to_numpy(dtype=float), 1.0)
        g_rows.append(
            {
                "Adj Strategy": "ALL",
                "Grade": "ALL",
                "life_phase": str(phase),
                "n_obs": int(np.sum(w)),
                "n_rep_gate": int(np.sum(g["n_rep_gate"])),
                "n_rep_pos": int(np.sum(g["n_rep_pos"])),
                "n_draw_pos": int(np.sum(g["n_draw_pos"])),
                "rep_p_mult": float(np.average(g["rep_p_mult"].to_numpy(dtype=float), weights=w)),
                "rep_ratio_mult": float(np.average(g["rep_ratio_mult"].to_numpy(dtype=float), weights=w)),
                "draw_p_mult": float(np.average(g["draw_p_mult"].to_numpy(dtype=float), weights=w)),
                "draw_ratio_mult": float(np.average(g["draw_ratio_mult"].to_numpy(dtype=float), weights=w)),
                "omega_bump": float(np.average(g["omega_bump"].to_numpy(dtype=float), weights=w)),
                "nav_floor_ratio": float(np.average(g["nav_floor_ratio"].to_numpy(dtype=float), weights=w)),
                "p_value_rep_prob": np.nan,
                "p_value_rep_ratio": np.nan,
                "p_value_draw_prob": np.nan,
                "p_value_draw_ratio": np.nan,
                "p_value_omega": np.nan,
            }
        )

    all_rows = pd.concat([out, s_df, pd.DataFrame(g_rows)], ignore_index=True)

    # Ensure a full phase grid for each strategy-grade combo with hierarchical fallbacks.
    strategies = sorted(all_rows["Adj Strategy"].astype(str).unique().tolist())
    grades = sorted([g for g in all_rows["Grade"].astype(str).unique().tolist() if g != "ALL"])
    combos = (
        all_rows[all_rows["Grade"] != "ALL"][["Adj Strategy", "Grade"]]
        .drop_duplicates()
        .sort_values(["Adj Strategy", "Grade"])
        .to_records(index=False)
        .tolist()
    )
    combos = [(str(s), str(g)) for s, g in combos]
    full_rows: list[dict[str, Any]] = []
    for strat, grade in combos + [("ALL", "ALL")]:
        for phase in LIFECYCLE_PHASES:
            sub = all_rows[
                (all_rows["Adj Strategy"].astype(str) == strat)
                & (all_rows["Grade"].astype(str) == grade)
                & (all_rows["life_phase"].astype(str) == phase)
            ]
            if len(sub):
                full_rows.append(sub.iloc[0].to_dict())
                continue
            s_sub = all_rows[
                (all_rows["Adj Strategy"].astype(str) == strat)
                & (all_rows["Grade"].astype(str) == "ALL")
                & (all_rows["life_phase"].astype(str) == phase)
            ]
            g_sub = all_rows[
                (all_rows["Adj Strategy"].astype(str) == "ALL")
                & (all_rows["Grade"].astype(str) == "ALL")
                & (all_rows["life_phase"].astype(str) == phase)
            ]
            if len(s_sub):
                row = s_sub.iloc[0].to_dict()
                row["Adj Strategy"] = strat
                row["Grade"] = grade
                full_rows.append(row)
            elif len(g_sub):
                row = g_sub.iloc[0].to_dict()
                row["Adj Strategy"] = strat
                row["Grade"] = grade
                full_rows.append(row)
            else:
                full_rows.append(
                    {
                        "Adj Strategy": strat,
                        "Grade": grade,
                        "life_phase": phase,
                        "n_obs": 0,
                        "n_rep_gate": 0,
                        "n_rep_pos": 0,
                        "n_draw_pos": 0,
                        "rep_p_mult": 1.0,
                        "rep_ratio_mult": 1.0,
                        "draw_p_mult": 1.0,
                        "draw_ratio_mult": 1.0,
                        "omega_bump": 0.0,
                        "nav_floor_ratio": 0.0,
                        "p_value_rep_prob": np.nan,
                        "p_value_rep_ratio": np.nan,
                        "p_value_draw_prob": np.nan,
                        "p_value_draw_ratio": np.nan,
                        "p_value_omega": np.nan,
                    }
                )

    return pd.DataFrame(full_rows).sort_values(["Adj Strategy", "Grade", "life_phase"]).reset_index(drop=True)


def _fit_stability_slope(x: np.ndarray, y: np.ndarray, w: np.ndarray, clip: tuple[float, float] = (0.0, 1.0)) -> tuple[float, int]:
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if m.sum() < 2:
        return 1.0, int(m.sum())
    xv = x[m]
    yv = y[m]
    wv = w[m]
    den = float(np.sum(wv * xv * xv))
    if den <= 1e-12:
        return 1.0, int(m.sum())
    k = float(np.sum(wv * xv * yv) / den)
    return float(np.clip(k, clip[0], clip[1])), int(m.sum())


def _compute_stability_attenuation(
    prepared: _PreparedHistory,
    cfg: CalibrationConfig,
    fit_cfg: FitConfig,
) -> pd.DataFrame:
    h = prepared.data
    if h.empty:
        return pd.DataFrame()
    ords = np.array(sorted(pd.to_numeric(h["q_ord"], errors="coerce").dropna().astype(int).unique()), dtype=int)
    if ords.size == 0:
        return pd.DataFrame()

    hold = int(cfg.holdout_quarters)
    start = int(ords.min() + 2 * hold)
    end = int(ords.max() - hold)
    if start > end:
        return pd.DataFrame()

    xs: dict[str, list[float]] = {
        "p_draw_mult": [],
        "p_rep_mult": [],
        "rep_ratio_scale": [],
        "draw_ratio_scale": [],
        "omega_bump": [],
        "rep_timing_shift_q": [],
    }
    ys: dict[str, list[float]] = {
        "p_draw_mult": [],
        "p_rep_mult": [],
        "rep_ratio_scale": [],
        "draw_ratio_scale": [],
        "omega_bump": [],
        "rep_timing_shift_q": [],
    }
    ws: dict[str, list[float]] = {
        "p_draw_mult": [],
        "p_rep_mult": [],
        "rep_ratio_scale": [],
        "draw_ratio_scale": [],
        "omega_bump": [],
        "rep_timing_shift_q": [],
    }

    for e in range(start, end + 1, max(int(cfg.stability_step_q), 1)):
        tr = h[h["q_ord"] <= (e - hold)].copy()
        ho = h[(h["q_ord"] > (e - hold)) & (h["q_ord"] <= e)].copy()
        fw = h[(h["q_ord"] > e) & (h["q_ord"] <= (e + hold))].copy()
        if tr.empty or ho.empty or fw.empty:
            continue

        d_ho = _compute_deltas(tr, ho, cfg, fit_cfg, apply_significance_gate=False)
        d_fw = _compute_deltas(tr, fw, cfg, fit_cfg, apply_significance_gate=False)

        x_pd = float(d_ho["raw"].get("p_draw_mult", 1.0)) - 1.0
        y_pd = float(d_fw["raw"].get("p_draw_mult", 1.0)) - 1.0
        w_pd = float(min(d_ho["counts"].get("n_holdout", 0), d_fw["counts"].get("n_holdout", 0)))
        xs["p_draw_mult"].append(x_pd)
        ys["p_draw_mult"].append(y_pd)
        ws["p_draw_mult"].append(w_pd)

        x_p = float(d_ho["raw"].get("p_rep_mult", 1.0)) - 1.0
        y_p = float(d_fw["raw"].get("p_rep_mult", 1.0)) - 1.0
        w_p = float(min(d_ho["counts"].get("n_holdout_rep_gate", 0), d_fw["counts"].get("n_holdout_rep_gate", 0)))
        xs["p_rep_mult"].append(x_p)
        ys["p_rep_mult"].append(y_p)
        ws["p_rep_mult"].append(w_p)

        x_rr = float(d_ho["raw"].get("rep_ratio_scale", 1.0)) - 1.0
        y_rr = float(d_fw["raw"].get("rep_ratio_scale", 1.0)) - 1.0
        w_rr = float(min(d_ho["counts"].get("n_holdout_rep_pos", 0), d_fw["counts"].get("n_holdout_rep_pos", 0)))
        xs["rep_ratio_scale"].append(x_rr)
        ys["rep_ratio_scale"].append(y_rr)
        ws["rep_ratio_scale"].append(w_rr)

        x_dr = float(d_ho["raw"].get("draw_ratio_scale", 1.0)) - 1.0
        y_dr = float(d_fw["raw"].get("draw_ratio_scale", 1.0)) - 1.0
        w_dr = float(min(d_ho["counts"].get("n_holdout_draw_pos", 0), d_fw["counts"].get("n_holdout_draw_pos", 0)))
        xs["draw_ratio_scale"].append(x_dr)
        ys["draw_ratio_scale"].append(y_dr)
        ws["draw_ratio_scale"].append(w_dr)

        x_om = float(d_ho["raw"].get("omega_bump", 0.0))
        y_om = float(d_fw["raw"].get("omega_bump", 0.0))
        w_om = float(min(d_ho["counts"].get("n_holdout_omega", 0), d_fw["counts"].get("n_holdout_omega", 0)))
        xs["omega_bump"].append(x_om)
        ys["omega_bump"].append(y_om)
        ws["omega_bump"].append(w_om)

        x_ts = float(d_ho["raw"].get("timing_shift_q", 0.0))
        y_ts = float(d_fw["raw"].get("timing_shift_q", 0.0))
        w_ts = float(min(d_ho["counts"].get("n_holdout_rep_gate", 0), d_fw["counts"].get("n_holdout_rep_gate", 0)))
        xs["rep_timing_shift_q"].append(x_ts)
        ys["rep_timing_shift_q"].append(y_ts)
        ws["rep_timing_shift_q"].append(w_ts)

    rows: list[dict[str, Any]] = []
    for metric in ["p_draw_mult", "p_rep_mult", "rep_ratio_scale", "draw_ratio_scale", "omega_bump", "rep_timing_shift_q"]:
        x = np.asarray(xs[metric], dtype=float)
        y = np.asarray(ys[metric], dtype=float)
        w = np.asarray(ws[metric], dtype=float)
        slope, n_pts = _fit_stability_slope(x, y, w, clip=(0.0, 1.0))
        rows.append(
            {
                "metric": metric,
                "attenuation": float(slope),
                "n_points": int(n_pts),
                "x_mean": float(np.nanmean(x)) if x.size else np.nan,
                "y_mean": float(np.nanmean(y)) if y.size else np.nan,
                "corr": float(np.corrcoef(x, y)[0, 1]) if x.size >= 3 and np.nanstd(x) > 0 and np.nanstd(y) > 0 else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["attenuation"] = np.where(out["n_points"] >= int(cfg.stability_min_points), out["attenuation"], 1.0)
    return out


def _apply_stability_attenuation(global_delta: dict[str, Any], stability: pd.DataFrame) -> dict[str, Any]:
    if stability is None or stability.empty:
        return global_delta
    att = dict(zip(stability["metric"], stability["attenuation"]))
    out = dict(global_delta)

    k_pd = float(att.get("p_draw_mult", 1.0))
    k_p = float(att.get("p_rep_mult", 1.0))
    k_rr = float(att.get("rep_ratio_scale", 1.0))
    k_dr = float(att.get("draw_ratio_scale", 1.0))
    k_om = float(att.get("omega_bump", 1.0))
    k_ts = float(att.get("rep_timing_shift_q", 1.0))

    out["delta_p_draw_mult"] = 1.0 + k_pd * (float(global_delta.get("delta_p_draw_mult", 1.0)) - 1.0)
    out["delta_p_rep_mult"] = 1.0 + k_p * (float(global_delta.get("delta_p_rep_mult", 1.0)) - 1.0)
    out["delta_rep_ratio_scale"] = 1.0 + k_rr * (float(global_delta.get("delta_rep_ratio_scale", 1.0)) - 1.0)
    out["delta_draw_ratio_scale"] = 1.0 + k_dr * (float(global_delta.get("delta_draw_ratio_scale", 1.0)) - 1.0)
    out["delta_omega_bump"] = k_om * float(global_delta.get("delta_omega_bump", 0.0))
    out["delta_rep_timing_shift_q"] = int(round(k_ts * int(global_delta.get("delta_rep_timing_shift_q", 0))))
    return out


def _compute_post_invest_draw_multipliers(
    hist_to_cutoff: pd.DataFrame,
    fit_cfg: FitConfig,
    cfg: CalibrationConfig,
) -> pd.DataFrame:
    h = hist_to_cutoff.sort_values(["FundID", "quarter_end"]).copy()
    if h.empty:
        return pd.DataFrame()

    invest_end = compute_invest_end_by_fund(h)
    end_info = compute_fund_end_dates(h)
    commit_max = pd.to_numeric(h["Commitment EUR"], errors="coerce").groupby(h["FundID"]).transform("max").replace(0, np.nan)
    draw_abs = pd.to_numeric(h["Adj Drawdown EUR"], errors="coerce").abs().fillna(0.0)

    h["invest_end"] = h["FundID"].map(invest_end)
    h["fund_end_qe"] = h["FundID"].map(end_info.fund_end_qe)
    h["draw_event"] = (draw_abs > 0.0).astype(int)
    h["draw_ratio_fund"] = np.where(commit_max > 0, draw_abs / commit_max, np.nan)

    h["in_invest"] = h["quarter_end"] <= h["invest_end"]
    h["after_invest_pre_end"] = (h["quarter_end"] > h["invest_end"]) & (
        h["fund_end_qe"].isna() | (h["quarter_end"] <= h["fund_end_qe"])
    )

    rows: list[dict[str, Any]] = []
    for strat, g in h.groupby("Adj Strategy"):
        pre = g[g["in_invest"]].copy()
        post = g[g["after_invest_pre_end"]].copy()
        if len(pre) < 40 or len(post) < 40:
            rows.append(
                {
                    "Adj Strategy": str(strat),
                    "n_pre": int(len(pre)),
                    "n_post": int(len(post)),
                    "draw_p_mult": 1.0,
                    "draw_ratio_mult": 1.0,
                    "p_value_draw_prob": np.nan,
                    "p_value_draw_ratio": np.nan,
                }
            )
            continue

        p_pre = float(pre["draw_event"].mean())
        p_post = float(post["draw_event"].mean())
        raw_p = _safe_ratio(p_post, p_pre, default=1.0)
        pval_p = _two_prop_pvalue(int(pre["draw_event"].sum()), int(len(pre)), int(post["draw_event"].sum()), int(len(post)))

        r_pre = pre[(pre["draw_event"] == 1) & np.isfinite(pre["draw_ratio_fund"])]["draw_ratio_fund"]
        r_post = post[(post["draw_event"] == 1) & np.isfinite(post["draw_ratio_fund"])]["draw_ratio_fund"]
        mean_pre = _trimmed_mean(r_pre, lo=0.01, hi=0.99)
        mean_post = _trimmed_mean(r_post, lo=0.01, hi=0.99)
        raw_r = _safe_ratio(mean_post, mean_pre, default=1.0)
        pval_r = _mean_diff_pvalue(r_pre, r_post, log_transform=True)

        if not np.isfinite(pval_p) or pval_p >= cfg.significance_alpha:
            raw_p = 1.0
        if not np.isfinite(pval_r) or pval_r >= cfg.significance_alpha:
            raw_r = 1.0

        # Post-invest drawdowns should not exceed invest-period intensity in this business setup.
        raw_p = float(np.clip(raw_p, cfg.post_invest_draw_prob_clip[0], cfg.post_invest_draw_prob_clip[1]))
        raw_r = float(np.clip(raw_r, cfg.post_invest_draw_ratio_clip[0], cfg.post_invest_draw_ratio_clip[1]))

        shrink = float(len(post) / (len(post) + cfg.shrink_n))
        p_mult = float(np.clip(1.0 + shrink * (raw_p - 1.0), cfg.post_invest_draw_prob_clip[0], cfg.post_invest_draw_prob_clip[1]))
        r_mult = float(np.clip(1.0 + shrink * (raw_r - 1.0), cfg.post_invest_draw_ratio_clip[0], cfg.post_invest_draw_ratio_clip[1]))

        rows.append(
            {
                "Adj Strategy": str(strat),
                "n_pre": int(len(pre)),
                "n_post": int(len(post)),
                "draw_p_mult": p_mult,
                "draw_ratio_mult": r_mult,
                "p_value_draw_prob": float(pval_p),
                "p_value_draw_ratio": float(pval_r),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    w = np.maximum(out["n_post"].to_numpy(dtype=float), 1.0)
    all_row = pd.DataFrame(
        [
            {
                "Adj Strategy": "ALL",
                "n_pre": int(out["n_pre"].sum()),
                "n_post": int(out["n_post"].sum()),
                "draw_p_mult": float(np.average(out["draw_p_mult"].to_numpy(dtype=float), weights=w)),
                "draw_ratio_mult": float(np.average(out["draw_ratio_mult"].to_numpy(dtype=float), weights=w)),
                "p_value_draw_prob": np.nan,
                "p_value_draw_ratio": np.nan,
            }
        ]
    )
    out = pd.concat([out, all_row], ignore_index=True)
    return out.sort_values("Adj Strategy").reset_index(drop=True)


def _compute_pre_end_repayment_multipliers(
    hist_to_cutoff: pd.DataFrame,
    fit_cfg: FitConfig,
    cfg: CalibrationConfig,
) -> pd.DataFrame:
    h = hist_to_cutoff.sort_values(["FundID", "quarter_end"]).copy()
    if h.empty:
        return pd.DataFrame()

    end_info = compute_fund_end_dates(h)
    nav_prev = pd.to_numeric(h["NAV Adjusted EUR"], errors="coerce").clip(lower=0.0).groupby(h["FundID"]).shift(1)
    rep_abs = pd.to_numeric(h["Adj Repayment EUR"], errors="coerce").abs().fillna(0.0)

    h["nav_prev"] = nav_prev
    h["rep_event"] = ((rep_abs > 0.0) & (nav_prev > fit_cfg.nav_gate)).astype(int)
    h["rep_ratio_cond"] = np.where(nav_prev > fit_cfg.nav_gate, rep_abs / nav_prev, np.nan)

    h["q_ord"] = h["quarter_end"].dt.to_period("Q").map(lambda p: p.ordinal if pd.notna(p) else np.nan)
    h["end_ord"] = h["FundID"].map(end_info.fund_end_qe).dt.to_period("Q").map(lambda p: p.ordinal if pd.notna(p) else np.nan)
    h["q_to_end"] = h["end_ord"] - h["q_ord"]
    pre = h[(h["q_to_end"] >= 1) & (h["nav_prev"] > fit_cfg.nav_gate)].copy()
    if pre.empty:
        return pd.DataFrame()

    def _phase(q_to_end: float) -> str:
        if not np.isfinite(q_to_end):
            return "unknown"
        q = int(q_to_end)
        if q > 24:
            return "far"
        if q >= 13:
            return "mid"
        return "near"

    pre["pre_end_phase"] = pre["q_to_end"].apply(_phase)

    rows: list[dict[str, Any]] = []
    for strat, g in pre.groupby("Adj Strategy"):
        base_p = float(g["rep_event"].mean())
        base_r = _trimmed_mean(g[g["rep_event"] == 1]["rep_ratio_cond"], lo=0.01, hi=0.99)
        if not np.isfinite(base_p) or base_p <= 0:
            base_p = 1e-9
        if not np.isfinite(base_r) or base_r <= 0:
            base_r = 1e-9

        for phase in ["far", "mid", "near"]:
            gp = g[g["pre_end_phase"] == phase].copy()
            if len(gp) < int(cfg.pre_end_repay_min_obs):
                rows.append(
                    {
                        "Adj Strategy": str(strat),
                        "pre_end_phase": phase,
                        "n_obs": int(len(gp)),
                        "rep_p_mult": 1.0,
                        "rep_ratio_mult": 1.0,
                        "p_value_rep_prob": np.nan,
                        "p_value_rep_ratio": np.nan,
                    }
                )
                continue

            if phase == "near":
                # Near-end behavior is handled by endgame ramp; keep multiplier neutral to avoid double counting.
                rows.append(
                    {
                        "Adj Strategy": str(strat),
                        "pre_end_phase": phase,
                        "n_obs": int(len(gp)),
                        "rep_p_mult": 1.0,
                        "rep_ratio_mult": 1.0,
                        "p_value_rep_prob": np.nan,
                        "p_value_rep_ratio": np.nan,
                    }
                )
                continue

            p_phase = float(gp["rep_event"].mean())
            raw_p = _safe_ratio(p_phase, base_p, default=1.0)
            pval_p = _two_prop_pvalue(int(g["rep_event"].sum()), int(len(g)), int(gp["rep_event"].sum()), int(len(gp)))

            r_phase = _trimmed_mean(gp[gp["rep_event"] == 1]["rep_ratio_cond"], lo=0.01, hi=0.99)
            raw_r = _safe_ratio(r_phase, base_r, default=1.0)
            pval_r = _mean_diff_pvalue(g[g["rep_event"] == 1]["rep_ratio_cond"], gp[gp["rep_event"] == 1]["rep_ratio_cond"], log_transform=True)

            if not np.isfinite(pval_p) or pval_p >= cfg.significance_alpha:
                raw_p = 1.0
            if not np.isfinite(pval_r) or pval_r >= cfg.significance_alpha:
                raw_r = 1.0

            shrink = float(len(gp) / (len(gp) + cfg.shrink_n))
            # Pre-end layer is attenuation-only; dedicated near-end ramp handles acceleration.
            p_mult = float(
                np.clip(
                    1.0 + shrink * (raw_p - 1.0),
                    cfg.pre_end_repay_prob_clip[0],
                    cfg.pre_end_repay_prob_clip[1],
                )
            )
            r_mult = float(
                np.clip(
                    1.0 + shrink * (raw_r - 1.0),
                    cfg.pre_end_repay_ratio_clip[0],
                    cfg.pre_end_repay_ratio_clip[1],
                )
            )

            rows.append(
                {
                    "Adj Strategy": str(strat),
                    "pre_end_phase": phase,
                    "n_obs": int(len(gp)),
                    "rep_p_mult": p_mult,
                    "rep_ratio_mult": r_mult,
                    "p_value_rep_prob": float(pval_p),
                    "p_value_rep_ratio": float(pval_r),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    all_rows: list[dict[str, Any]] = []
    for phase, gp in out.groupby("pre_end_phase"):
        w = np.maximum(gp["n_obs"].to_numpy(dtype=float), 1.0)
        all_rows.append(
            {
                "Adj Strategy": "ALL",
                "pre_end_phase": phase,
                "n_obs": int(np.sum(w)),
                "rep_p_mult": float(np.average(gp["rep_p_mult"].to_numpy(dtype=float), weights=w)),
                "rep_ratio_mult": float(np.average(gp["rep_ratio_mult"].to_numpy(dtype=float), weights=w)),
                "p_value_rep_prob": np.nan,
                "p_value_rep_ratio": np.nan,
            }
        )
    out = pd.concat([out, pd.DataFrame(all_rows)], ignore_index=True)

    # Ensure complete fallback grid.
    strategies = sorted([s for s in out["Adj Strategy"].unique() if s != "ALL"]) + ["ALL"]
    full_rows = []
    for s in strategies:
        for phase in ["far", "mid", "near"]:
            sub = out[(out["Adj Strategy"] == s) & (out["pre_end_phase"] == phase)]
            if len(sub):
                full_rows.append(sub.iloc[0].to_dict())
            else:
                glob = out[(out["Adj Strategy"] == "ALL") & (out["pre_end_phase"] == phase)]
                if len(glob):
                    row = glob.iloc[0].to_dict()
                    row["Adj Strategy"] = s
                    full_rows.append(row)
                else:
                    full_rows.append(
                        {
                            "Adj Strategy": s,
                            "pre_end_phase": phase,
                            "n_obs": 0,
                            "rep_p_mult": 1.0,
                            "rep_ratio_mult": 1.0,
                            "p_value_rep_prob": np.nan,
                            "p_value_rep_ratio": np.nan,
                        }
                    )
    return pd.DataFrame(full_rows).sort_values(["Adj Strategy", "pre_end_phase"]).reset_index(drop=True)


def calibrate_from_history(
    hist_to_cutoff: pd.DataFrame,
    *,
    cutoff_quarter: str,
    fit_cfg: FitConfig,
    cal_cfg: CalibrationConfig,
) -> CalibrationArtifacts:
    cutoff_qe = parse_quarter_text(cutoff_quarter)
    prepared = _prepare_history(hist_to_cutoff, cutoff_qe=cutoff_qe, fit_cfg=fit_cfg)
    tr, ho = _split_train_holdout(prepared, cal_cfg.holdout_quarters)

    global_delta = _compute_deltas(tr, ho, cal_cfg, fit_cfg)
    stability = _compute_stability_attenuation(prepared, cal_cfg, fit_cfg)
    if cal_cfg.apply_stability_attenuation:
        global_delta = _apply_stability_attenuation(global_delta, stability)

    _, rc_prop_global = _recall_propensity_by_group(prepared, ("Adj Strategy", "Grade"))
    global_delta.setdefault("delta_p_rc_mult", 1.0)
    global_delta.setdefault("delta_rc_ratio_scale", 1.0)
    global_delta["rc_propensity"] = float(rc_prop_global)

    group_delta = _compute_group_deltas(prepared, cal_cfg, fit_cfg, global_delta, group_cols=("Adj Strategy", "Grade"))
    tail_floors = _compute_tail_floors(hist_to_cutoff, fit_cfg)
    endgame_ramp = _compute_endgame_ramp(hist_to_cutoff, fit_cfg, cal_cfg)
    lifecycle_phase = _compute_lifecycle_multipliers(prepared, fit_cfg, cal_cfg)
    post_invest_draw = _compute_post_invest_draw_multipliers(hist_to_cutoff, fit_cfg, cal_cfg)
    pre_end_repayment = _compute_pre_end_repayment_multipliers(hist_to_cutoff, fit_cfg, cal_cfg)

    summary_rows = [
        {
            "metric": "p_draw_mult",
            "delta": float(global_delta.get("delta_p_draw_mult", 1.0)),
            "raw": float(global_delta.get("raw", {}).get("p_draw_mult", 1.0))
            if np.isfinite(global_delta.get("raw", {}).get("p_draw_mult", np.nan))
            else np.nan,
            "p_value": float(global_delta.get("pvalues", {}).get("p_draw", np.nan)),
            "n_holdout": int(global_delta.get("counts", {}).get("n_holdout", 0)),
        },
        {
            "metric": "p_rep_mult",
            "delta": float(global_delta.get("delta_p_rep_mult", 1.0)),
            "raw": float(global_delta.get("raw", {}).get("p_rep_mult", 1.0))
            if np.isfinite(global_delta.get("raw", {}).get("p_rep_mult", np.nan))
            else np.nan,
            "p_value": float(global_delta.get("pvalues", {}).get("p_rep", np.nan)),
            "n_holdout": int(global_delta.get("counts", {}).get("n_holdout_rep_gate", 0)),
        },
        {
            "metric": "rep_ratio_scale",
            "delta": float(global_delta.get("delta_rep_ratio_scale", 1.0)),
            "raw": float(global_delta.get("raw", {}).get("rep_ratio_scale", 1.0))
            if np.isfinite(global_delta.get("raw", {}).get("rep_ratio_scale", np.nan))
            else np.nan,
            "p_value": float(global_delta.get("pvalues", {}).get("rep_ratio", np.nan)),
            "n_holdout": int(global_delta.get("counts", {}).get("n_holdout_rep_pos", 0)),
        },
        {
            "metric": "draw_ratio_scale",
            "delta": float(global_delta.get("delta_draw_ratio_scale", 1.0)),
            "raw": float(global_delta.get("raw", {}).get("draw_ratio_scale", 1.0))
            if np.isfinite(global_delta.get("raw", {}).get("draw_ratio_scale", np.nan))
            else np.nan,
            "p_value": float(global_delta.get("pvalues", {}).get("draw_ratio", np.nan)),
            "n_holdout": int(global_delta.get("counts", {}).get("n_holdout_draw_pos", 0)),
        },
        {
            "metric": "omega_bump",
            "delta": float(global_delta.get("delta_omega_bump", 0.0)),
            "raw": float(global_delta.get("raw", {}).get("omega_bump", 0.0))
            if np.isfinite(global_delta.get("raw", {}).get("omega_bump", np.nan))
            else np.nan,
            "p_value": float(global_delta.get("pvalues", {}).get("omega", np.nan)),
            "n_holdout": int(global_delta.get("counts", {}).get("n_holdout_omega", 0)),
        },
        {
            "metric": "rep_timing_shift_q",
            "delta": int(global_delta.get("delta_rep_timing_shift_q", 0)),
            "raw": int(global_delta.get("raw", {}).get("timing_shift_q", 0)),
            "p_value": np.nan,
            "n_holdout": int(global_delta.get("counts", {}).get("n_holdout_rep_gate", 0)),
        },
    ]

    return CalibrationArtifacts(
        global_deltas=global_delta,
        group_deltas=group_delta,
        tail_floors=tail_floors,
        endgame_ramp=endgame_ramp,
        lifecycle_phase=lifecycle_phase,
        post_invest_draw=post_invest_draw,
        pre_end_repayment=pre_end_repayment,
        stability_attenuation=stability,
        summary=pd.DataFrame(summary_rows),
    )


def load_calibration_artifacts(calibration_dir: Path) -> CalibrationArtifacts:
    with (calibration_dir / "holdout_recalibration.json").open("r", encoding="utf-8") as f:
        global_deltas = json.load(f)
    group_path = calibration_dir / "holdout_group_recalibration.csv"
    tail_path = calibration_dir / "tail_repayment_floors.csv"
    endgame_path = calibration_dir / "endgame_repayment_ramp.csv"
    lifecycle_path = calibration_dir / "lifecycle_phase_multipliers.csv"
    post_invest_draw_path = calibration_dir / "post_invest_draw_multipliers.csv"
    pre_end_repay_path = calibration_dir / "pre_end_repayment_multipliers.csv"
    stability_path = calibration_dir / "stability_attenuation.csv"
    summary_path = calibration_dir / "holdout_recalibration_summary.csv"

    group = pd.read_csv(group_path) if group_path.exists() else pd.DataFrame()
    tail = pd.read_csv(tail_path) if tail_path.exists() else pd.DataFrame()
    endgame = pd.read_csv(endgame_path) if endgame_path.exists() else pd.DataFrame()
    lifecycle = pd.read_csv(lifecycle_path) if lifecycle_path.exists() else pd.DataFrame()
    post_invest_draw = pd.read_csv(post_invest_draw_path) if post_invest_draw_path.exists() else pd.DataFrame()
    pre_end_repay = pd.read_csv(pre_end_repay_path) if pre_end_repay_path.exists() else pd.DataFrame()
    stability = pd.read_csv(stability_path) if stability_path.exists() else pd.DataFrame()
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    return CalibrationArtifacts(
        global_deltas=global_deltas,
        group_deltas=group,
        tail_floors=tail,
        endgame_ramp=endgame,
        lifecycle_phase=lifecycle,
        post_invest_draw=post_invest_draw,
        pre_end_repayment=pre_end_repay,
        stability_attenuation=stability,
        summary=summary,
    )


__all__ = ["CalibrationArtifacts", "calibrate_from_history", "load_calibration_artifacts"]
