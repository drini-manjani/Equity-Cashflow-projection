from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .calibration import CalibrationArtifacts
from .config import FitConfig, SimConfig
from .features import FundState
from .fitting import FitArtifacts
from .utils import ensure_dir, make_age_bucket


TAIL_LABELS = ["0-3", "4-7", "8-11", "12+"]


@dataclass
class SimulationOutputs:
    portfolio_series: pd.DataFrame
    fund_quarterly_mean: pd.DataFrame
    fund_end_summary: pd.DataFrame
    final_distribution_summary: pd.DataFrame | None = None

    def save(self, projection_dir: Path) -> None:
        out = projection_dir / "sim_outputs"
        ensure_dir(out)
        self.portfolio_series.to_csv(out / "sim_portfolio_series.csv", index=False)
        self.fund_quarterly_mean.to_csv(out / "sim_fund_quarterly_mean.csv", index=False)
        self.fund_end_summary.to_csv(out / "sim_fund_end_summary.csv", index=False)
        if self.final_distribution_summary is not None and not self.final_distribution_summary.empty:
            self.final_distribution_summary.to_csv(out / "sim_portfolio_final_distribution.csv", index=False)


class _Lookup:
    def __init__(self, fits: FitArtifacts, calibs: CalibrationArtifacts | None) -> None:
        self.timing_idx = self._build_timing(fits.timing_probs)
        self.ratio_idx = self._build_ratio(fits.ratio_fits)
        self.omega_idx = self._build_omega(fits.omega_fits)
        self.global_deltas = calibs.global_deltas if calibs is not None else {}
        self.group_deltas = self._build_group_deltas(calibs.group_deltas if calibs is not None else pd.DataFrame())
        self.tail_idx = self._build_tail(calibs.tail_floors if calibs is not None else pd.DataFrame())
        self.endgame_idx = self._build_endgame(calibs.endgame_ramp if calibs is not None else pd.DataFrame())
        self.lifecycle_idx = self._build_lifecycle(calibs.lifecycle_phase if calibs is not None else pd.DataFrame())
        self.post_invest_draw_idx = self._build_post_invest_draw(calibs.post_invest_draw if calibs is not None else pd.DataFrame())
        self.pre_end_repay_idx = self._build_pre_end_repay(calibs.pre_end_repayment if calibs is not None else pd.DataFrame())

    @staticmethod
    def _build_timing(df: pd.DataFrame) -> dict[tuple[str, str, str, str, str], dict[str, float]]:
        idx: dict[tuple[str, str, str, str, str], dict[str, float]] = {}
        if df is None or df.empty:
            return idx
        d = df.copy()
        for c in ["Adj Strategy", "Grade", "AgeBucket", "size_bucket", "level"]:
            if c not in d.columns:
                d[c] = "ALL"
            d[c] = d[c].astype(str)
        for _, r in d.iterrows():
            key = (r["level"], r["Adj Strategy"], r["Grade"], r["AgeBucket"], r["size_bucket"])
            idx[key] = {
                "p_draw": float(r.get("p_draw", 0.0)),
                "p_rep": float(r.get("p_rep", 0.0)),
                "p_rc": float(r.get("p_rc", 0.0)),
            }
        return idx

    @staticmethod
    def _build_ratio(df: pd.DataFrame) -> dict[tuple[str, str, str, str, str, str], dict[str, Any]]:
        idx: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
        if df is None or df.empty:
            return idx
        d = df.copy()
        for c in ["metric", "level", "Adj Strategy", "Grade", "AgeBucket", "size_bucket"]:
            if c not in d.columns:
                d[c] = "ALL"
            d[c] = d[c].astype(str)
        for _, r in d.iterrows():
            key = (r["metric"], r["level"], r["Adj Strategy"], r["Grade"], r["AgeBucket"], r["size_bucket"])
            params_raw = r.get("params", "[]")
            try:
                params = json.loads(params_raw) if isinstance(params_raw, str) else list(params_raw)
            except Exception:
                params = []
            idx[key] = {
                "dist": str(r.get("dist", "empirical")),
                "params": [float(x) for x in params] if len(params) else [],
                "mean": float(r.get("mean", 0.0)),
            }
        return idx

    @staticmethod
    def _build_omega(df: pd.DataFrame) -> dict[tuple[str, str, str, str, str], dict[str, float]]:
        idx: dict[tuple[str, str, str, str, str], dict[str, float]] = {}
        if df is None or df.empty:
            return idx
        d = df.copy()
        for c in ["level", "Adj Strategy", "Grade", "AgeBucket", "size_bucket"]:
            if c not in d.columns:
                d[c] = "ALL"
            d[c] = d[c].astype(str)
        for _, r in d.iterrows():
            key = (r["level"], r["Adj Strategy"], r["Grade"], r["AgeBucket"], r["size_bucket"])
            idx[key] = {
                "a_intercept": float(r.get("a_intercept", 0.0)),
                "b0": float(r.get("b0", 0.0)),
                "b1": float(r.get("b1", 0.0)),
                "b_age": float(r.get("b_age", 0.0)),
                "b_age2": float(r.get("b_age2", 0.0)),
                "b_q_to_end": float(r.get("b_q_to_end", 0.0)),
                "b_pre_end": float(r.get("b_pre_end", 0.0)),
                "b_post_end": float(r.get("b_post_end", 0.0)),
                "alpha": float(r.get("alpha", 0.0)),
                "sigma": float(max(float(r.get("sigma", 0.0)), 1e-6)),
            }
        return idx

    @staticmethod
    def _build_group_deltas(df: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
        out: dict[tuple[str, str], dict[str, Any]] = {}
        if df is None or df.empty:
            return out
        d = df.copy()
        if "Adj Strategy" not in d.columns or "Grade" not in d.columns:
            return out
        for _, r in d.iterrows():
            key = (str(r.get("Adj Strategy", "ALL")), str(r.get("Grade", "ALL")))
            out[key] = {
                "delta_p_draw_mult": float(r.get("delta_p_draw_mult", 1.0)),
                "delta_p_rep_mult": float(r.get("delta_p_rep_mult", 1.0)),
                "delta_p_rc_mult": float(r.get("delta_p_rc_mult", 1.0)),
                "delta_rep_ratio_scale": float(r.get("delta_rep_ratio_scale", 1.0)),
                "delta_draw_ratio_scale": float(r.get("delta_draw_ratio_scale", 1.0)),
                "delta_rc_ratio_scale": float(r.get("delta_rc_ratio_scale", 1.0)),
                "delta_omega_bump": float(r.get("delta_omega_bump", 0.0)),
                "delta_rep_timing_shift_q": int(r.get("delta_rep_timing_shift_q", 0)),
                "rc_propensity": float(r.get("rc_propensity", 1.0)),
            }
        return out

    @staticmethod
    def _build_tail(df: pd.DataFrame) -> dict[tuple[str, str], dict[str, float]]:
        out: dict[tuple[str, str], dict[str, float]] = {}
        if df is None or df.empty:
            return out
        d = df.copy()
        for _, r in d.iterrows():
            k = (str(r.get("Adj Strategy", "ALL")), str(r.get("tail_bucket", "12+")))
            out[k] = {
                "p_rep_floor": float(r.get("p_rep_floor", 0.0)),
                "rep_ratio_floor": float(r.get("rep_ratio_floor", 0.0)),
                "draw_prob_cap": float(r.get("draw_prob_cap", 1.0)),
                "omega_drag": float(r.get("omega_drag", 0.0)),
            }
        return out

    @staticmethod
    def _build_endgame(df: pd.DataFrame) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        if df is None or df.empty:
            return out
        d = df.copy()
        for _, r in d.iterrows():
            strat = str(r.get("Adj Strategy", "ALL"))
            out[strat] = {
                "p_rep_mult": float(r.get("p_rep_mult", 1.0)),
                "rep_ratio_mult": float(r.get("rep_ratio_mult", 1.0)),
            }
        return out

    @staticmethod
    def _build_post_invest_draw(df: pd.DataFrame) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        if df is None or df.empty:
            return out
        d = df.copy()
        for _, r in d.iterrows():
            strat = str(r.get("Adj Strategy", "ALL"))
            out[strat] = {
                "draw_p_mult": float(r.get("draw_p_mult", 1.0)),
                "draw_ratio_mult": float(r.get("draw_ratio_mult", 1.0)),
            }
        return out

    @staticmethod
    def _build_pre_end_repay(df: pd.DataFrame) -> dict[tuple[str, str], dict[str, float]]:
        out: dict[tuple[str, str], dict[str, float]] = {}
        if df is None or df.empty:
            return out
        d = df.copy()
        for _, r in d.iterrows():
            strat = str(r.get("Adj Strategy", "ALL"))
            phase = str(r.get("pre_end_phase", "far"))
            out[(strat, phase)] = {
                "rep_p_mult": float(r.get("rep_p_mult", 1.0)),
                "rep_ratio_mult": float(r.get("rep_ratio_mult", 1.0)),
            }
        return out

    @staticmethod
    def _build_lifecycle(df: pd.DataFrame) -> dict[tuple[str, str, str], dict[str, float]]:
        out: dict[tuple[str, str, str], dict[str, float]] = {}
        if df is None or df.empty:
            return out
        d = df.copy()
        for c in ["Adj Strategy", "Grade", "life_phase"]:
            if c not in d.columns:
                d[c] = "ALL"
            d[c] = d[c].astype(str)
        for _, r in d.iterrows():
            key = (str(r.get("Adj Strategy", "ALL")), str(r.get("Grade", "ALL")), str(r.get("life_phase", "far")))
            out[key] = {
                "rep_p_mult": float(r.get("rep_p_mult", 1.0)),
                "rep_ratio_mult": float(r.get("rep_ratio_mult", 1.0)),
                "draw_p_mult": float(r.get("draw_p_mult", 1.0)),
                "draw_ratio_mult": float(r.get("draw_ratio_mult", 1.0)),
                "omega_bump": float(r.get("omega_bump", 0.0)),
                "nav_floor_ratio": float(r.get("nav_floor_ratio", 0.0)),
            }
        return out

    def get_deltas(self, strategy: str, grade: str) -> dict[str, Any]:
        g = {
            "delta_p_draw_mult": float(self.global_deltas.get("delta_p_draw_mult", 1.0)),
            "delta_p_rep_mult": float(self.global_deltas.get("delta_p_rep_mult", 1.0)),
            "delta_p_rc_mult": float(self.global_deltas.get("delta_p_rc_mult", 1.0)),
            "delta_rep_ratio_scale": float(self.global_deltas.get("delta_rep_ratio_scale", 1.0)),
            "delta_draw_ratio_scale": float(self.global_deltas.get("delta_draw_ratio_scale", 1.0)),
            "delta_rc_ratio_scale": float(self.global_deltas.get("delta_rc_ratio_scale", 1.0)),
            "delta_omega_bump": float(self.global_deltas.get("delta_omega_bump", 0.0)),
            "delta_rep_timing_shift_q": int(self.global_deltas.get("delta_rep_timing_shift_q", 0)),
            "rc_propensity": float(self.global_deltas.get("rc_propensity", 1.0)),
        }
        row = self.group_deltas.get((strategy, grade))
        if row is not None:
            g.update(row)
        return g

    def get_tail(self, strategy: str, past_end_q: int) -> dict[str, float]:
        if past_end_q <= 3:
            tb = "0-3"
        elif past_end_q <= 7:
            tb = "4-7"
        elif past_end_q <= 11:
            tb = "8-11"
        else:
            tb = "12+"
        return self.tail_idx.get(
            (strategy, tb),
            self.tail_idx.get(
                ("ALL", tb),
                {"p_rep_floor": 0.0, "rep_ratio_floor": 0.0, "draw_prob_cap": 1.0, "omega_drag": 0.0},
            ),
        )

    def get_endgame(self, strategy: str) -> dict[str, float]:
        return self.endgame_idx.get(strategy, self.endgame_idx.get("ALL", {"p_rep_mult": 1.0, "rep_ratio_mult": 1.0}))

    def get_post_invest_draw(self, strategy: str) -> dict[str, float]:
        return self.post_invest_draw_idx.get(
            strategy,
            self.post_invest_draw_idx.get("ALL", {"draw_p_mult": 1.0, "draw_ratio_mult": 1.0}),
        )

    def get_pre_end_repay(self, strategy: str, q_to_end: int) -> dict[str, float]:
        if q_to_end > 24:
            phase = "far"
        elif q_to_end >= 13:
            phase = "mid"
        else:
            phase = "near"
        return self.pre_end_repay_idx.get(
            (strategy, phase),
            self.pre_end_repay_idx.get(("ALL", phase), {"rep_p_mult": 1.0, "rep_ratio_mult": 1.0}),
        )

    def get_lifecycle(self, strategy: str, grade: str, q_to_end: int) -> dict[str, float]:
        if q_to_end > 24:
            phase = "far"
        elif q_to_end >= 13:
            phase = "mid"
        elif q_to_end >= 5:
            phase = "late"
        elif q_to_end >= 1:
            phase = "final"
        elif q_to_end >= -4:
            phase = "post_0_4"
        elif q_to_end >= -8:
            phase = "post_5_8"
        elif q_to_end >= -12:
            phase = "post_9_12"
        else:
            phase = "post_12p"
        return self.lifecycle_idx.get(
            (strategy, grade, phase),
            self.lifecycle_idx.get(
                (strategy, "ALL", phase),
                self.lifecycle_idx.get(
                    ("ALL", "ALL", phase),
                    {
                        "rep_p_mult": 1.0,
                        "rep_ratio_mult": 1.0,
                        "draw_p_mult": 1.0,
                        "draw_ratio_mult": 1.0,
                        "omega_bump": 0.0,
                        "nav_floor_ratio": 0.0,
                    },
                ),
            ),
        )

    def timing(self, strategy: str, grade: str, age_bucket: str, size_bucket: str) -> dict[str, float]:
        keys = [
            ("strategy_grade_age_size", strategy, grade, age_bucket, size_bucket),
            ("strategy_grade_age", strategy, grade, age_bucket, "ALL"),
            ("strategy_age", strategy, "ALL", age_bucket, "ALL"),
            ("strategy", strategy, "ALL", "ALL", "ALL"),
            ("global", "ALL", "ALL", "ALL", "ALL"),
        ]
        for k in keys:
            if k in self.timing_idx:
                return self.timing_idx[k]
        return {"p_draw": 0.0, "p_rep": 0.0, "p_rc": 0.0}

    def ratio(self, metric: str, strategy: str, grade: str, age_bucket: str, size_bucket: str) -> dict[str, Any] | None:
        keys = [
            (metric, "strategy_grade_age_size", strategy, grade, age_bucket, size_bucket),
            (metric, "strategy_grade_age", strategy, grade, age_bucket, "ALL"),
            (metric, "strategy_age", strategy, "ALL", age_bucket, "ALL"),
            (metric, "strategy", strategy, "ALL", "ALL", "ALL"),
            (metric, "global", "ALL", "ALL", "ALL", "ALL"),
        ]
        for k in keys:
            if k in self.ratio_idx:
                return self.ratio_idx[k]
        return None

    def omega(self, strategy: str, grade: str, age_bucket: str, size_bucket: str) -> dict[str, float]:
        keys = [
            ("strategy_grade_age_size", strategy, grade, age_bucket, size_bucket),
            ("strategy_grade_age", strategy, grade, age_bucket, "ALL"),
            ("strategy", strategy, "ALL", "ALL", "ALL"),
            ("global", "ALL", "ALL", "ALL", "ALL"),
        ]
        for k in keys:
            if k in self.omega_idx:
                return self.omega_idx[k]
        return {
            "a_intercept": 0.0,
            "b0": 0.0,
            "b1": 0.0,
            "b_age": 0.0,
            "b_age2": 0.0,
            "b_q_to_end": 0.0,
            "b_pre_end": 0.0,
            "b_post_end": 0.0,
            "alpha": 0.0,
            "sigma": 1e-6,
        }


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


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
    if not (np.any(cfs < 0.0) and np.any(cfs > 0.0)):
        return math.nan

    lo, hi = -0.9999, 10.0
    f_lo, f_hi = _xnpv(lo, cfs, dts), _xnpv(hi, cfs, dts)
    k = 0
    while np.isfinite(f_lo) and np.isfinite(f_hi) and f_lo * f_hi > 0.0 and k < 80:
        hi *= 2.0
        f_hi = _xnpv(hi, cfs, dts)
        k += 1
    if (not np.isfinite(f_lo)) or (not np.isfinite(f_hi)) or f_lo * f_hi > 0.0:
        return math.nan

    for _ in range(240):
        mid = (lo + hi) / 2.0
        f_mid = _xnpv(mid, cfs, dts)
        if not np.isfinite(f_mid):
            return math.nan
        if abs(f_mid) < 1e-8:
            return mid
        if f_lo * f_mid <= 0.0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return (lo + hi) / 2.0


def _copula_uniform(rng: np.random.Generator, z_common: float, rho: float) -> float:
    r = float(np.clip(float(rho), 0.0, 0.999))
    if r <= 1e-12:
        return float(rng.random())
    z = math.sqrt(r) * float(z_common) + math.sqrt(1.0 - r) * float(rng.standard_normal())
    return float(np.clip(_norm_cdf(z), 1e-12, 1.0 - 1e-12))


def _sample_ratio(rng: np.random.Generator, fit: dict[str, Any] | None, u: float | None = None) -> float:
    if fit is None:
        return 0.0
    dist = str(fit.get("dist", "empirical")).lower()
    params = fit.get("params", [])
    mean = float(max(fit.get("mean", 0.0), 0.0))
    uu = float(np.clip(u, 1e-12, 1.0 - 1e-12)) if u is not None else None

    if dist == "lognorm" and len(params) >= 3:
        shape, loc, scale = float(params[0]), float(params[1]), float(params[2])
        if scale <= 0:
            return mean
        if uu is None:
            return float(max(loc + np.exp(rng.normal(np.log(scale), shape)), 0.0))
        z = float(stats.norm.ppf(uu))
        return float(max(loc + scale * np.exp(shape * z), 0.0))

    if dist == "gamma" and len(params) >= 3:
        a, loc, scale = float(params[0]), float(params[1]), float(params[2])
        if a <= 0 or scale <= 0:
            return mean
        if uu is None:
            return float(max(loc + rng.gamma(shape=a, scale=scale), 0.0))
        x = float(stats.gamma.ppf(uu, a, loc=loc, scale=scale))
        return float(max(x, 0.0)) if np.isfinite(x) else mean

    return mean


def _msci_arrays(msci_paths: pd.DataFrame, n_sims: int) -> tuple[list[pd.Timestamp], np.ndarray, np.ndarray]:
    q = sorted(pd.to_datetime(msci_paths["quarter_end"].dropna().unique()).tolist())
    if not q:
        raise ValueError("MSCI projection paths are empty")

    piv = msci_paths.pivot_table(index="sim_id", columns="quarter_end", values="msci_ret_q", aggfunc="first")
    piv_lag = msci_paths.pivot_table(index="sim_id", columns="quarter_end", values="msci_ret_q_lag1", aggfunc="first")

    sim_ids = list(range(1, n_sims + 1))
    q_idx = pd.Index(q)
    piv = piv.reindex(index=sim_ids, columns=q_idx).fillna(0.0)
    piv_lag = piv_lag.reindex(index=sim_ids, columns=q_idx).fillna(0.0)

    return q, piv.to_numpy(dtype=float), piv_lag.to_numpy(dtype=float)


def _quarter_ordinal(ts: pd.Timestamp | pd.NaT) -> int:
    if pd.isna(ts):
        return -10**9
    return int(pd.Period(ts, freq="Q").ordinal)


def simulate_portfolio(
    states: dict[str, FundState],
    fit_artifacts: FitArtifacts,
    calibration: CalibrationArtifacts | None,
    msci_paths: pd.DataFrame,
    sim_cfg: SimConfig,
    fit_cfg: FitConfig,
) -> SimulationOutputs:
    if not states:
        raise ValueError("No funds to simulate")

    lookup = _Lookup(fit_artifacts, calibration)
    n_sims = int(sim_cfg.n_sims)

    quarters, msci_r, msci_r_lag = _msci_arrays(msci_paths, n_sims=n_sims)
    horizon = len(quarters)

    fund_ids = sorted(states.keys())
    n_funds = len(fund_ids)

    strategy = np.array([states[f].strategy for f in fund_ids], dtype=object)
    grade = np.array([states[f].grade for f in fund_ids], dtype=object)
    size_bucket = np.array([states[f].size_bucket for f in fund_ids], dtype=object)

    commitment0 = np.array([max(states[f].commitment, 0.0) for f in fund_ids], dtype=float)
    nav0 = np.array([max(states[f].nav, 0.0) for f in fund_ids], dtype=float)
    draw_cum0 = np.array([max(states[f].draw_cum, 0.0) for f in fund_ids], dtype=float)
    rep_cum0 = np.array([max(states[f].rep_cum, 0.0) for f in fund_ids], dtype=float)
    recall_cum0 = np.array([max(states[f].recall_cum, 0.0) for f in fund_ids], dtype=float)
    age0 = np.array([max(states[f].age_q, 0) for f in fund_ids], dtype=int)

    # Recallable capital available to redraw.
    recall_bal0 = np.maximum(recall_cum0 - np.maximum(draw_cum0 - commitment0, 0.0), 0.0)

    first_close_ord = np.array([_quarter_ordinal(states[f].first_close_qe) for f in fund_ids], dtype=int)
    invest_end_ord = np.array([_quarter_ordinal(states[f].invest_end) for f in fund_ids], dtype=int)
    fund_end_ord = np.array([_quarter_ordinal(states[f].fund_end_qe) for f in fund_ids], dtype=int)

    deltas = [lookup.get_deltas(strategy[i], grade[i]) for i in range(n_funds)]

    draw_sum = np.zeros((n_funds, horizon), dtype=float)
    rep_sum = np.zeros((n_funds, horizon), dtype=float)
    nav_sum = np.zeros((n_funds, horizon), dtype=float)
    rc_sum = np.zeros((n_funds, horizon), dtype=float)

    p_draw_sims = np.zeros((n_sims, horizon), dtype=float)
    p_rep_sims = np.zeros((n_sims, horizon), dtype=float)
    p_nav_sims = np.zeros((n_sims, horizon), dtype=float)
    p_rc_sims = np.zeros((n_sims, horizon), dtype=float)

    q_ord = np.array([_quarter_ordinal(pd.Timestamp(x)) for x in quarters], dtype=int)
    rng = np.random.default_rng(sim_cfg.seed)
    copula_enabled = bool(getattr(sim_cfg, "copula_enabled", False))
    rho_raw = getattr(sim_cfg, "copula_rho", None)
    try:
        copula_rho = float(rho_raw) if rho_raw is not None else 0.35
    except Exception:
        copula_rho = 0.35
    if not np.isfinite(copula_rho):
        copula_rho = 0.35
    copula_rho = float(np.clip(copula_rho, 0.0, 0.999))

    for s in range(n_sims):
        nav = nav0.copy()
        draw_cum = draw_cum0.copy()
        rep_cum = rep_cum0.copy()
        recall_cum = recall_cum0.copy()
        recall_bal = recall_bal0.copy()
        age = age0.copy()
        rc_enabled = np.array([rng.random() < float(d.get("rc_propensity", 1.0)) for d in deltas], dtype=bool)
        for t in range(horizon):
            qe_ord = q_ord[t]
            msci_now = float(msci_r[s, t])
            msci_lag = float(msci_r_lag[s, t])
            # One-factor Gaussian copula common shocks by quarter/channel.
            # This induces cross-fund dependence in the same quarter while avoiding
            # hard-coded within-fund persistence across all quarters.
            if copula_enabled:
                z_draw_evt_common = float(rng.standard_normal())
                z_draw_ratio_common = float(rng.standard_normal())
                z_rep_evt_common = float(rng.standard_normal())
                z_rep_ratio_common = float(rng.standard_normal())
                z_rc_evt_common = float(rng.standard_normal())
                z_rc_ratio_common = float(rng.standard_normal())
            else:
                z_draw_evt_common = 0.0
                z_draw_ratio_common = 0.0
                z_rep_evt_common = 0.0
                z_rep_ratio_common = 0.0
                z_rc_evt_common = 0.0
                z_rc_ratio_common = 0.0

            q_draw = np.zeros(n_funds, dtype=float)
            q_rep = np.zeros(n_funds, dtype=float)
            q_nav = np.zeros(n_funds, dtype=float)
            q_rc = np.zeros(n_funds, dtype=float)

            for i in range(n_funds):
                strat = str(strategy[i])
                grd = str(grade[i])
                sz = str(size_bucket[i])
                d = deltas[i]

                pre_first_close = first_close_ord[i] > -10**8 and qe_ord < first_close_ord[i]
                if pre_first_close:
                    # Fund is not active yet: keep balances flat and do not age.
                    q_draw[i] = 0.0
                    q_rep[i] = 0.0
                    q_nav[i] = nav[i]
                    continue

                age_draw = int(max(age[i], 0))
                age_rep = int(max(age[i] + int(d["delta_rep_timing_shift_q"]), 0))

                age_bucket_draw = make_age_bucket(age_draw)
                age_bucket_rep = make_age_bucket(age_rep)

                tim_draw = lookup.timing(strat, grd, age_bucket_draw, sz)
                tim_rep = lookup.timing(strat, grd, age_bucket_rep, sz)

                p_draw = float(np.clip(tim_draw["p_draw"], 0.0, 1.0))
                p_rep = float(np.clip(tim_rep["p_rep"] * float(d["delta_p_rep_mult"]), 0.0, 1.0))
                p_draw = float(np.clip(p_draw * float(d["delta_p_draw_mult"]), 0.0, 1.0))
                p_rc_joint = float(np.clip(tim_rep["p_rc"] * float(d.get("delta_p_rc_mult", 1.0)), 0.0, 1.0))
                p_rc_cond = float(np.clip(p_rc_joint / max(tim_rep["p_rep"], 1e-9), 0.0, 1.0)) if tim_rep["p_rep"] > 0 else 0.0
                if not rc_enabled[i]:
                    p_rc_cond = 0.0

                past_end_q = qe_ord - fund_end_ord[i]
                in_post_end = fund_end_ord[i] > -10**8 and past_end_q > 0
                after_invest_pre_end = (qe_ord > invest_end_ord[i]) and (not in_post_end)
                q_to_end = fund_end_ord[i] - qe_ord
                in_endgame = fund_end_ord[i] > -10**8 and (1 <= q_to_end <= 12)
                q_to_end_lifecycle = q_to_end if fund_end_ord[i] > -10**8 else 999
                life = lookup.get_lifecycle(strat, grd, int(q_to_end_lifecycle))
                rep_life_mult = 1.0 + float(sim_cfg.lifecycle_rep_strength) * (float(life.get("rep_p_mult", 1.0)) - 1.0)
                rep_ratio_life_mult = 1.0 + float(sim_cfg.lifecycle_rep_strength) * (float(life.get("rep_ratio_mult", 1.0)) - 1.0)
                draw_life_mult = 1.0 + float(sim_cfg.lifecycle_draw_strength) * (float(life.get("draw_p_mult", 1.0)) - 1.0)
                draw_ratio_life_mult = 1.0 + float(sim_cfg.lifecycle_draw_strength) * (float(life.get("draw_ratio_mult", 1.0)) - 1.0)
                omega_life_bump = float(sim_cfg.lifecycle_omega_strength) * float(life.get("omega_bump", 0.0))
                nav_floor_ratio_life = float(max(life.get("nav_floor_ratio", 0.0), 0.0))
                endgame_p_mult = 1.0
                endgame_rr_mult = 1.0
                draw_p_mult_phase = 1.0
                draw_r_mult_phase = 1.0
                rep_p_mult_phase = 1.0
                rep_rr_mult_phase = 1.0

                p_draw = float(np.clip(p_draw * draw_life_mult, 0.0, 1.0))
                p_rep = float(np.clip(p_rep * rep_life_mult, 0.0, 1.0))

                # Apply statistically estimated pre-end repayment attenuation for far/mid phases.
                # Near-end stays with dedicated endgame ramp to avoid double counting.
                pre_end_repay_active = (
                    sim_cfg.pre_end_repay_enabled
                    and after_invest_pre_end
                    and fund_end_ord[i] > -10**8
                    and (q_to_end >= 13)
                )
                if pre_end_repay_active:
                    rm = lookup.get_pre_end_repay(strat, int(q_to_end))
                    rep_p_mult_phase = float(min(float(rm.get("rep_p_mult", 1.0)), 1.0))
                    rep_rr_mult_phase = float(min(float(rm.get("rep_ratio_mult", 1.0)), 1.0))
                    p_rep = float(np.clip(p_rep * rep_p_mult_phase, 0.0, 1.0))

                if in_endgame:
                    eg = lookup.get_endgame(strat)
                    ramp = float((12 - q_to_end) / 11.0) if q_to_end < 12 else 0.0
                    endgame_p_mult = 1.0 + (float(eg.get("p_rep_mult", 1.0)) - 1.0) * ramp
                    endgame_rr_mult = 1.0 + (float(eg.get("rep_ratio_mult", 1.0)) - 1.0) * ramp
                    p_rep = float(np.clip(p_rep * endgame_p_mult, 0.0, 1.0))

                if after_invest_pre_end and sim_cfg.post_invest_draws_enabled:
                    dm = lookup.get_post_invest_draw(strat)
                    draw_p_mult_phase = float(dm.get("draw_p_mult", 1.0))
                    draw_r_mult_phase = float(dm.get("draw_ratio_mult", 1.0))
                    p_draw = float(np.clip(p_draw * draw_p_mult_phase, 0.0, 1.0))

                if in_post_end and sim_cfg.post_end_runoff_enabled:
                    tail = lookup.get_tail(strat, past_end_q)
                    p_rep = max(p_rep, float(np.clip(tail.get("p_rep_floor", 0.0), 0.0, 1.0)))
                    p_draw = min(p_draw, float(np.clip(tail.get("draw_prob_cap", 1.0), 0.0, 1.0)))

                draw_allowed = qe_ord <= invest_end_ord[i]
                if after_invest_pre_end and sim_cfg.post_invest_draws_enabled:
                    draw_allowed = True
                if in_post_end and sim_cfg.post_end_no_draws:
                    draw_allowed = False

                draw_cap = max(commitment0[i] + recall_bal[i] - draw_cum[i], 0.0)
                u_draw_evt = _copula_uniform(rng, z_draw_evt_common, copula_rho) if copula_enabled else float(rng.random())
                draw_event = draw_allowed and (draw_cap > 1.0) and (u_draw_evt < p_draw)

                draw_amt = 0.0
                if draw_event:
                    rfit = lookup.ratio("draw_ratio", strat, grd, age_bucket_draw, sz)
                    u_draw_ratio = _copula_uniform(rng, z_draw_ratio_common, copula_rho) if copula_enabled else None
                    r = _sample_ratio(rng, rfit, u=u_draw_ratio) * float(d["delta_draw_ratio_scale"]) * float(draw_r_mult_phase)
                    r *= draw_ratio_life_mult
                    r = float(max(r, 0.0))
                    draw_denom = max(commitment0[i] + recall_bal[i], 0.0)
                    draw_amt = min(draw_cap, r * draw_denom)
                    draw_amt = min(draw_amt, sim_cfg.draw_ratio_cap * max(commitment0[i], 0.0) + recall_bal[i])

                    commit_remaining = max(commitment0[i] - draw_cum[i], 0.0)
                    from_recall = max(draw_amt - commit_remaining, 0.0)
                    recall_bal[i] = max(recall_bal[i] - from_recall, 0.0)

                u_rep_evt = _copula_uniform(rng, z_rep_evt_common, copula_rho) if copula_enabled else float(rng.random())
                rep_event = (nav[i] > fit_cfg.nav_gate) and (u_rep_evt < p_rep)
                rep_amt = 0.0
                if rep_event:
                    rfit = lookup.ratio("rep_ratio", strat, grd, age_bucket_rep, sz)
                    u_rep_ratio = _copula_uniform(rng, z_rep_ratio_common, copula_rho) if copula_enabled else None
                    rr = _sample_ratio(rng, rfit, u=u_rep_ratio) * float(d["delta_rep_ratio_scale"])
                    rr *= rep_ratio_life_mult
                    rr *= float(rep_rr_mult_phase)
                    rr *= float(endgame_rr_mult)
                    rr = float(max(rr, 0.0))
                    if in_post_end and sim_cfg.post_end_runoff_enabled:
                        tail = lookup.get_tail(strat, past_end_q)
                        rr = max(rr, float(max(tail.get("rep_ratio_floor", 0.0), 0.0)))
                    rep_amt = min(rr * nav[i], nav[i] + draw_amt)

                rc_amt = 0.0
                u_rc_evt = _copula_uniform(rng, z_rc_evt_common, copula_rho) if copula_enabled else float(rng.random())
                if rep_amt > 0 and (u_rc_evt < p_rc_cond):
                    rfit_rc = lookup.ratio("rc_ratio_given_rep", strat, grd, age_bucket_rep, sz)
                    u_rc_ratio = _copula_uniform(rng, z_rc_ratio_common, copula_rho) if copula_enabled else None
                    rc_ratio = max(_sample_ratio(rng, rfit_rc, u=u_rc_ratio), 0.0) * float(d.get("delta_rc_ratio_scale", 1.0))
                    rc_amt = rep_amt * rc_ratio

                # Cap cumulative recallables by cumulative repayments.
                max_rc_add = max((rep_cum[i] + rep_amt) - recall_cum[i], 0.0)
                rc_amt = min(rc_amt, max_rc_add)

                nav_after_flow = max(nav[i] + draw_amt - rep_amt, 0.0)

                om = lookup.omega(strat, grd, age_bucket_draw, sz)
                age_years = float(max(age[i], 0)) / 4.0
                qte_years = float(np.clip(float(q_to_end_lifecycle) / 4.0, -4.0, 12.0))
                is_pre_end = 1.0 if (fund_end_ord[i] > -10**8 and (1 <= q_to_end <= 12)) else 0.0
                is_post_end = 1.0 if in_post_end else 0.0
                omega = (
                    float(om["a_intercept"])
                    + float(om["alpha"])
                    + float(d["delta_omega_bump"])
                    + float(om["b0"]) * msci_now
                    + float(om["b1"]) * msci_lag
                    + float(om.get("b_age", 0.0)) * age_years
                    + float(om.get("b_age2", 0.0)) * (age_years**2)
                    + float(om.get("b_q_to_end", 0.0)) * qte_years
                    + float(om.get("b_pre_end", 0.0)) * is_pre_end
                    + float(om.get("b_post_end", 0.0)) * is_post_end
                    + omega_life_bump
                    + float(om["sigma"]) * float(rng.standard_normal())
                )
                if in_post_end and sim_cfg.post_end_runoff_enabled:
                    tail = lookup.get_tail(strat, past_end_q)
                    omega += float(tail.get("omega_drag", 0.0))
                omega = float(np.clip(omega, sim_cfg.omega_clip[0], sim_cfg.omega_clip[1]))

                nav_next = max(nav_after_flow * (1.0 + omega), 0.0)
                if in_post_end and nav_next < 1.0:
                    nav_next = 0.0

                # Preserve historically observed residual NAV profiles near and after end-of-life.
                if (
                    age0[i] >= int(sim_cfg.lifecycle_nav_floor_min_start_age_q)
                    and fund_end_ord[i] > -10**8
                    and q_to_end_lifecycle <= 24
                    and nav_floor_ratio_life > 0.0
                ):
                    target_nav_floor = nav_floor_ratio_life * max(commitment0[i], 0.0)
                    if q_to_end_lifecycle >= 13:
                        floor_strength = float(sim_cfg.lifecycle_nav_floor_strength_mid)
                    elif q_to_end_lifecycle >= 5:
                        floor_strength = 0.65 * float(sim_cfg.lifecycle_nav_floor_strength)
                    elif q_to_end_lifecycle >= 1:
                        floor_strength = 0.85 * float(sim_cfg.lifecycle_nav_floor_strength)
                    else:
                        floor_strength = float(sim_cfg.lifecycle_nav_floor_strength)
                    if target_nav_floor > nav_next and floor_strength > 0.0:
                        nav_next = nav_next + floor_strength * (target_nav_floor - nav_next)

                draw_cum[i] += draw_amt
                rep_cum[i] += rep_amt
                recall_cum[i] += rc_amt
                recall_bal[i] += rc_amt
                nav[i] = nav_next
                age[i] += 1

                q_draw[i] = draw_amt
                q_rep[i] = rep_amt
                q_nav[i] = nav_next
                q_rc[i] = rc_amt

            draw_sum[:, t] += q_draw
            rep_sum[:, t] += q_rep
            nav_sum[:, t] += q_nav
            rc_sum[:, t] += q_rc

            p_draw_sims[s, t] = float(np.sum(q_draw))
            p_rep_sims[s, t] = float(np.sum(q_rep))
            p_nav_sims[s, t] = float(np.sum(q_nav))
            p_rc_sims[s, t] = float(np.sum(q_rc))

    draw_mean = draw_sum / n_sims
    rep_mean = rep_sum / n_sims
    nav_mean = nav_sum / n_sims
    rc_mean = rc_sum / n_sims

    # DPI distribution across simulation paths (to-date by quarter).
    cum_draw_sims = np.cumsum(p_draw_sims, axis=1)
    cum_rep_sims = np.cumsum(p_rep_sims, axis=1)
    dpi_sims = np.full_like(cum_draw_sims, np.nan, dtype=float)
    np.divide(cum_rep_sims, cum_draw_sims, out=dpi_sims, where=cum_draw_sims > 1e-12)

    def _path_quantile(a: np.ndarray, q: float) -> np.ndarray:
        out = np.full(a.shape[1], np.nan, dtype=float)
        for j in range(a.shape[1]):
            col = a[:, j]
            col = col[np.isfinite(col)]
            if len(col):
                out[j] = float(np.quantile(col, q))
        return out

    port = pd.DataFrame(
        {
            "quarter_end": quarters,
            "sim_draw_mean": p_draw_sims.mean(axis=0),
            "sim_rep_mean": p_rep_sims.mean(axis=0),
            "sim_nav_mean": p_nav_sims.mean(axis=0),
            "sim_rc_mean": p_rc_sims.mean(axis=0),
            "sim_draw_p05": np.quantile(p_draw_sims, 0.05, axis=0),
            "sim_draw_p10": np.quantile(p_draw_sims, 0.10, axis=0),
            "sim_draw_p50": np.quantile(p_draw_sims, 0.50, axis=0),
            "sim_draw_p90": np.quantile(p_draw_sims, 0.90, axis=0),
            "sim_draw_p95": np.quantile(p_draw_sims, 0.95, axis=0),
            "sim_rep_p05": np.quantile(p_rep_sims, 0.05, axis=0),
            "sim_rep_p10": np.quantile(p_rep_sims, 0.10, axis=0),
            "sim_rep_p50": np.quantile(p_rep_sims, 0.50, axis=0),
            "sim_rep_p90": np.quantile(p_rep_sims, 0.90, axis=0),
            "sim_rep_p95": np.quantile(p_rep_sims, 0.95, axis=0),
            "sim_nav_p05": np.quantile(p_nav_sims, 0.05, axis=0),
            "sim_nav_p10": np.quantile(p_nav_sims, 0.10, axis=0),
            "sim_nav_p50": np.quantile(p_nav_sims, 0.50, axis=0),
            "sim_nav_p90": np.quantile(p_nav_sims, 0.90, axis=0),
            "sim_nav_p95": np.quantile(p_nav_sims, 0.95, axis=0),
            "sim_rc_p05": np.quantile(p_rc_sims, 0.05, axis=0),
            "sim_rc_p10": np.quantile(p_rc_sims, 0.10, axis=0),
            "sim_rc_p50": np.quantile(p_rc_sims, 0.50, axis=0),
            "sim_rc_p90": np.quantile(p_rc_sims, 0.90, axis=0),
            "sim_rc_p95": np.quantile(p_rc_sims, 0.95, axis=0),
            "sim_dpi_p05": _path_quantile(dpi_sims, 0.05),
            "sim_dpi_p50": _path_quantile(dpi_sims, 0.50),
            "sim_dpi_p95": _path_quantile(dpi_sims, 0.95),
        }
    )

    fund_rows: list[dict[str, Any]] = []
    for i, fid in enumerate(fund_ids):
        for t, qe in enumerate(quarters):
            fund_rows.append(
                {
                    "FundID": fid,
                    "quarter_end": qe,
                    "sim_draw_mean": float(draw_mean[i, t]),
                    "sim_rep_mean": float(rep_mean[i, t]),
                    "sim_nav_mean": float(nav_mean[i, t]),
                    "sim_rc_mean": float(rc_mean[i, t]),
                }
            )
    fund_q = pd.DataFrame(fund_rows)

    end = (
        fund_q.sort_values(["FundID", "quarter_end"]).groupby("FundID", as_index=False).tail(1)[
            ["FundID", "quarter_end", "sim_nav_mean"]
        ]
    )
    end = end.rename(columns={"quarter_end": "projection_end_qe", "sim_nav_mean": "sim_nav_end_mean"})
    final_draw = cum_draw_sims[:, -1] if cum_draw_sims.shape[1] else np.array([], dtype=float)
    final_rep = cum_rep_sims[:, -1] if cum_rep_sims.shape[1] else np.array([], dtype=float)
    final_nav = p_nav_sims[:, -1] if p_nav_sims.shape[1] else np.array([], dtype=float)
    final_dpi = np.where(final_draw > 1e-12, final_rep / final_draw, np.nan)
    final_tvpi = np.where(final_draw > 1e-12, (final_rep + final_nav) / final_draw, np.nan)

    irr_paths = np.full(n_sims, np.nan, dtype=float)
    if horizon > 0:
        cfs_q = p_rep_sims - p_draw_sims
        dts = np.array(quarters, dtype="datetime64[ns]")
        for s in range(n_sims):
            cfs = np.append(cfs_q[s, :], float(final_nav[s]))
            dts_s = np.append(dts, dts[-1])
            irr = _xirr(cfs, dts_s)
            if np.isfinite(irr) and (-0.95 <= irr <= 2.0):
                irr_paths[s] = float(irr)

    def _q(x: np.ndarray, q: float) -> float:
        v = x[np.isfinite(x)]
        if len(v) == 0:
            return math.nan
        return float(np.quantile(v, q))

    final_dist = pd.DataFrame(
        [
            {
                "quarter_end": pd.Timestamp(quarters[-1]) if len(quarters) else pd.NaT,
                "metric": "cum_draw",
                "p05": _q(final_draw, 0.05),
                "p50": _q(final_draw, 0.50),
                "p95": _q(final_draw, 0.95),
            },
            {
                "quarter_end": pd.Timestamp(quarters[-1]) if len(quarters) else pd.NaT,
                "metric": "cum_repay",
                "p05": _q(final_rep, 0.05),
                "p50": _q(final_rep, 0.50),
                "p95": _q(final_rep, 0.95),
            },
            {
                "quarter_end": pd.Timestamp(quarters[-1]) if len(quarters) else pd.NaT,
                "metric": "end_nav",
                "p05": _q(final_nav, 0.05),
                "p50": _q(final_nav, 0.50),
                "p95": _q(final_nav, 0.95),
            },
            {
                "quarter_end": pd.Timestamp(quarters[-1]) if len(quarters) else pd.NaT,
                "metric": "dpi",
                "p05": _q(final_dpi, 0.05),
                "p50": _q(final_dpi, 0.50),
                "p95": _q(final_dpi, 0.95),
            },
            {
                "quarter_end": pd.Timestamp(quarters[-1]) if len(quarters) else pd.NaT,
                "metric": "tvpi",
                "p05": _q(final_tvpi, 0.05),
                "p50": _q(final_tvpi, 0.50),
                "p95": _q(final_tvpi, 0.95),
            },
            {
                "quarter_end": pd.Timestamp(quarters[-1]) if len(quarters) else pd.NaT,
                "metric": "irr",
                "p05": _q(irr_paths, 0.05),
                "p50": _q(irr_paths, 0.50),
                "p95": _q(irr_paths, 0.95),
            },
        ]
    )

    return SimulationOutputs(
        portfolio_series=port,
        fund_quarterly_mean=fund_q,
        fund_end_summary=end,
        final_distribution_summary=final_dist,
    )


__all__ = ["SimulationOutputs", "simulate_portfolio"]
