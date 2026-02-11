from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from .utils import assign_size_bucket, make_age_bucket


@dataclass
class FundState:
    fund_id: str
    strategy: str
    grade: str
    size_bucket: str
    commitment: float
    nav: float
    draw_cum: float
    rep_cum: float
    recall_cum: float
    age_q: int
    start_qe: pd.Timestamp
    first_close_qe: pd.Timestamp | pd.NaT
    fund_end_qe: pd.Timestamp | pd.NaT
    invest_end: pd.Timestamp | pd.NaT


@dataclass
class EndDateInfo:
    fund_end_qe: pd.Series
    avg_overrun_by_strategy: pd.Series


def add_size_bucket(df: pd.DataFrame, bins: list[float], labels: list[str], column: str = "Target Fund Size") -> pd.DataFrame:
    out = df.copy()
    out["size_bucket"] = out[column].apply(lambda x: assign_size_bucket(float(x), bins, labels))
    return out


def compute_invest_end_by_fund(hist: pd.DataFrame) -> pd.Series:
    first_close = hist.groupby("FundID")["First Closing Date"].min()
    fallback = hist.groupby("FundID")["quarter_end"].min()
    first_close = first_close.fillna(fallback)

    first_repay = (
        hist[hist["Adj Repayment EUR"].abs() > 0]
        .groupby("FundID")["quarter_end"]
        .min()
        .reindex(first_close.index)
    )
    repay_within_5y = first_repay.notna() & (first_repay <= (first_close + pd.DateOffset(years=5)))
    base_years = np.where(repay_within_5y, 5, 6)
    invest_years = pd.Series(base_years + 1, index=first_close.index)

    invest_end = pd.Series(index=first_close.index, dtype="datetime64[ns]")
    for fid, fc in first_close.items():
        if pd.isna(fc):
            invest_end.loc[fid] = pd.NaT
        else:
            invest_end.loc[fid] = fc + pd.DateOffset(years=int(invest_years.loc[fid]))
    return invest_end


def _quarters_diff(a: pd.Timestamp, b: pd.Timestamp) -> float:
    if pd.isna(a) or pd.isna(b):
        return np.nan
    return float(pd.Period(a, freq="Q").ordinal - pd.Period(b, freq="Q").ordinal)


def compute_fund_end_dates(
    hist: pd.DataFrame,
    avg_overrun_override: pd.Series | dict[str, float] | None = None,
) -> EndDateInfo:
    h = hist.sort_values(["FundID", "quarter_end"]).copy()
    planned_end = h.groupby("FundID")["Planned End Date"].last()
    planned_end_qe = planned_end.dt.to_period("Q").dt.to_timestamp("Q")
    first_close_qe = h.groupby("FundID")["First Closing Date"].min().dt.to_period("Q").dt.to_timestamp("Q")
    last_obs = h.groupby("FundID")["quarter_end"].max()
    fund_strategy = h.groupby("FundID")["Adj Strategy"].agg(lambda s: s.mode().iat[0] if len(s.mode()) else s.iloc[0]).astype(str)
    fund_grade = h.groupby("FundID")["Grade"].agg(lambda s: s.mode().iat[0] if len(s.mode()) else s.iloc[0]).astype(str)

    overrun_q = (
        last_obs.to_frame("last_qe")
        .join(planned_end_qe.rename("planned_end_qe"))
        .apply(
            lambda r: max(_quarters_diff(r["last_qe"], r["planned_end_qe"]), 0.0)
            if pd.notna(r["planned_end_qe"])
            else np.nan,
            axis=1,
        )
    )
    overran = overrun_q[overrun_q.notna() & (overrun_q > 0.0)]

    def _blend_center(median_v: float, mean_v: float) -> float:
        med = float(median_v) if np.isfinite(median_v) else np.nan
        avg = float(mean_v) if np.isfinite(mean_v) else np.nan
        if np.isfinite(med) and np.isfinite(avg):
            return 0.75 * med + 0.25 * avg
        if np.isfinite(med):
            return med
        if np.isfinite(avg):
            return avg
        return np.nan

    over_df = overran.to_frame("overrun_q").join(fund_strategy.rename("Adj Strategy")).join(fund_grade.rename("Grade"))
    sg_stats = (
        over_df.groupby(["Adj Strategy", "Grade"], as_index=False)["overrun_q"]
        .agg(n="size", median="median", mean="mean")
        .sort_values("n", ascending=False)
    )
    s_stats = over_df.groupby("Adj Strategy")["overrun_q"].agg(n="size", median="median", mean="mean")
    global_overrun = float(overran.median()) if len(overran) else 0.0

    strat_center = (
        s_stats.apply(lambda r: _blend_center(r["median"], r["mean"]), axis=1).clip(lower=0.0, upper=20.0)
        if len(s_stats)
        else pd.Series(dtype=float)
    )

    if avg_overrun_override is not None:
        if isinstance(avg_overrun_override, dict):
            override = pd.Series({str(k): float(v) for k, v in avg_overrun_override.items()}, dtype=float)
        else:
            override = pd.to_numeric(avg_overrun_override, errors="coerce")
            override.index = override.index.astype(str)
        strat_center = override.combine_first(strat_center)

    def _est_overrun(strat: str, grade: str) -> float:
        s0 = float(strat_center.get(strat, global_overrun)) if len(strat_center) else global_overrun
        if not np.isfinite(s0):
            s0 = global_overrun
        sg = sg_stats[(sg_stats["Adj Strategy"] == str(strat)) & (sg_stats["Grade"] == str(grade))]
        if len(sg):
            r = sg.iloc[0]
            sg_center = _blend_center(float(r["median"]), float(r["mean"]))
            lam = float(r["n"]) / (float(r["n"]) + 12.0)
            if np.isfinite(sg_center):
                s0 = s0 + lam * (sg_center - s0)
        if strat in s_stats.index:
            ns = float(s_stats.loc[strat, "n"])
            lam_s = ns / (ns + 20.0)
            s0 = global_overrun + lam_s * (s0 - global_overrun)
        return float(np.clip(s0, 0.0, 20.0))

    # Planned life fallback when planned end date is missing.
    planned_life_q = (
        planned_end_qe.to_frame("planned_end_qe")
        .join(first_close_qe.rename("first_close_qe"))
        .apply(
            lambda r: _quarters_diff(r["planned_end_qe"], r["first_close_qe"])
            if pd.notna(r["planned_end_qe"]) and pd.notna(r["first_close_qe"])
            else np.nan,
            axis=1,
        )
    )
    planned_life_q = planned_life_q[(planned_life_q >= 8.0) & (planned_life_q <= 80.0)]
    life_df = planned_life_q.to_frame("life_q").join(fund_strategy.rename("Adj Strategy")).join(fund_grade.rename("Grade"))
    sg_life = life_df.groupby(["Adj Strategy", "Grade"], as_index=False)["life_q"].agg(n="size", median="median")
    s_life = life_df.groupby("Adj Strategy")["life_q"].agg(n="size", median="median")
    global_life_q = float(planned_life_q.median()) if len(planned_life_q) else 40.0

    def _est_planned_life(strat: str, grade: str) -> float:
        s0 = float(s_life.loc[strat, "median"]) if strat in s_life.index else global_life_q
        sg = sg_life[(sg_life["Adj Strategy"] == str(strat)) & (sg_life["Grade"] == str(grade))]
        if len(sg):
            n = float(sg["n"].iloc[0])
            med = float(sg["median"].iloc[0])
            lam = n / (n + 16.0)
            s0 = s0 + lam * (med - s0)
        if strat in s_life.index:
            ns = float(s_life.loc[strat, "n"])
            lam_s = ns / (ns + 24.0)
            s0 = global_life_q + lam_s * (s0 - global_life_q)
        return float(np.clip(s0, 24.0, 80.0))

    fund_end_qe = pd.Series(index=planned_end_qe.index, dtype="datetime64[ns]")
    min_life_q = 32

    for fid in fund_end_qe.index:
        strat = str(fund_strategy.get(fid, "Unknown"))
        grade = str(fund_grade.get(fid, "D"))
        fc = first_close_qe.get(fid, pd.NaT)
        pe = planned_end_qe.get(fid, pd.NaT)
        over_q = _est_overrun(strat, grade)

        if pd.isna(pe):
            if pd.notna(fc):
                life_q = int(round(_est_planned_life(strat, grade)))
                pe = (pd.Period(fc, freq="Q") + life_q).to_timestamp("Q")
            else:
                lo = last_obs.get(fid, pd.NaT)
                if pd.notna(lo):
                    pe = (pd.Period(lo, freq="Q") + min_life_q).to_timestamp("Q")
                else:
                    pe = pd.NaT

        if pd.isna(pe):
            fund_end_qe.loc[fid] = pd.NaT
            continue

        end_qe = (pd.Period(pe, freq="Q") + int(round(over_q))).to_timestamp("Q")
        if pd.notna(fc):
            min_end = (pd.Period(fc, freq="Q") + min_life_q).to_timestamp("Q")
            if end_qe < min_end:
                end_qe = min_end
        fund_end_qe.loc[fid] = end_qe

    # Last fallback for any unresolved funds.
    missing = fund_end_qe[fund_end_qe.isna()].index
    if len(missing):
        for fid in missing:
            lo = last_obs.get(fid, pd.NaT)
            strat = str(fund_strategy.get(fid, "Unknown"))
            grade = str(fund_grade.get(fid, "D"))
            over_q = max(_est_overrun(strat, grade), 8.0)
            if pd.notna(lo):
                fund_end_qe.loc[fid] = (pd.Period(lo, freq="Q") + int(round(over_q))).to_timestamp("Q")

    avg_overrun_by_strategy = strat_center.astype(float).clip(lower=0.0, upper=20.0)
    return EndDateInfo(fund_end_qe=fund_end_qe, avg_overrun_by_strategy=avg_overrun_by_strategy)


def build_fund_states(
    hist_to_cutoff: pd.DataFrame,
    size_bins: list[float],
    size_labels: list[str],
    avg_overrun_by_strategy: pd.Series | dict[str, float] | None = None,
) -> Dict[str, FundState]:
    hist = add_size_bucket(hist_to_cutoff, size_bins, size_labels)
    invest_end = compute_invest_end_by_fund(hist)
    end_info = compute_fund_end_dates(hist, avg_overrun_override=avg_overrun_by_strategy)

    last = hist.sort_values(["FundID", "quarter_end"]).groupby("FundID").tail(1)

    draw_cum = hist.groupby("FundID")["Adj Drawdown EUR"].sum()
    rep_cum = hist.groupby("FundID")["Adj Repayment EUR"].sum()
    recall_cum = hist.groupby("FundID")["Recallable"].sum()
    commit_max = hist.groupby("FundID")["Commitment EUR"].max()
    start_qe = hist.groupby("FundID")["quarter_end"].min()
    first_close = hist.groupby("FundID")["First Closing Date"].min()

    states: Dict[str, FundState] = {}
    for _, row in last.iterrows():
        fid = row["FundID"]
        status = str(row.get("Fund Workflow Stage", "")).lower()
        if "terminated" in status:
            continue

        age_q = int(pd.to_numeric(row.get("Fund_Age_Quarters", 0), errors="coerce") or 0)
        if age_q <= 0 and pd.notna(row.get("First Closing Date")):
            age_q = int(
                max(
                    0,
                    _quarters_diff(row["quarter_end"], pd.Period(row["First Closing Date"], freq="Q").to_timestamp("Q")),
                )
            )

        states[fid] = FundState(
            fund_id=fid,
            strategy=str(row.get("Adj Strategy", "Unknown")),
            grade=str(row.get("Grade", "D")),
            size_bucket=str(row.get("size_bucket", size_labels[-1])),
            commitment=float(commit_max.get(fid, row.get("Commitment EUR", 0.0))),
            nav=float(row.get("NAV Adjusted EUR", 0.0)),
            draw_cum=float(draw_cum.get(fid, 0.0)),
            rep_cum=float(rep_cum.get(fid, 0.0)),
            recall_cum=float(recall_cum.get(fid, 0.0)),
            age_q=age_q,
            start_qe=start_qe.get(fid, row["quarter_end"]),
            first_close_qe=first_close.get(fid, pd.NaT),
            fund_end_qe=end_info.fund_end_qe.get(fid, pd.NaT),
            invest_end=invest_end.get(fid, pd.NaT),
        )
    return states


def append_age_bucket(df: pd.DataFrame, age_col: str = "Fund_Age_Quarters") -> pd.DataFrame:
    out = df.copy()
    out["AgeBucket"] = out[age_col].apply(make_age_bucket)
    return out
