#!/usr/bin/env python3
"""
Simulate cashflows using fitted timing + ratio distributions and grade transitions.
This is a lightweight simulator for validation/experimentation (not the full backtest).
"""

import argparse
import ast
import json
import os
from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats


AGE_BINS_Q = [-1, 3, 7, 11, 15, 19, 1000]
AGE_LABELS = ["0-3", "4-7", "8-11", "12-15", "16-19", "20+"]
GRADE_STATES = ["A", "B", "C", "D"]


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
    rename[_get("NAV Adjusted EUR")] = "NAV Adjusted EUR"
    rename[_get("Commitment EUR")] = "Commitment EUR"
    rename[_get("Signed Amount EUR")] = "Signed Amount EUR"
    rename[_get("Capacity")] = "Capacity"
    rename[_get("Fund_Age_Quarters")] = "Fund_Age_Quarters"
    rename[_get("draw_cum_prev")] = "draw_cum_prev"
    rename[_get("Recallable_Percentage_Decimal")] = "Recallable_Percentage_Decimal"
    rename[_get("Expiration_Quarters")] = "Expiration_Quarters"
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


def make_age_bucket(age_q: int) -> str:
    for i in range(len(AGE_BINS_Q) - 1):
        if AGE_BINS_Q[i] < age_q <= AGE_BINS_Q[i + 1]:
            return AGE_LABELS[i]
    return AGE_LABELS[-1]


def inv_norm(u: float) -> float:
    u = float(np.clip(u, 1e-6, 1 - 1e-6))
    return stats.norm.ppf(u)


def one_factor_uniforms(n: int, rng: np.random.Generator, rho: float) -> np.ndarray:
    Z = rng.standard_normal()
    eps = rng.standard_normal(n)
    z = rho * Z + sqrt(1.0 - rho * rho) * eps
    return stats.norm.cdf(z)


@dataclass
class RecallableBucket:
    created_q: int
    expiry_q: int
    amount_remaining: float


@dataclass
class RecallableLedger:
    rho: float
    expiry_quarters: int
    commitment: float
    buckets: List[RecallableBucket] = field(default_factory=list)

    def _rc_cap(self) -> float:
        return max(float(self.rho), 0.0) * max(float(self.commitment), 0.0)

    def drop_expired(self, q: int) -> None:
        if int(self.expiry_quarters) <= 0:
            self.buckets = []
            return
        self.buckets = [b for b in self.buckets if b.expiry_q >= q and b.amount_remaining > 0]

    def available(self, q: int) -> float:
        self.drop_expired(q)
        return float(sum(b.amount_remaining for b in self.buckets))

    def add_recallable(self, q: int, rc_amount: float, enforce_cap: bool = True) -> float:
        self.drop_expired(q)
        x = max(float(rc_amount or 0.0), 0.0)
        if x <= 0.0 or int(self.expiry_quarters) <= 0:
            return 0.0
        add_amt = x
        if enforce_cap:
            cap = self._rc_cap()
            cur = self.available(q)
            room = max(cap - cur, 0.0)
            add_amt = min(add_amt, room)
        if add_amt <= 0.0:
            return 0.0
        self.buckets.append(RecallableBucket(
            created_q=q,
            expiry_q=q + int(self.expiry_quarters),
            amount_remaining=float(add_amt)
        ))
        return float(add_amt)

    def consume_for_drawdown(self, q: int, draw_amount: float) -> Dict[str, float]:
        self.drop_expired(q)
        need = max(float(draw_amount or 0.0), 0.0)
        if need <= 0.0:
            return {"use_rc": 0.0, "use_commitment": 0.0}
        self.buckets.sort(key=lambda b: b.created_q)
        use_rc = 0.0
        for b in self.buckets:
            if need <= 0:
                break
            take = min(b.amount_remaining, need)
            b.amount_remaining -= take
            need -= take
            use_rc += take
        return {"use_rc": use_rc, "use_commitment": max(draw_amount - use_rc, 0.0)}


def load_fit_table(path: str, key_cols: List[str]) -> Dict:
    df = pd.read_csv(path)
    out = {}
    for _, r in df.iterrows():
        key = tuple(r[c] for c in key_cols)
        out[key] = r.to_dict()
    return out


def sample_from_dist(dist_name: str, params, u: float) -> float:
    if dist_name == "error" or params is None:
        return 0.0
    dist = getattr(stats, dist_name)
    return float(dist.ppf(u, *params))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="anonymized.csv")
    ap.add_argument("--fit-dir", default="model_fits/outputs")
    ap.add_argument("--trans-dir", default="model_fits/outputs/transitions")
    ap.add_argument("--copula", default="model_fits/outputs/copula_params.json")
    ap.add_argument("--start-year", type=int, required=True)
    ap.add_argument("--start-quarter", type=str, required=True)
    ap.add_argument("--horizon-quarters", type=int, default=20)
    ap.add_argument("--n-sims", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--omega-mode", choices=["none", "global"], default="none")
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
    df = apply_current_grade(df, context="simulation")
    df = df.sort_values(["FundID", "quarter_end"])

    start_qe = pd.Period(f"{args.start_year}Q{args.start_quarter[-1]}", freq="Q").to_timestamp("Q")
    horizon = args.horizon_quarters
    quarters = [start_qe + pd.offsets.QuarterEnd(i) for i in range(1, horizon + 1)]

    # Load fitted tables
    ratio_sel = load_fit_table(os.path.join(args.fit_dir, "ratio_fit_selected.csv"),
                               ["Adj Strategy", "Grade", "AgeBucket", "ratio"])
    timing_sel = load_fit_table(os.path.join(args.fit_dir, "timing_probs_selected.csv"),
                                ["Adj Strategy", "Grade", "AgeBucket"])

    # Copula params
    rho_event = 0.25
    rho_size = 0.15
    if os.path.exists(args.copula):
        with open(args.copula, "r") as f:
            cop = json.load(f)
            rho_event = float(cop.get("rho_event", rho_event))
            rho_size = float(cop.get("rho_size", rho_size))

    # Transition matrices
    trans_all_path = os.path.join(args.trans_dir, "grade_transition_1y_all.csv")
    trans_all = pd.read_csv(trans_all_path, index_col=0) if os.path.exists(trans_all_path) else None

    # Fund states at start_qe
    hist = df[df["quarter_end"] <= start_qe].copy()
    last = hist.sort_values(["FundID", "quarter_end"]).groupby("FundID").tail(1)

    fund_states = {}
    for _, r in last.iterrows():
        fid = r["FundID"]
        strategy = r.get("Adj Strategy", "Unknown")
        grade = r.get("Grade", "D")
        age_q = int(pd.to_numeric(r.get("Fund_Age_Quarters", 0), errors="coerce") or 0)
        nav = float(pd.to_numeric(r.get("NAV Adjusted EUR", 0), errors="coerce") or 0.0)
        commitment = float(pd.to_numeric(r.get("Commitment EUR", 0), errors="coerce") or
                           pd.to_numeric(r.get("Signed Amount EUR", 0), errors="coerce") or 0.0)
        dd_cum = float(pd.to_numeric(r.get("draw_cum_prev", 0), errors="coerce") or 0.0)
        rho = float(pd.to_numeric(r.get("Recallable_Percentage_Decimal", 0), errors="coerce") or 0.0)
        exp_q = int(pd.to_numeric(r.get("Expiration_Quarters", 0), errors="coerce") or 0)

        fund_states[fid] = {
            "strategy": strategy,
            "grade": grade,
            "age0": age_q,
            "nav": nav,
            "dd_commit": dd_cum,
            "commitment": commitment,
            "ledger": RecallableLedger(rho=rho, expiry_quarters=exp_q, commitment=commitment),
        }

    fund_ids = list(fund_states.keys())
    n_funds = len(fund_ids)

    # Omega stats (global)
    omega_mu = 0.0
    omega_sig = 0.0
    if args.omega_mode == "global":
        df2 = df.sort_values(["FundID", "quarter_end"]).copy()
        df2["nav_prev"] = df2.groupby("FundID")["NAV Adjusted EUR"].shift(1)
        df2["flow_net"] = pd.to_numeric(df2["Adj Drawdown EUR"], errors="coerce").fillna(0.0) - \
                          pd.to_numeric(df2["Adj Repayment EUR"], errors="coerce").fillna(0.0)
        m = df2["nav_prev"].abs() > 1.0
        omega = ((df2.loc[m, "NAV Adjusted EUR"] - df2.loc[m, "nav_prev"]) - df2.loc[m, "flow_net"]) / df2.loc[m, "nav_prev"]
        omega = omega.replace([np.inf, -np.inf], np.nan).dropna()
        if len(omega):
            omega_mu = float(omega.mean())
            omega_sig = float(omega.std(ddof=1))

    rng = np.random.default_rng(args.seed)

    # Prepare outputs
    sim_nav = np.zeros((args.n_sims, horizon))
    sim_draw = np.zeros((args.n_sims, horizon))
    sim_rep = np.zeros((args.n_sims, horizon))

    for s in range(args.n_sims):
        # copy state
        state = {
            fid: {
                **st,
                "ledger": RecallableLedger(
                    rho=st["ledger"].rho,
                    expiry_quarters=st["ledger"].expiry_quarters,
                    commitment=st["ledger"].commitment,
                    buckets=[RecallableBucket(b.created_q, b.expiry_q, b.amount_remaining) for b in st["ledger"].buckets],
                ),
            } for fid, st in fund_states.items()
        }

        for t, qe in enumerate(quarters):
            # copula uniforms for this quarter
            U = {
                "draw_event": one_factor_uniforms(n_funds, rng, rho_event),
                "draw_size": one_factor_uniforms(n_funds, rng, rho_size),
                "rep_event": one_factor_uniforms(n_funds, rng, rho_event),
                "rep_size": one_factor_uniforms(n_funds, rng, rho_size),
                "rc_event": one_factor_uniforms(n_funds, rng, rho_event),
                "rc_size": one_factor_uniforms(n_funds, rng, rho_size),
            }

            for i, fid in enumerate(fund_ids):
                st = state[fid]
                age_q = int(st["age0"] + t + 1)
                age_bucket = make_age_bucket(age_q)
                strategy = st["strategy"]
                grade = st["grade"]

                # grade transition yearly
                if t > 0 and t % 4 == 0 and trans_all is not None:
                    P = trans_all.reindex(index=GRADE_STATES, columns=GRADE_STATES).fillna(0.0)
                    row = P.loc[grade].to_numpy(dtype=float)
                    if row.sum() > 0:
                        row = row / row.sum()
                        grade = str(rng.choice(GRADE_STATES, p=row))
                        st["grade"] = grade

                # timing probabilities
                key = (strategy, grade, age_bucket)
                tp = timing_sel.get(key, None)
                if tp is None:
                    # fallback to global
                    tp = next(iter([v for k, v in timing_sel.items() if k[0] == strategy]), None)
                if tp is None:
                    p_draw = 0.0
                    p_rep = 0.0
                    p_rc = 0.0
                else:
                    p_draw = float(tp.get("p_draw", 0.0))
                    p_rep = float(tp.get("p_rep", 0.0))
                    p_rc = float(tp.get("p_rc_given_rep", 0.0))

                draw_event = U["draw_event"][i] < p_draw
                rep_event = U["rep_event"][i] < p_rep

                # draw ratio distribution
                rkey = (strategy, grade, age_bucket, "draw_ratio")
                rr = ratio_sel.get(rkey)
                draw_ratio = 0.0
                if draw_event and rr is not None:
                    dist = rr.get("dist")
                    params = rr.get("params")
                    try:
                        params = ast.literal_eval(params) if isinstance(params, str) else params
                        draw_ratio = sample_from_dist(dist, params, float(U["draw_size"][i]))
                    except Exception:
                        draw_ratio = 0.0
                draw_ratio = float(np.clip(draw_ratio, 0.0, 1.0))

                # capacity
                ledger = st["ledger"]
                rc_avail = ledger.available(t)
                remaining_commit = max(st["commitment"] - st["dd_commit"], 0.0)
                capacity = remaining_commit + rc_avail

                draw_amt = draw_ratio * capacity
                cons = ledger.consume_for_drawdown(t, draw_amt)
                st["dd_commit"] += cons["use_commitment"]

                # repayment ratio distribution
                NAV_prev = float(st["nav"])
                rep_ratio = 0.0
                if rep_event and NAV_prev > 1.0:
                    rkey = (strategy, grade, age_bucket, "rep_ratio")
                    rr = ratio_sel.get(rkey)
                    if rr is not None:
                        dist = rr.get("dist")
                        params = rr.get("params")
                        try:
                            params = ast.literal_eval(params) if isinstance(params, str) else params
                            rep_ratio = sample_from_dist(dist, params, float(U["rep_size"][i]))
                        except Exception:
                            rep_ratio = 0.0
                rep_ratio = float(np.clip(rep_ratio, 0.0, 1.0))
                rep_amt = rep_ratio * NAV_prev

                # recallable
                rc_amt = 0.0
                if rep_amt > 0 and (U["rc_event"][i] < p_rc):
                    rkey = (strategy, grade, age_bucket, "rc_ratio_given_rep")
                    rr = ratio_sel.get(rkey)
                    if rr is not None:
                        dist = rr.get("dist")
                        params = rr.get("params")
                        try:
                            params = ast.literal_eval(params) if isinstance(params, str) else params
                            rc_ratio = sample_from_dist(dist, params, float(U["rc_size"][i]))
                            rc_ratio = float(np.clip(rc_ratio, 0.0, 1.0))
                            rc_amt = ledger.add_recallable(t, rc_ratio * rep_amt, enforce_cap=True)
                        except Exception:
                            rc_amt = 0.0

                # NAV update
                nav_after_flow = max(NAV_prev + draw_amt - rep_amt, 0.0)
                if args.omega_mode == "global" and omega_sig > 0:
                    omega = float(rng.normal(omega_mu, omega_sig))
                else:
                    omega = 0.0
                nav_after = max(nav_after_flow * (1.0 + omega), 0.0)
                st["nav"] = nav_after

                sim_draw[s, t] += draw_amt
                sim_rep[s, t] += rep_amt
                sim_nav[s, t] += nav_after

    out_dir = os.path.join(args.fit_dir, "sim_outputs")
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({
        "quarter_end": quarters,
        "sim_draw_mean": sim_draw.mean(axis=0),
        "sim_rep_mean": sim_rep.mean(axis=0),
        "sim_nav_mean": sim_nav.mean(axis=0),
    }).to_csv(os.path.join(out_dir, "sim_portfolio_series.csv"), index=False)

    print("Wrote:", os.path.join(out_dir, "sim_portfolio_series.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
