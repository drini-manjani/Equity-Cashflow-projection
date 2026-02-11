#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ecf_model.calibration import load_calibration_artifacts
from ecf_model.config import PipelineConfig
from ecf_model.data_io import load_cashflow_table
from ecf_model.pipeline import run_fit_pipeline, run_projection_pipeline
from ecf_model.schema import REQUIRED_COLUMNS
from ecf_model.tuning import run_neutral_tuning_pipeline
from ecf_model.utils import parse_quarter_text

TIMING_PROFILE_V2: dict[str, float] = {
    "rep_scale": 1.025185,
    "far_relax": 0.707559,
    "end_relax": 0.811656,
    "tail_scale": 0.166693,
    "late_scale": 0.649157,
}


def _resolve(path_text: str) -> Path:
    p = Path(path_text)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def _can_load_as_canonical(path: Path) -> bool:
    try:
        _ = load_cashflow_table(path)
        return True
    except Exception:
        return False


def _snapshot_to_canonical(
    input_path: Path,
    out_path: Path,
    cutoff_quarter: str,
    planned_life_years: int,
) -> Path:
    if input_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(input_path)
    else:
        df = pd.read_csv(input_path)

    if set(REQUIRED_COLUMNS).issubset(set(df.columns)):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        return out_path

    required_snapshot = ["VC Fund Name", "Strategy", "Grading", "First Closing Date", "Commitment amount"]
    missing = [c for c in required_snapshot if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input '{input_path}' is neither canonical nor supported snapshot template. Missing: {missing}"
        )

    cutoff_qe = parse_quarter_text(cutoff_quarter)
    year = int(cutoff_qe.year)
    quarter = int(((cutoff_qe.month - 1) // 3) + 1)
    cutoff_ord = int(pd.Period(cutoff_qe, freq="Q").ordinal)

    fc = pd.to_datetime(df["First Closing Date"], errors="coerce")
    fc_qe = fc.dt.to_period("Q").dt.to_timestamp("Q")

    fund_age_q = []
    for x in fc_qe:
        if pd.isna(x):
            fund_age_q.append(0)
            continue
        age = cutoff_ord - int(pd.Period(x, freq="Q").ordinal)
        fund_age_q.append(int(max(age, 0)))

    commit = pd.to_numeric(df["Commitment amount"], errors="coerce").fillna(0.0)
    draw = pd.to_numeric(df.get("Drawn amount", 0.0), errors="coerce").fillna(0.0)
    rep = pd.to_numeric(df.get("Reflows amount", 0.0), errors="coerce").fillna(0.0)
    nav = pd.to_numeric(df.get("Adjusted NAV", 0.0), errors="coerce").fillna(0.0)
    target_size = pd.to_numeric(df.get("Target Fund Size", commit), errors="coerce").fillna(commit)

    planned_end = fc + pd.DateOffset(years=int(planned_life_years))

    out = pd.DataFrame(
        {
            "FundID": df["VC Fund Name"].astype(str).str.strip(),
            "Year": year,
            "Quarter": quarter,
            "Adj Strategy": df["Strategy"].astype(str).str.strip(),
            "VC Fund Status": "Signed",
            "Fund Workflow Stage": "Signed",
            "First Closing Date": fc,
            "Planned End Date": planned_end,
            "Transaction Quarter": f"{year}-Q{quarter}",
            "Commitment EUR": commit,
            "Signed Amount EUR": commit,
            "Target Fund Size": target_size,
            "Adj Drawdown EUR": draw,
            "Adj Repayment EUR": rep,
            "Recallable": 0.0,
            "NAV Adjusted EUR": nav,
            "Grade": df["Grading"].astype(str).str.strip(),
            "Fund_Age_Quarters": fund_age_q,
        }
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out_path


def _prepare_portfolio_input(
    input_path: Path,
    cutoff_quarter: str,
    prepared_dir: Path,
    planned_life_years: int,
) -> Path:
    if _can_load_as_canonical(input_path):
        return input_path
    out_name = f"{input_path.stem}_{cutoff_quarter}.csv"
    out_path = prepared_dir / out_name
    return _snapshot_to_canonical(
        input_path=input_path,
        out_path=out_path,
        cutoff_quarter=cutoff_quarter,
        planned_life_years=planned_life_years,
    )


def _scale_repayment_intensity_inplace(group_deltas: pd.DataFrame, scale: float) -> pd.DataFrame:
    gd = group_deltas.copy()
    if gd.empty:
        return gd
    rr = pd.to_numeric(gd.get("delta_rep_ratio_scale", 1.0), errors="coerce").fillna(1.0)
    pp = pd.to_numeric(gd.get("delta_p_rep_mult", 1.0), errors="coerce").fillna(1.0)
    om = pd.to_numeric(gd.get("delta_omega_bump", 0.0), errors="coerce").fillna(0.0)
    gd["delta_rep_ratio_scale"] = np.clip(1.0 + (rr - 1.0) * float(scale), 0.70, 1.90)
    gd["delta_p_rep_mult"] = np.clip(1.0 + (pp - 1.0) * float(scale), 0.75, 1.40)
    gd["delta_omega_bump"] = np.clip(om * float(scale), -0.10, 0.10)
    return gd


def apply_timing_profile_v2(calibration_dir: Path) -> None:
    cal = load_calibration_artifacts(calibration_dir)
    p = TIMING_PROFILE_V2

    cal.group_deltas = _scale_repayment_intensity_inplace(cal.group_deltas, p["rep_scale"])

    pre = cal.pre_end_repayment.copy()
    m_pre = pre["Adj Strategy"].isin(["Private Equity", "Venture Capital"]) & (pre["pre_end_phase"] == "far")
    pre.loc[m_pre, "rep_p_mult"] = pre.loc[m_pre, "rep_p_mult"].astype(float) + p["far_relax"] * (
        1.0 - pre.loc[m_pre, "rep_p_mult"].astype(float)
    )
    pre.loc[m_pre, "rep_ratio_mult"] = pre.loc[m_pre, "rep_ratio_mult"].astype(float) + p["far_relax"] * (
        1.0 - pre.loc[m_pre, "rep_ratio_mult"].astype(float)
    )
    cal.pre_end_repayment = pre

    eg = cal.endgame_ramp.copy()
    m_eg = eg["Adj Strategy"].isin(["Private Equity", "Venture Capital"])
    eg.loc[m_eg, "rep_ratio_mult"] = 1.0 + (
        eg.loc[m_eg, "rep_ratio_mult"].astype(float) - 1.0
    ) * (1.0 - p["end_relax"])
    cal.endgame_ramp = eg

    tail = cal.tail_floors.copy()
    m_tail = tail["Adj Strategy"].isin(["Private Equity", "Venture Capital"])
    tail.loc[m_tail, "p_rep_floor"] = np.clip(
        tail.loc[m_tail, "p_rep_floor"].astype(float) * p["tail_scale"],
        0.0,
        1.0,
    )
    tail.loc[m_tail, "rep_ratio_floor"] = np.clip(
        tail.loc[m_tail, "rep_ratio_floor"].astype(float) * p["tail_scale"],
        0.0,
        10.0,
    )
    cal.tail_floors = tail

    lp = cal.lifecycle_phase.copy()
    m_lp = lp["Adj Strategy"].isin(["Private Equity", "Venture Capital"]) & lp["life_phase"].isin(
        ["late", "final", "post_0_4", "post_5_8", "post_9_12", "post_12p"]
    )
    lp.loc[m_lp, "rep_p_mult"] = 1.0 + (lp.loc[m_lp, "rep_p_mult"].astype(float) - 1.0) * p["late_scale"]
    lp.loc[m_lp, "rep_ratio_mult"] = 1.0 + (lp.loc[m_lp, "rep_ratio_mult"].astype(float) - 1.0) * p[
        "late_scale"
    ]
    cal.lifecycle_phase = lp

    cal.save(calibration_dir)
    (calibration_dir / "timing_adjustment_applied.json").write_text(json.dumps(p, indent=2), encoding="utf-8")


def _build_cfg(
    cutoff: str,
    run_tag: str,
    scenario: str,
    seed: int,
    n_sims: int,
    volatility_scale: float,
    copula_enabled: bool,
    copula_rho: float | None,
) -> PipelineConfig:
    cfg = PipelineConfig(cutoff_quarter=cutoff, run_tag=run_tag, scenario=scenario)
    cfg.simulation.seed = int(seed)
    cfg.simulation.n_sims = int(n_sims)
    cfg.simulation.copula_enabled = bool(copula_enabled)
    cfg.simulation.copula_rho = float(copula_rho) if copula_rho is not None else None
    cfg.scenarios.volatility_scale = float(volatility_scale)
    return cfg


def run(args: argparse.Namespace) -> dict[str, Any]:
    cutoff = str(args.cutoff).strip().upper()
    hist_path = _resolve(args.hist)
    msci_path = _resolve(args.msci)
    portfolio_path = _resolve(args.portfolio)
    sentral_path = _resolve(args.portfolio_sentral)

    prepared_dir = PROJECT_ROOT / "inputs" / cutoff
    portfolio_input = _prepare_portfolio_input(
        input_path=portfolio_path,
        cutoff_quarter=cutoff,
        prepared_dir=prepared_dir,
        planned_life_years=int(args.planned_life_years),
    )
    sentral_input = _prepare_portfolio_input(
        input_path=sentral_path,
        cutoff_quarter=cutoff,
        prepared_dir=prepared_dir,
        planned_life_years=int(args.planned_life_years),
    )

    fit_tag = f"{cutoff}/_fit_base"
    primary_tag = f"{cutoff}/test_portfolio"
    sentral_tag = f"{cutoff}/test_portfolio_Sentral"

    fit_cfg = _build_cfg(
        cutoff=cutoff,
        run_tag=fit_tag,
        scenario="neutral",
        seed=int(args.seed),
        n_sims=int(args.n_sims_final),
        volatility_scale=float(args.volatility_scale),
        copula_enabled=bool(args.copula_enabled),
        copula_rho=args.copula_rho,
    )
    fit_out = run_fit_pipeline(hist_path=hist_path, msci_path=msci_path, cfg=fit_cfg)

    tune_cfg = _build_cfg(
        cutoff=cutoff,
        run_tag=primary_tag,
        scenario="neutral",
        seed=int(args.seed),
        n_sims=int(args.n_sims_final),
        volatility_scale=float(args.volatility_scale),
        copula_enabled=bool(args.copula_enabled),
        copula_rho=args.copula_rho,
    )
    tune_out = run_neutral_tuning_pipeline(
        hist_path=hist_path,
        portfolio_path=portfolio_input,
        base_calibration_dir=fit_cfg.calibration_dir,
        cfg=tune_cfg,
        n_sims_tune=int(args.n_sims_tune),
        n_iters=int(args.iters),
        n_sims_final=int(args.n_sims_final),
        irr_target_mode="pooled",
    ).to_dict()

    if bool(args.apply_timing_profile_v2):
        apply_timing_profile_v2(tune_cfg.calibration_dir)
        primary_proj_cfg = _build_cfg(
            cutoff=cutoff,
            run_tag=primary_tag,
            scenario="neutral",
            seed=int(args.seed),
            n_sims=int(args.n_sims_final),
            volatility_scale=float(args.volatility_scale),
            copula_enabled=bool(args.copula_enabled),
            copula_rho=args.copula_rho,
        )
        _ = run_projection_pipeline(
            portfolio_path=portfolio_input,
            calibration_dir=tune_cfg.calibration_dir,
            cfg=primary_proj_cfg,
        )

    sentral_cfg = _build_cfg(
        cutoff=cutoff,
        run_tag=sentral_tag,
        scenario="neutral",
        seed=int(args.seed),
        n_sims=int(args.n_sims_final),
        volatility_scale=float(args.volatility_scale),
        copula_enabled=bool(args.copula_enabled),
        copula_rho=args.copula_rho,
    )
    sentral_out = run_projection_pipeline(
        portfolio_path=sentral_input,
        calibration_dir=tune_cfg.calibration_dir,
        cfg=sentral_cfg,
    )

    if bool(args.cleanup_intermediate):
        shutil.rmtree(fit_cfg.run_root, ignore_errors=True)

    summary = {
        "project_root": str(PROJECT_ROOT),
        "cutoff": cutoff,
        "inputs": {
            "hist": str(hist_path),
            "msci": str(msci_path),
            "portfolio": str(portfolio_input),
            "portfolio_sentral": str(sentral_input),
        },
        "runs": {
            "test_portfolio": str((PROJECT_ROOT / "runs_v2" / primary_tag).resolve()),
            "test_portfolio_sentral": str((PROJECT_ROOT / "runs_v2" / sentral_tag).resolve()),
        },
        "fit": fit_out,
        "tune": tune_out,
        "sentral_projection": sentral_out,
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cross-platform quarter workflow (fit + tune + project).")
    p.add_argument("--cutoff", default="2025Q3", help="Quarter label, e.g. 2025Q3")
    p.add_argument("--hist", default="anonymized.csv", help="Historical anonymized dataset")
    p.add_argument("--msci", default="msci.xlsx", help="MSCI history file")
    p.add_argument("--portfolio", default="test_portfolio.xlsx", help="Primary portfolio input")
    p.add_argument(
        "--portfolio-sentral",
        default="tests/test_portfolio_Sentral.xlsx",
        help="Second portfolio input",
    )
    p.add_argument("--planned-life-years", type=int, default=10, help="Planned life used for snapshot templates")
    p.add_argument("--n-sims-tune", type=int, default=120, help="Sim count during tuning")
    p.add_argument("--n-sims-final", type=int, default=300, help="Sim count for final projections")
    p.add_argument("--iters", type=int, default=10, help="Neutral tuning iterations")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--copula-rho", type=float, default=None, help="One-factor copula loading within fund")
    p.add_argument("--copula-enabled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--volatility-scale", type=float, default=1.0)
    p.add_argument(
        "--apply-timing-profile-v2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the latest timing profile after neutral tuning",
    )
    p.add_argument(
        "--cleanup-intermediate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete the intermediate fit run after workflow completes",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    out = run(args)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
