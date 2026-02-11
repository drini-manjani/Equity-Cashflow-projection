from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from .config import PipelineConfig
from .backtest import run_backtest_suite
from .pipeline import run_fit_pipeline, run_projection_pipeline
from .tuning import run_neutral_tuning_pipeline


def _base_cfg(args: argparse.Namespace) -> PipelineConfig:
    cfg = PipelineConfig(
        cutoff_quarter=getattr(args, "cutoff", "2025Q3"),
        run_tag=getattr(args, "run_tag", "2025Q3_v2"),
        scenario=getattr(args, "scenario", "neutral"),
    )
    if getattr(args, "n_sims", None) is not None:
        cfg.simulation.n_sims = int(args.n_sims)
    if getattr(args, "seed", None) is not None:
        cfg.simulation.seed = int(args.seed)
    if getattr(args, "horizon_q", None) is not None:
        cfg.simulation.horizon_q = int(args.horizon_q)
    if getattr(args, "copula_rho", None) is not None:
        cfg.simulation.copula_rho = float(args.copula_rho)
    if hasattr(args, "copula_enabled") and getattr(args, "copula_enabled") is not None:
        cfg.simulation.copula_enabled = bool(args.copula_enabled)
    if getattr(args, "volatility_scale", None) is not None:
        cfg.scenarios.volatility_scale = float(args.volatility_scale)
    return cfg


def _cmd_fit(args: argparse.Namespace) -> int:
    cfg = _base_cfg(args)
    out = run_fit_pipeline(hist_path=args.hist, msci_path=args.msci, cfg=cfg)
    print(json.dumps(out, indent=2))
    return 0


def _cmd_project(args: argparse.Namespace) -> int:
    cfg = _base_cfg(args)
    out = run_projection_pipeline(portfolio_path=args.input, calibration_dir=args.calibration, cfg=cfg)
    print(json.dumps(out, indent=2))
    return 0


def _cmd_project_all(args: argparse.Namespace) -> int:
    base = _base_cfg(args)
    summaries = []
    for scenario in ["neutral", "bullish", "bearish"]:
        run_tag = f"{base.run_tag}_{scenario}"
        cfg = replace(base, run_tag=run_tag, scenario=scenario)
        cfg.simulation = base.simulation
        cfg.fit = base.fit
        cfg.calibration = base.calibration
        cfg.scenarios = base.scenarios
        summaries.append(run_projection_pipeline(portfolio_path=args.input, calibration_dir=args.calibration, cfg=cfg))
    print(json.dumps(summaries, indent=2))
    return 0


def _cmd_backtest(args: argparse.Namespace) -> int:
    cfg = _base_cfg(args)
    cutoffs = [c.strip() for c in str(args.cutoffs).split(",") if c.strip()]
    if not cutoffs:
        raise ValueError("No cutoffs provided")
    out = run_backtest_suite(
        hist_path=args.hist,
        msci_path=args.msci,
        cutoffs=cutoffs,
        eval_to_quarter=args.eval_to,
        cfg=cfg,
    )
    print(out.to_json(orient="records", indent=2))
    return 0


def _cmd_tune_neutral(args: argparse.Namespace) -> int:
    cfg = _base_cfg(args)
    cfg.scenario = "neutral"
    irr_target_mode = str(getattr(args, "irr_target_mode", "pooled")).strip().lower().replace("-", "_")
    out = run_neutral_tuning_pipeline(
        hist_path=args.hist,
        portfolio_path=args.input,
        base_calibration_dir=args.calibration,
        cfg=cfg,
        n_sims_tune=int(args.n_sims_tune),
        n_iters=int(args.iters),
        n_sims_final=int(args.n_sims_final),
        irr_target_mode=irr_target_mode,
    )
    print(json.dumps(out.to_dict(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ecf", description="Equity cashflow projection pipeline")
    sub = p.add_subparsers(dest="command")

    fit = sub.add_parser("fit", help="Fit model artifacts and calibration")
    fit.add_argument("--hist", required=True, help="Historical cashflow table (csv/xlsx)")
    fit.add_argument("--msci", required=True, help="MSCI index file (xlsx)")
    fit.add_argument("--cutoff", default="2025Q3", help="Historical cutoff quarter, e.g. 2025Q3")
    fit.add_argument("--run-tag", default="2025Q3_v2", help="Run tag written under runs_v2/")
    fit.set_defaults(func=_cmd_fit)

    proj = sub.add_parser("project", help="Project portfolio using fitted artifacts")
    proj.add_argument("--input", required=True, help="Portfolio snapshot/history table (csv/xlsx)")
    proj.add_argument("--calibration", required=True, help="Calibration directory from fit run")
    proj.add_argument("--scenario", choices=["neutral", "bullish", "bearish"], default="neutral")
    proj.add_argument("--cutoff", default="2025Q3")
    proj.add_argument("--run-tag", default="projection_v2")
    proj.add_argument("--n-sims", type=int, default=200)
    proj.add_argument("--seed", type=int, default=1234)
    proj.add_argument("--horizon-q", type=int, default=24)
    proj.add_argument("--copula-rho", type=float, default=None)
    proj.add_argument("--copula-enabled", action=argparse.BooleanOptionalAction, default=True)
    proj.add_argument("--volatility-scale", type=float, default=1.0)
    proj.set_defaults(func=_cmd_project)

    proj_all = sub.add_parser("project-all", help="Run neutral/bullish/bearish projection")
    proj_all.add_argument("--input", required=True)
    proj_all.add_argument("--calibration", required=True)
    proj_all.add_argument("--cutoff", default="2025Q3")
    proj_all.add_argument("--run-tag", default="projection_v2")
    proj_all.add_argument("--n-sims", type=int, default=200)
    proj_all.add_argument("--seed", type=int, default=1234)
    proj_all.add_argument("--horizon-q", type=int, default=24)
    proj_all.add_argument("--copula-rho", type=float, default=None)
    proj_all.add_argument("--copula-enabled", action=argparse.BooleanOptionalAction, default=True)
    proj_all.add_argument("--volatility-scale", type=float, default=1.0)
    proj_all.set_defaults(func=_cmd_project_all)

    bt = sub.add_parser("backtest", help="Run walk-forward backtest suite")
    bt.add_argument("--hist", required=True, help="Historical cashflow table (csv/xlsx)")
    bt.add_argument("--msci", required=True, help="MSCI index file (xlsx)")
    bt.add_argument("--cutoffs", required=True, help="Comma-separated cutoff quarters, e.g. 2019Q3,2021Q3")
    bt.add_argument("--eval-to", default="2025Q3", help="Evaluation end quarter")
    bt.add_argument("--scenario", choices=["neutral", "bullish", "bearish"], default="neutral")
    bt.add_argument("--run-tag", default="backtest_suite")
    bt.add_argument("--n-sims", type=int, default=20)
    bt.add_argument("--seed", type=int, default=1234)
    bt.add_argument("--horizon-q", type=int, default=24)
    bt.add_argument("--copula-rho", type=float, default=None)
    bt.add_argument("--copula-enabled", action=argparse.BooleanOptionalAction, default=True)
    bt.add_argument("--volatility-scale", type=float, default=1.0)
    bt.set_defaults(func=_cmd_backtest)

    tune = sub.add_parser("tune-neutral", help="Tune neutral calibration to historical targets")
    tune.add_argument("--hist", required=True, help="Historical cashflow table (csv/xlsx)")
    tune.add_argument("--input", required=True, help="Portfolio snapshot/history table (csv/xlsx)")
    tune.add_argument("--calibration", required=True, help="Base calibration directory from fit run")
    tune.add_argument("--cutoff", default="2025Q3")
    tune.add_argument("--run-tag", default="neutral_tuned")
    tune.add_argument("--n-sims-tune", type=int, default=120, help="Simulation count during tuning iterations")
    tune.add_argument("--n-sims-final", type=int, default=300, help="Final neutral projection simulation count")
    tune.add_argument("--iters", type=int, default=10, help="Number of tuning iterations")
    tune.add_argument(
        "--irr-target-mode",
        choices=["pooled", "mix-median"],
        default="pooled",
        help="IRR target source: pooled historical cashflows (recommended) or mix-weighted median fund IRR",
    )
    tune.add_argument("--seed", type=int, default=1234)
    tune.add_argument("--horizon-q", type=int, default=24)
    tune.add_argument("--copula-rho", type=float, default=None)
    tune.add_argument("--copula-enabled", action=argparse.BooleanOptionalAction, default=True)
    tune.add_argument("--volatility-scale", type=float, default=1.0)
    tune.set_defaults(func=_cmd_tune_neutral)

    return p


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    # Allow ecf-fit/ecf-project script entry without explicit subcommand.
    prog = Path(sys.argv[0]).name
    if argv and argv[0] in {"fit", "project", "project-all"}:
        pass
    elif prog.endswith("ecf-fit"):
        argv = ["fit", *argv]
    elif prog.endswith("ecf-project"):
        argv = ["project", *argv]

    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
