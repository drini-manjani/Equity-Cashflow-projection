from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import ensure_dir, parse_quarter_text


@dataclass
class MSCIModel:
    last_level: float
    overall_mu: float
    phi: float
    q1: float
    q2: float
    regime_mu: dict[int, float]
    regime_sigma: dict[int, float]
    transition: dict[int, list[float]]

    def to_dict(self) -> dict:
        return {
            "last_level": self.last_level,
            "overall_mu": self.overall_mu,
            "phi": self.phi,
            "q1": self.q1,
            "q2": self.q2,
            "regime_mu": self.regime_mu,
            "regime_sigma": self.regime_sigma,
            "transition": self.transition,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "MSCIModel":
        return cls(
            last_level=float(payload["last_level"]),
            overall_mu=float(payload["overall_mu"]),
            phi=float(payload["phi"]),
            q1=float(payload["q1"]),
            q2=float(payload["q2"]),
            regime_mu={int(k): float(v) for k, v in payload["regime_mu"].items()},
            regime_sigma={int(k): float(v) for k, v in payload["regime_sigma"].items()},
            transition={int(k): [float(x) for x in v] for k, v in payload["transition"].items()},
        )


def _detect_date_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if "date" in str(c).lower():
            return c
    return df.columns[0]


def _detect_level_column(df: pd.DataFrame, date_col: str) -> str:
    candidates = [c for c in df.columns if c != date_col and np.issubdtype(df[c].dtype, np.number)]
    for c in candidates:
        if "index" in str(c).lower() or "msci" in str(c).lower() or "scxp" in str(c).lower():
            return c
    if candidates:
        return candidates[0]
    raise ValueError("Could not detect MSCI level column")


def load_msci_quarterly(msci_path: str | Path) -> pd.DataFrame:
    p = Path(msci_path)
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_excel(p)
    dcol = _detect_date_column(df)
    lcol = _detect_level_column(df, dcol)

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[dcol], errors="coerce"),
            "index_level": pd.to_numeric(df[lcol], errors="coerce"),
        }
    ).dropna()

    out["quarter_end"] = out["date"].dt.to_period("Q").dt.to_timestamp("Q")
    q = out.groupby("quarter_end", as_index=False)["index_level"].last().sort_values("quarter_end")
    q["msci_ret_q"] = q["index_level"].pct_change().fillna(0.0)
    q["msci_ret_q_lag1"] = q["msci_ret_q"].shift(1).fillna(0.0)
    return q


def fit_msci_model(msci_q: pd.DataFrame, cutoff_quarter: str) -> MSCIModel:
    cutoff_qe = parse_quarter_text(cutoff_quarter)
    fit = msci_q[msci_q["quarter_end"] <= cutoff_qe].copy().sort_values("quarter_end")
    if len(fit) < 16:
        raise ValueError("Not enough MSCI history before cutoff to fit model")

    r = fit["msci_ret_q"].to_numpy(dtype=float)
    lag = np.roll(r, 1)
    lag[0] = 0.0

    # AR(1) coefficient with clipping for stability.
    denom = float(np.dot(lag, lag))
    phi = float(np.dot(lag, r) / denom) if denom > 0 else 0.0
    phi = float(np.clip(phi, -0.75, 0.75))

    q1, q2 = float(np.quantile(r, 0.33)), float(np.quantile(r, 0.67))

    def regime_of(x: float) -> int:
        if x <= q1:
            return 0  # bearish regime
        if x >= q2:
            return 2  # bullish regime
        return 1  # neutral regime

    regimes = np.array([regime_of(x) for x in r], dtype=int)
    overall_mu = float(np.mean(r))

    regime_mu: dict[int, float] = {}
    regime_sigma: dict[int, float] = {}
    for k in (0, 1, 2):
        vals = r[regimes == k]
        if len(vals) < 4:
            regime_mu[k] = overall_mu
            regime_sigma[k] = float(np.std(r, ddof=1))
        else:
            regime_mu[k] = float(np.mean(vals))
            regime_sigma[k] = float(np.std(vals, ddof=1))
        regime_sigma[k] = max(regime_sigma[k], 1e-4)

    transition = {k: [1 / 3, 1 / 3, 1 / 3] for k in (0, 1, 2)}
    counts = {k: np.zeros(3, dtype=float) for k in (0, 1, 2)}
    for prev, nxt in zip(regimes[:-1], regimes[1:]):
        counts[int(prev)][int(nxt)] += 1.0
    for k in (0, 1, 2):
        row = counts[k] + 1.0  # Laplace smoothing
        transition[k] = (row / row.sum()).tolist()

    return MSCIModel(
        last_level=float(fit["index_level"].iloc[-1]),
        overall_mu=overall_mu,
        phi=phi,
        q1=q1,
        q2=q2,
        regime_mu=regime_mu,
        regime_sigma=regime_sigma,
        transition=transition,
    )


def _scenario_shift(scenario: str, bullish_shift: float = 0.0075, bearish_shift: float = -0.0075, neutral_shift: float = 0.0) -> float:
    sc = scenario.lower()
    if sc == "bullish":
        return bullish_shift
    if sc == "bearish":
        return bearish_shift
    return neutral_shift


def _transition_for_scenario(base_row: np.ndarray, scenario: str) -> np.ndarray:
    row = base_row.copy()
    if scenario == "bullish":
        row[2] += 0.08
        row[0] -= 0.08
    elif scenario == "bearish":
        row[0] += 0.08
        row[2] -= 0.08
    row = np.clip(row, 0.01, None)
    return row / row.sum()


def project_msci_paths(
    model: MSCIModel,
    start_quarter: str,
    horizon_q: int,
    n_sims: int,
    seed: int,
    scenario: str,
    drift_shift_q: float | None = None,
    volatility_scale: float = 1.0,
) -> pd.DataFrame:
    start_qe = parse_quarter_text(start_quarter)
    future_qe = pd.period_range(start=start_qe.to_period("Q") + 1, periods=horizon_q, freq="Q").to_timestamp("Q")

    rng = np.random.default_rng(seed)
    rows = []
    drift_shift = float(drift_shift_q) if drift_shift_q is not None else _scenario_shift(scenario)

    base_transition = {k: np.array(v, dtype=float) for k, v in model.transition.items()}
    trans = {k: _transition_for_scenario(base_transition[k], scenario.lower()) for k in (0, 1, 2)}

    for sim in range(1, n_sims + 1):
        level = model.last_level
        prev_ret = model.overall_mu
        regime = 1
        for qe in future_qe:
            p = trans.get(regime, np.array([1 / 3, 1 / 3, 1 / 3], dtype=float))
            regime = int(rng.choice([0, 1, 2], p=p))

            mu = model.regime_mu[regime] + drift_shift
            sig = model.regime_sigma[regime] * volatility_scale
            shock = rng.normal(0.0, sig)
            ret_q = mu + model.phi * (prev_ret - model.overall_mu) + shock
            prev_ret = ret_q
            level = level * (1.0 + ret_q)

            rows.append(
                {
                    "sim_id": sim,
                    "quarter_end": qe,
                    "msci_ret_q": float(ret_q),
                    "index_level": float(level),
                }
            )

    out = pd.DataFrame(rows)
    out = out.sort_values(["sim_id", "quarter_end"]).reset_index(drop=True)
    out["msci_ret_q_lag1"] = out.groupby("sim_id")["msci_ret_q"].shift(1).fillna(0.0)
    return out


def save_msci_model(path: Path, model: MSCIModel) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(model.to_dict(), f, indent=2)


def load_msci_model(path: Path) -> MSCIModel:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return MSCIModel.from_dict(payload)
