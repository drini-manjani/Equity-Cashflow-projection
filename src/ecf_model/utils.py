from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


AGE_BUCKETS: Sequence[tuple[int, int | None, str]] = (
    (0, 3, "0-3"),
    (4, 7, "4-7"),
    (8, 11, "8-11"),
    (12, 15, "12-15"),
    (16, 19, "16-19"),
    (20, None, "20+"),
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_quarter_text(value: str) -> pd.Timestamp:
    token = value.strip().upper().replace("-", "")
    if len(token) != 6 or not token.startswith("20") or "Q" not in token:
        raise ValueError(f"Invalid quarter token: {value}")
    year = int(token[:4])
    q = int(token[-1])
    return pd.Period(year=year, quarter=q, freq="Q").to_timestamp("Q")


def format_quarter(qe: pd.Timestamp) -> str:
    p = pd.Period(qe, freq="Q")
    return f"{p.year}Q{p.quarter}"


def make_age_bucket(age_q: int | float | None) -> str:
    if age_q is None or not np.isfinite(float(age_q)):
        return "20+"
    age = int(max(float(age_q), 0))
    for lo, hi, label in AGE_BUCKETS:
        if hi is None:
            if age >= lo:
                return label
        elif lo <= age <= hi:
            return label
    return "20+"


def recallable_to_flow(series: pd.Series) -> pd.Series:
    """
    Convert recallable series to per-period flow.

    If the series is non-decreasing (cumulative-style), use first difference
    (floored at 0). Otherwise treat it as already a flow and take abs().
    """
    vals = pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if len(vals) <= 1:
        return pd.Series(np.abs(vals), index=series.index)

    diffs = np.diff(vals, prepend=vals[0])
    if (np.diff(vals) >= -1e-6).all():
        out = np.maximum(diffs, 0.0)
    else:
        out = np.abs(vals)
    return pd.Series(out, index=series.index)


def assign_size_bucket(value: float, bins: Sequence[float], labels: Sequence[str]) -> str:
    if not np.isfinite(value):
        return labels[-1]
    idx = np.digitize([value], bins[1:-1], right=True)[0]
    return labels[int(idx)]


def json_dumps(value: Any) -> str:
    if is_dataclass(value):
        payload = asdict(value)
    else:
        payload = value
    return json.dumps(payload, sort_keys=True)


def json_dump_file(path: Path, value: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        if is_dataclass(value):
            json.dump(asdict(value), f, indent=2, sort_keys=True)
        else:
            json.dump(value, f, indent=2, sort_keys=True)


def json_load_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def capped_ratio(num: float, den: float, min_den: float = 1e-9) -> float:
    if not np.isfinite(num) or not np.isfinite(den):
        return np.nan
    if den <= min_den:
        return np.nan
    return float(num / den)


def clip01(x: float) -> float:
    if not np.isfinite(x):
        return 0.0
    return float(np.clip(x, 0.0, 1.0))


def winsorize_series(values: pd.Series, lower_q: float, upper_q: float) -> pd.Series:
    if values.empty:
        return values
    lo = values.quantile(lower_q)
    hi = values.quantile(upper_q)
    return values.clip(lower=lo, upper=hi)


def weighted_mean(values: Iterable[float], weights: Iterable[float]) -> float:
    vals = np.asarray(list(values), dtype=float)
    w = np.asarray(list(weights), dtype=float)
    m = np.isfinite(vals) & np.isfinite(w) & (w > 0)
    if not m.any():
        return np.nan
    return float(np.sum(vals[m] * w[m]) / np.sum(w[m]))
