from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schema import add_quarter_end, canonicalize_columns, normalize_core_types, validate_required_columns
from .utils import parse_quarter_text


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        # Historical data is semicolon; test portfolio CSV is comma.
        try:
            df = pd.read_csv(path, sep=";")
            if len(df.columns) == 1:
                return pd.read_csv(path)
            return df
        except Exception:
            return pd.read_csv(path)
    raise ValueError(f"Unsupported file format: {path}")


def load_cashflow_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    df = _read_table(p)
    df = canonicalize_columns(df)
    v = validate_required_columns(df)
    if not v.ok:
        raise ValueError(f"Missing required columns: {v.missing_columns}")
    df = add_quarter_end(df)
    df = normalize_core_types(df)
    df = df.dropna(subset=["FundID", "quarter_end"]).sort_values(["FundID", "quarter_end"]).reset_index(drop=True)
    return df


def slice_to_cutoff(df: pd.DataFrame, cutoff_quarter: str) -> pd.DataFrame:
    cutoff = parse_quarter_text(cutoff_quarter)
    return df[df["quarter_end"] <= cutoff].copy()


def infer_test_portfolio_csv(input_path: str | Path, output_csv: str | Path) -> Path:
    src = Path(input_path)
    out = Path(output_csv)
    if src.suffix.lower() == ".csv":
        return src
    df = load_cashflow_table(src)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out
