from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from .utils import make_age_bucket


CANONICAL_RENAME: Dict[str, str] = {
    "Adj strategy": "Adj Strategy",
    "Adj Strategy": "Adj Strategy",
    "Year of Transaction Date": "Year",
    "Quarter of Transaction Date": "Quarter",
    "Planned end date with add. years as per legal doc": "Planned End Date",
    "First Closing Date": "First Closing Date",
    "NAV Adjusted EUR": "NAV Adjusted EUR",
    "Adj Drawdown EUR": "Adj Drawdown EUR",
    "Adj Repayment EUR": "Adj Repayment EUR",
    "Recallable": "Recallable",
    "Transaction Quarter": "Transaction Quarter",
    "Fund Workflow Stage": "Fund Workflow Stage",
    "Fund_Age_Quarters": "Fund_Age_Quarters",
    "Target Fund Size": "Target Fund Size",
    "Grade": "Grade",
    "P-Grade": "Grade",
    "P Grade": "Grade",
}


REQUIRED_COLUMNS = (
    "FundID",
    "Adj Strategy",
    "Grade",
    "Year",
    "Quarter",
    "First Closing Date",
    "Planned End Date",
    "Commitment EUR",
    "Adj Drawdown EUR",
    "Adj Repayment EUR",
    "Recallable",
    "NAV Adjusted EUR",
)


@dataclass
class SchemaValidationResult:
    missing_columns: list[str]

    @property
    def ok(self) -> bool:
        return not self.missing_columns


def parse_quarter(value: object) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    token = str(value).strip().upper()
    if token.startswith("Q"):
        token = token[1:]
    try:
        return float(token)
    except Exception:
        return np.nan


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.rename(columns=CANONICAL_RENAME).copy()
    if "Year" not in renamed.columns and "Year of Transaction Date" in df.columns:
        renamed["Year"] = df["Year of Transaction Date"]
    if "Quarter" not in renamed.columns and "Quarter of Transaction Date" in df.columns:
        renamed["Quarter"] = df["Quarter of Transaction Date"]
    return renamed


def validate_required_columns(df: pd.DataFrame) -> SchemaValidationResult:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return SchemaValidationResult(missing_columns=missing)


def add_quarter_end(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Quarter"] = out["Quarter"].apply(parse_quarter)
    out["Year"] = pd.to_numeric(out["Year"], errors="coerce")
    mask = out["Year"].notna() & out["Quarter"].notna()
    if mask.any():
        out.loc[mask, "quarter_end"] = pd.PeriodIndex.from_fields(
            year=out.loc[mask, "Year"].astype(int),
            quarter=out.loc[mask, "Quarter"].astype(int),
            freq="Q",
        ).to_timestamp("Q")
    return out


def normalize_core_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["FundID"] = out["FundID"].astype(str).str.strip()
    out["Adj Strategy"] = out["Adj Strategy"].astype(str).str.strip().replace({"": "Unknown"})
    out["Grade"] = out["Grade"].astype(str).str.strip().replace({"": "D", "nan": "D", "None": "D"})

    for c in ["Adj Drawdown EUR", "Adj Repayment EUR", "Recallable", "NAV Adjusted EUR", "Commitment EUR", "Target Fund Size"]:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    for c in ["First Closing Date", "Planned End Date"]:
        if c not in out.columns:
            out[c] = pd.NaT
        out[c] = pd.to_datetime(out[c], errors="coerce")

    if "Fund_Age_Quarters" not in out.columns:
        out["Fund_Age_Quarters"] = 0
    out["Fund_Age_Quarters"] = pd.to_numeric(out["Fund_Age_Quarters"], errors="coerce").fillna(0).astype(int)

    out["AgeBucket"] = out["Fund_Age_Quarters"].apply(make_age_bucket)
    return out
