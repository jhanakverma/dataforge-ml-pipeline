"""
Phase 2 — Data Doctor
Statistical imputation, outlier correction, format standardization.
"""

import pandas as pd
import numpy as np
import re
from typing import List, Tuple
from dataclasses import dataclass

from phase1_detective import is_missing, infer_type

MISSING_TOKENS = {"null", "n/a", "none", "na", "nan", "", "undefined", "missing"}


@dataclass
class Fix:
    column: str
    row_index: int
    issue: str
    old_value: str
    new_value: str
    strategy: str


def _median(nums: List[float]) -> float:
    s = sorted(nums)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m-1] + s[m]) / 2


def _mode(values: List[str]) -> str:
    freq = {}
    for v in values:
        freq[v] = freq.get(v, 0) + 1
    return max(freq, key=freq.get)


def _standardize_date(val: str) -> str:
    """Convert MM/DD/YYYY -> YYYY-MM-DD."""
    match = re.match(r"(\d{2})/(\d{2})/(\d{4})", val.strip())
    if match:
        m, d, y = match.groups()
        return f"{y}-{m}-{d}"
    return val


def _is_outlier(val: float, nums: List[float]) -> bool:
    if len(nums) < 5:
        return False
    mean = np.mean(nums)
    std = np.std(nums)
    return abs(val - mean) > 2.5 * std


def fix_column(df: pd.DataFrame, col: str) -> Tuple[pd.DataFrame, List[Fix]]:
    fixes = []
    values = df[col].tolist()
    col_type = infer_type(pd.Series(values))
    clean_vals = [str(v).strip() for v in values if not is_missing(v)]
    clean_nums = []
    for v in clean_vals:
        try:
            clean_nums.append(float(str(v).replace(",", "")))
        except ValueError:
            pass

    new_values = list(values)

    for i, val in enumerate(values):
        str_val = str(val).strip() if val is not None else ""

        # ── Missing value ──────────────────────────────────────────────────
        if is_missing(val):
            if col_type in ("INTEGER", "FLOAT") and clean_nums:
                med = _median(clean_nums)
                new_val = str(int(med) if col_type == "INTEGER" else round(med, 2))
                strategy = f"Filled with median ({new_val})"
            elif col_type == "CATEGORY" and clean_vals:
                new_val = _mode(clean_vals)
                strategy = f"Filled with mode ('{new_val}')"
            elif col_type == "EMAIL":
                new_val = f"unknown_{i+1}@placeholder.com"
                strategy = "Replaced with placeholder email"
            elif col_type == "DATE":
                new_val = "1970-01-01"
                strategy = "Replaced with default date (1970-01-01)"
            else:
                new_val = "Unknown"
                strategy = "Filled with 'Unknown'"

            fixes.append(Fix(col, i, "Missing value", str_val or "(empty)", new_val, strategy))
            new_values[i] = new_val

        # ── Invalid email ──────────────────────────────────────────────────
        elif col_type == "EMAIL" and not re.match(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", str_val):
            new_val = f"fixed_{i+1}@placeholder.com"
            fixes.append(Fix(col, i, "Invalid email", str_val, new_val, "Replaced with placeholder"))
            new_values[i] = new_val

        # ── Date format standardization ────────────────────────────────────
        elif col_type == "DATE" and re.match(r"\d{2}/\d{2}/\d{4}", str_val):
            new_val = _standardize_date(str_val)
            fixes.append(Fix(col, i, "Non-standard date format", str_val, new_val, "MM/DD/YYYY → YYYY-MM-DD"))
            new_values[i] = new_val

        # ── Outlier ────────────────────────────────────────────────────────
        elif col_type in ("INTEGER", "FLOAT") and clean_nums:
            try:
                num = float(str_val.replace(",", ""))
                if _is_outlier(num, clean_nums):
                    med = _median(clean_nums)
                    new_val = str(int(med) if col_type == "INTEGER" else round(med, 2))
                    fixes.append(Fix(col, i, "Statistical outlier (>2.5σ)", str_val, new_val,
                                     f"Replaced with median ({new_val})"))
                    new_values[i] = new_val
            except ValueError:
                pass

    df = df.copy()
    df[col] = new_values
    return df, fixes


def run_doctor(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Fix]]:
    all_fixes = []
    for col in df.columns:
        df, fixes = fix_column(df, col)
        all_fixes.extend(fixes)
    return df, all_fixes
