"""
Phase 1 — Data Detective
Automatically infers schema: column types, missing values, format issues.
"""

import pandas as pd
import numpy as np
import re
from dataclasses import dataclass, field
from typing import List, Optional

MISSING_TOKENS = {"null", "n/a", "none", "na", "nan", "", "undefined", "missing"}


def is_missing(val) -> bool:
    if val is None:
        return True
    return str(val).strip().lower() in MISSING_TOKENS


@dataclass
class ColumnProfile:
    name: str
    inferred_type: str          # INTEGER, FLOAT, DATE, EMAIL, CATEGORY, TEXT
    total: int
    missing_count: int
    unique_count: int
    issues: List[str] = field(default_factory=list)
    sample_values: List[str] = field(default_factory=list)

    @property
    def missing_pct(self) -> float:
        return round(self.missing_count / self.total * 100, 1) if self.total else 0


def infer_type(values: pd.Series) -> str:
    clean = [str(v).strip() for v in values if not is_missing(v)]
    if not clean:
        return "TEXT"

    # Email
    email_re = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
    if all(email_re.match(v) for v in clean):
        return "EMAIL"

    # Date
    date_patterns = [
        r"\d{4}-\d{2}-\d{2}",
        r"\d{2}/\d{2}/\d{4}",
        r"\d{2}-\d{2}-\d{4}",
    ]
    date_hits = sum(
        1 for v in clean if any(re.search(p, v) for p in date_patterns)
    )
    if date_hits / len(clean) > 0.6:
        return "DATE"

    # Numeric
    numeric_clean = []
    for v in clean:
        try:
            numeric_clean.append(float(v.replace(",", "")))
        except ValueError:
            pass

    if len(numeric_clean) == len(clean):
        if all(float(v).is_integer() for v in numeric_clean):
            return "INTEGER"
        return "FLOAT"

    # Category (low cardinality)
    unique = set(v.lower() for v in clean)
    if len(unique) <= max(3, len(clean) * 0.3):
        return "CATEGORY"

    return "TEXT"


def detect_issues(name: str, values: pd.Series, inferred_type: str) -> List[str]:
    issues = []
    clean = [str(v).strip() for v in values if not is_missing(v)]
    missing_n = sum(1 for v in values if is_missing(v))

    if missing_n > 0:
        issues.append(f"{missing_n} missing value(s)")

    if inferred_type == "DATE":
        formats_found = set()
        for v in clean:
            if re.search(r"\d{4}-\d{2}-\d{2}", v):
                formats_found.add("YYYY-MM-DD")
            if re.search(r"\d{2}/\d{2}/\d{4}", v):
                formats_found.add("MM/DD/YYYY")
        if len(formats_found) > 1:
            issues.append(f"Mixed date formats: {', '.join(formats_found)}")

    if inferred_type == "EMAIL":
        email_re = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
        bad = [v for v in clean if not email_re.match(v)]
        if bad:
            issues.append(f"{len(bad)} malformed email(s)")

    if inferred_type in ("INTEGER", "FLOAT") and len(clean) > 4:
        nums = []
        for v in clean:
            try:
                nums.append(float(v.replace(",", "")))
            except ValueError:
                pass
        if nums:
            mean = np.mean(nums)
            std = np.std(nums)
            outliers = [v for v in nums if abs(v - mean) > 2.5 * std]
            if outliers:
                issues.append(f"{len(outliers)} statistical outlier(s) (>2.5σ)")

    return issues


def profile_dataframe(df: pd.DataFrame) -> List[ColumnProfile]:
    profiles = []
    for col in df.columns:
        values = df[col].tolist()
        inferred_type = infer_type(pd.Series(values))
        missing_count = sum(1 for v in values if is_missing(v))
        clean = [str(v).strip() for v in values if not is_missing(v)]
        unique_count = len(set(v.lower() for v in clean))
        issues = detect_issues(col, pd.Series(values), inferred_type)
        sample = clean[:3]

        profiles.append(ColumnProfile(
            name=col,
            inferred_type=inferred_type,
            total=len(values),
            missing_count=missing_count,
            unique_count=unique_count,
            issues=issues,
            sample_values=sample,
        ))
    return profiles


def run_detective(df: pd.DataFrame) -> dict:
    profiles = profile_dataframe(df)
    return {
        "profiles": profiles,
        "summary": {
            "total_columns": len(profiles),
            "total_rows": len(df),
            "columns_with_issues": sum(1 for p in profiles if p.issues),
            "type_distribution": {
                t: sum(1 for p in profiles if p.inferred_type == t)
                for t in ["INTEGER", "FLOAT", "DATE", "EMAIL", "CATEGORY", "TEXT"]
            }
        }
    }
