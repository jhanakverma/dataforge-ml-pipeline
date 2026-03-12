"""
Phase 3 — The Translator
One-hot encoding, min-max normalization, date feature extraction.
Converts all data into pure numbers that ML models can consume.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

from phase1_detective import infer_type


@dataclass
class TransformLog:
    original_column: str
    col_type: str
    action: str           # ONE_HOT, NORMALIZED, DATE_FEATURES, DROPPED
    new_columns: List[str]
    reason: str


def _min_max_normalize(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series([0.0] * len(series), index=series.index)
    return ((series - mn) / (mx - mn)).round(4)


def _extract_date_features(series: pd.Series, col_name: str) -> Tuple[pd.DataFrame, List[str]]:
    dates = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
    new_cols = {
        f"{col_name}_year":        dates.dt.year.fillna(0).astype(int),
        f"{col_name}_month":       dates.dt.month.fillna(0).astype(int),
        f"{col_name}_day_of_week": dates.dt.dayofweek.fillna(0).astype(int),
    }
    return pd.DataFrame(new_cols), list(new_cols.keys())


def translate(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[TransformLog]]:
    result_frames = []
    logs = []

    for col in df.columns:
        col_type = infer_type(df[col])

        # ── Drop email / free text ────────────────────────────────────────
        if col_type in ("EMAIL", "TEXT"):
            logs.append(TransformLog(
                original_column=col, col_type=col_type, action="DROPPED",
                new_columns=[],
                reason="Free-form text cannot be directly consumed by ML models. "
                       "NLP-specific techniques (TF-IDF, embeddings) would handle this separately."
            ))
            continue

        # ── One-hot encode categories ─────────────────────────────────────
        if col_type == "CATEGORY":
            dummies = pd.get_dummies(df[col].str.strip(), prefix=col).astype(int)
            result_frames.append(dummies)
            logs.append(TransformLog(
                original_column=col, col_type=col_type, action="ONE_HOT",
                new_columns=list(dummies.columns),
                reason=f"Categories like {list(df[col].unique()[:3])} are meaningless to math. "
                       "One-hot encoding creates a binary column per category: 1=present, 0=absent."
            ))

        # ── Extract date features ─────────────────────────────────────────
        elif col_type == "DATE":
            date_df, new_cols = _extract_date_features(df[col], col)
            result_frames.append(date_df)
            logs.append(TransformLog(
                original_column=col, col_type=col_type, action="DATE_FEATURES",
                new_columns=new_cols,
                reason="Raw date strings are meaningless to ML. Extracting year, month, and "
                       "day-of-week gives the model actual numeric signals about time patterns."
            ))

        # ── Normalize numerics ────────────────────────────────────────────
        elif col_type in ("INTEGER", "FLOAT"):
            nums = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median())
            new_col_name = f"{col}_norm"
            normalized = _min_max_normalize(nums).rename(new_col_name)
            result_frames.append(normalized.to_frame())
            logs.append(TransformLog(
                original_column=col, col_type=col_type, action="NORMALIZED",
                new_columns=[new_col_name],
                reason=f"Range was {nums.min():.1f}–{nums.max():.1f}. Large ranges dominate small ones "
                       "during training. Min-max normalization squishes everything into 0–1."
            ))

    translated_df = pd.concat(result_frames, axis=1) if result_frames else pd.DataFrame()
    return translated_df, logs


def run_translator(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[TransformLog]]:
    return translate(df)
