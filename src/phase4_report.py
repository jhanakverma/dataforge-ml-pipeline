"""
Phase 4 — Report Card + Model Recommender (Secret Sauce)
Full pipeline audit, data quality scoring, and ML model recommendation
based on dataset characteristics.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List
from phase1_detective import infer_type


@dataclass
class ModelRecommendation:
    name: str
    confidence: int          # 0-100
    why: str
    avoid: str
    best_for: str


@dataclass
class PipelineReport:
    # Raw stats
    input_rows: int
    input_cols: int
    output_cols: int
    total_fixes: int
    data_quality_score: float   # 0-100
    type_distribution: dict

    # Phase summaries
    fix_summary: dict           # {issue_type: count}
    transform_summary: dict     # {action: count}

    # Model recommendations
    recommendations: List[ModelRecommendation] = field(default_factory=list)


def compute_quality_score(input_rows: int, input_cols: int, total_fixes: int) -> float:
    """Simple quality score: penalise fixes relative to total cells."""
    total_cells = input_rows * input_cols
    if total_cells == 0:
        return 100.0
    penalty = (total_fixes / total_cells) * 100
    return round(max(0, min(100, 100 - penalty * 2)), 1)


def recommend_models(df_raw: pd.DataFrame) -> List[ModelRecommendation]:
    n_rows, n_cols = df_raw.shape
    col_types = [infer_type(df_raw[c]) for c in df_raw.columns]
    n_cat = sum(1 for t in col_types if t == "CATEGORY")
    n_num = sum(1 for t in col_types if t in ("INTEGER", "FLOAT"))
    mostly_categorical = n_cat > n_num

    recommendations = []

    if n_rows < 500:
        recommendations += [
            ModelRecommendation(
                name="Decision Tree",
                confidence=95,
                why="Small datasets (< 500 rows) are where Decision Trees excel. They find clear rules quickly without needing thousands of examples.",
                avoid="Neural networks — they need far more data before they learn anything meaningful.",
                best_for="Quick baseline, interpretable results, small data"
            ),
            ModelRecommendation(
                name="Random Forest",
                confidence=88,
                why="Builds many Decision Trees and takes a majority vote. More stable than a single tree even on small datasets.",
                avoid="Overkill for tiny datasets but still solid — good second choice.",
                best_for="When you want more reliability than a single Decision Tree"
            ),
        ]
    elif n_rows < 10_000:
        recommendations += [
            ModelRecommendation(
                name="Random Forest",
                confidence=93,
                why="Medium datasets are the sweet spot for Random Forest. Handles mixed numeric/categorical data well with minimal tuning.",
                avoid="Deep learning — still needs more data and compute to justify the complexity.",
                best_for="General-purpose tabular ML, mixed feature types"
            ),
            ModelRecommendation(
                name="Gradient Boosting (XGBoost)",
                confidence=90,
                why="Learns iteratively from its own mistakes. Often wins on medium tabular datasets. Industry workhorse.",
                avoid="Can overfit if hyperparameters aren't tuned carefully on small-to-medium data.",
                best_for="Competition-grade accuracy on structured data"
            ),
        ]
    else:
        recommendations += [
            ModelRecommendation(
                name="XGBoost / LightGBM",
                confidence=95,
                why="Large tabular datasets are where boosting algorithms dominate — fast, accurate, and widely deployed in industry pipelines.",
                avoid="Simple models like linear regression will underfit this volume of data.",
                best_for="Production ML, large tabular datasets, speed + accuracy"
            ),
            ModelRecommendation(
                name="Neural Network (MLP)",
                confidence=85,
                why="Large enough data to justify neural network complexity. Can capture non-linear interactions that tree-based models miss.",
                avoid="Without careful architecture choices and regularisation, overfitting is a real risk.",
                best_for="Complex feature interactions, large-scale tabular data"
            ),
        ]

    if mostly_categorical:
        recommendations.append(ModelRecommendation(
            name="Naive Bayes",
            confidence=72,
            why="Your data is mostly categorical. Naive Bayes handles category-heavy datasets quickly and makes a great fast baseline.",
            avoid="Don't use it as your final model — the 'naive' independence assumption rarely holds in practice.",
            best_for="Fast baseline, category-heavy datasets, text classification"
        ))

    return recommendations


def generate_report(
    df_raw: pd.DataFrame,
    df_fixed: pd.DataFrame,
    df_translated: pd.DataFrame,
    fixes: list,
    transform_logs: list,
) -> PipelineReport:

    fix_summary = {}
    for f in fixes:
        fix_summary[f.issue] = fix_summary.get(f.issue, 0) + 1

    transform_summary = {}
    for t in transform_logs:
        transform_summary[t.action] = transform_summary.get(t.action, 0) + 1

    col_types = {c: infer_type(df_raw[c]) for c in df_raw.columns}
    type_dist = {}
    for t in col_types.values():
        type_dist[t] = type_dist.get(t, 0) + 1

    quality = compute_quality_score(len(df_raw), len(df_raw.columns), len(fixes))
    recs = recommend_models(df_raw)

    return PipelineReport(
        input_rows=len(df_raw),
        input_cols=len(df_raw.columns),
        output_cols=len(df_translated.columns),
        total_fixes=len(fixes),
        data_quality_score=quality,
        type_distribution=type_dist,
        fix_summary=fix_summary,
        transform_summary=transform_summary,
        recommendations=recs,
    )
