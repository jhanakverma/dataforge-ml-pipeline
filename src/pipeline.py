"""
pipeline.py — Full DataForge pipeline runner
Chains all 4 phases: Detect → Fix → Translate → Report
"""

import pandas as pd
import io

from phase1_detective import run_detective
from phase2_doctor import run_doctor
from phase3_translator import run_translator
from phase4_report import generate_report


def run_pipeline(csv_text: str) -> dict:
    """
    Run the complete DataForge pipeline on a raw CSV string.

    Returns a dict with:
        - raw_df, fixed_df, translated_df  (DataFrames)
        - profiles     (Phase 1 column profiles)
        - fixes        (Phase 2 fix list)
        - transform_logs (Phase 3 transform log)
        - report       (Phase 4 PipelineReport)
    """
    # Parse
    df_raw = pd.read_csv(io.StringIO(csv_text))

    # Phase 1 — Detect
    detection = run_detective(df_raw)
    profiles = detection["profiles"]

    # Phase 2 — Fix
    df_fixed, fixes = run_doctor(df_raw.copy())

    # Phase 3 — Translate
    df_translated, transform_logs = run_translator(df_fixed.copy())

    # Phase 4 — Report + Recommend
    report = generate_report(df_raw, df_fixed, df_translated, fixes, transform_logs)

    return {
        "raw_df": df_raw,
        "fixed_df": df_fixed,
        "translated_df": df_translated,
        "profiles": profiles,
        "fixes": fixes,
        "transform_logs": transform_logs,
        "report": report,
    }


if __name__ == "__main__":
    sample = """name,age,salary,join_date,department,score,email
Alice,28,75000,2021-03-15,Engineering,8.5,alice@email.com
Bob,,82000,2020-11-02,Marketing,7.2,bob@email.com
Charlie,35,NULL,2019-07-20,Engineering,9.1,charlie@email.com
Diana,29,91000,2022-01-10,Design,,diana@email.com
Eve,42,67000,2018-05-30,Marketing,6.8,not-an-email
Frank,31,88000,03/15/2021,Engineering,8.0,frank@email.com
Grace,N/A,95000,2023-02-14,Design,9.5,grace@email.com
Ivan,150,88000,2021-06-01,Engineering,8.2,ivan@email.com"""

    result = run_pipeline(sample)
    r = result["report"]
    print(f"\n{'='*50}")
    print(f"DataForge Pipeline Complete")
    print(f"{'='*50}")
    print(f"Rows: {r.input_rows}  |  Input cols: {r.input_cols}  →  Output cols: {r.output_cols}")
    print(f"Total fixes: {r.total_fixes}  |  Quality score: {r.data_quality_score}/100")
    print(f"\nFix breakdown: {r.fix_summary}")
    print(f"\nTop model recommendation: {r.recommendations[0].name} ({r.recommendations[0].confidence}% fit)")
    print(f"Why: {r.recommendations[0].why}")
