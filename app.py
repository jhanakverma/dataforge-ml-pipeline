"""
app.py — DataForge Hugging Face Spaces entry point
Interactive Gradio UI for the full preprocessing pipeline.
"""

import gradio as gr
import pandas as pd
import io
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pipeline import run_pipeline

SAMPLE_CSV = """name,age,salary,join_date,department,score,email
Alice,28,75000,2021-03-15,Engineering,8.5,alice@email.com
Bob,,82000,2020-11-02,Marketing,7.2,bob@email.com
Charlie,35,NULL,2019-07-20,Engineering,9.1,charlie@email.com
Diana,29,91000,2022-01-10,Design,,diana@email.com
Eve,42,67000,2018-05-30,Marketing,6.8,not-an-email
Frank,31,88000,03/15/2021,Engineering,8.0,frank@email.com
Grace,N/A,95000,2023-02-14,Design,9.5,grace@email.com
Henry,38,71000,2020-08-19,Marketing,7.7,henry@email.com
Ivan,150,88000,2021-06-01,Engineering,8.2,ivan@email.com
Julia,27,72000,2022-09-10,Design,7.9,julia@email.com"""


def process(csv_text: str):
    if not csv_text.strip():
        return [None] * 6 + ["Please paste CSV data or click 'Load Sample'."]

    try:
        result = run_pipeline(csv_text)
    except Exception as e:
        return [None] * 6 + [f"Error: {str(e)}"]

    r = result["report"]

    # ── Summary text ──────────────────────────────────────────────────────
    summary = f"""## 📊 Pipeline Report

**Input:** {r.input_rows} rows × {r.input_cols} columns  
**Output:** {r.input_rows} rows × {r.output_cols} columns  
**Total fixes applied:** {r.total_fixes}  
**Data quality score:** {r.data_quality_score} / 100

### 🩺 Fixes Breakdown
{chr(10).join(f'- **{k}**: {v}' for k, v in r.fix_summary.items()) or '- No fixes needed!'}

### 🔄 Transformations
{chr(10).join(f'- **{k}**: {v} column(s)' for k, v in r.transform_summary.items())}

### ✨ Model Recommendation
"""
    for i, rec in enumerate(r.recommendations[:2]):
        summary += f"\n**#{i+1} {rec.name}** — {rec.confidence}% fit  \n✅ {rec.why}  \n⚠️ Avoid: {rec.avoid}\n"

    # ── Schema table ──────────────────────────────────────────────────────
    schema_data = []
    for p in result["profiles"]:
        schema_data.append([
            p.name,
            p.inferred_type,
            f"{p.missing_count} ({p.missing_pct}%)",
            p.unique_count,
            "; ".join(p.issues) if p.issues else "✅ Clean",
        ])
    schema_df = pd.DataFrame(schema_data, columns=["Column", "Type", "Missing", "Unique", "Issues"])

    # ── Fix log table ─────────────────────────────────────────────────────
    if result["fixes"]:
        fix_data = [[f.column, f"Row {f.row_index+2}", f.issue, f.old_value, f.new_value, f.strategy]
                    for f in result["fixes"]]
        fix_df = pd.DataFrame(fix_data, columns=["Column", "Row", "Issue", "Old Value", "New Value", "Strategy"])
    else:
        fix_df = pd.DataFrame([["No fixes needed", "", "", "", "", ""]], 
                               columns=["Column", "Row", "Issue", "Old Value", "New Value", "Strategy"])

    # ── Transform log ─────────────────────────────────────────────────────
    trans_data = [[t.original_column, t.col_type, t.action, ", ".join(t.new_columns[:4]) + ("..." if len(t.new_columns) > 4 else ""), t.reason]
                  for t in result["transform_logs"]]
    trans_df = pd.DataFrame(trans_data, columns=["Original Column", "Type", "Action", "New Columns", "Reason"])

    return (
        result["raw_df"],
        schema_df,
        fix_df,
        result["fixed_df"],
        trans_df,
        result["translated_df"],
        summary,
    )


def load_sample():
    return SAMPLE_CSV


# ── Build UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="DataForge — ML Data Preprocessing Engine",
    theme=gr.themes.Soft(primary_hue="blue"),
    css="""
    .gr-button-primary { background: linear-gradient(135deg, #4cc9f0, #7209b7) !important; }
    h1 { font-size: 2rem !important; }
    """
) as demo:

    gr.Markdown("""
    # 🔬 DataForge — Intelligent ML Data Preprocessing Engine
    > Automated schema inference · Statistical imputation · Feature engineering · Model recommendation
    
    Paste your raw CSV data below and run the full 4-phase pipeline. DataForge will detect issues,
    fix them with statistical methods, translate everything into AI-ready numbers, and recommend
    which ML model best fits your data.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            csv_input = gr.Textbox(
                label="📋 Raw CSV Input",
                placeholder="Paste your CSV here, or click 'Load Sample Data'...",
                lines=12,
                max_lines=30,
            )
            with gr.Row():
                sample_btn = gr.Button("Load Sample Data", variant="secondary")
                run_btn = gr.Button("🚀 Run Full Pipeline", variant="primary")

        with gr.Column(scale=2):
            summary_output = gr.Markdown(label="Pipeline Report")

    gr.Markdown("---")

    with gr.Tabs():
        with gr.Tab("🔍 Phase 1 — Schema (Raw Data)"):
            raw_table = gr.Dataframe(label="Raw Data", interactive=False)
            schema_table = gr.Dataframe(label="Column Profiles", interactive=False)

        with gr.Tab("🩺 Phase 2 — Fixes"):
            fix_table = gr.Dataframe(label="Fix Log", interactive=False)
            fixed_table = gr.Dataframe(label="Cleaned Data", interactive=False)

        with gr.Tab("🔄 Phase 3 — Translations"):
            trans_table = gr.Dataframe(label="Transform Log", interactive=False)
            translated_table = gr.Dataframe(label="AI-Ready Data (all numbers)", interactive=False)

    # ── Wire up ───────────────────────────────────────────────────────────
    sample_btn.click(fn=load_sample, outputs=csv_input)
    run_btn.click(
        fn=process,
        inputs=csv_input,
        outputs=[raw_table, schema_table, fix_table, fixed_table, trans_table, translated_table, summary_output],
    )

if __name__ == "__main__":
    demo.launch()
