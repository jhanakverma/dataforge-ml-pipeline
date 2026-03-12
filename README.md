# 🔬 DataForge — Intelligent ML Data Preprocessing Engine

> Automated schema inference · Statistical imputation · Feature engineering · Model recommendation

DataForge is an end-to-end data preprocessing pipeline that transforms raw, messy tabular data into AI-ready feature sets — automatically. No manual cleaning. No guesswork.

---

## 🧠 What It Does

Most real-world datasets are broken. Missing values, mixed date formats, outliers, text categories that ML models can't read. DataForge fixes all of it in four stages:

| Phase | Name | What it does |
|-------|------|--------------|
| 1 | 🔍 Data Detective | Automatically infers schema — column types, missing value patterns, format inconsistencies |
| 2 | 🩺 Data Doctor | Statistical imputation (median/mode fill), outlier detection (2.5σ), date standardization |
| 3 | 🔄 Translator | One-hot encoding, min-max normalization, date feature extraction |
| 4 + ✨ | 📊 Report Card | Full pipeline audit, data quality scoring, and ML model recommender |

---

## ✨ Key Features

- **Zero configuration** — paste a CSV, get clean AI-ready data
- **Smart imputation** — uses median for numerics, mode for categories (not naive zero/mean fill)
- **Outlier detection** — flags and replaces values beyond 2.5 standard deviations
- **Date intelligence** — detects mixed formats, standardizes to ISO 8601, extracts year/month/day-of-week features
- **One-hot encoding** — automatically expands categorical columns into binary feature sets
- **Model Recommender** — analyzes dataset characteristics and recommends the optimal ML model family with reasoning

---

## 🗂️ Project Structure

```
dataforge/
├── src/
│   ├── phase1_detective.py     # Schema inference engine
│   ├── phase2_doctor.py        # Statistical imputation & outlier correction
│   ├── phase3_translator.py    # Feature engineering & encoding
│   ├── phase4_report.py        # Pipeline audit & model recommender
│   └── pipeline.py             # Full pipeline runner
├── app.py                      # Hugging Face Spaces entry point (Gradio)
├── requirements.txt
└── README.md
```

---

## 🚀 Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/dataforge
cd dataforge
pip install -r requirements.txt
python app.py
```

---

## 🤗 Live Demo

[**Try it on Hugging Face Spaces →**](https://huggingface.co/spaces/YOUR_USERNAME/dataforge)

---

## 🛠️ Tech Stack

- **Python** — core pipeline logic
- **Pandas** — data manipulation
- **NumPy** — statistical operations
- **Scikit-learn** — preprocessing utilities
- **Gradio** — interactive web UI
- **Matplotlib / Seaborn** — data quality visualizations

---

## 📊 Example

**Input (raw messy data):**
```
name,age,salary,join_date,department,score
Alice,28,75000,2021-03-15,Engineering,8.5
Bob,,82000,2020-11-02,Marketing,7.2
Charlie,35,NULL,2019-07-20,Engineering,9.1
Grace,N/A,95000,2023-02-14,Design,9.5
Ivan,150,88000,2021-06-01,Engineering,8.2   ← outlier age
Frank,31,88000,03/15/2021,Engineering,8.0   ← wrong date format
```

**Output (AI-ready):**
```
age_norm, salary_norm, join_date_year, join_date_month, join_date_dow,
department_Design, department_Engineering, department_Marketing, score_norm
0.031, 0.296, 2021, 3, 1, 0, 1, 0, 0.81
...
```

**Model Recommendation:** `Random Forest` — medium dataset, mixed feature types, handles categorical well with minimal tuning.

---

## 📄 License

MIT
