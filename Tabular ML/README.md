# Telco Customer Churn — Tabular ML

End-to-end tabular ML pipeline on the [Kaggle Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

## Project Phases

- [x] Phase 1 — EDA & Data Understanding
- [x] Phase 2 — Preprocessing Pipeline (ColumnTransformer + sklearn Pipeline)
- [x] Phase 3 — Baseline Models (LR, DT, RF)
- [x] Phase 4 — Class Imbalance (SMOTE, class_weight)
- [ ] Phase 5 — Gradient Boosting (XGBoost / LightGBM + tuning)
- [ ] Phase 6 — Final Evaluation & Write-up

## Key Findings (EDA)

- Dataset: 7,043 customers, 19 features, 1 target (`Churn`)
- Class imbalance: ~73.5% No, ~26.5% Yes
- `TotalCharges` has a dtype bug (whitespace strings) — fixed via `pd.to_numeric(..., errors='coerce')`
- Strongest churn signals: contract type, tenure, monthly charges
- Month-to-month contracts churn at ~42% vs two-year at ~3%

## Stack

## Dataset

Not included. Download from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the project root.