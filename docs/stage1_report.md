# Immortals AI - Dataset Exploration and Baseline

Dataset: Statlog (German Credit Data) (UCI ID 144), 1000 applicants and 20 input features.

Target values: 1 = good credit risk and 2 = bad credit risk. Missing values: 0; duplicate complete rows: 0. See `../outputs/reports/dataset_profile.json` and `../outputs/reports/feature_dictionary.csv` for details.

## Baseline model

The data is split into 80% training and 20% test rows, stratified by target, with random seed 42. Numeric columns are median-imputed and standardized. Categories are most-frequent-imputed and one-hot encoded. The classifier is Logistic Regression with `max_iter=1000`. Good credit risk is the positive class.

Financial error cost = (false approvals x 5) + (false rejections x 1).

| Model | Accuracy | Balanced accuracy | ROC-AUC | False approvals | False rejections | Cost |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.780 | 0.710 | 0.804 | 28 | 16 | 156 |
| All-good dummy | 0.700 | 0.500 | - | 60 | 0 | 300 |

## Plots

The target and selected feature charts are in `../data/eda/`. Age-group plots and feature comparisons are in `../outputs/plots/`. These plots show associations and distributions; they do not by themselves establish bias or unfairness.

Full baseline metrics are in `../models/stage1_metrics.json`. The separate age-group, proxy, and data-quality analysis is in `../outputs/reports/data_audit_report.md`.
