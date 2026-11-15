# Immortals AI

**Fairness-Aware Credit Intelligence**  
*Intelligence. Transparency. Fairness.*  
Team: **The Immortals**

Link: https://immortals-ai.streamlit.app/

Immortals AI uses the Statlog German Credit dataset (UCI ID 144) to study loan-risk
predictions and explore differences between applicants under 25 and applicants
aged 25 or older. It is a small research prototype, not a production loan
decision system.

## Run it

From the project folder in PowerShell:

```powershell
python -m pip install -r requirements.txt
python main.py
```

The program downloads UCI dataset 144, trains the baseline, compares the
model's outcomes by age group, explores features that may act as proxies for
age, and writes reports and charts. Internet access is needed to download the
dataset.

## Run the Immortals AI dashboard

After installing the dependencies in `requirements.txt`, launch the local
Streamlit application from the project folder:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The dashboard loads the saved age-excluded Model B and existing fairness,
proxy-ablation, and repeated-validation outputs. It does not fit models or
rerun experiments at startup. Applicant assessment, affordability, and
research results are shown in separate tabs. Credit-risk inputs follow the
saved model schema and do not include age. The affordability percentage is
adjustable and displayed with its illustrative assumption. Outputs support
analysis and do not approve or reject a loan.

### Windows: Application Control blocks a `.venv` package

If startup reports `DLL load failed ... An Application Control policy has
blocked this file` while importing Pandas from `.venv`, Windows is blocking a
compiled package file in that virtual environment before the app runs. On the
development workstation, launch with the tested base Python installation
instead of the blocked project virtual environment:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python314\python.exe" -m streamlit run app.py
```

This selects the base interpreter directly even if VS Code has activated
`.venv`. On another machine, use an installed Python interpreter whose
dependencies are permitted by that machine's application-control policy.

To run only the age fairness audit on the saved baseline without retraining:

```powershell
python main.py --fairness-only
```

This recreates the original stratified test split (20%, seed 42), loads the
saved pipeline, and predicts that split. The target is kept out of prediction
inputs. The saved baseline was trained with age among its input features; this
audit preserves that model and uses age to group its predictions.

## What the project found in its baseline

The dataset target is **1 = good credit risk** and **2 = bad credit risk**.
A false approval means a bad-risk applicant was predicted as good. A false
rejection means a good-risk applicant was predicted as bad. The financial cost
is:

```text
(False Approvals × 5) + (False Rejections × 1)
```

| Model | Accuracy | Balanced accuracy | ROC-AUC | False approvals | False rejections | Cost |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 78% | 70.95% | 0.804 | 28 | 16 | 156 |
| Always predict good | 70% | 50% | — | 60 | 0 | 300 |

The split is stratified 80/20 with random seed 42. The complete saved scores
are in [`models/stage1_metrics.json`](models/stage1_metrics.json). The original
baseline plots are in `data/eda/`; the audit's charts and reports are in
`outputs/`.

## How the code is organized

```text
main.py                 Runs the baseline and the audit
src/
  data_loader.py        Downloads data and reads UCI descriptions
  model.py              Preprocesses, trains, evaluates, saves, and predicts
  data_analysis.py      Creates data, age-group, and proxy reports and charts
  fairness.py           Compares model errors and approvals across groups
data/eda/               Original baseline plots
models/                 Saved pipeline, preprocessor, and baseline scores
outputs/plots/          Age and feature comparison charts
outputs/reports/        Dataset, age-group, proxy, and fairness reports
outputs/metrics/        Proxy experiment scores
tests/                  Small checks for age groups and analysis measures
```

### In plain language

`main.py` loads the data once, asks `model.py` to train Logistic Regression,
then passes the results to `data_analysis.py`. `model.py` can also use the saved
pipeline to score new applicant feature rows. The analysis describes the data,
compares age groups, and tests whether other features can predict age group
without using age or the credit-risk target. `fairness.py` reports approval,
false-approval, and false-rejection rates for each age group. The plots and
written reports are saved under `outputs/`.

The age-group and proxy results are exploratory. A feature predicting age
does not prove that it causes unfair treatment. The dataset is small and old,
and its good/bad labels reflect historical lending outcomes. See
[`outputs/reports/data_audit_report.md`](outputs/reports/data_audit_report.md)
for limitations.

## Explainability module

Run the explainability analysis from the project folder:

```powershell
python -m src.explainability
```

It reuses the saved age-excluded Model B at
`outputs/age_ablation/model_b_without_age.joblib`; it does not retrain the
prediction model. The command loads the German Credit dataset through UCI, so
it needs internet access. Outputs are isolated under `outputs/explainability/`:
the global influence ranking and chart, the transformed Logistic Regression
coefficients, three synthetic dataset-backed applicant examples, and a written
report. Global rankings use mean absolute per-row contributions to Good-risk
log odds (transformed values times fitted coefficients); one-hot category
levels are grouped under their original input feature. Individual explanations
use the same transformed values and coefficients, so the terms plus the
intercept reproduce the model's probability. Probabilities are estimates, not
guarantees, and explanations do not establish fairness or correctness.

## Affordability engine

Run the separate rule-based affordability demonstration from the project
folder:

```powershell
python -m src.affordability
```

The reusable API is `src.affordability.assess_affordability(...)`; it accepts
monthly income, existing loan repayments, other recurring obligations, annual
interest rate (as a percentage), tenure in whole months, an optional proposed
principal, and a configurable `affordability_percentage`. The default is 30% of
nonnegative disposable income. This is an illustrative prototype assumption,
not a validated lending standard. Inputs must be finite and nonnegative where
applicable; annual interest is limited to 0–100%, tenure must be positive, and
the affordability percentage must be 0–100%.

The engine calculates EMI and inverse EMI principal using the annuity formula,
with a separate zero-interest case. It reports an indicative budget limit and,
when a proposed amount is provided, its EMI, total repayment, total interest,
remaining income, and relation to the illustrative limit. It does not read
German Credit features, access or alter either credit-risk model, or approve or
reject applicants. Its assessment is not a repayment guarantee or a
creditworthiness decision. A synthetic example and assumptions are written to
`outputs/affordability/`.

## Libraries

- `pandas` and `numpy` handle tables and numeric calculations.
- `scikit-learn` preprocesses the data, trains the models, and calculates model scores.
- `ucimlrepo` downloads the dataset and its descriptions.
- `matplotlib` and `seaborn` create charts.
- `scipy` calculates feature-association statistics.
- `joblib` saves the trained model.
