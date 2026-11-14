# Veritas AI: Loan Risk and Fairness Research

Veritas uses the Statlog German Credit dataset (UCI ID 144) to study loan-risk
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

## Libraries

- `pandas` and `numpy` handle tables and numeric calculations.
- `scikit-learn` preprocesses the data, trains the models, and calculates model scores.
- `ucimlrepo` downloads the dataset and its descriptions.
- `matplotlib` and `seaborn` create charts.
- `scipy` calculates feature-association statistics.
- `joblib` saves the trained model.
