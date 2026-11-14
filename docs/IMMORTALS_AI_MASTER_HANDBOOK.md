# Immortals AI: Complete Project Handbook

**Fairness-Aware Credit Intelligence**  
Team: **The Immortals**  
Project type: research and demonstration prototype

> **Prototype notice:** Immortals AI is for learning and research. It is not financial advice, a credit bureau, or a loan decision system. Its predictions and affordability examples must not be used to make decisions about real applicants.

## How to read this handbook

This handbook explains what the repository and recorded project history show was built. It distinguishes implemented behavior from historical plans and future ideas. Source tags appear throughout; the complete source key is in [Sources and source index](#sources-and-source-index).

The four supplied PDFs are competition guidance and reference material. Their directions describe what a hackathon team is expected to submit; they do not change the user's instructions for this documentation task. In particular, this handbook does not alter application code, retrain models, or publish conversation content. The supplied project brief and Hacker Guide differ on whether there is a live kickoff: the guide says briefs are revealed at kickoff, while the detailed brief says there is no live kickoff and contains the challenge details. This handbook records the discrepancy rather than resolving it by assumption. [PDF-2, pp. 3, 16; PDF-4, pp. 1-3]

## 1. Project in one minute

Immortals AI is a small credit-risk research project built around the Statlog German Credit dataset. It trains Logistic Regression to distinguish historical “good” and “bad” credit-risk labels, compares it with an “always predict good” baseline, examines differences between applicants younger than 25 and those 25 or older, and tests whether removing age-related inputs changes model results. A Streamlit dashboard demonstrates applicant scoring, explanations, a separate monthly-payment affordability calculator, and stored fairness experiments. [README.md; `src/`; `app.py`; `outputs/`]

The “AI” part is the Logistic Regression model. The fairness audits and explanations are analyses around that model. The affordability tab is a separate calculator based on arithmetic rules; it is not a machine-learning feature and does not feed the credit-risk model. [ `src/model.py`; `src/fairness.py`; `src/explainability.py`; `src/affordability.py`; `app.py`]

### What is implemented

| Status | What it means in this project |
|---|---|
| **Implemented** | A UCI dataset loader; preprocessing and Logistic Regression; a held-out baseline evaluation; age-group metrics; a model that omits age; feature ablations; repeated cross-validation; coefficient-based explanations; a stand-alone affordability calculator; and a three-tab Streamlit application that loads saved results. |
| **Planned or suggested in project material** | The conversations and competition references discuss improving, presenting, or extending the prototype. Ideas such as collecting current local loan data, intersectional auditing, policy-grade fairness mitigation, user studies, and validating with a lender are future work unless specifically listed as implemented above. |
| **Not implemented** | A real lender integration, an actual loan approval/denial service, a validated lending policy, a repayment guarantee, live applicant storage, a deployed production API, or proof that the model is fair. |

Sources: [README.md], [`src/`], [`app.py`], [PDF-1, pp. 1-3], [PDF-4, section 4.2.3], and the project conversations *Baseline*, *Implement Fairness Auditing*, and *Dashboard*.

## 2. Project history and goals

The starting point was a narrow question: can a simple model predict the German Credit dataset’s good/bad risk label, and how does its behavior compare across younger and older applicants? The project chose a small public tabular dataset and a model that can run on an ordinary computer, which matches the weekend-project guidance to define a measurable question, make a baseline first, and leave time to explain failures. [ChatGPT project conversation: *Baseline*; PDF-2, pp. 4-12; PDF-3, section 4.2.3]

The work then grew in stages:

1. **Baseline:** download and inspect UCI Statlog German Credit; preprocess numeric and categorical columns; fit Logistic Regression; compare with a simple all-good predictor; save metrics, a model, and exploratory plots. [*Baseline*; `main.py`; `src/data_loader.py`; `src/model.py`; `docs/stage1_report.md`]
2. **Fairness audit:** use age groups (`<25` and `25+`) to measure approval and error rates. The audit uses the saved baseline and does not retrain it. [*Implement Fairness Auditing*; `src/fairness.py`; `outputs/reports/age_fairness_audit.md`]
3. **Data and proxy analysis:** inspect group composition and feature associations, then train an age-group prediction experiment without age or the credit-risk label. This asks whether remaining columns contain age information; it does not establish unfairness. [*Implement Fairness Auditing*; `src/data_analysis.py`; `outputs/reports/proxy_analysis_report.md`]
4. **Age ablation:** compare the saved all-feature baseline with a new 19-feature model that drops only age. [Codex conversation: *Baseline*; `src/age_ablation.py`; `outputs/age_ablation/age_ablation_report.md`]
5. **Proxy-feature ablation and repeated validation:** preselect candidate columns, remove them individually or as a group, then repeat stratified validation so the result is not judged only from one test split. [Codex conversation: *Organizing*; `src/proxy_ablation.py`; `src/repeated_validation.py`; `outputs/proxy_ablation/`; `outputs/repeated_validation/`]
6. **Explainability:** show which transformed model inputs move a prediction toward the good-risk or bad-risk score, including synthetic examples. [Codex conversation: *Dashboard*; `src/explainability.py`; `outputs/explainability/`]
7. **Affordability and dashboard:** build a separate annuity-payment calculator and join the saved analysis into a Streamlit interface with assessment, affordability, and fairness/validation tabs. [*Dashboard*; `src/affordability.py`; `src/application.py`; `app.py`]

The project’s original purpose is exploratory learning, not building a fair-lending product. The reference brief asks for a model, a baseline comparison, approval-rate comparisons between groups, and an original extension such as checking proxy features or trying a change. Immortals AI’s extensions are the proxy analyses, ablations, and repeated validation described below. [PDF-4, section 4.2.3; PDF-1, pp. 2-3]

## 3. Data: what the model sees

### Dataset overview

The loader requests **Statlog (German Credit Data)**, UCI Machine Learning Repository dataset **144**. It contains 1,000 historical applicants, 20 input attributes, and a binary target: `1 = Good risk`, `2 = Bad risk`. The dataset has 700 Good labels and 300 Bad labels. The repository does not store a raw dataset copy; `ucimlrepo` fetches it at runtime. [ `src/data_loader.py`; `data/README.md`; `outputs/reports/dataset_profile.json`; PDF-4, section 4.2.3]

There are 7 numeric attributes and 13 categorical attributes. The project audit found no missing cells and no duplicate complete rows, so its audit did not need to impute or remove original observations. The machine-learning pipeline still contains imputers so it can handle missing values in future inputs. [ `outputs/reports/dataset_profile.md`; `outputs/reports/data_quality_report.md`; `src/model.py`]

The project uses the dataset’s historically assigned outcome labels. These are not a perfect measure of a person’s true ability to repay, and the dataset is small and old. Its observations reflect who was evaluated under past lending practices; people previously denied credit may be missing. The attached fairness note also calls out old data, sample size, label concerns, socioeconomic context, and selection effects. [ `outputs/reports/data_audit_report.md`; PDF-1, pp. 1-3]

### The 20 original attributes

The UCI source labels each input `Attribute1` through `Attribute20`. The descriptions below are simplified from the repository feature dictionary. Some are category codes rather than dollar amounts in modern currency. Do not interpret this historic data as current local lending data. [ `outputs/reports/feature_dictionary.csv`; `src/data_loader.py`]

| # | Field | Plain-language meaning |
|---:|---|---|
| 1 | `Attribute1` | Existing checking-account status or balance band |
| 2 | `Attribute2` | Requested loan duration, in months |
| 3 | `Attribute3` | Existing credit history |
| 4 | `Attribute4` | Purpose of the loan |
| 5 | `Attribute5` | Requested credit amount |
| 6 | `Attribute6` | Savings account or bonds band |
| 7 | `Attribute7` | Time in current employment |
| 8 | `Attribute8` | Installment size as a percentage of disposable income |
| 9 | `Attribute9` | Combined personal status and sex code |
| 10 | `Attribute10` | Other debtors or guarantors |
| 11 | `Attribute11` | Years at current residence |
| 12 | `Attribute12` | Property or asset category |
| 13 | `Attribute13` | Age in years |
| 14 | `Attribute14` | Other installment plans |
| 15 | `Attribute15` | Housing category (rent, own, or free) |
| 16 | `Attribute16` | Number of existing credits at this bank |
| 17 | `Attribute17` | Job category |
| 18 | `Attribute18` | Number of people liable for maintenance |
| 19 | `Attribute19` | Telephone availability/category |
| 20 | `Attribute20` | Foreign-worker indicator |

`Attribute9` is already a combined personal-status-and-sex field in the source data; the repository does not have separate sex and marital-status columns. A fairness audit by sex alone cannot be claimed from this field without additional assumptions or a reliable recoding. [ `outputs/reports/data_audit_report.md`; PDF-1, p. 1]

### Data path

```text
UCI dataset 144
        |
        v
1,000 rows: 20 inputs + class label
        |
        +--> age and feature audits
        |
        +--> stratified train/test split
                 |
                 +--> preprocessing fit on training rows
                 +--> Logistic Regression fit on training rows
                 +--> one-time evaluation on held-out rows
```

The baseline split is stratified, so the Good/Bad label proportions are preserved approximately in both portions. It uses 80% for training and 20% for held-out testing, with random seed 42. Fitting preprocessing only on the training portion avoids letting test-set averages or category information leak into training. [ `src/model.py`; `outputs/age_ablation/age_ablation_report.md`; PDF-3, section 3]

## 4. Credit-risk model and baseline

### Logistic Regression in plain language

Logistic Regression is a model for a two-choice question. It combines input signals using learned weights, converts the combined score to a number between 0 and 1, and uses that score to estimate the probability of one class. Here class 1 is Good risk, so a larger Good-risk probability leans toward class 1. The name says “regression,” but it is commonly used for classification. It is not the same thing as a human-readable rulebook or a causal explanation. [ `src/model.py`; `src/explainability.py`]

The pipeline first prepares each input. Numeric values are median-imputed and standardized. Categorical values are filled with the most frequent training category and one-hot encoded, which creates a yes/no indicator for each category level. Unknown categories are ignored rather than causing a crash. Logistic Regression uses `max_iter=1000` and `random_state=42`. [ `src/model.py`]

For prepared inputs `x`, the model forms a weighted score `z` and converts it into a Good-risk probability:

```text
z = β0 + β1x1 + β2x2 + ... + βpxp
p(Good | x) = 1 / (1 + e^(-z))
p(Bad | x) = 1 - p(Good | x)
```

Each `β` is learned from training examples. With the usual 0.5 cutoff, the classifier predicts Good when its Good probability is at least 0.5; otherwise it predicts Bad. A score is an estimate based on the training data, not a promise about an applicant. [ `src/model.py`; `src/explainability.py`]

```text
Raw applicant fields
      -> fill/scale numbers and encode categories
      -> weighted Logistic Regression score
      -> probability estimates: Good risk / Bad risk
      -> predicted label at the classifier threshold
```

### Confusion matrix and mistakes

The held-out test set has 200 people: 140 Good and 60 Bad. The saved baseline confusion matrix, with actual classes as rows and predicted classes as columns, is:

| Actual  Predicted | Good (`1`) | Bad (`2`) |
|---|---:|---:|
| Good (`1`) | 124 | 16 |
| Bad (`2`) | 28 | 32 |

Thus, 28 actual Bad risks were predicted Good (false approvals) and 16 actual Good risks were predicted Bad (false rejections). The model correctly classified 156 of 200 test examples. [ `models/stage1_metrics.json`; `docs/stage1_report.md`]

### Baseline comparison

The baseline is “predict Good for everyone,” which is simple enough to reveal why accuracy alone can mislead. Since 70% of the dataset is labeled Good, it gets 70% accuracy by always choosing that majority class. But it approves all 60 Bad examples in the held-out set. Logistic Regression scores better on accuracy, balanced accuracy, and ROC-AUC, and has lower cost under the project’s chosen toy error weights. [ `models/stage1_metrics.json`; `docs/stage1_report.md`; PDF-4, section 4.2.3]

| Held-out result | Always predict Good | Logistic Regression |
|---|---:|---:|
| Accuracy | 70.0% | 78.0% |
| Balanced accuracy | 50.0% | 70.95% |
| ROC-AUC | Not applicable | 0.8040 |
| False approvals | 60 | 28 |
| False rejections | 0 | 16 |
| Weighted error cost | 300 | 156 |

The cost formula is `5 × false approvals + 1 × false rejections`: approving a Bad risk is weighted five times as costly as rejecting a Good risk. The all-Good baseline cost is `5×60 + 0 = 300`; Logistic Regression costs `5×28 + 16 = 156`. These reproduce the values in the saved report. [ `models/stage1_metrics.json`; `src/model.py`; `docs/stage1_report.md`]

**Other metrics:** accuracy is the share of all predictions that are right. Balanced accuracy averages recall for the two classes, which gives the smaller Bad class equal weight. Precision for Good risk asks “of predicted-Good applicants, how many were truly Good?”; recall for Good asks “of truly Good applicants, how many did the model identify?” ROC-AUC measures how well scores rank positive examples above negative ones across thresholds. The saved report gives Good precision 0.816, recall 0.886, and F1 0.849. [ `models/stage1_metrics.json`]

For a binary class, the common score formulas are:

```text
Accuracy = (true positives + true negatives) / all examples
Precision = true positives / (true positives + false positives)
Recall = true positives / (true positives + false negatives)
F1 = 2 × Precision × Recall / (Precision + Recall)
Balanced accuracy = (recall for class 1 + recall for class 2) / 2
```

For this credit task, it is often clearer to name the mistakes directly rather than depend on positive/negative terminology: a **false approval** is actual Bad predicted Good, and a **false rejection** is actual Good predicted Bad. [ `src/model.py`; `src/fairness.py`]

The PDF’s 1,500 vs 700 example describes applying the 5:1 weights to all 1,000 true labels for always-approve versus always-reject. That is a different accounting scope from the repository’s 200-row held-out test metrics (300 for 60 false approvals). The figures are not contradictory once the denominator is stated. [PDF-1, p. 2; `models/stage1_metrics.json`]

## 5. Age fairness investigation

### What was measured

The project divides applicants into **Under 25** and **25 or older**. It reports:

- **Approval rate:** predicted-Good applicants divided by all applicants in a group.
- **False-approval rate:** actual-Bad applicants predicted Good, divided by actual-Bad applicants in that group.
- **False-rejection rate:** actual-Good applicants predicted Bad, divided by actual-Good applicants in that group.
- **True-positive rate / Good-risk recall:** actual-Good applicants predicted Good, divided by actual-Good applicants in that group.

If group `g` contains `N_g` applicants, its approval rate is `predicted Good in g / N_g`. The approval gap is the absolute difference between the two group rates, expressed in percentage points. The false-approval denominator includes only actual Bad cases in that group; the false-rejection denominator includes only actual Good cases. [ `src/fairness.py`]

The data audit has 149 under-25 applicants and 851 aged 25+. In the held-out set there are only 39 under-25 applicants and 161 older applicants. Small subgroup counts make each rate less stable. [ `outputs/reports/data_audit_report.md`; `outputs/reports/age_fairness_audit.md`]

The supplied fairness note gives an approximate age split of about 810 versus 190. The repository’s executed `<25` grouping reports 149 versus 851. This handbook uses the repository’s count for project results; the PDF number is general context and should not be substituted into the project’s denominators. [PDF-1, p. 1; `outputs/reports/data_audit_report.md`]

### Original all-feature baseline on the test split

| Group | Test applicants | Approval | False approval | False rejection | Good-risk recall |
|---|---:|---:|---:|---:|---:|
| Under 25 | 39 | 61.5% (24/39) | 42.1% (8/19 Bad) | 20.0% (4/20 Good) | 80.0% |
| 25 or older | 161 | 79.5% (128/161) | 48.8% (20/41 Bad) | 10.0% (12/120 Good) | 90.0% |

Absolute group gaps are 17.96 percentage points for approval, 6.68 points for false approvals, and 10.00 points for false rejections. “Percentage points” means subtracting rates directly; a 79.5% minus 61.5% difference is 18 points, not a relative 18% change. These are descriptive measurements from one split, not proof of discrimination or proof of fairness. [ `outputs/reports/age_fairness_audit.md`]

### Removing the direct age input

Model A is the saved baseline with all 20 inputs. Model B is a separately trained Logistic Regression model with only `Attribute13` (age) removed. It uses the same fixed split, preprocessing, and settings. Age remains available to the fairness report as a group label, but is not passed into Model B for predictions. [ `src/age_ablation.py`; `outputs/age_ablation/age_ablation_report.md`]

| Test metric | Model A: all features | Model B: age removed |
|---|---:|---:|
| Accuracy | 78.0% | 77.5% |
| Balanced accuracy | 70.95% | 70.60% |
| ROC-AUC | 0.8040 | 0.8017 |
| False approvals | 28 | 28 |
| False rejections | 16 | 17 |
| Weighted error cost | 156 | 157 |

Model B’s approval gap changes from 17.96 to 17.34 points, and false-rejection gap from 10.00 to 9.17 points; the false-approval gap remains 6.68 points. The small change in this particular split shows why merely removing age is not enough to claim a fair model. [ `outputs/age_ablation/age_ablation_report.md`; PDF-1, p. 2]

## 6. Proxy-feature investigation

A **proxy feature** is an input that can carry information about another attribute even when that other attribute is omitted. For example, housing or employment history may be statistically related to age. The project uses two different types of check; neither proves that a feature causes discrimination. [ `src/data_analysis.py`; PDF-1, p. 2]

### Associations with age group

The data-analysis step compares each feature to age group using Cramer’s V for categorical data, Spearman association for numeric data, and supporting group plots. The strongest reported associations include Housing (`Attribute15`, Cramer’s V 0.314), residence duration (`Attribute11`, Spearman 0.235), and employment duration (`Attribute7`, Cramer’s V 0.212). These are candidates to investigate, not a causal ranking or proof of unfairness. [ `outputs/reports/age_group_feature_comparison.csv`; `outputs/reports/proxy_analysis_report.md`; `outputs/plots/feature_association_heatmap.png`]

### Predict age without age

The proxy-prediction experiment tries to predict whether a person is under 25 using the other features, excluding both `Attribute13` (age) and `class` (credit label). Its held-out balanced accuracy is 0.6382 versus a majority baseline of 0.5000; ROC-AUC is 0.7782. Five-fold cross-validation on the training rows has mean ROC-AUC 0.8004 with standard deviation 0.0496. This indicates that other columns contain information associated with the age grouping in this dataset. It does not show that the credit model necessarily uses that information unfairly. [ `outputs/metrics/proxy_prediction_metrics.json`; `outputs/reports/proxy_analysis_report.md`]

### Remove selected candidate features

The project selected checking-account status, loan purpose, savings, employment duration, and housing based on their definitions and observed associations, then compared individual removals and a predeclared group removal against age-excluded Model B. These are fixed-split experiments, not a search over every possible combination. [ `src/proxy_ablation.py`; `outputs/proxy_ablation/candidate_features.csv`; `outputs/proxy_ablation/proxy_ablation_report.md`]

| Age-excluded feature set | Accuracy | Balanced accuracy | ROC-AUC | Cost |
|---|---:|---:|---:|---:|
| Reference Model B | 77.5% | 70.60% | 0.8017 | 157 |
| Minus checking status (`Attribute1`) | 72.0% | 63.81% | 0.7380 | 192 |
| Minus loan purpose (`Attribute4`) | 75.0% | 67.38% | 0.7779 | 174 |
| Minus savings (`Attribute6`) | 80.0% | 72.38% | 0.8087 | 152 |
| Minus employment (`Attribute7`) | 78.0% | 71.43% | 0.8040 | 152 |
| Minus housing (`Attribute15`) | 78.5% | 71.31% | 0.8107 | 155 |
| Minus savings + employment + housing | 80.5% | 72.74% | 0.8220 | 151 |

On this particular test split, the joint removal shrank the approval-rate gap to about 1.96 points but increased the false-approval gap to about 24.13 points. That is an example of fairness measures moving in different directions. A change should not be called a success by selecting only the metric that improved. [ `outputs/proxy_ablation/fairness_gaps.csv`; `outputs/proxy_ablation/proxy_ablation_report.md`; PDF-1, p. 2]

## 7. Repeated validation

One split can produce a result that depends on which applicants happened to land in the test rows. The repeated-validation experiment uses **5-fold stratified cross-validation repeated 10 times**. Every feature-set comparison uses matching folds; preprocessing is fitted only on each training fold. Each repeat produces out-of-fold predictions for the 1,000 rows. [ `src/repeated_validation.py`; `outputs/repeated_validation/experiment_metadata.json`; `outputs/repeated_validation/repeated_validation_report.md`]

| Model | Mean accuracy | Mean balanced accuracy | Mean ROC-AUC | Mean cost per 100 |
|---|---:|---:|---:|---:|
| A: all features | 75.2% | 67.4% | 0.784 | 87.31 |
| B: age excluded | 75.2% | 67.2% | 0.784 | 88.13 |
| C: age + checking excluded | 73.7% | 63.8% | 0.746 | 99.43 |
| D: age + purpose excluded | 75.1% | 66.6% | 0.774 | 90.58 |
| E: age + savings excluded | 75.2% | 66.9% | 0.780 | 89.49 |
| F: age + employment excluded | 75.2% | 67.1% | 0.783 | 88.48 |
| G: age + housing excluded | 75.4% | 67.4% | 0.784 | 87.57 |
| H: age + savings + employment + housing excluded | 75.4% | 66.9% | 0.776 | 89.81 |

Across the ten repeats, mean absolute approval-rate gaps are 15.73 points for A, 13.02 for B, 8.50 for C, and 6.64 for H. For Model H, mean false-approval gap is 2.64 points and mean false-rejection gap is 5.32 points. The report’s paired comparisons show that removing checking status consistently narrowed the approval gap but reduced predictive performance and raised mean cost. Combined removal results do not reproduce the unusually large false-approval gap from the single fixed split. [ `outputs/repeated_validation/fairness_gap_summary.csv`; `outputs/repeated_validation/paired_changes_vs_age_excluded.csv`; `outputs/repeated_validation/repeated_validation_report.md`]

The 2.5th and 97.5th percentiles in these reports summarize the **ten observed repeat results**. They are empirical ranges, not formal confidence intervals for all future borrowers. Repeats reuse the same 1,000 people, so they are not ten independent new samples. The dataset remains small and old, and the experiments do not provide external validation. [ `src/repeated_validation.py`; `outputs/repeated_validation/repeated_validation_report.md`; PDF-1, pp. 1-3]

## 8. Explainability

The explanation module uses saved age-excluded Model B without retraining. For a given applicant, each transformed input value is multiplied by its fitted Logistic Regression coefficient. Adding these contributions and the intercept gives the Good-versus-Bad log-odds score; applying the logistic function turns it into a probability. [ `src/explainability.py`; `outputs/explainability/explainability_report.md`]

The project’s global ranking averages the absolute per-applicant contribution for each original feature, grouping one-hot category levels back under their parent attribute. The top reported features are checking status (`Attribute1`), foreign-worker status (`Attribute20`), savings (`Attribute6`), other installment plans (`Attribute14`), installment rate (`Attribute8`), loan purpose (`Attribute4`), combined personal status (`Attribute9`), property (`Attribute12`), credit history (`Attribute3`), and duration (`Attribute2`). The sign of a contribution describes movement in this fitted model’s score, not a causal effect on a real loan outcome. [ `outputs/explainability/explainability_report.md`; `outputs/explainability/global_feature_importance.csv`]

Three synthetic dataset-backed examples demonstrate a Good prediction (99.8% Good probability), a Bad prediction (2.6% Good probability), and a near-boundary case (49.9% Good / 50.1% Bad). A warning is shown when the Good probability is within 10 percentage points of 50%. These are explanatory examples, not real customers. [ `outputs/explainability/demonstration_cases.json`; `outputs/explainability/explainability_report.md`]

## 9. Affordability engine and EMI

The affordability engine is a separate calculator in `src/affordability.py`. It takes monthly income, existing loan payments, other recurring obligations, annual interest rate, loan duration in whole months, and optionally a proposed loan principal. It estimates a maximum monthly payment as a configurable percentage (default 30%) of nonnegative disposable income. That 30% is an illustrative assumption chosen for the prototype, not a lending standard validated by the project. [ `src/affordability.py`; `outputs/affordability/affordability_report.md`; `README.md`]

### Formulas

Let:

- `I` = monthly income
- `E` = existing monthly loan payments
- `O` = other monthly obligations
- `D` = disposable income before the proposed loan
- `q` = chosen affordability percentage as a decimal (default `0.30`)
- `P` = principal (amount borrowed)
- `a` = nominal annual interest rate in percent
- `r = a / 100 / 12` = monthly interest rate as a decimal
- `n` = number of monthly payments
- `M` = monthly EMI (equated monthly installment)

```text
D = I - E - O
Maximum illustrative EMI = max(D, 0) × q

If r > 0:
M = P × r × (1 + r)^n / ((1 + r)^n - 1)

If r = 0:
M = P / n

Maximum principal for a given EMI limit, if r > 0:
P = M × (1 - (1 + r)^(-n)) / r

If r = 0:
P = M × n

Total repayment = M × n
Total interest = total repayment - P
Income remaining after proposed payment = D - M
```

An EMI is like a fixed monthly slice of the repayment plan: each payment covers interest for that month plus some principal. The code uses numerically stable `expm1` / `log1p` forms for positive interest and a separate zero-rate formula to avoid dividing by zero. Input checks require finite nonnegative money values, a 0%-100% annual rate, positive whole-month tenure, and a 0%-100% affordability share. [ `src/affordability.py`]

### Recorded synthetic example

The demonstration uses monthly income 60,000, current payments/obligations 20,000, disposable income 40,000, a 30% share, 12% annual rate, and 36 months. The illustrative EMI cap is 12,000; the corresponding indicative principal is 361,290.06. A proposed principal of 700,000 has EMI 23,250.02, total repayment 837,000.61, interest 137,000.61, and leaves 16,749.98 after current obligations and this EMI. The example has no named currency and uses synthetic values. “Above illustrative limit” is not a credit-risk classification. [ `outputs/affordability/demonstration_assessment.json`; `outputs/affordability/affordability_report.md`]

## 10. Streamlit dashboard

The Streamlit application is `app.py`; reusable input/schema and inference checks live in `src/application.py`. It loads the saved age-excluded Model B, validates its expected 19-column input schema, and loads previously generated fairness, proxy-ablation, and repeated-validation results. It does not train a model or recompute the research experiments when a visitor opens the dashboard. [ `app.py`; `src/application.py`; `README.md`]

### Tabs

1. **Applicant assessment:** collects the 19 model inputs (all original attributes except age), reports Good/Bad probabilities and the predicted label, gives model-contribution explanations, and warns near the 50% decision boundary. Age can be used separately in group analysis but is not part of the prediction form. [ `app.py`; `src/application.py`; `src/explainability.py`]
2. **Affordability:** accepts monthly budget and loan assumptions, allows the affordability share to be changed, and shows EMI, indicative principal, repayment totals, interest, and the illustrative-limit comparison. This is the independent rule-based calculator. [ `app.py`; `src/affordability.py`]
3. **Fairness & validation:** displays saved age-group findings, feature-ablation comparisons, and repeated-validation summaries. These are the checked-in results, not a live experiment runner. [ `app.py`; `outputs/repeated_validation/`; `outputs/proxy_ablation/`]

The dashboard includes inline visual elements and tables, caches loaded artifacts/results, checks the saved model’s schema, and includes prototype limitations. It does not persist visitor-entered application data as a project database. [ `app.py`; `src/application.py`; `tests/test_application.py`]

## 11. Running and deployment

### Run locally

From the repository root, install the pinned requirements and start Streamlit:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The older research workflow can be started with `python main.py`; it downloads dataset 144 and generates baseline/audit outputs. Explainability and affordability have separate module commands documented in `README.md`. [ `README.md`; `requirements.txt`]

### GitHub and Streamlit hosting

The project history records a GitHub repository and a Streamlit Community Cloud deployment, and the README contains the local launch/dependency instructions. The recorded app address is `https://immortals-ai.streamlit.app/`; live availability was not checked as part of this handbook. Hosting state and URLs can change. The project’s model artifacts and precomputed reports let the dashboard load without training at startup. [Codex conversations: *Baseline*, *Organizing*, *Dashboard*; `README.md`; `requirements.txt`; `app.py`]

For a typical Streamlit Community Cloud deployment, the repository must contain the app entry point, requirements file, and the saved model/results the app loads. A user selects the GitHub repository and `app.py` in the hosting interface; the service installs dependencies and starts the app. This is an explanation of the deployment shape, not a claim that every current cloud setting has been rechecked. The README also documents a Windows Application Control issue encountered in one local environment and a base-Python workaround; it is environment-specific, not a requirement for every user. [ `README.md`; `requirements.txt`; Codex conversation: *Dashboard*]

### Historical repository state

At investigation time, the local checkout was on `fairloan-core-implementation` and had deployment-related work ahead of its tracked remote-main reference. This handbook does not change branches, commit, or push. Treat Git branch and remote details as a snapshot, not a promise about later state. [Local repository Git metadata; Codex conversation: *Dashboard*]

## 12. Testing and verification

The project contains unittest modules for data auditing, age ablation, proxy ablation, repeated validation, explainability, affordability, and application/schema behavior. Together these modules contain 38 test cases by source inspection. The Codex history reports that the project test suite passed in a prior run; this handbook task did not rerun tests. `tests/results.md` and `tests/demo_script.md` are present but empty, so they do not contain a saved test transcript or demo script. [ `tests/`; Codex conversations: *Organizing*, *Dashboard*]

Several checks specifically protect artifacts and app behavior: schema and prediction consistency, age exclusion in the 19-field model, loading saved research results, and avoiding model fitting or model-artifact changes during dashboard integration. The model artifacts were inspected read-only during this investigation. [ `tests/test_application.py`; `tests/test_explainability.py`; `outputs/age_ablation/model_b_without_age.joblib`]

All reported experiment numbers in this handbook were cross-checked against repository JSON/CSV/report outputs, including the fixed-split metrics, group numerators and denominators, proxy-prediction scores, repeated-validation summaries, explanation examples, and affordability demonstration. Source outputs are identified next to each result. This is artifact verification, not a new model run. [Files cited in Sections 4-9]

## 13. Limitations and responsible use

1. **Old and small sample:** 1,000 historical cases cannot represent modern applicants, lending markets, countries, or current economic conditions. [ `outputs/reports/data_audit_report.md`; PDF-1, p. 1]
2. **Historical labels:** the Good/Bad target reflects old recorded outcomes and decisions, not a perfect or morally neutral label. Previously rejected applicants may be absent. [PDF-1, p. 1; `outputs/reports/data_audit_report.md`]
3. **Small younger subgroup:** only 39 under-25 people are in the original 20% test split; rates can move noticeably when a few cases change. [ `outputs/reports/age_fairness_audit.md`]
4. **Fairness has multiple definitions:** equal approval rates, equal error rates, and other goals can conflict. The project reports multiple metrics and does not declare a single universal fairness definition. [PDF-1, p. 2]
5. **Removing a protected input is not enough:** other features can correlate with age, and feature ablation can change several errors in different directions. [ `outputs/reports/proxy_analysis_report.md`; `outputs/proxy_ablation/`; PDF-1, p. 2]
6. **No causal conclusions:** feature associations and model contributions show statistical relationships or model mechanics, not why people receive outcomes or what would happen under a policy change. [ `outputs/reports/data_audit_report.md`; `outputs/explainability/explainability_report.md`]
7. **No external or temporal validation:** repeated splits reuse the same dataset; there is no current lender dataset or real deployment evaluation. [ `outputs/repeated_validation/repeated_validation_report.md`]
8. **Not a lending policy:** the model probability, class label, 5:1 error cost, and 30% affordability share are prototype choices. They are not adopted by a bank or validated legal standards. [ `src/model.py`; `src/affordability.py`; PDF-1, p. 3]
9. **No applicant appeal or human-review workflow:** the interface is a research demo; it is not built to issue regulated decisions, explain a decision to a real applicant, or correct a source-data error. [ `app.py`; `src/application.py`]
10. **Limited group detail:** the project compares age groups only. It does not establish intersectional fairness, and the source personal-status field combines attributes. [ `outputs/reports/data_audit_report.md`; PDF-1, pp. 1-2]

Responsible reporting means explaining the sample, metrics, mistakes, uncertainty, assumptions, and data limits; not presenting one good score as proof of a safe system. That emphasis is consistent with the reference material’s instructions to use held-out data, report failures honestly, cite public data/tools, and label finance work as a prototype rather than financial advice. [PDF-2, pp. 9-16; PDF-3, pp. 11-15; PDF-4, sections 2-3, 4.2]

## 14. Beginner glossary

| Term | Plain-language meaning |
|---|---|
| Applicant / row | One record in the dataset. Here, one row represents one historical loan applicant. |
| Attribute / feature | An input column the model can use, such as loan duration or savings band. |
| Target / label | The answer the model is trained to predict; here, Good (`1`) or Bad (`2`). |
| Categorical value | A named group such as housing type, stored as a code. |
| Numeric feature | A number such as age or requested amount. |
| Preprocessing | Preparing raw inputs into a consistent form a model can use. |
| Imputation | Filling in a missing value using a chosen rule. |
| Standardization | Rescaling a numeric feature so its magnitude is comparable with others. |
| One-hot encoding | Turning each category choice into a separate yes/no input column. |
| Training set | Examples used to learn model weights. |
| Test set | Held-back examples used to estimate performance after training. |
| Data leakage | Test information accidentally influencing training, making results look too good. |
| Baseline | A simple reference prediction, such as always choosing the most common label. |
| Accuracy | Fraction of predictions that match the recorded labels. |
| Precision | Of the examples predicted as a class, the share that truly belong to it. |
| Recall / true-positive rate | Of the examples truly in a class, the share the model finds. |
| False approval | In this project, a Bad-risk example predicted Good. |
| False rejection | A Good-risk example predicted Bad. |
| Confusion matrix | A table counting correct predictions and each type of mistake. |
| ROC-AUC | A score for how well probability scores rank positives above negatives across thresholds. |
| Logistic Regression | A model that maps a weighted combination of inputs to a probability for a class. |
| Coefficient | A learned weight showing how an encoded input shifts the model score. |
| Proxy | A feature that carries information about another attribute, even if that attribute is removed. |
| Ablation | An experiment that retrains or evaluates a model after removing one or more inputs. |
| Cross-validation | Repeatedly holding out different folds of the same dataset to examine variability. |
| Out-of-fold prediction | A prediction for a row made by a model that did not train on that row in that fold. |
| Percentage point | Direct difference between two percentages; 80% minus 60% is 20 points. |
| EMI | Equated monthly installment: the fixed monthly payment in a loan schedule. |
| Principal | The amount originally borrowed, before interest. |
| Disposable income | Income remaining after subtracting existing repayments and obligations. |
| Calibration | Whether predicted probabilities match observed outcome frequencies. This project does not report a dedicated calibration study. |
| Intersectionality | Studying combined groups, such as younger women, rather than only one attribute at a time. |

## 15. Likely judge questions and accurate answers

These answers describe the checked-in project, not a claim of production readiness. Relevant evidence is cited in each answer.

1. **What problem does Immortals AI address?** It studies whether a simple credit-risk model’s predictions and errors differ across age groups, and demonstrates model explanations and a separate affordability calculator. [README.md; PDF-4, section 4.2.3]
2. **Who is the user?** The dashboard is a prototype for exploring model behavior and illustrating loan-payment calculations. It has not been tested with loan officers or applicants. [ `app.py`; `README.md`]
3. **Which dataset did you use?** Statlog German Credit from UCI, dataset 144: 1,000 rows, 20 attributes, 700 Good and 300 Bad labels. [ `src/data_loader.py`; `outputs/reports/dataset_profile.json`]
4. **Why this dataset?** It is a small, public, tabular dataset suitable for a weekend-scale Logistic Regression and it is the dataset named in the finance-track loan fairness brief. [ `data/README.md`; PDF-4, section 4.2.3]
5. **What does “Good” mean?** It is the source dataset’s target code `1`, not a guarantee that a person will repay. Code `2` is labeled Bad. [ `src/data_loader.py`; `outputs/reports/feature_dictionary.csv`]
6. **Why Logistic Regression?** It is a fast, understandable two-class baseline that produces probability scores and coefficients. It is still a statistical model and does not establish causality. [ `src/model.py`; PDF-2, p. 9]
7. **How did you split the data?** The baseline uses a stratified 80/20 split and seed 42; preprocessing is fit on training data. [ `src/model.py`]
8. **What was the baseline?** Predict Good risk for every applicant. It gets 70% accuracy because 70% of labels are Good, but misses all Bad cases in the test split. [ `models/stage1_metrics.json`]
9. **How did the model compare?** On the fixed held-out split: 78% accuracy, 70.95% balanced accuracy, 0.804 ROC-AUC, 28 false approvals, 16 false rejections, and weighted cost 156. [ `models/stage1_metrics.json`]
10. **Why isn’t accuracy enough?** A majority-class prediction can have high accuracy while missing the minority class; false approvals and false rejections have different consequences. [PDF-1, p. 2; `docs/stage1_report.md`]
11. **What cost weights did you choose?** The project assigns five units to each false approval and one to each false rejection. This follows the German Credit task’s 5:1 example but is a prototype assumption, not a lender’s measured loss function. [ `src/model.py`; PDF-4, section 4.2.3]
12. **What age groups did you compare?** Applicants under 25 versus applicants at least 25 years old. [ `src/fairness.py`]
13. **What was the approval-rate gap?** On the original 200-person test split, 61.5% under 25 versus 79.5% at least 25, an absolute gap of 17.96 percentage points. [ `outputs/reports/age_fairness_audit.md`]
14. **Does that prove discrimination?** No. It is a descriptive difference on a small old dataset and does not separate legitimate factors from bias or establish causation. [ `outputs/reports/data_audit_report.md`; PDF-1, pp. 1-2]
15. **What happens when age is removed?** Accuracy changes from 78.0% to 77.5%, cost from 156 to 157, and the one-split approval gap from 17.96 to 17.34 points. Other group error gaps remain. [ `outputs/age_ablation/age_ablation_report.md`]
16. **What is a proxy feature?** A remaining input that may encode information about age. The dataset shows age associations in housing, residence duration, and employment duration. [ `outputs/reports/age_group_feature_comparison.csv`]
17. **Can the other columns predict age group?** The proxy audit gets 0.638 balanced accuracy and 0.778 ROC-AUC without age or the credit label, above its 0.5 balanced-accuracy majority baseline. That means age-group information remains in the inputs; it does not prove unfair model use. [ `outputs/metrics/proxy_prediction_metrics.json`]
18. **Did proxy removal fix fairness?** No single result establishes that. In one split the joint removal greatly reduced the approval gap but increased the false-approval gap; repeated validation gives a more modest and different pattern. [ `outputs/proxy_ablation/proxy_ablation_report.md`; `outputs/repeated_validation/repeated_validation_report.md`]
19. **Why do the fixed-split and repeated-validation numbers differ?** They use different evaluation procedures: one fixed 200-row holdout versus repeated cross-validation with out-of-fold predictions over the same dataset. [ `src/repeated_validation.py`; reports in `outputs/`]
20. **Are the repeated ranges confidence intervals?** No. They are empirical percentiles over ten repeats that reuse the same 1,000 rows, not formal population confidence intervals. [ `outputs/repeated_validation/repeated_validation_report.md`]
21. **How do you explain an individual prediction?** Multiply each transformed input by its fitted coefficient, sum with the intercept, and show the strongest positive and negative log-odds contributions. [ `src/explainability.py`]
22. **Do explanations prove fairness?** No. They describe how the saved model score was produced; coefficients are not causal effects and do not prove a decision is correct or fair. [ `outputs/explainability/explainability_report.md`]
23. **What does the affordability calculator do?** It subtracts existing monthly obligations from income, applies a configurable share to get an illustrative EMI ceiling, and calculates EMI or a corresponding principal using an annuity formula. [ `src/affordability.py`]
24. **Is the 30% affordability share a standard?** No. It is the prototype’s default illustrative setting and can be changed. The project has not validated it against lenders or households. [ `src/affordability.py`; `README.md`]
25. **Does the affordability tab affect credit risk?** No. It is a separate rule-based calculator; it does not use German Credit features or alter the saved model. [ `src/affordability.py`; `README.md`]
26. **Does the app train models when opened?** No. It loads saved Model B and research outputs; research commands generate those outputs separately. [ `app.py`; `src/application.py`; `README.md`]
27. **Does the app use age to predict?** No. The deployed assessment schema has 19 inputs and excludes age. Age is used separately to group predictions in the fairness analysis. [ `src/application.py`; `tests/test_application.py`]
28. **What does the 50% warning mean?** The app flags probabilities within 10 percentage points of 50% as near the classifier boundary. It indicates model uncertainty around that cutoff, not uncertainty quantified by a confidence interval. [ `src/explainability.py`; `app.py`]
29. **How many tests are there?** The repository has seven test modules with 38 test cases by inspection. Prior Codex history reports the suite passed; this handbook run did not rerun it. [ `tests/`; Codex conversation: *Dashboard*]
30. **What is the biggest limitation?** A 1,000-row old dataset with historic labels and no external validation cannot support present-day real lending decisions or broad fairness conclusions. [ `outputs/reports/data_audit_report.md`; PDF-1, pp. 1-3]
31. **What would you do next with more time?** Validate on a current, appropriately governed dataset; predefine fairness goals with affected stakeholders; examine uncertainty and intersectional groups; and test explanations and usability with people. These are proposed next steps, not implemented features. [PDF-1, pp. 2-3; PDF-3, pp. 12-15]
32. **Is this financial advice?** No. It is a prototype for research and demonstration and should not be used to approve, reject, or price real credit. [ `README.md`; PDF-3, p. 15; PDF-4, section 2]

## 16. Repository file index

This index covers project-owned source, documentation, test, data, model, and result files. Virtual-environment package contents, `.git` internals, and Python cache contents were inventoried but are machine-generated/runtime internals, not project deliverables; they are not enumerated file-by-file here.

### Folder map

| Folder | Purpose / observed contents |
|---|---|
| `docs/` | Baseline report and this handbook. |
| `src/` | Reusable data, modeling, analysis, fairness, explanation, application, and affordability modules. |
| `tests/` | Unit and integration checks plus two empty placeholder Markdown files. |
| `notebooks/` | Baseline metric-reading notebook. |
| `models/` | Saved baseline pipeline, preprocessor, and baseline metrics. |
| `data/eda/` | Eight checked-in exploratory charts. |
| `data/raw/` | Empty in the inspected project tree; raw UCI rows are fetched by the loader rather than stored here. |
| `data/processed/` | Empty in the inspected project tree. |
| `outputs/reports/`, `outputs/metrics/`, `outputs/plots/` | Dataset, fairness, proxy reports and exploratory figures. |
| `outputs/age_ablation/`, `outputs/proxy_ablation/`, `outputs/repeated_validation/` | Experiment reports, structured metrics, comparison tables, split/fold details, and selected fitted models. |
| `outputs/explainability/`, `outputs/affordability/` | Saved interpretation and synthetic calculator demonstrations. |
| `.venv/`, `.venv-1/`, `.venv312/` | Local Python environments; package files are runtime dependencies, not project source. |
| `__pycache__/` | Python-generated bytecode cache. |
| `.git/` | Git object database and metadata. |

### Root and documentation

| File | Purpose |
|---|---|
| `README.md` | Project summary, local commands, model results, application behavior, limitations, affordability notes, and Windows-specific startup note. |
| `main.py` | Command-line entry point for baseline modeling and analysis; includes fairness-only mode using the saved baseline. |
| `app.py` | Streamlit user interface with applicant, affordability, and fairness/validation tabs. |
| `requirements.txt` | Pinned Python dependency versions for reproducible setup. |
| `.gitignore` | Excludes generated/runtime data such as virtual environments, caches, and raw data. |
| `.env.example` | Empty environment-variable template; no configured secrets or settings. |
| `docs/stage1_report.md` | Original baseline summary, split, metrics, and plot references. |
| `docs/IMMORTALS_AI_MASTER_HANDBOOK.md` | This project-history and technical handbook. |
| `notebooks/baseline_exploration.ipynb` | Small notebook that reads saved baseline metrics for display; it does not fetch data or train a model. |
| `data/README.md` | Explains the UCI dataset source and that raw data is not stored in the repository. |

### Python package: `src/`

| File | Purpose |
|---|---|
| `src/__init__.py` | Marks `src` as an importable Python package. |
| `src/data_loader.py` | Fetches UCI dataset 144, validates its metadata/target, and exposes feature descriptions. |
| `src/model.py` | Builds preprocessing and Logistic Regression pipeline; splits, trains, evaluates, saves, loads, and predicts. |
| `src/data_analysis.py` | Produces dataset-quality, age-group, association, proxy-prediction analyses, reports, and exploratory charts. |
| `src/fairness.py` | Computes age-group approval, false-approval, false-rejection, and true-positive metrics and gaps. |
| `src/age_ablation.py` | Compares the saved all-feature model with a model trained without age on a matched split. |
| `src/proxy_ablation.py` | Runs candidate single-feature and grouped feature-removal experiments. |
| `src/repeated_validation.py` | Runs repeated stratified 5-fold validation with matched feature-set folds and out-of-fold predictions. |
| `src/explainability.py` | Calculates global feature contribution summaries and individual synthetic case explanations from saved Model B. |
| `src/affordability.py` | Validates budget inputs and calculates EMI, total repayment, interest, and indicative principal. |
| `src/application.py` | Loads and validates saved inference artifacts, feature schema, predictions, explanations, and research data for the app. |

### Tests

| File | Purpose |
|---|---|
| `tests/test_data_audit.py` | Checks dataset audit and descriptive-statistic behavior. |
| `tests/test_age_ablation.py` | Checks age-removal experiment setup and comparison. |
| `tests/test_proxy_ablation.py` | Checks candidate ablation configuration and output behavior. |
| `tests/test_repeated_validation.py` | Checks repeated validation folds, summaries, and metric calculations. |
| `tests/test_explainability.py` | Checks contributions and probability reconstruction. |
| `tests/test_affordability.py` | Checks EMI formulas, input validation, zero-rate case, and assessment outputs. |
| `tests/test_application.py` | Checks 19-field schema, saved-artifact integration, UI contract, and no fit/retrain at app startup. |
| `tests/results.md` | Empty placeholder; contains no recorded results. |
| `tests/demo_script.md` | Empty placeholder; contains no recorded script. |

### Saved baseline model files

| File | Purpose |
|---|---|
| `models/logistic_regression_pipeline.joblib` | Saved fitted all-feature baseline pipeline (includes age among the 20 model inputs). |
| `models/preprocessor.joblib` | Saved preprocessing artifact associated with the baseline workflow. |
| `models/stage1_metrics.json` | Baseline metrics, confusion matrix, and evaluation details. |

### Data visuals: `data/eda/`

| File | Purpose |
|---|---|
| `data/eda/target_distribution.png` | Counts of Good and Bad labels. |
| `data/eda/age_attribute13.png` | Age distribution in the source dataset. |
| `data/eda/credit_amount_attribute5.png` | Credit amount distribution/comparison. |
| `data/eda/credit_history_attribute3.png` | Credit-history category distribution. |
| `data/eda/duration_attribute2.png` | Loan duration distribution/comparison. |
| `data/eda/employment_attribute7.png` | Employment-duration categories. |
| `data/eda/loan_purpose_attribute4.png` | Loan-purpose categories. |
| `data/eda/savings_attribute6.png` | Savings categories. |

### Research figures: `outputs/plots/`

| File | Purpose |
|---|---|
| `outputs/plots/age_distribution.png` | Distribution of age values. |
| `outputs/plots/age_group_distribution.png` | Counts for Under 25 versus 25+. |
| `outputs/plots/categorical_feature_comparisons.png` | Categorical inputs compared across risk/group labels. |
| `outputs/plots/credit_risk_by_age_group.png` | Historical target composition by age group. |
| `outputs/plots/credit_risk_distribution.png` | Risk-label distribution chart. |
| `outputs/plots/feature_association_heatmap.png` | Feature-to-age association overview. |
| `outputs/plots/numerical_feature_comparisons.png` | Numeric feature comparisons across groups/labels. |

### Reports and source dictionary: `outputs/reports/`

| File | Purpose |
|---|---|
| `outputs/reports/dataset_profile.json` | Machine-readable dataset size, feature types, and quality summary. |
| `outputs/reports/dataset_profile.md` | Short human-readable dataset profile. |
| `outputs/reports/data_quality_report.md` | Missingness and duplicate-row audit. |
| `outputs/reports/data_audit_report.md` | Overall dataset, age-group, proxy, and limitation summary. |
| `outputs/reports/feature_dictionary.csv` | Original attribute descriptions and category code meanings. |
| `outputs/reports/age_group_outcomes.csv` | Observed Good/Bad outcome proportions and intervals by age group. |
| `outputs/reports/age_group_feature_comparison.csv` | Statistics for comparing features across age groups. |
| `outputs/reports/age_group_model_fairness.csv` | Baseline model metrics by actual age group. |
| `outputs/reports/age_group_fairness_table.csv` | Human-readable fairness rate table. |
| `outputs/reports/age_group_fairness_gaps.csv` | Age-group rate-gap summary. |
| `outputs/reports/age_fairness_audit.md` | Explanation and interpretation of fixed-split age audit. |
| `outputs/reports/proxy_analysis_report.md` | Summary of age associations and age-prediction experiment. |

### Proxy prediction and fixed-split age ablation

| File | Purpose |
|---|---|
| `outputs/metrics/proxy_prediction_metrics.json` | Held-out and cross-validation scores for predicting age group without age or credit label. |
| `outputs/age_ablation/age_ablation_report.md` | Human-readable comparison of Model A and Model B. |
| `outputs/age_ablation/experiment_results.json` | Structured model metrics for the age ablation. |
| `outputs/age_ablation/model_comparison.csv` | Side-by-side predictive metrics. |
| `outputs/age_ablation/fairness_metrics.csv` | Group rates for each model. |
| `outputs/age_ablation/fairness_gaps.csv` | Absolute group gaps by model. |
| `outputs/age_ablation/fairness_table.csv` | Formatted subgroup comparison. |
| `outputs/age_ablation/split_indices.csv` | Fixed train/test row indices to make paired comparisons reproducible. |
| `outputs/age_ablation/model_b_without_age.joblib` | Saved age-excluded 19-input Model B used by dashboard and explanations. |

### Proxy feature ablation: `outputs/proxy_ablation/`

| File | Purpose |
|---|---|
| `outputs/proxy_ablation/proxy_ablation_report.md` | Experiment design, results, candidate definitions, and interpretation. |
| `outputs/proxy_ablation/experiment_results.json` | Structured fixed-split experiment metrics. |
| `outputs/proxy_ablation/candidate_features.csv` | Preselected candidate features and descriptions. |
| `outputs/proxy_ablation/model_comparison.csv` | Predictive score comparison for each feature set. |
| `outputs/proxy_ablation/performance_changes.csv` | Changes from age-excluded reference model. |
| `outputs/proxy_ablation/fairness_metrics.csv` | Group rate metrics for candidate models. |
| `outputs/proxy_ablation/fairness_gaps.csv` | Absolute gaps for each fairness metric/model. |
| `outputs/proxy_ablation/fairness_table.csv` | Tabular group rates. |
| `outputs/proxy_ablation/split_indices.csv` | Shared training/test rows across ablation comparisons. |
| `outputs/proxy_ablation/models/without_checking_account_status_attribute1.joblib` | Saved pipeline without checking status. |
| `outputs/proxy_ablation/models/without_loan_purpose_attribute4.joblib` | Saved pipeline without loan purpose. |
| `outputs/proxy_ablation/models/without_savings_attribute6.joblib` | Saved pipeline without savings. |
| `outputs/proxy_ablation/models/without_employment_duration_attribute7.joblib` | Saved pipeline without employment duration. |
| `outputs/proxy_ablation/models/without_housing_attribute15.joblib` | Saved pipeline without housing. |
| `outputs/proxy_ablation/models/without_savings_and_employment_duration_and_housing.joblib` | Saved pipeline without the three-feature group. |

### Repeated validation: `outputs/repeated_validation/`

| File | Purpose |
|---|---|
| `outputs/repeated_validation/experiment_metadata.json` | Fold/repeat settings, seed, feature sets, and run configuration. |
| `outputs/repeated_validation/fold_assignments.csv` | Row-to-fold assignment for each repeat. |
| `outputs/repeated_validation/fold_sizes.csv` | Training/validation fold sizes. |
| `outputs/repeated_validation/oof_predictions.csv` | Out-of-fold predictions and scores for each row/repeat/model. |
| `outputs/repeated_validation/repeat_performance_metrics.csv` | Performance metrics for each repeat and feature set. |
| `outputs/repeated_validation/performance_summary.csv` | Mean and empirical summary of predictive performance. |
| `outputs/repeated_validation/subgroup_denominators.csv` | Actual subgroup denominators used for group-rate calculations. |
| `outputs/repeated_validation/repeat_fairness_metrics.csv` | Age-group rates for each repeat and model. |
| `outputs/repeated_validation/fairness_summary.csv` | Summarized group rates across repeats. |
| `outputs/repeated_validation/fairness_gap_summary.csv` | Mean and empirical ranges of absolute fairness gaps. |
| `outputs/repeated_validation/paired_changes_vs_age_excluded.csv` | Per-repeat paired metric differences relative to Model B. |
| `outputs/repeated_validation/paired_change_summary.csv` | Summary and sign counts of paired differences. |
| `outputs/repeated_validation/repeated_validation_report.md` | Full experiment description, tables, interpretation, and limitations. |

### Explainability and affordability outputs

| File | Purpose |
|---|---|
| `outputs/explainability/explainability_report.md` | Global feature influence and synthetic case explanations. |
| `outputs/explainability/global_feature_importance.csv` | Aggregated per-feature contribution ranking. |
| `outputs/explainability/logistic_regression_coefficients.csv` | Coefficients for transformed numeric and one-hot inputs. |
| `outputs/explainability/demonstration_cases.json` | Input values, probabilities, and contributions for three synthetic examples. |
| `outputs/explainability/global_feature_importance.png` | Bar chart of global average absolute contributions. |
| `outputs/affordability/affordability_report.md` | EMI formula, assumptions, and sample calculation. |
| `outputs/affordability/demonstration_assessment.json` | Structured synthetic affordability example. |

## 17. Sources and source index

The shorthand used throughout the handbook points to the following material. Local chat titles are provided as source identifiers without copying private conversation contents into the repository.

### Repository sources

Use the exact repository paths listed in Section 16; experiment results cite the corresponding output report/JSON/CSV in their section. Key implementation sources are `src/data_loader.py`, `src/model.py`, `src/fairness.py`, `src/data_analysis.py`, `src/age_ablation.py`, `src/proxy_ablation.py`, `src/repeated_validation.py`, `src/explainability.py`, `src/affordability.py`, `src/application.py`, `main.py`, `app.py`, and `README.md`.

### ChatGPT project conversations

1. *Baseline* — original dataset, first model and baseline objectives.
2. *Implement Fairness Auditing* — age-group fairness and proxy analysis.
3. *Summarizing Resources* — project/reference research.
4. *Summarizing Resources (1)* — continued project/reference research.
5. *Summarizing Resources (2)* — continued project/reference research.
6. *Summarizing Resources (3)* — continued project/reference research; the recovered long response repeats the handbook prompt and shared-link recommendation.
7. *Before (Done)* — preparation and project planning context.
8. *Create Hackathon Timetable* — hackathon planning context.

These are the eight local Veritas project histories reviewed for this handbook. The share links were supplied in the user’s task; the handbook cites the conversation titles rather than republishing their URLs or contents.

### Codex conversations

- *Baseline* — repository baseline implementation and review.
- *Organizing* (the local history matching the requested “Organization”) — analysis organization, ablations, and validation work.
- *Dashboard* — explainability, affordability, Streamlit integration, and deployment history.

The provided `chatgpt.com/s/cx_...` links were not the mechanism by which these histories were read; matching histories were accessed locally by title. Only project facts corroborated by the repository are presented as implementation facts.

### Supplied reference PDFs

- **PDF-1:** *Fairness of Loan Models Problems to Know.pdf*, especially dataset limits, proxy concerns, metric tradeoffs, repeated validation, and responsible interpretation (pp. 1-3).
- **PDF-2:** *How to scope a weekend AI project.pdf*, especially problem scoping, data-first planning, baseline-first workflow, and demo/testing schedule (pp. 4-16).
- **PDF-3:** *Veritas AI - Hackathon Guide.pdf*, especially project planning, evaluation, submission, and responsible AI (pp. 3-15).
- **PDF-4:** *Veritas AI International High School Hackathon 2026 Project Briefs.pdf*, especially rules, project expectations, the AI + Finance track, section 4.2.3, submission requirements, judging, and FAQ (pp. 2-4, 8-11, 18-22).

The PDF filenames identify reference documents supplied by the user and are not copies inside this repository. Competition instructions are presented here as context for the project and do not supersede the user’s explicit constraints for this handbook task.
