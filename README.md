# Immortals AI
### Fairness-Aware Credit Intelligence

**Track:** AI + Finance  
**Team:** The Immortals  
**Live application:** [immortals-ai.streamlit.app](https://immortals-ai.streamlit.app/)  
**Source code:** [GitHub Repository](https://github.com/srixtan95/veritas-ai-hackathon)

---

## The problem

AI-assisted credit scoring can make loan assessments faster, but a model can produce different outcomes for different demographic groups—even when a sensitive attribute such as age is excluded from its inputs. Accuracy alone does not reveal these differences: a model may perform well overall while making unequal errors or approval decisions across groups.

Immortals AI investigates this problem using the Statlog German Credit dataset. We focus on applicants under 25 compared with applicants aged 25 or older, examining model performance, approval outcomes, and the possibility that other applicant features act as proxies for age.

## What we built

Immortals AI is a research prototype combining a credit-risk prediction dashboard with fairness analysis and supporting financial tools.

Users can:
- Explore credit-risk predictions using an age-excluded model.
- Examine differences in approval rates and prediction errors across age groups.
- Investigate whether non-age features can act as proxies for age.
- Explore model explanations and feature influence.
- Experiment with a separate, rule-based affordability calculator.

The application is designed to make model behaviour easier to inspect—not to make real lending decisions.

## What we added

Building beyond a basic credit-risk classifier, we developed an integrated analysis workflow:

- **Age-excluded modelling:** A second model removes age from its prediction inputs, allowing us to examine whether group differences remain.
- **Proxy analysis:** We test whether other features can predict age group after age and the credit-risk target are excluded.
- **Fairness auditing:** We compare group-level approval rates, false approvals and false rejections.
- **Explainability:** We examine feature contributions to model predictions and provide illustrative applicant examples.
- **Affordability demonstration:** We built a separate EMI-based calculator using an adjustable affordability assumption.

These components help examine different aspects of credit assessment rather than relying on a single accuracy score.

## How it works

**Data.** We use the [Statlog German Credit dataset (UCI Machine Learning Repository, ID 144)](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data), containing 1,000 applicants, 20 attributes, 700 good-credit labels and 300 bad-credit labels.

**Model.** The baseline uses Logistic Regression with a stratified 80/20 train-test split and random seed 42. The application also uses a saved age-excluded Model B for its applicant assessment workflow. The dashboard loads saved models and research outputs rather than retraining experiments at startup.

**Evaluation.** We report accuracy alongside balanced accuracy, ROC-AUC and cost-weighted errors. Under the dataset's cost convention, falsely approving a bad-risk applicant costs five times as much as falsely rejecting a good-risk applicant:

`Total cost = (False approvals × 5) + (False rejections × 1)`

We also compare group-level outcomes and investigate potential proxy features.

## Results

On the held-out test split, the original Logistic Regression baseline achieved:

| Metric | Logistic Regression | Always predict good |
|---|---:|---:|
| Accuracy | 78% | 70% |
| Balanced accuracy | 70.95% | 50% |
| ROC-AUC | 0.804 | — |
| False approvals | 28 | 60 |
| False rejections | 16 | 0 |
| Cost | 156 | 300 |

The model reduces the cost-weighted errors compared with the simple always-good baseline on this split. However, overall performance does not establish that the model is fair. Group-level results and proxy analyses are important because removing age from model inputs does not necessarily remove information associated with age.

The figures above describe the original baseline experiment; they should not be interpreted as the performance of the separate age-excluded Model B.

## Limitations and what we'd do next

The German Credit dataset is small, historical and not representative of every modern lending population. Its labels reflect historical credit-risk classifications, not a complete account of an applicant's financial circumstances. Small subgroup sizes also make fairness estimates uncertain. An observed approval-rate gap does not, by itself, prove discrimination, and removing age does not guarantee fair outcomes.

With more time, we would validate the findings on larger and more recent datasets, report uncertainty around subgroup metrics, and evaluate mitigation methods against both fairness measures and predictive performance.

**This project is a prototype and is not financial advice.**

## Run locally

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Launch the Streamlit dashboard:

```bash
python -m streamlit run app.py
```

Run the baseline and research workflow:

```bash
python main.py
```

Run the saved-baseline age fairness audit without retraining:

```bash
python main.py --fairness-only
```

Run the explainability analysis:

```bash
python -m src.explainability
```

Run the affordability demonstration:

```bash
python -m src.affordability
```

The dashboard uses saved model and research artifacts. The research scripts that download the dataset require an internet connection.

## Project structure

```text
app.py                  Streamlit dashboard
main.py                 Baseline and research workflow
src/
  data_loader.py        Dataset loading
  model.py              Model training and prediction
  data_analysis.py      Dataset and proxy analysis
  fairness.py           Group-level fairness metrics
  explainability.py     Model explanation analysis
  affordability.py      EMI-based affordability calculator
models/                 Saved baseline model artifacts
outputs/                Research results, reports and charts
data/eda/               Exploratory data analysis charts
tests/                  Automated tests
```

## Sources and credits

- **Dataset:** [Statlog German Credit — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
- **Machine learning:** [scikit-learn](https://scikit-learn.org/)
- **Data processing:** [pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/)
- **Dataset access:** [ucimlrepo](https://github.com/uci-ml-repo/ucimlrepo)
- **Visualisation:** [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/)
- **Statistical analysis:** [SciPy](https://scipy.org/)
- **Model persistence:** [Joblib](https://joblib.readthedocs.io/)

**AI tools and learning resources:** Add the AI tools, tutorials and other external resources actually used during development, with links. Do not omit tools used to generate or modify project code.
