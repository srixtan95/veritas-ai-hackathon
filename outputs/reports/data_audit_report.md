# German Credit Data Audit

The dataset has 1000 applicants, 20 features, and target labels 1 = good and 2 = bad credit risk. The personal-status field combines status and sex; the dataset does not provide separate gender or marital-status fields.

## Data quality

Missing cells: 0; duplicate rows: 0. No observations were changed.

## Age-group comparison

Under 25: 149 applicants. 25 or older: 851 applicants. Good/bad label rates and Wilson 95% intervals are in `age_group_outcomes.csv`. Baseline model rates are in `age_group_model_fairness.csv`.

## Feature and proxy investigation

The strongest observed age associations include Attribute15, Attribute11, Attribute7. These are candidate proxy signals, not evidence of unfairness. See `age_group_feature_comparison.csv` and `proxy_analysis_report.md`.

## Limitations

The dataset contains only 1,000 applicants and reflects an older lending context. Age groups are imbalanced, and the smaller under-25 group makes estimates less certain. Historical good/bad labels may reflect past lending and measurement choices. Repayment outcomes can be unavailable for people who were previously denied credit. Savings, housing, employment, and account status also reflect socioeconomic opportunity. All reported differences are descriptive, not causal, and do not establish discrimination or fairness.
