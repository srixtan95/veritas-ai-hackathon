# Age ablation experiment

Model A reuses the saved all-feature Logistic Regression pipeline. Model B is trained with only the age feature removed. Both use the same stratified train/test rows, seed 42, preprocessing method, classifier settings, and evaluation functions. Preprocessing is fitted on training data only.
Age feature: `Attribute13`. Target: `class`, with 1 = Good risk and 2 = Bad risk. Model A exactly matches the saved baseline metrics.

## Predictive performance

| model | accuracy | balanced_accuracy | roc_auc | false_approvals | false_rejections | error_cost |
|---|---|---|---|---|---|---|
| Model A (all features) | 0.78 | 0.7095238095238094 | 0.8040476190476191 | 28 | 16 | 156 |
| Model B (without age) | 0.775 | 0.7059523809523809 | 0.8016666666666667 | 28 | 17 | 157 |

Cost is 5 × false approvals + false rejections. A false approval is actual Bad risk predicted Good; a false rejection is actual Good risk predicted Bad.

## Fairness rates by actual age group

| Model | Age group | Group size | Approval rate | False approval rate | False rejection rate | True positive rate |
|---|---|---|---|---|---|---|
| Model A (all features) | Under 25 | 39 | 61.5% (24/39) | 42.1% (8/19) | 20.0% (4/20) | 80.0% (16/20) |
| Model A (all features) | 25 or older | 161 | 79.5% (128/161) | 48.8% (20/41) | 10.0% (12/120) | 90.0% (108/120) |
| Model B (without age) | Under 25 | 39 | 61.5% (24/39) | 42.1% (8/19) | 20.0% (4/20) | 80.0% (16/20) |
| Model B (without age) | 25 or older | 161 | 78.9% (127/161) | 48.8% (20/41) | 10.8% (13/120) | 89.2% (107/120) |

Each rate includes numerator/denominator. Age is used from the original test rows for grouping, not from Model B inputs.

## Absolute age-group gaps

| model | metric | absolute_gap_percentage_points |
|---|---|---|
| Model A (all features) | Approval rate | 17.964644051600565 |
| Model A (all features) | False approval rate | 6.675224646983313 |
| Model A (all features) | False rejection rate | 10.0 |
| Model B (without age) | Approval rate | 17.343526039178215 |
| Model B (without age) | False approval rate | 6.675224646983313 |
| Model B (without age) | False rejection rate | 9.166666666666668 |

Differences are descriptive, not proof of discrimination. The under-25 test group is small; interpret its rates cautiously. Removing the age column does not remove possible age proxies from the remaining features.
