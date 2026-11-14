# Proxy feature ablation experiment

The reference is the already-saved age-excluded Model B. Each candidate is removed individually in a separate Logistic Regression model. The pre-defined group experiment removes savings, employment duration, and housing together after the individual experiments. Candidate choices come from UCI feature descriptions and category mappings, not test-set scores.
All models use the same stratified 80/20 split (random_state=42), class 1 = Good / class 2 = Bad, the same Logistic Regression settings, and the existing preprocessing function. Preprocessing is fit on training rows only. Test rows are used only for final evaluation.

## Candidate features and descriptions

| candidate | column | uci_description | uci_code_meanings | observed_code_counts | experiment |
|---|---|---|---|---|---|
| Checking account status | Attribute1 | Status of existing checking account | A11: ... < 0 DM; A12: 0 <= ... < 200 DM; A13: >= 200 DM / salary assignments for at least 1 year; A14: No checking account | A14: 394; A11: 274; A12: 269; A13: 63 | Without Checking account status (Attribute1) |
| Loan purpose | Attribute4 | Purpose | A40: Car (new); A41: Car (used); A42: Furniture/equipment; A43: Radio/television; A44: Domestic appliances; A45: Repairs; A46: Education; A47: Vacation (does not exist?); A48: Retraining; A49: Business; A410: Others | A43: 280; A40: 234; A42: 181; A41: 103; A49: 97; A46: 50; A45: 22; A44: 12; A410: 12; A48: 9 | Without Loan purpose (Attribute4) |
| Savings | Attribute6 | Savings account/bonds | A61: < 100 DM; A62: 100 <= ... < 500 DM; A63: 500 <= ... < 1000 DM; A64: >= 1000 DM; A65: Unknown / no savings account | A61: 603; A65: 183; A62: 103; A63: 63; A64: 48 | Without Savings (Attribute6) |
| Employment duration | Attribute7 | Present employment since | A71: Unemployed; A72: < 1 year; A73: 1 <= ... < 4 years; A74: 4 <= ... < 7 years; A75: >= 7 years | A73: 339; A75: 253; A74: 174; A72: 172; A71: 62 | Without Employment duration (Attribute7) |
| Housing | Attribute15 | Housing | A151: Rent; A152: Own; A153: For free | A152: 713; A151: 179; A153: 108 | Without Housing (Attribute15) |

Group ablation: Savings + employment duration + housing (Attribute6, Attribute7, Attribute15). These three UCI descriptions refer to savings, present employment duration, and housing.

## Predictive performance

| experiment | accuracy | balanced_accuracy | roc_auc | false_approvals | false_rejections | error_cost |
|---|---|---|---|---|---|---|
| Reference (age excluded) | 0.775 | 0.7059523809523809 | 0.8016666666666667 | 28 | 17 | 157 |
| Without Checking account status (Attribute1) | 0.72 | 0.638095238095238 | 0.7379761904761903 | 34 | 22 | 192 |
| Without Loan purpose (Attribute4) | 0.75 | 0.6738095238095239 | 0.7778571428571428 | 31 | 19 | 174 |
| Without Savings (Attribute6) | 0.8 | 0.7238095238095238 | 0.8086904761904762 | 28 | 12 | 152 |
| Without Employment duration (Attribute7) | 0.78 | 0.7142857142857143 | 0.8040476190476191 | 27 | 17 | 152 |
| Without Housing (Attribute15) | 0.785 | 0.7130952380952381 | 0.8107142857142857 | 28 | 15 | 155 |
| Without Savings + employment duration + housing | 0.805 | 0.7273809523809524 | 0.8220238095238094 | 28 | 11 | 151 |

## Performance change from reference

| experiment | accuracy_change_percentage_points | balanced_accuracy_change_percentage_points | roc_auc_change | cost_change |
|---|---|---|---|---|
| Without Checking account status (Attribute1) | -5.500000000000005 | -6.785714285714284 | -0.06369047619047641 | 35 |
| Without Loan purpose (Attribute4) | -2.500000000000002 | -3.214285714285703 | -0.023809523809523947 | 17 |
| Without Savings (Attribute6) | 2.500000000000002 | 1.7857142857142905 | 0.007023809523809432 | -5 |
| Without Employment duration (Attribute7) | 0.5000000000000004 | 0.8333333333333415 | 0.0023809523809523725 | -5 |
| Without Housing (Attribute15) | 1.0000000000000009 | 0.7142857142857229 | 0.009047619047618971 | -2 |
| Without Savings + employment duration + housing | 3.0000000000000027 | 2.1428571428571463 | 0.02035714285714263 | -6 |

## Age-group fairness metrics

| Experiment | Age group | Group size | Approval rate | False approval rate | False rejection rate | True positive rate |
|---|---|---|---|---|---|---|
| Reference (age excluded) | Under 25 | 39 | 61.5% (24/39) | 42.1% (8/19) | 20.0% (4/20) | 80.0% (16/20) |
| Reference (age excluded) | 25 or older | 161 | 78.9% (127/161) | 48.8% (20/41) | 10.8% (13/120) | 89.2% (107/120) |
| Without Checking account status (Attribute1) | Under 25 | 39 | 71.8% (28/39) | 57.9% (11/19) | 15.0% (3/20) | 85.0% (17/20) |
| Without Checking account status (Attribute1) | 25 or older | 161 | 77.0% (124/161) | 56.1% (23/41) | 15.8% (19/120) | 84.2% (101/120) |
| Without Loan purpose (Attribute4) | Under 25 | 39 | 64.1% (25/39) | 47.4% (9/19) | 20.0% (4/20) | 80.0% (16/20) |
| Without Loan purpose (Attribute4) | 25 or older | 161 | 78.9% (127/161) | 53.7% (22/41) | 12.5% (15/120) | 87.5% (105/120) |
| Without Savings (Attribute6) | Under 25 | 39 | 66.7% (26/39) | 52.6% (10/19) | 20.0% (4/20) | 80.0% (16/20) |
| Without Savings (Attribute6) | 25 or older | 161 | 80.7% (130/161) | 43.9% (18/41) | 6.7% (8/120) | 93.3% (112/120) |
| Without Employment duration (Attribute7) | Under 25 | 39 | 66.7% (26/39) | 47.4% (9/19) | 15.0% (3/20) | 85.0% (17/20) |
| Without Employment duration (Attribute7) | 25 or older | 161 | 77.0% (124/161) | 43.9% (18/41) | 11.7% (14/120) | 88.3% (106/120) |
| Without Housing (Attribute15) | Under 25 | 39 | 66.7% (26/39) | 47.4% (9/19) | 15.0% (3/20) | 85.0% (17/20) |
| Without Housing (Attribute15) | 25 or older | 161 | 78.9% (127/161) | 46.3% (19/41) | 10.0% (12/120) | 90.0% (108/120) |
| Without Savings + employment duration + housing | Under 25 | 39 | 76.9% (30/39) | 63.2% (12/19) | 10.0% (2/20) | 90.0% (18/20) |
| Without Savings + employment duration + housing | 25 or older | 161 | 78.9% (127/161) | 39.0% (16/41) | 7.5% (9/120) | 92.5% (111/120) |

## Absolute age-group gaps

| experiment | metric | absolute_gap_percentage_points |
|---|---|---|
| Reference (age excluded) | Approval rate | 17.343526039178215 |
| Reference (age excluded) | False approval rate | 6.675224646983313 |
| Reference (age excluded) | False rejection rate | 9.166666666666668 |
| Without Checking account status (Attribute1) | Approval rate | 5.223761745500877 |
| Without Checking account status (Attribute1) | False approval rate | 1.797175866495504 |
| Without Checking account status (Attribute1) | False rejection rate | 0.8333333333333331 |
| Without Loan purpose (Attribute4) | Approval rate | 14.779423475075648 |
| Without Loan purpose (Attribute4) | False approval rate | 6.290115532734281 |
| Without Loan purpose (Attribute4) | False rejection rate | 7.500000000000001 |
| Without Savings (Attribute6) | Approval rate | 14.078674948240167 |
| Without Savings (Attribute6) | False approval rate | 8.729139922978174 |
| Without Savings (Attribute6) | False rejection rate | 13.333333333333336 |
| Without Employment duration (Attribute7) | Approval rate | 10.351966873706008 |
| Without Employment duration (Attribute7) | False approval rate | 3.4659820282413323 |
| Without Employment duration (Attribute7) | False rejection rate | 3.3333333333333326 |
| Without Housing (Attribute15) | Approval rate | 12.215320910973093 |
| Without Housing (Attribute15) | False approval rate | 1.0269576379974277 |
| Without Housing (Attribute15) | False rejection rate | 4.999999999999999 |
| Without Savings + employment duration + housing | Approval rate | 1.9589106545628288 |
| Without Savings + employment duration + housing | False approval rate | 24.13350449293966 |
| Without Savings + employment duration + housing | False rejection rate | 2.500000000000001 |

## Interpretation

Among individual removals, excluding checking-account status had the largest predictive decline: accuracy changed by -5.50 percentage points, balanced accuracy by -6.79 points, ROC-AUC by -0.064, and cost by +35. Excluding savings gave the largest accuracy and balanced-accuracy increases among the individual removals (+2.50 and +1.79 points; cost -5); these are single-split results, not evidence that savings harms generalization.

The group removal changed accuracy by +3.00 points and cost by -6. Its approval-rate gap fell from 17.34 to 1.96 points, while its false-approval gap rose from 6.68 to 24.13 points. The group experiment therefore does not show a uniform fairness improvement.

Rates include numerator/denominator in `fairness_table.csv`. Approval denominator is all applicants in that age group; false approval denominator is actual Bad-risk applicants in that group; false rejection denominator is actual Good-risk applicants in that group. `candidate_features.csv` lists UCI code meanings and observed feature-column category counts.

A feature removal that changes model scores or subgroup gaps does not prove the feature causes unfair outcomes. The dataset has only 1,000 historical cases and the under-25 test group is small. Remaining features may also encode age-related information. These results are exploratory and use one fixed split.
