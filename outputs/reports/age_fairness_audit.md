# Age-based fairness audit

This audit uses predictions from the saved Logistic Regression model on the existing stratified 20% test split (seed 42). It does not fit or change the model.
The target is not included in prediction inputs. The saved baseline was trained with all 20 original features, including the age feature, so this audit preserves that baseline model as requested.

## Test-set comparison

| Age group | Group size | Approval rate | False approval rate | False rejection rate | True positive rate |
|---|---|---|---|---|---|
| Under 25 | 39 | 61.5% (24/39) | 42.1% (8/19) | 20.0% (4/20) | 80.0% (16/20) |
| 25 or older | 161 | 79.5% (128/161) | 48.8% (20/41) | 10.0% (12/120) | 90.0% (108/120) |

Rates show their numerator and denominator in parentheses. Approval rate = predicted-good applicants / all applicants in group. False approval rate = actual-bad applicants predicted good / actual-bad applicants in group. False rejection rate = actual-good applicants predicted bad / actual-good applicants in group. True positive rate = actual-good applicants predicted good / actual-good applicants in group.

## Absolute rate gaps

- Approval rate: 17.96 percentage points.
- False approval rate: 6.68 percentage points.
- False rejection rate: 10.00 percentage points.

## Interpretation

- Approval rate is higher for 25 or older; false approval rate is higher for 25 or older; false rejection rate is higher for Under 25; true positive rate is higher for 25 or older.
- These are descriptive differences, not proof of discrimination. Only 39 under-25 applicants are in the held-out test set, so those subgroup rates are less stable. The German Credit dataset is small and reflects historical lending outcomes.
