# Immortals AI explainability

Primary model: saved age-excluded Model B (`outputs/age_ablation/model_b_without_age.joblib`). It is reused as-is; no prediction model is rebuilt. Age is excluded from model inputs and remains available separately to the fairness audit.

## Global influence

Importance is the dataset average of the absolute contribution to the Good-versus-Bad log odds. Each contribution is the fitted transformed value multiplied by its Logistic Regression coefficient; one-hot levels are combined under their original categorical feature for the global report. Numeric values therefore use the fitted scaler, and categorical values use their actual encoded activations. The signed mean indicates average direction. This is model influence, not causal effect.

| feature | mean_absolute_log_odds_contribution | mean_log_odds_contribution | direction |
| --- | --- | --- | --- |
| Attribute1 | 0.6070 | 0.0967 | towards Good risk |
| Attribute20 | 0.5756 | -0.5327 | towards Bad risk |
| Attribute6 | 0.5373 | -0.3190 | towards Bad risk |
| Attribute14 | 0.3869 | 0.2738 | towards Good risk |
| Attribute8 | 0.3361 | -0.0107 | towards Bad risk |
| Attribute4 | 0.3320 | -0.0382 | towards Bad risk |
| Attribute9 | 0.3222 | 0.2278 | towards Good risk |
| Attribute12 | 0.3096 | 0.0891 | towards Good risk |
| Attribute3 | 0.2874 | 0.1925 | towards Good risk |
| Attribute2 | 0.2673 | -0.0037 | towards Bad risk |

Green bars have an average direction towards Good risk; red bars have an average direction towards Bad risk. The chart length ranks mean absolute log-odds contribution.

The complete transformed Logistic Regression coefficient table is saved as `logistic_regression_coefficients.csv`. Positive coefficients increase Good-versus-Bad log odds; negative coefficients decrease them. For categorical inputs, each row is a one-hot encoded level.

## Individual demonstration cases

These are synthetic demonstrations selected from valid rows in the dataset, not real customer applications. For each applicant, transformed values are multiplied by the saved coefficients. Contributions sum with the intercept to the Good-risk log odds; applying the logistic function reproduces Model B's Good-risk probability.

### Good risk

- Predicted class: Good risk (1)
- Probabilities: Good 99.8%; Bad 0.2%
- Near-threshold warning: no
- Explanation: The model predicts Good risk (Good-risk probability 99.8%). The listed factors add to a Good-versus-Bad log-odds score of 6.011. The probability is a model estimate, not a guarantee.

Strongest factors towards Good risk:
- Attribute1 = A14: +0.884 log odds
- Attribute4 = A41: +0.709 log odds
- Attribute6 = A64: +0.635 log odds

Strongest factors towards Bad risk:
- Attribute19 = A191: -0.164 log odds
- Attribute17 = A173: -0.131 log odds
- Attribute10 = A101: -0.069 log odds

### Bad risk

- Predicted class: Bad risk (2)
- Probabilities: Good 2.6%; Bad 97.4%
- Near-threshold warning: no
- Explanation: The model predicts Bad risk (Good-risk probability 2.6%). The listed factors add to a Good-versus-Bad log-odds score of -3.642. The probability is a model estimate, not a guarantee.

Strongest factors towards Good risk:
- Attribute9 = A93: +0.500 log odds
- Attribute14 = A143: +0.406 log odds
- Attribute19 = A192: +0.168 log odds

Strongest factors towards Bad risk:
- Attribute5: -1.546 log odds
- Attribute2: -0.937 log odds
- Attribute12 = A124: -0.716 log odds

### Relatively uncertain

- Predicted class: Bad risk (2)
- Probabilities: Good 49.9%; Bad 50.1%
- Near-threshold warning: yes
- Explanation: The model predicts Bad risk (Good-risk probability 49.9%). The listed factors add to a Good-versus-Bad log-odds score of -0.004. The probabilities are close to 50%, so this prediction is relatively uncertain; a probability is not a guarantee.

Strongest factors towards Good risk:
- Attribute6 = A64: +0.635 log odds
- Attribute14 = A143: +0.406 log odds
- Attribute12 = A122: +0.211 log odds

Strongest factors towards Bad risk:
- Attribute1 = A11: -0.659 log odds
- Attribute15 = A151: -0.584 log odds
- Attribute20 = A201: -0.575 log odds

## Limits

Predicted probabilities are estimates, not guarantees. A case is flagged relatively uncertain when the Good-risk probability is within 10 percentage points of 50%. Explanations describe how this fitted model produced a score; they do not prove a prediction is fair, correct, or causally justified.
