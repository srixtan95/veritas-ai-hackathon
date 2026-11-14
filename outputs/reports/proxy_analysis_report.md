# Proxy analysis

Age group is predicted without the age column `Attribute13` or the credit-risk target `class`. The model uses a stratified split and five-fold cross-validation on training rows.

Test scores: `{"balanced_accuracy": 0.638235294117647, "precision": 0.6923076923076923, "recall": 0.3, "f1": 0.4186046511627907, "roc_auc": 0.7782352941176471}`. Majority baseline: `{"balanced_accuracy": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0}`.

Predictability suggests the other features contain information about age group. It does not prove causation or discrimination.