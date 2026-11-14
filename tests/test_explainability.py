import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.explainability import explain_applicant, global_feature_importance
from src.model import make_preprocessor


def fitted_model():
    rows = 120
    features = pd.DataFrame({
        "amount": np.tile([100, 250, 400, 550], rows // 4),
        "purpose": np.tile(["car", "furniture", "radio", "car"], rows // 4),
        "history": np.tile(["good", "poor", "good", "poor", "good", "poor"], rows // 6),
    })
    target = pd.Series(np.where(
        (features["amount"] < 400) & (features["history"] == "good"), 1, 2
    ))
    model = Pipeline([
        ("preprocessor", make_preprocessor(features)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    model.fit(features, target)
    return model, features


class ExplainabilityTests(unittest.TestCase):
    def test_contributions_sum_to_pipeline_probability_and_prediction(self):
        model, features = fitted_model()
        applicant = features.iloc[0]
        explanation = explain_applicant(model, applicant)
        self.assertEqual(explanation["predicted_class"], model.predict(applicant.to_frame().T)[0])
        self.assertAlmostEqual(
            explanation["probabilities"]["Good risk"],
            model.predict_proba(applicant.to_frame().T)[0, list(model.classes_).index(1)],
        )
        total = explanation["intercept_good_log_odds"] + sum(
            item["good_log_odds_contribution"] for item in explanation["contributions"]
        )
        self.assertAlmostEqual(total, explanation["good_log_odds"])

    def test_one_hot_levels_are_named_and_grouped_to_source_feature(self):
        model, features = fitted_model()
        explanation = explain_applicant(model, features.iloc[0])
        self.assertTrue(any(item["feature"] == "purpose" and " = " in item["transformed_feature"]
                            for item in explanation["contributions"]))
        importance = global_feature_importance(model, features)
        self.assertEqual(set(importance["feature"]), set(features.columns))
        self.assertTrue(importance["mean_absolute_log_odds_contribution"].is_monotonic_decreasing)

    def test_age_is_rejected_as_model_input_and_model_feature(self):
        model, features = fitted_model()
        with self.assertRaisesRegex(ValueError, "cannot be supplied"):
            explain_applicant(model, {**features.iloc[0].to_dict(), "Attribute13": 30})
        age_features = features.assign(Attribute13=30)
        with self.assertRaisesRegex(ValueError, "cannot be used"):
            global_feature_importance(model, age_features)

    def test_applicant_schema_is_checked(self):
        model, features = fitted_model()
        with self.assertRaisesRegex(ValueError, "missing"):
            explain_applicant(model, {"amount": 100, "purpose": "car"})


if __name__ == "__main__":
    unittest.main()
