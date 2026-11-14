import hashlib
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.application import (
    FAIRNESS_FILES,
    MODEL_B_PATH,
    PROJECT_ROOT,
    assess_credit_risk,
    load_credit_model,
    load_research_results,
    model_input_schema,
    validate_applicant_input,
)
from src.affordability import assess_affordability


def artifact_hashes():
    paths = [MODEL_B_PATH]
    for folder in ("models", "outputs/age_ablation", "outputs/proxy_ablation",
                   "outputs/repeated_validation", "outputs/explainability"):
        paths.extend(path for path in (PROJECT_ROOT / folder).rglob("*") if path.is_file())
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


class ApplicationServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_credit_model()
        cls.schema = model_input_schema(cls.model)
        cls.valid_applicant = {
            field["name"]: (field["default"] if field["kind"] == "numeric" else field["choices"][0])
            for field in cls.schema
        }

    def test_schema_has_exactly_19_features_and_excludes_age(self):
        self.assertEqual(len(self.schema), 19)
        self.assertEqual(list(self.valid_applicant), list(self.model.feature_names_in_))
        self.assertNotIn("Attribute13", self.valid_applicant)
        self.assertEqual(sum(field["kind"] == "categorical" for field in self.schema), 13)
        self.assertEqual(sum(field["kind"] == "numeric" for field in self.schema), 6)

    def test_successful_submission_prediction_and_explanation_match_pipeline(self):
        result = assess_credit_risk(self.model, self.valid_applicant)
        applicant = pd.DataFrame([self.valid_applicant], columns=self.model.feature_names_in_)
        self.assertEqual(result["predicted_class"], self.model.predict(applicant)[0])
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0)
        self.assertEqual(result["predicted_class"], result["explanation"]["predicted_class"])
        self.assertEqual(result["probabilities"], result["explanation"]["probabilities"])
        self.assertEqual(result["near_threshold"], result["explanation"]["near_threshold"])

    def test_missing_invalid_and_age_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_applicant_input(self.model, {})
        with self.assertRaisesRegex(ValueError, "Age"):
            validate_applicant_input(self.model, self.valid_applicant | {"Attribute13": 45})
        numeric_name = next(field["name"] for field in self.schema if field["kind"] == "numeric")
        with self.assertRaisesRegex(ValueError, "whole number"):
            validate_applicant_input(self.model, self.valid_applicant | {numeric_name: -1})
        category = next(field for field in self.schema if field["kind"] == "categorical")
        with self.assertRaisesRegex(ValueError, "valid dataset category"):
            validate_applicant_input(self.model, self.valid_applicant | {category["name"]: "NOT_A_CATEGORY"})

    def test_affordability_is_integrated_as_a_separate_calculation(self):
        result = assess_affordability(50000, 6000, 9000, 10, 36, 250000, 25)
        self.assertEqual(result["affordability_percentage"], 25)
        self.assertEqual(result["maximum_illustrative_affordable_emi"], 8750)
        self.assertNotIn("predicted_class", result)

    def test_research_dashboard_loads_saved_results_without_rerunning(self):
        results = load_research_results()
        self.assertEqual(set(results), set(FAIRNESS_FILES))
        self.assertEqual(set(results["age_fairness"]["model"]), {"Model A (all features)", "Model B (without age)"})
        self.assertEqual(len(results["repeated_fairness"]["model"].unique()), 8)

    def test_loading_and_assessing_never_calls_pipeline_fit_or_changes_research_artifacts(self):
        before = artifact_hashes()
        with patch("sklearn.pipeline.Pipeline.fit", side_effect=AssertionError("unexpected retraining")):
            model = load_credit_model()
            assess_credit_risk(model, self.valid_applicant)
            load_research_results()
        self.assertEqual(before, artifact_hashes())


class StreamlitIntegrationTests(unittest.TestCase):
    def test_dashboard_renders_and_accepts_applicant_submission(self):
        before = artifact_hashes()
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=15).run()
        self.assertEqual(len(app.exception), 0, [exception.message for exception in app.exception])
        self.assertGreaterEqual(len(app.selectbox), 13)  # Model B's 13 categorical inputs
        self.assertGreaterEqual(len(app.number_input), 6)  # Model B's 6 numeric inputs
        app.form_submit_button[0].click().run()
        self.assertEqual(len(app.exception), 0, [exception.message for exception in app.exception])
        result = app.session_state["credit_result"]
        self.assertEqual(len(result["features"]), 19)
        self.assertNotIn("Attribute13", result["features"])
        self.assertEqual(result["predicted_class"], result["explanation"]["predicted_class"])
        self.assertEqual(before, artifact_hashes())

    def test_dashboard_runs_affordability_form(self):
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=15).run()
        app.form_submit_button[1].click().run()
        self.assertEqual(len(app.exception), 0, [exception.message for exception in app.exception])
        result = app.session_state["affordability_result"]
        self.assertEqual(result["affordability_percentage"], 30)
        self.assertIsNotNone(result["proposed_monthly_emi"])


if __name__ == "__main__":
    unittest.main()
