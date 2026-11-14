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
    def _complete_applicant_inputs(self, app):
        for field in model_input_schema(load_credit_model()):
            key = f"applicant_{field['name']}"
            if field["kind"] == "categorical":
                app.selectbox(key=key).set_value(field["choices"][0])
            else:
                app.number_input(key=key).set_value(field["default"])

    def test_dashboard_renders_four_sections_and_live_applicant_assessment(self):
        before = artifact_hashes()
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=15).run()
        self.assertEqual(len(app.exception), 0, [exception.message for exception in app.exception])
        self.assertEqual([tab.label for tab in app.tabs], [
            "Overview", "Applicant Assessment", "Fairness & Research", "Features & Technology",
        ])
        self.assertGreaterEqual(len(app.selectbox), 13)  # Model B's 13 categorical inputs
        self.assertGreaterEqual(len(app.number_input), 6)  # Model B's 6 numeric inputs
        self.assertIn("Enter ", " ".join(element.value for element in app.info))
        self._complete_applicant_inputs(app)
        app.run()
        self.assertEqual(len(app.exception), 0, [exception.message for exception in app.exception])
        result = app.session_state["live_credit_result"]
        self.assertEqual(len(result["features"]), 19)
        self.assertNotIn("Attribute13", result["features"])
        self.assertEqual(result["predicted_class"], result["explanation"]["predicted_class"])
        self.assertAlmostEqual(result["probabilities"]["Good risk"], result["explanation"]["probabilities"]["Good risk"])
        contribution_sum = sum(item["good_log_odds_contribution"] for item in result["explanation"]["contributions"])
        self.assertAlmostEqual(
            result["explanation"]["intercept_good_log_odds"] + contribution_sum,
            result["explanation"]["good_log_odds"],
        )
        self.assertEqual(before, artifact_hashes())

    def test_changing_applicant_input_updates_live_explanation(self):
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=15).run()
        self._complete_applicant_inputs(app)
        app.run()
        self.assertEqual(len(app.exception), 0, [exception.message for exception in app.exception])
        before = app.session_state["live_credit_result"]["probabilities"]["Good risk"]
        numeric_field = next(field for field in model_input_schema(load_credit_model()) if field["kind"] == "numeric")
        widget = app.number_input(key=f"applicant_{numeric_field['name']}")
        widget.set_value(int(numeric_field["default"]) + 1)
        app.run()
        self.assertEqual(len(app.exception), 0, [exception.message for exception in app.exception])
        self.assertIn("live_credit_result", app.session_state)
        after = app.session_state["live_credit_result"]["probabilities"]["Good risk"]
        self.assertNotEqual(before, after)

    def test_live_affordability_matches_existing_calculation(self):
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=15).run()
        app.number_input(key="finance_income").set_value(50000)
        app.number_input(key="finance_repayments").set_value(6000)
        app.number_input(key="finance_other").set_value(9000)
        app.number_input(key="finance_rate").set_value(10)
        app.number_input(key="finance_tenure").set_value(36)
        app.number_input(key="finance_amount").set_value(250000)
        app.run()
        self.assertEqual(len(app.exception), 0, [exception.message for exception in app.exception])
        result = app.session_state["live_affordability_result"]
        expected = assess_affordability(50000, 6000, 9000, 10, 36, 250000, 30)
        self.assertEqual(result, expected)
        app.slider(key="finance_percentage").set_value(25)
        app.run()
        self.assertEqual(app.session_state["live_affordability_result"]["affordability_percentage"], 25)

    def test_research_and_global_explainability_render_from_saved_files(self):
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=15).run()
        self.assertEqual(len(app.exception), 0, [exception.message for exception in app.exception])
        self.assertTrue(any("Top Model B global feature influence" in str(element.value) for element in app.markdown))
        self.assertTrue(any("Baseline comparison" in str(element.value) for element in app.markdown))
        self.assertTrue(any("Repeated validation across eight configurations" in str(element.value) for element in app.markdown))


if __name__ == "__main__":
    unittest.main()
