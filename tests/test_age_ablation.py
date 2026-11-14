import unittest

import pandas as pd

from src.age_ablation import (
    fairness_metrics_for_ages,
    make_model_inputs,
    make_shared_split,
    validate_dataset,
)
from src.model import evaluate_predictions


def small_dataset():
    """Build a small deterministic dataset for testing split and inputs."""
    row_count = 60
    return {
        "features": pd.DataFrame({
            "Attribute13": [18 + (row % 50) for row in range(row_count)],
            "Attribute1": ["A" if row % 2 else "B" for row in range(row_count)],
        }),
        "target": pd.Series([1, 2] * (row_count // 2), name="class"),
        "target_name": "class",
        "good_label": 1,
        "bad_label": 2,
        "feature_descriptions": {"Attribute13": "Age in years"},
        "metadata": {"name": "Small test data"},
    }


class AgeAblationTests(unittest.TestCase):
    def test_model_a_keeps_age_and_model_b_drops_only_age(self):
        split = make_shared_split(small_dataset())
        inputs = make_model_inputs(split)
        model_a_train, model_a_test = inputs["Model A (all features)"]
        model_b_train, model_b_test = inputs["Model B (without age)"]

        self.assertIn("Attribute13", model_a_train.columns)
        self.assertIn("Attribute13", model_a_test.columns)
        self.assertNotIn("Attribute13", model_b_train.columns)
        self.assertNotIn("Attribute13", model_b_test.columns)
        self.assertEqual(list(model_b_train.columns), ["Attribute1"])
        self.assertEqual(list(model_b_test.columns), ["Attribute1"])
        self.assertNotIn("class", model_a_train.columns)

    def test_both_models_share_the_same_split_indices(self):
        dataset = small_dataset()
        split = make_shared_split(dataset)
        inputs = make_model_inputs(split)

        self.assertEqual(split["X_train"].index.tolist(), split["y_train"].index.tolist())
        self.assertEqual(split["X_test"].index.tolist(), split["y_test"].index.tolist())
        self.assertEqual(
            inputs["Model A (all features)"][0].index.tolist(),
            inputs["Model B (without age)"][0].index.tolist(),
        )
        self.assertEqual(
            inputs["Model A (all features)"][1].index.tolist(),
            inputs["Model B (without age)"][1].index.tolist(),
        )

    def test_target_mapping_is_good_one_bad_two(self):
        dataset = small_dataset()
        validate_dataset(dataset)
        dataset["good_label"] = 2
        with self.assertRaises(ValueError):
            validate_dataset(dataset)

    def test_cost_uses_false_approval_and_false_rejection_definitions(self):
        scores = evaluate_predictions(
            actual=[1, 1, 2, 2],
            predicted=[2, 1, 1, 2],
            good_label=1,
            bad_label=2,
        )
        self.assertEqual(scores["false_approvals"], 1)
        self.assertEqual(scores["false_rejections"], 1)
        self.assertEqual(scores["error_cost"], 6)

    def test_fairness_uses_actual_age_values_separately(self):
        # These model inputs contain no age column; the supplied age series is
        # the actual age from the original test rows and determines the groups.
        model_b_inputs = pd.DataFrame({"Attribute1": ["a", "b", "c", "d"]})
        actual_ages = pd.Series([24, 24, 25, 25])
        results = fairness_metrics_for_ages(
            actual=[1, 2, 1, 2],
            predicted=[1, 1, 2, 1],
            actual_ages=actual_ages,
        ).set_index("group")

        self.assertNotIn("Attribute13", model_b_inputs.columns)
        self.assertEqual(results.loc["Under 25", "group_size"], 2)
        self.assertEqual(results.loc["25 or older", "group_size"], 2)
        self.assertEqual(results.loc["Under 25", "approval_count"], 2)
        self.assertEqual(results.loc["25 or older", "false_rejection_count"], 1)


if __name__ == "__main__":
    unittest.main()
