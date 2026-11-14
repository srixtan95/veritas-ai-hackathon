import unittest

import numpy as np
import pandas as pd

from src.repeated_validation import (
    FEATURES,
    calculate_repeat_metrics,
    evaluate_configurations,
    make_feature_configurations,
    make_shared_fold_plan,
    validate_dataset,
)


def small_dataset(row_count=100):
    columns = {}
    categorical = {"Attribute1", "Attribute4", "Attribute6", "Attribute7", "Attribute15"}
    for number in range(1, 21):
        column = f"Attribute{number}"
        if column == "Attribute13":
            columns[column] = [18 + index % 50 for index in range(row_count)]
        elif column in categorical:
            columns[column] = [f"{column}_v{index % 4}" for index in range(row_count)]
        else:
            columns[column] = [index % 17 for index in range(row_count)]
    target = pd.Series([1 if index % 5 else 2 for index in range(row_count)], name="class")
    return {
        "features": pd.DataFrame(columns),
        "target": target,
        "target_name": "class",
        "good_label": 1,
        "bad_label": 2,
        "feature_descriptions": {"Attribute13": "Age in years"},
        "metadata": {"name": "synthetic test dataset"},
    }


class RepeatedValidationTests(unittest.TestCase):
    def test_eight_feature_configurations_exclude_only_declared_columns(self):
        dataset = small_dataset()
        configured = make_feature_configurations(dataset["features"], "Attribute13")
        self.assertEqual(list(configured), list(FEATURES))
        self.assertEqual(len(configured["A: All features"].columns), 20)
        for name, excluded in FEATURES.items():
            expected = set(dataset["features"].columns) - set(excluded)
            self.assertEqual(set(configured[name].columns), expected)
            self.assertNotIn("class", configured[name].columns)

    def test_fold_plan_is_shared_and_each_repeat_holds_out_every_row_once(self):
        dataset = small_dataset()
        plan = make_shared_fold_plan(
            dataset["features"], dataset["target"], n_splits=5, n_repeats=3, random_state=42
        )
        repeated_plan = make_shared_fold_plan(
            dataset["features"], dataset["target"], n_splits=5, n_repeats=3, random_state=42
        )
        self.assertEqual(len(plan), 15)
        for first, second in zip(plan, repeated_plan):
            self.assertEqual(first["repeat"], second["repeat"])
            self.assertEqual(first["fold"], second["fold"])
            np.testing.assert_array_equal(first["train_positions"], second["train_positions"])
            np.testing.assert_array_equal(first["test_positions"], second["test_positions"])
            self.assertTrue(set(first["train_positions"]).isdisjoint(first["test_positions"]))
        for repeat in range(1, 4):
            test_positions = np.concatenate([
                fold["test_positions"] for fold in plan if fold["repeat"] == repeat
            ])
            self.assertEqual(sorted(test_positions.tolist()), list(range(len(dataset["target"]))))

    def test_target_mapping_is_validated(self):
        dataset = small_dataset()
        validate_dataset(dataset)
        dataset["bad_label"] = 0
        with self.assertRaises(ValueError):
            validate_dataset(dataset)

    def test_cost_rate_and_gap_calculations(self):
        actual = np.array([1, 1, 2, 2, 1, 2, 1, 2])
        predicted = np.array([1, 2, 1, 2, 1, 1, 2, 2])
        ages = np.array([20, 20, 30, 30, 22, 30, 25, 40])
        probabilities = np.array([0.9, 0.4, 0.8, 0.2, 0.7, 0.6, 0.3, 0.1])
        scores, fairness, gaps = calculate_repeat_metrics(
            actual, predicted, probabilities, ages
        )
        self.assertEqual(scores["false_approvals"], 2)
        self.assertEqual(scores["false_rejections"], 2)
        self.assertEqual(scores["error_cost"], 12)
        self.assertEqual(scores["error_cost_per_100"], 150.0)
        grouped = fairness.set_index("group")
        self.assertEqual(grouped.loc["Under 25", "group_size"], 3)
        self.assertEqual(grouped.loc["25 or older", "false_approval_denominator"], 4)
        self.assertAlmostEqual(grouped.loc["25 or older", "false_approval_rate"], 0.5)
        self.assertAlmostEqual(
            gaps["approval_rate_gap_pp"], abs((2 / 3) - (2 / 5)) * 100
        )
        self.assertEqual(
            gaps["true_positive_rate_gap_pp"], gaps["false_rejection_rate_gap_pp"]
        )

    def test_all_models_use_the_same_fold_schedule(self):
        dataset = small_dataset(50)
        plan = make_shared_fold_plan(
            dataset["features"], dataset["target"], n_splits=2, n_repeats=2, random_state=42
        )
        outputs = evaluate_configurations(dataset, plan)
        self.assertEqual(len(outputs["performance"]), 8 * 2)
        self.assertEqual(len(outputs["fold_assignments"]), 2 * 50)
        self.assertEqual(len(outputs["oof_predictions"]), 8 * 2 * 50)
        prediction_counts = outputs["oof_predictions"].groupby(["model", "repeat", "row_position"]).size()
        self.assertTrue((prediction_counts == 1).all())
        for repeat in [1, 2]:
            assignments = outputs["fold_assignments"]
            repeat_assignments = assignments[assignments["repeat"] == repeat]
            self.assertEqual(repeat_assignments["row_position"].nunique(), 50)
            self.assertEqual(repeat_assignments["row_position"].value_counts().max(), 1)


if __name__ == "__main__":
    unittest.main()
