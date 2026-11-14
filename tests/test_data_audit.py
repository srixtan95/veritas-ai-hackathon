import unittest

import pandas as pd

from src.data_analysis import cramers_v, wilson_interval
from src.fairness import age_groups, format_fairness_table, group_metrics, rate_gaps


class DataAuditTests(unittest.TestCase):
    def test_age_boundary_is_under_25_vs_25_or_older(self):
        actual = age_groups(pd.Series([18, 24, 25, 40]))
        self.assertEqual(actual.tolist(), ["Under 25", "Under 25", "25 or older", "25 or older"])

    def test_wilson_interval_contains_observed_rate(self):
        low, high = wilson_interval(7, 10)
        self.assertLessEqual(low, 0.7)
        self.assertGreaterEqual(high, 0.7)

    def test_cramers_v_is_bounded(self):
        value = cramers_v(pd.Series(["a", "a", "b", "b"]), pd.Series(["x", "x", "y", "y"]))
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)

    def test_group_metrics_counts_approval_and_errors(self):
        result = group_metrics(
            actual=pd.Series([1, 1, 2, 2]),
            predicted=pd.Series([1, 2, 1, 2]),
            groups=pd.Series(["A", "A", "B", "B"]),
            good_label=1,
            bad_label=2,
        ).set_index("group")
        self.assertEqual(result.loc["A", "approval_rate"], 0.5)
        self.assertEqual(result.loc["A", "false_rejection_rate"], 0.5)
        self.assertEqual(result.loc["B", "false_approval_rate"], 0.5)

    def test_zero_denominator_displays_na_and_skips_gap(self):
        metrics = group_metrics(
            actual=pd.Series([1, 1]),
            predicted=pd.Series([1, 2]),
            groups=pd.Series(["Under 25", "25 or older"]),
            good_label=1,
            bad_label=2,
        )
        display = format_fairness_table(metrics)
        self.assertIn("N/A (0/0)", display.loc[0, "False approval rate"])
        gap = rate_gaps(metrics).set_index("metric").loc["False approval rate", "absolute_gap_percentage_points"]
        self.assertTrue(pd.isna(gap))


if __name__ == "__main__":
    unittest.main()
