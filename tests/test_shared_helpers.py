"""Compatibility checks for extracted schema and report helpers."""

import unittest

import pandas as pd

from src.age_ablation import _markdown_table as age_table
from src.data_analysis import find_age_column
from src.proxy_ablation import _markdown_table as proxy_table
from src.repeated_validation import _markdown_table as repeated_table


class SharedHelperTests(unittest.TestCase):
    def test_age_lookup_keeps_order_and_word_matching(self):
        self.assertEqual(
            find_age_column(["storage", "first", "second"],
                            {"storage": "Storage", "first": "AGE in years", "second": "Age"}),
            "first",
        )
        with self.assertRaisesRegex(ValueError, "Could not find an age feature"):
            find_age_column(["storage"], {"storage": "Storage"})

    def test_ablation_tables_preserve_text_and_empty_headers(self):
        table = pd.DataFrame({"name": ["Under 25", "25 or older"], "value": [0.5, None]})
        expected = "| name | value |\n|---|---|\n| Under 25 | 0.5 |\n| 25 or older | nan |"
        for render in (age_table, proxy_table):
            self.assertEqual(render(table), expected)
            self.assertEqual(render(table.iloc[:0]), "| name | value |\n|---|---|")

    def test_repeat_table_preserves_precision_na_and_input(self):
        table = pd.DataFrame({"name": ["A", "B"], "value": [0.12345, None], "count": [2, 3]})
        before = table.copy(deep=True)
        self.assertEqual(
            repeated_table(table),
            "| name | value | count |\n|---|---|---|\n| A | 0.123 | 2.000 |\n| B | N/A | 3.000 |",
        )
        pd.testing.assert_frame_equal(table, before)


if __name__ == "__main__":
    unittest.main()
