import unittest

import pandas as pd

from src.proxy_ablation import (
    PROXY_CANDIDATES,
    PROXY_GROUP,
    create_feature_variants,
    make_shared_split,
)


def small_dataset():
    row_count = 100
    features = pd.DataFrame({
        "Attribute1": ["A11", "A12", "A13", "A14"] * 25,
        "Attribute4": ["A40", "A41", "A42", "A43", "A44"] * 20,
        "Attribute6": ["A61", "A62", "A63", "A64", "A65"] * 20,
        "Attribute7": ["A71", "A72", "A73", "A74", "A75"] * 20,
        "Attribute13": [18 + row % 50 for row in range(row_count)],
        "Attribute15": ["A151", "A152", "A153", "A151"] * 25,
        "Attribute2": [6 + row % 40 for row in range(row_count)],
    })
    return {
        "features": features,
        "target": pd.Series([1, 2] * (row_count // 2), name="class"),
        "target_name": "class",
        "good_label": 1,
        "bad_label": 2,
        "feature_descriptions": {"Attribute13": "Age in years"},
    }


class ProxyAblationTests(unittest.TestCase):
    def test_variants_exclude_age_and_only_the_declared_proxy_feature(self):
        split = make_shared_split(small_dataset())
        variants = create_feature_variants(split)
        reference_columns = set(split["X_train"].columns) - {"Attribute13"}

        reference = variants["Reference (age excluded)"]
        self.assertEqual(set(reference["X_train"].columns), reference_columns)
        self.assertNotIn("Attribute13", reference["X_train"].columns)
        self.assertNotIn("class", reference["X_train"].columns)

        for candidate in PROXY_CANDIDATES:
            name = f"Without {candidate['name']} ({candidate['column']})"
            variant = variants[name]
            expected_columns = reference_columns - {candidate["column"]}
            self.assertEqual(set(variant["X_train"].columns), expected_columns)
            self.assertEqual(set(variant["X_test"].columns), expected_columns)
            self.assertNotIn("Attribute13", variant["X_train"].columns)
            self.assertNotIn("class", variant["X_train"].columns)

    def test_all_variants_keep_the_same_train_and_test_rows(self):
        split = make_shared_split(small_dataset())
        variants = create_feature_variants(split)
        expected_train = split["y_train"].index.tolist()
        expected_test = split["y_test"].index.tolist()

        for variant in variants.values():
            self.assertEqual(variant["X_train"].index.tolist(), expected_train)
            self.assertEqual(variant["X_test"].index.tolist(), expected_test)

    def test_group_ablation_is_the_predeclared_three_feature_set(self):
        split = make_shared_split(small_dataset())
        variants = create_feature_variants(split)
        group_name = f"Without {PROXY_GROUP['name']}"
        group_variant = variants[group_name]
        expected_removed = {"Attribute13", *PROXY_GROUP["columns"]}

        self.assertEqual(set(group_variant["removed_features"]), expected_removed)
        self.assertEqual(
            set(group_variant["X_train"].columns),
            set(split["X_train"].columns) - expected_removed,
        )


if __name__ == "__main__":
    unittest.main()
