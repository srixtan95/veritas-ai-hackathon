"""Compare the saved credit-risk model with a model trained without age."""

import json
from pathlib import Path

import pandas as pd
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .data_analysis import find_age_column
from .fairness import age_groups, format_fairness_table, group_metrics, rate_gaps
from .model import MODEL_DIR, evaluate_predictions, make_preprocessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "age_ablation"
TEST_SIZE = 0.20
RANDOM_STATE = 42


def validate_dataset(dataset):
    """Check that the data uses the expected German Credit target labels."""
    if dataset["target_name"] != "class":
        raise ValueError(f"Expected target 'class', got {dataset['target_name']!r}.")
    if dataset["good_label"] != 1 or dataset["bad_label"] != 2:
        raise ValueError("Expected class 1 = Good risk and class 2 = Bad risk.")
    if dataset["target_name"] in dataset["features"].columns:
        raise ValueError("The target column must not be included among model features.")
    labels = set(pd.unique(dataset["target"]))
    if labels != {1, 2}:
        raise ValueError(f"Expected target values 1 and 2, got {labels!r}.")


def make_shared_split(dataset):
    """Create the original stratified split once for both models."""
    validate_dataset(dataset)
    age_column = find_age_column(dataset["features"], dataset["feature_descriptions"])
    X_train, X_test, y_train, y_test = train_test_split(
        dataset["features"],
        dataset["target"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=dataset["target"],
    )
    return {
        "age_column": age_column,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def make_model_inputs(split):
    """Return all features for Model A and age-excluded features for Model B."""
    age_column = split["age_column"]
    X_train = split["X_train"]
    X_test = split["X_test"]
    if age_column not in X_train.columns or age_column not in X_test.columns:
        raise ValueError(f"Age feature {age_column!r} is missing from the split.")

    return {
        "Model A (all features)": (X_train, X_test),
        "Model B (without age)": (
            X_train.drop(columns=[age_column]),
            X_test.drop(columns=[age_column]),
        ),
    }


def fairness_metrics_for_ages(actual, predicted, actual_ages, good_label=1, bad_label=2):
    """Group test predictions by actual age, independent of model inputs."""
    groups = age_groups(actual_ages)
    return group_metrics(actual, predicted, groups, good_label, bad_label)


def _score_model(model, features, actual, actual_ages, good_label, bad_label):
    """Evaluate one fitted pipeline on test rows and the separate age values."""
    predictions = model.predict(features)
    good_index = list(model.named_steps["classifier"].classes_).index(good_label)
    good_probabilities = model.predict_proba(features)[:, good_index]
    scores = evaluate_predictions(actual, predictions, good_label, bad_label, good_probabilities)
    by_age = fairness_metrics_for_ages(actual, predictions, actual_ages, good_label, bad_label)
    return scores, predictions, by_age


def _baseline_matches(saved_scores, recalculated_scores):
    """Check the all-feature model against the preserved baseline metrics."""
    metric_names = [
        "accuracy", "balanced_accuracy", "roc_auc", "false_approvals",
        "false_rejections", "error_cost", "confusion_matrix",
    ]
    return all(saved_scores[name] == recalculated_scores[name] for name in metric_names)


def _markdown_table(table):
    """Turn a small DataFrame into a Markdown table without extra packages."""
    lines = [
        "| " + " | ".join(str(column) for column in table.columns) + " |",
        "|" + "|".join(["---"] * len(table.columns)) + "|",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def run_age_ablation(dataset, output_dir=OUTPUT_DIR):
    """Run the age ablation and write results only to its dedicated folder.

    Model A is the already-trained baseline artifact. Model B is the only
    model fitted by this experiment. Existing project outputs are not touched.
    """
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Experiment output folder is not empty: {output_dir}. "
            "Choose a new output folder to avoid replacing prior results."
        )

    split = make_shared_split(dataset)
    model_inputs = make_model_inputs(split)
    age_column = split["age_column"]

    # Reuse the saved baseline model; do not fit or replace Model A.
    model_a = load(MODEL_DIR / "logistic_regression_pipeline.joblib")
    expected_columns = list(model_a.feature_names_in_)
    model_a_train, model_a_test = model_inputs["Model A (all features)"]
    if list(model_a_train.columns) != expected_columns:
        raise ValueError("Dataset features do not match the saved all-feature baseline model.")

    # Fit Model B on the same training rows after dropping only age.
    model_b_train, model_b_test = model_inputs["Model B (without age)"]
    model_b = Pipeline([
        ("preprocessor", make_preprocessor(model_b_train)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])
    model_b.fit(model_b_train, split["y_train"])

    # Keep actual ages separate so Model B never receives them as inputs.
    actual_ages = split["X_test"][age_column]
    model_outputs = {
        "Model A (all features)": _score_model(
            model_a,
            model_a_test,
            split["y_test"],
            actual_ages,
            dataset["good_label"],
            dataset["bad_label"],
        ),
        "Model B (without age)": _score_model(
            model_b,
            model_b_test,
            split["y_test"],
            actual_ages,
            dataset["good_label"],
            dataset["bad_label"],
        ),
    }

    reference = json.loads((MODEL_DIR / "stage1_metrics.json").read_text(encoding="utf-8"))
    model_a_scores = model_outputs["Model A (all features)"][0]
    baseline_matches = _baseline_matches(reference["metrics"]["logistic_regression"], model_a_scores)

    metric_rows = []
    fairness_rows = []
    gap_rows = []
    display_rows = []
    for model_name, (scores, _, fairness) in model_outputs.items():
        metric_rows.append({"model": model_name, **scores})
        fairness["group"] = pd.Categorical(
            fairness["group"], categories=["Under 25", "25 or older"], ordered=True
        )
        fairness = fairness.sort_values("group").reset_index(drop=True)
        fairness.insert(0, "model", model_name)
        fairness_rows.extend(fairness.to_dict(orient="records"))

        model_gaps = rate_gaps(fairness)
        model_gaps.insert(0, "model", model_name)
        gap_rows.extend(model_gaps.to_dict(orient="records"))

        display = format_fairness_table(fairness)
        display.insert(0, "Model", model_name)
        display_rows.extend(display.to_dict(orient="records"))

    metrics_table = pd.DataFrame(metric_rows)
    fairness_table = pd.DataFrame(display_rows)
    fairness_frame = pd.DataFrame(fairness_rows)
    gaps_frame = pd.DataFrame(gap_rows)

    # The output folder must be new or empty; no baseline output path is used.
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_table.to_csv(output_dir / "model_comparison.csv", index=False)
    fairness_frame.to_csv(output_dir / "fairness_metrics.csv", index=False)
    fairness_table.to_csv(output_dir / "fairness_table.csv", index=False)
    gaps_frame.to_csv(output_dir / "fairness_gaps.csv", index=False)
    dump(model_b, output_dir / "model_b_without_age.joblib")

    split_rows = [
        {"row_index": index, "split": "train"} for index in split["X_train"].index
    ] + [
        {"row_index": index, "split": "test"} for index in split["X_test"].index
    ]
    pd.DataFrame(split_rows).to_csv(output_dir / "split_indices.csv", index=False)

    result_data = {
        "dataset": dataset["metadata"],
        "target": {
            "name": dataset["target_name"],
            "good_label": int(dataset["good_label"]),
            "bad_label": int(dataset["bad_label"]),
        },
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratified": True,
            "train_rows": len(split["y_train"]),
            "test_rows": len(split["y_test"]),
            "train_indices": list(split["X_train"].index),
            "test_indices": list(split["X_test"].index),
        },
        "age_column": age_column,
        "model_a_reproduces_saved_baseline": baseline_matches,
        "models": {row["model"]: row for row in metric_rows},
        "fairness_gaps": gap_rows,
    }
    (output_dir / "experiment_results.json").write_text(
        json.dumps(result_data, indent=2, default=lambda value: value.item() if hasattr(value, "item") else str(value)),
        encoding="utf-8",
    )

    baseline_note = (
        "Model A exactly matches the saved baseline metrics."
        if baseline_matches
        else "WARNING: Model A does not match the saved baseline metrics; check the fetched dataset and environment."
    )
    report = [
        "# Age ablation experiment",
        "",
        "Model A reuses the saved all-feature Logistic Regression pipeline. Model B is trained with only the age feature removed. Both use the same stratified train/test rows, seed 42, preprocessing method, classifier settings, and evaluation functions. Preprocessing is fitted on training data only.",
        f"Age feature: `{age_column}`. Target: `class`, with 1 = Good risk and 2 = Bad risk. {baseline_note}",
        "",
        "## Predictive performance",
        "",
        _markdown_table(metrics_table[[
            "model", "accuracy", "balanced_accuracy", "roc_auc",
            "false_approvals", "false_rejections", "error_cost",
        ]]),
        "",
        "Cost is 5 × false approvals + false rejections. A false approval is actual Bad risk predicted Good; a false rejection is actual Good risk predicted Bad.",
        "",
        "## Fairness rates by actual age group",
        "",
        _markdown_table(fairness_table),
        "",
        "Each rate includes numerator/denominator. Age is used from the original test rows for grouping, not from Model B inputs.",
        "",
        "## Absolute age-group gaps",
        "",
        _markdown_table(gaps_frame),
        "",
        "Differences are descriptive, not proof of discrimination. The under-25 test group is small; interpret its rates cautiously. Removing the age column does not remove possible age proxies from the remaining features.",
        "",
    ]
    (output_dir / "age_ablation_report.md").write_text("\n".join(report), encoding="utf-8")
    return {
        "metrics": metrics_table,
        "fairness": fairness_table,
        "gaps": gaps_frame,
        "baseline_matches": baseline_matches,
        "output_dir": output_dir,
    }


def main():
    """Download the dataset, run the experiment, and print its tables."""
    from .data_loader import load_credit_dataset

    results = run_age_ablation(load_credit_dataset())
    print("Predictive performance:")
    print(results["metrics"].to_string(index=False))
    print("\nFairness metrics:")
    print(results["fairness"].to_string(index=False))
    print("\nAbsolute percentage-point gaps:")
    print(results["gaps"].to_string(index=False))
    print(f"\nNew experiment files: {results['output_dir']}")


if __name__ == "__main__":
    main()
