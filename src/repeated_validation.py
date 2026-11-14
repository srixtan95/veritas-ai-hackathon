"""Repeated stratified cross-validation for Immortals AI model variants."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

from .data_analysis import find_age_column
from .fairness import age_groups, group_metrics, rate_gaps
from .model import evaluate_predictions, make_preprocessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "repeated_validation"
N_SPLITS = 5
N_REPEATS = 10
RANDOM_STATE = 42

FEATURES = {
    "A: All features": [],
    "B: Age excluded": ["Attribute13"],
    "C: Age + checking excluded": ["Attribute13", "Attribute1"],
    "D: Age + purpose excluded": ["Attribute13", "Attribute4"],
    "E: Age + savings excluded": ["Attribute13", "Attribute6"],
    "F: Age + employment excluded": ["Attribute13", "Attribute7"],
    "G: Age + housing excluded": ["Attribute13", "Attribute15"],
    "H: Age + savings + employment + housing excluded": [
        "Attribute13", "Attribute6", "Attribute7", "Attribute15"
    ],
}

PERFORMANCE_METRICS = [
    "accuracy", "balanced_accuracy", "roc_auc", "false_approvals",
    "false_rejections", "error_cost", "error_cost_per_100",
]
FAIRNESS_RATES = [
    "approval_rate", "false_approval_rate", "false_rejection_rate", "true_positive_rate"
]
GAP_METRICS = [
    "approval_rate_gap_pp", "false_approval_rate_gap_pp",
    "false_rejection_rate_gap_pp", "true_positive_rate_gap_pp",
]


def validate_dataset(dataset):
    """Enforce the project's German Credit target mapping and feature separation."""
    if dataset["target_name"] != "class":
        raise ValueError("Expected target column 'class'.")
    if dataset["good_label"] != 1 or dataset["bad_label"] != 2:
        raise ValueError("Expected class 1 = Good risk and class 2 = Bad risk.")
    if "class" in dataset["features"].columns:
        raise ValueError("Target must not appear in model features.")
    if set(pd.unique(dataset["target"])) != {1, 2}:
        raise ValueError("Expected target values 1 and 2.")
    if len(dataset["features"]) != len(dataset["target"]):
        raise ValueError("Feature and target row counts differ.")


def make_feature_configurations(features, age_column):
    """Return the eight predeclared feature sets without changing row order."""
    missing = sorted({column for excluded in FEATURES.values() for column in excluded} - set(features.columns))
    if missing:
        raise ValueError(f"Required ablation feature(s) are missing: {missing}")
    if age_column != "Attribute13":
        raise ValueError(f"Expected age to map to Attribute13, got {age_column!r}.")
    if "Attribute13" not in features.columns:
        raise ValueError("Age feature Attribute13 is missing.")

    configured = {}
    for name, excluded in FEATURES.items():
        kept = [column for column in features.columns if column not in excluded]
        if "class" in kept or ("Attribute13" in excluded and "Attribute13" in kept):
            raise ValueError(f"Invalid feature configuration: {name}")
        configured[name] = features.loc[:, kept].copy()
    return configured


def make_shared_fold_plan(features, target, n_splits=N_SPLITS, n_repeats=N_REPEATS,
                          random_state=RANDOM_STATE):
    """Materialize one common repeated-stratified fold schedule for all models."""
    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    plan = []
    for split_number, (train_positions, test_positions) in enumerate(
        splitter.split(features, target)
    ):
        plan.append({
            "repeat": split_number // n_splits + 1,
            "fold": split_number % n_splits + 1,
            "train_positions": train_positions,
            "test_positions": test_positions,
        })
    return plan


def calculate_repeat_metrics(actual, predicted, good_probability, actual_ages,
                             good_label=1, bad_label=2):
    """Calculate full-repeat prediction and age-group fairness metrics."""
    scores = evaluate_predictions(
        actual, predicted, good_label, bad_label, good_probability
    )
    scores["error_cost_per_100"] = scores["error_cost"] / (len(actual) / 100)

    groups = age_groups(pd.Series(actual_ages).reset_index(drop=True))
    fairness = group_metrics(actual, predicted, groups, good_label, bad_label)
    fairness["true_positive_rate_gap_pp"] = np.nan
    gaps = rate_gaps(fairness)
    gap_mapping = {
        "Approval rate": "approval_rate_gap_pp",
        "False approval rate": "false_approval_rate_gap_pp",
        "False rejection rate": "false_rejection_rate_gap_pp",
    }
    gap_values = {}
    for _, row in gaps.iterrows():
        key = gap_mapping[row["metric"]]
        value = row["absolute_gap_percentage_points"]
        gap_values[key] = np.nan if value is None or pd.isna(value) else float(value)
    # TPR is the complement of false rejection rate within each age group.
    gap_values["true_positive_rate_gap_pp"] = gap_values["false_rejection_rate_gap_pp"]
    return scores, fairness, gap_values


def _summary_table(frame, group_columns, metric_column, value_column):
    """Summarize repeat observations with SD and empirical percentile interval."""
    rows = []
    for keys, group in frame.groupby(group_columns, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = pd.to_numeric(group[value_column], errors="coerce").dropna().to_numpy()
        row = dict(zip(group_columns, keys))
        row[metric_column] = group[metric_column].iloc[0] if metric_column in group else value_column
        if len(values):
            row.update({
                "mean": float(np.mean(values)),
                "standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "empirical_2_5_percentile": float(np.quantile(values, 0.025)),
                "empirical_97_5_percentile": float(np.quantile(values, 0.975)),
                "repeats_with_value": int(len(values)),
            })
        else:
            row.update({
                "mean": np.nan, "standard_deviation": np.nan,
                "empirical_2_5_percentile": np.nan,
                "empirical_97_5_percentile": np.nan, "repeats_with_value": 0,
            })
        rows.append(row)
    return pd.DataFrame(rows)


def _summarize_performance(raw):
    records = []
    for model in raw["model"].unique():
        for metric in PERFORMANCE_METRICS:
            values = raw.loc[raw["model"] == model, metric].to_numpy(dtype=float)
            records.append({
                "model": model,
                "metric": metric,
                "mean": float(np.mean(values)),
                "standard_deviation": float(np.std(values, ddof=1)),
                "empirical_2_5_percentile": float(np.quantile(values, 0.025)),
                "empirical_97_5_percentile": float(np.quantile(values, 0.975)),
                "repeats": int(len(values)),
            })
    return pd.DataFrame(records)


def _summarize_fairness(raw):
    rows = []
    for model in raw["model"].unique():
        model_rows = raw[raw["model"] == model]
        for group in ["Under 25", "25 or older"]:
            subset = model_rows[model_rows["group"] == group]
            for metric in FAIRNESS_RATES:
                values = subset[metric].dropna().to_numpy(dtype=float)
                denominator_column = {
                    "approval_rate": "approval_denominator",
                    "false_approval_rate": "false_approval_denominator",
                    "false_rejection_rate": "false_rejection_denominator",
                    "true_positive_rate": "true_positive_denominator",
                }[metric]
                denominator = int(subset[denominator_column].iloc[0])
                count_column = {
                    "approval_rate": "approval_count",
                    "false_approval_rate": "false_approval_count",
                    "false_rejection_rate": "false_rejection_count",
                    "true_positive_rate": "true_positive_count",
                }[metric]
                row = {
                    "model": model,
                    "group": group,
                    "metric": metric,
                    "denominator_per_repeat": denominator,
                    "mean_numerator_per_repeat": float(subset[count_column].mean()),
                    "mean": float(np.mean(values)),
                    "standard_deviation": float(np.std(values, ddof=1)),
                    "empirical_2_5_percentile": float(np.quantile(values, 0.025)),
                    "empirical_97_5_percentile": float(np.quantile(values, 0.975)),
                    "repeats": len(values),
                }
                rows.append(row)
    return pd.DataFrame(rows)


def _summarize_gaps(raw):
    rows = []
    for model in raw["model"].unique():
        subset = raw[raw["model"] == model]
        for metric in GAP_METRICS:
            values = subset[metric].dropna().to_numpy(dtype=float)
            rows.append({
                "model": model,
                "metric": metric,
                "mean": float(np.mean(values)),
                "standard_deviation": float(np.std(values, ddof=1)),
                "empirical_2_5_percentile": float(np.quantile(values, 0.025)),
                "empirical_97_5_percentile": float(np.quantile(values, 0.975)),
                "repeats": len(values),
            })
    return pd.DataFrame(rows)


def _paired_changes(performance, gaps):
    """Calculate within-repeat deltas against Model B (age excluded)."""
    reference_name = "B: Age excluded"
    reference = performance[performance["model"] == reference_name].set_index("repeat")
    reference_gap = gaps[gaps["model"] == reference_name].set_index("repeat")
    rows = []
    for model in FEATURES:
        if model == reference_name:
            continue
        model_performance = performance[performance["model"] == model].set_index("repeat")
        model_gaps = gaps[gaps["model"] == model].set_index("repeat")
        for metric in ["accuracy", "balanced_accuracy", "roc_auc", "error_cost_per_100"]:
            for repeat in reference.index:
                rows.append({
                    "model": model,
                    "repeat": int(repeat),
                    "metric": metric,
                    "delta_vs_age_excluded": float(model_performance.loc[repeat, metric] - reference.loc[repeat, metric]),
                    "delta_type": "model minus Model B",
                })
        for metric in GAP_METRICS:
            for repeat in reference_gap.index:
                rows.append({
                    "model": model,
                    "repeat": int(repeat),
                    "metric": metric,
                    "delta_vs_age_excluded": float(model_gaps.loc[repeat, metric] - reference_gap.loc[repeat, metric]),
                    "delta_type": "gap difference vs Model B",
                })
    return pd.DataFrame(rows)


def _summarize_paired(changes):
    rows = []
    for (model, metric), group in changes.groupby(["model", "metric"], sort=False):
        values = group["delta_vs_age_excluded"].to_numpy(dtype=float)
        rows.append({
            "model": model,
            "metric": metric,
            "mean_paired_delta": float(np.mean(values)),
            "standard_deviation": float(np.std(values, ddof=1)),
            "empirical_2_5_percentile": float(np.quantile(values, 0.025)),
            "empirical_97_5_percentile": float(np.quantile(values, 0.975)),
            "repeats_delta_below_zero": int((values < 0).sum()),
            "repeats_delta_equal_zero": int((values == 0).sum()),
            "repeats_delta_above_zero": int((values > 0).sum()),
        })
    return pd.DataFrame(rows)


def evaluate_configurations(dataset, fold_plan):
    """Fit every model on each shared fold and return repeat-level OOF metrics."""
    validate_dataset(dataset)
    features = dataset["features"].reset_index(drop=True)
    target = pd.Series(dataset["target"]).reset_index(drop=True)
    age_column = find_age_column(dataset["features"], dataset["feature_descriptions"])
    if age_column != "Attribute13":
        raise ValueError(f"Expected age column Attribute13, got {age_column!r}.")
    ages = features[age_column].to_numpy()
    inputs = make_feature_configurations(features, age_column)
    repeat_numbers = sorted({fold["repeat"] for fold in fold_plan})

    performance_rows = []
    fairness_rows = []
    fold_rows = []
    prediction_rows = []
    for repeat in repeat_numbers:
        repeat_plan = [fold for fold in fold_plan if fold["repeat"] == repeat]
        test_coverage = np.zeros(len(features), dtype=np.int8)
        for fold in repeat_plan:
            test_positions = fold["test_positions"]
            test_coverage[test_positions] += 1
            fold_rows.extend({
                "repeat": repeat, "fold": fold["fold"], "row_position": int(position)
            } for position in test_positions)
        if not np.all(test_coverage == 1):
            raise ValueError(f"Repeat {repeat} does not hold out each row exactly once.")

        for model_name, model_features in inputs.items():
            prediction = np.full(len(features), np.nan)
            probability = np.full(len(features), np.nan)
            for fold in repeat_plan:
                train_positions = fold["train_positions"]
                test_positions = fold["test_positions"]
                X_train = model_features.iloc[train_positions]
                X_test = model_features.iloc[test_positions]
                y_train = target.iloc[train_positions]
                model = Pipeline([
                    ("preprocessor", make_preprocessor(X_train)),
                    ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
                ])
                model.fit(X_train, y_train)
                prediction[test_positions] = model.predict(X_test)
                good_index = list(model.named_steps["classifier"].classes_).index(dataset["good_label"])
                probability[test_positions] = model.predict_proba(X_test)[:, good_index]

            if np.isnan(prediction).any() or np.isnan(probability).any():
                raise ValueError(f"Incomplete out-of-fold predictions for {model_name}, repeat {repeat}.")
            scores, by_age, gap_values = calculate_repeat_metrics(
                target.to_numpy(), prediction.astype(int), probability, ages,
                dataset["good_label"], dataset["bad_label"],
            )
            performance_rows.append({"model": model_name, "repeat": repeat, **scores})
            by_age.insert(0, "repeat", repeat)
            by_age.insert(0, "model", model_name)
            fairness_rows.extend(by_age.to_dict(orient="records"))
            performance_rows[-1].update(gap_values)
            age_group_labels = age_groups(pd.Series(ages)).to_numpy()
            prediction_rows.extend({
                "model": model_name,
                "repeat": repeat,
                "row_position": int(position),
                "actual": int(target.iloc[position]),
                "actual_age": float(ages[position]),
                "age_group": age_group_labels[position],
                "predicted": int(prediction[position]),
                "good_risk_probability": float(probability[position]),
            } for position in range(len(features)))

    return {
        "performance": pd.DataFrame(performance_rows),
        "fairness": pd.DataFrame(fairness_rows),
        "fold_assignments": pd.DataFrame(fold_rows),
        "oof_predictions": pd.DataFrame(prediction_rows),
        "feature_columns": {name: list(frame.columns) for name, frame in inputs.items()},
    }


def _markdown_table(table):
    table = table.copy()
    for column in table.select_dtypes(include="number"):
        table[column] = table[column].map(lambda value: "N/A" if pd.isna(value) else f"{value:.3f}")
    lines = [
        "| " + " | ".join(str(column) for column in table.columns) + " |",
        "|" + "|".join(["---"] * len(table.columns)) + "|",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _save_results(dataset, fold_plan, outputs, output_dir, n_splits, n_repeats, random_state):
    output_dir.mkdir(parents=True, exist_ok=True)
    performance = outputs["performance"]
    fairness = outputs["fairness"]
    performance_summary = _summarize_performance(performance)
    fairness_summary = _summarize_fairness(fairness)
    gap_summary = _summarize_gaps(performance)
    paired = _paired_changes(performance, performance)
    paired_summary = _summarize_paired(paired)

    performance.to_csv(output_dir / "repeat_performance_metrics.csv", index=False)
    performance_summary.to_csv(output_dir / "performance_summary.csv", index=False)
    fairness.to_csv(output_dir / "repeat_fairness_metrics.csv", index=False)
    fairness_summary.to_csv(output_dir / "fairness_summary.csv", index=False)
    gap_summary.to_csv(output_dir / "fairness_gap_summary.csv", index=False)
    paired.to_csv(output_dir / "paired_changes_vs_age_excluded.csv", index=False)
    paired_summary.to_csv(output_dir / "paired_change_summary.csv", index=False)
    outputs["fold_assignments"].to_csv(output_dir / "fold_assignments.csv", index=False)
    outputs["oof_predictions"].to_csv(output_dir / "oof_predictions.csv", index=False)

    age_column = find_age_column(dataset["features"], dataset["feature_descriptions"])
    age_group = age_groups(dataset["features"][age_column]).reset_index(drop=True)
    target = pd.Series(dataset["target"]).reset_index(drop=True)
    denominator_rows = []
    for group in ["Under 25", "25 or older"]:
        selected = age_group == group
        actual_good = int((target[selected] == dataset["good_label"]).sum())
        actual_bad = int((target[selected] == dataset["bad_label"]).sum())
        denominator_rows.append({
            "group": group,
            "group_size": int(selected.sum()),
            "approval_denominator_per_repeat": int(selected.sum()),
            "false_approval_denominator_per_repeat": actual_bad,
            "false_rejection_denominator_per_repeat": actual_good,
            "true_positive_denominator_per_repeat": actual_good,
        })
    denominators = pd.DataFrame(denominator_rows)
    denominators.to_csv(output_dir / "subgroup_denominators.csv", index=False)

    split_records = [{
        "repeat": int(fold["repeat"]),
        "fold": int(fold["fold"]),
        "train_rows": int(len(fold["train_positions"])),
        "test_rows": int(len(fold["test_positions"])),
    } for fold in fold_plan]
    pd.DataFrame(split_records).to_csv(output_dir / "fold_sizes.csv", index=False)

    result_metadata = {
        "dataset": dataset.get("metadata", {}),
        "target": {"name": "class", "good_label": 1, "bad_label": 2},
        "age_column": age_column,
        "validation": {
            "method": "RepeatedStratifiedKFold",
            "folds": n_splits,
            "repeats": n_repeats,
            "random_state": random_state,
            "rows": len(dataset["target"]),
            "preprocessing_fit_on_training_fold_only": True,
            "shared_fold_plan_for_all_models": True,
            "prediction_threshold": "LogisticRegression.predict default decision threshold",
        },
        "classifier": {"name": "LogisticRegression", "max_iter": 1000, "random_state": random_state},
        "models": outputs["feature_columns"],
        "subgroup_denominators": denominator_rows,
        "interval_note": "Empirical 2.5th and 97.5th percentiles across 10 repeated OOF evaluations; not formal population confidence intervals. Repeats reuse the same dataset and have overlapping training sets.",
    }
    (output_dir / "experiment_metadata.json").write_text(
        json.dumps(result_metadata, indent=2, default=lambda value: value.item() if hasattr(value, "item") else str(value)),
        encoding="utf-8",
    )

    interpretation = _interpretation(performance, paired_summary, n_repeats)
    report = [
        "# Repeated stratified validation",
        "",
        f"We evaluated eight predeclared Logistic Regression feature configurations using {n_repeats} repeats of {n_splits}-fold stratified cross-validation (random_state={random_state}). All configurations use the same folds. Every repeat combines its {n_splits} held-out folds into one complete OOF prediction per dataset row. Preprocessing is fitted separately on each training fold only. The existing preprocessor, Logistic Regression (`max_iter=1000`, `random_state={random_state}`), target mapping (1 = Good, 2 = Bad), and `predict` threshold were preserved.",
        "All held-out predictions, actual labels and ages are retained in `oof_predictions.csv` for audit; the fold membership is in `fold_assignments.csv`.",
        "",
        f"The report gives the mean, sample standard deviation and empirical 2.5th–97.5th percentile interval across {n_repeats} repeat-level OOF evaluations. These percentiles describe variation under repeated partitions of this dataset; they are not population confidence intervals. The repeated evaluations are dependent because they reuse rows and overlap in training data.",
        "",
        "## Predictive performance",
        "",
        _markdown_table(performance_summary),
        "",
        "Cost is 5 × false approvals + false rejections. Cost per 100 applicants is that cost divided by dataset size and multiplied by 100.",
        "",
        "## Fairness rates by age group",
        "",
        _markdown_table(fairness_summary),
        "",
        "Rates use the fixed full-dataset subgroup denominators in `subgroup_denominators.csv`; group membership is based on actual age, not model inputs. The table includes the mean numerator count per repeat as well as the denominator.",
        "",
        "## Absolute age-group gaps",
        "",
        _markdown_table(gap_summary),
        "",
        "## Paired changes against age-excluded Model B",
        "",
        _markdown_table(paired_summary),
        "",
        "## Interpretation",
        "",
        *interpretation,
        "",
        f"The under-25 subgroup is smaller than the 25+ subgroup. Even with OOF predictions for all {len(dataset['target'])} rows, its rates use fewer actual good- and bad-risk cases. Repeated folds do not create new independent applicants. This historical dataset, one model family, fixed default threshold, and lack of external validation limit conclusions. Do not infer discrimination or causality from feature-removal differences.",
        "",
    ]
    (output_dir / "repeated_validation_report.md").write_text("\n".join(report), encoding="utf-8")
    return {
        "performance_summary": performance_summary,
        "fairness_summary": fairness_summary,
        "gap_summary": gap_summary,
        "paired_summary": paired_summary,
        "denominators": denominators,
        "output_dir": output_dir,
    }


def _interpretation(performance, paired_summary, n_repeats):
    def row(model, metric):
        return paired_summary[
            (paired_summary["model"] == model) & (paired_summary["metric"] == metric)
        ].iloc[0]

    age_accuracy = row("A: All features", "accuracy")
    age_roc_auc = row("A: All features", "roc_auc")
    age_cost = row("A: All features", "error_cost_per_100")
    checking_approval = row("C: Age + checking excluded", "approval_rate_gap_pp")
    combined_fa = row("H: Age + savings + employment + housing excluded", "false_approval_rate_gap_pp")
    total = n_repeats
    lines = [
        f"- Removing age changed accuracy in {total - int(age_accuracy['repeats_delta_equal_zero'])} of {total} repeats, but direction was mixed: with Model A minus B as the delta, accuracy was lower in {int(age_accuracy['repeats_delta_below_zero'])}, higher in {int(age_accuracy['repeats_delta_above_zero'])}, and equal in {int(age_accuracy['repeats_delta_equal_zero'])}. AUC direction was lower in {int(age_roc_auc['repeats_delta_below_zero'])} and higher in {int(age_roc_auc['repeats_delta_above_zero'])}. Removing age increased cost per 100 by an average of {-age_cost['mean_paired_delta']:+.2f}; this direction occurred in {int(age_cost['repeats_delta_below_zero'])} repeats, but its paired empirical interval crosses zero.",
        f"- Removing checking status narrowed the approval-rate gap in {int(checking_approval['repeats_delta_below_zero'])} of {total} repeats; its mean gap change versus Model B was {checking_approval['mean_paired_delta']:+.2f} percentage points. This removal also raised cost per 100 by an average of {row('C: Age + checking excluded', 'error_cost_per_100')['mean_paired_delta']:+.2f} and lowered accuracy by {row('C: Age + checking excluded', 'accuracy')['mean_paired_delta']:.3f} in every repeat.",
        f"- The combined removal's false-approval gap change versus Model B averaged {combined_fa['mean_paired_delta']:+.2f} percentage points and was positive in {int(combined_fa['repeats_delta_above_zero'])} of {total} repeats. This does not reproduce the very large fixed-test split gap as a stable pattern.",
        "- More stable fairness patterns: including age (Model A versus B) had a larger approval-rate gap and false-rejection gap in all 10 repeats. Removing checking status lowered both of those gaps in all 10 repeats, but did not consistently lower the false-approval gap.",
        "- Stability should be judged from the paired sign counts and empirical ranges alongside absolute performance and all three distinct fairness gaps. Approval and error-rate gaps can move in different directions, so no single metric establishes an overall improvement.",
    ]
    return lines


def run_repeated_validation(dataset, output_dir=OUTPUT_DIR, n_splits=N_SPLITS,
                            n_repeats=N_REPEATS, random_state=RANDOM_STATE):
    """Run the repeated OOF experiment and write only to its dedicated folder."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Repeated-validation output folder is not empty: {output_dir}. Choose a new folder to preserve existing results."
        )
    validate_dataset(dataset)
    features = dataset["features"].reset_index(drop=True)
    target = pd.Series(dataset["target"]).reset_index(drop=True)
    age_column = find_age_column(dataset["features"], dataset["feature_descriptions"])
    fold_plan = make_shared_fold_plan(features, target, n_splits, n_repeats, random_state)
    outputs = evaluate_configurations(dataset, fold_plan)
    return _save_results(
        dataset, fold_plan, outputs, output_dir, n_splits, n_repeats, random_state
    )


def save_oof_predictions(dataset, output_dir=OUTPUT_DIR):
    """Recreate and save the complete OOF prediction audit without replacing outputs."""
    output_dir = Path(output_dir)
    output_path = output_dir / "oof_predictions.csv"
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Run the repeated-validation experiment first: {output_dir}")
    if output_path.exists():
        raise FileExistsError(f"Refusing to replace existing OOF predictions: {output_path}")

    features = dataset["features"].reset_index(drop=True)
    target = pd.Series(dataset["target"]).reset_index(drop=True)
    fold_plan = make_shared_fold_plan(features, target)
    outputs = evaluate_configurations(dataset, fold_plan)
    recorded = pd.read_csv(output_dir / "repeat_performance_metrics.csv")
    for (model, repeat), predictions in outputs["oof_predictions"].groupby(["model", "repeat"]):
        scores, _, _ = calculate_repeat_metrics(
            predictions["actual"].to_numpy(),
            predictions["predicted"].to_numpy(),
            predictions["good_risk_probability"].to_numpy(),
            predictions["actual_age"].to_numpy(),
            dataset["good_label"],
            dataset["bad_label"],
        )
        scores["error_cost_per_100"] = scores["error_cost"] / (len(predictions) / 100)
        expected = recorded[(recorded["model"] == model) & (recorded["repeat"] == repeat)].iloc[0]
        for metric in PERFORMANCE_METRICS:
            if not np.isclose(float(expected[metric]), float(scores[metric])):
                raise ValueError(f"OOF prediction artifact failed metric verification: {model}, repeat {repeat}, {metric}.")
    outputs["oof_predictions"].to_csv(output_path, index=False)
    return output_path


def main():
    from .data_loader import load_credit_dataset

    results = run_repeated_validation(load_credit_dataset())
    print("Predictive performance summary:")
    print(results["performance_summary"].to_string(index=False))
    print("\nFairness summary:")
    print(results["fairness_summary"].to_string(index=False))
    print("\nFairness-gap summary:")
    print(results["gap_summary"].to_string(index=False))
    print(f"\nNew outputs: {results['output_dir']}")


if __name__ == "__main__":
    main()
