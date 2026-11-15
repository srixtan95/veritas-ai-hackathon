"""Test how selected non-age features affect loan risk and age-group rates."""

import json
from pathlib import Path

import pandas as pd
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .data_analysis import find_age_column
from .fairness import age_groups, format_fairness_table, group_metrics, rate_gaps
from .model import evaluate_predictions, make_preprocessor
from .reporting import markdown_table as _markdown_table


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGE_ABLATION_DIR = PROJECT_ROOT / "outputs" / "age_ablation"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "proxy_ablation"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# These candidates are chosen from UCI feature descriptions before evaluating
# this experiment's test results.
PROXY_CANDIDATES = [
    {"name": "Checking account status", "column": "Attribute1"},
    {"name": "Loan purpose", "column": "Attribute4"},
    {"name": "Savings", "column": "Attribute6"},
    {"name": "Employment duration", "column": "Attribute7"},
    {"name": "Housing", "column": "Attribute15"},
]

UCI_CATEGORY_MEANINGS = {
    "Attribute1": {
        "A11": "... < 0 DM", "A12": "0 <= ... < 200 DM",
        "A13": ">= 200 DM / salary assignments for at least 1 year",
        "A14": "No checking account",
    },
    "Attribute4": {
        "A40": "Car (new)", "A41": "Car (used)", "A42": "Furniture/equipment",
        "A43": "Radio/television", "A44": "Domestic appliances", "A45": "Repairs",
        "A46": "Education", "A47": "Vacation (does not exist?)",
        "A48": "Retraining", "A49": "Business", "A410": "Others",
    },
    "Attribute6": {
        "A61": "< 100 DM", "A62": "100 <= ... < 500 DM",
        "A63": "500 <= ... < 1000 DM", "A64": ">= 1000 DM",
        "A65": "Unknown / no savings account",
    },
    "Attribute7": {
        "A71": "Unemployed", "A72": "< 1 year", "A73": "1 <= ... < 4 years",
        "A74": "4 <= ... < 7 years", "A75": ">= 7 years",
    },
    "Attribute15": {"A151": "Rent", "A152": "Own", "A153": "For free"},
}

# A small, pre-defined group of socioeconomic stability attributes.
PROXY_GROUP = {
    "name": "Savings + employment duration + housing",
    "columns": ["Attribute6", "Attribute7", "Attribute15"],
    "reason": "These three UCI descriptions refer to savings, present employment duration, and housing.",
}


def validate_dataset(dataset):
    """Check the expected target mapping and prevent target leakage."""
    if dataset["target_name"] != "class":
        raise ValueError(f"Expected target 'class', got {dataset['target_name']!r}.")
    if dataset["good_label"] != 1 or dataset["bad_label"] != 2:
        raise ValueError("Expected class 1 = Good risk and class 2 = Bad risk.")
    if dataset["target_name"] in dataset["features"].columns:
        raise ValueError("The target must stay outside the feature table.")
    if set(pd.unique(dataset["target"])) != {1, 2}:
        raise ValueError("Expected both target values 1 and 2.")


def make_shared_split(dataset):
    """Recreate the fixed stratified split used by the age-ablation reference."""
    validate_dataset(dataset)
    age_column = find_age_column(dataset["features"], dataset["feature_descriptions"])
    X = dataset["features"]
    y = dataset["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return {
        "age_column": age_column,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def create_feature_variants(split, candidate_features=PROXY_CANDIDATES, proxy_group=PROXY_GROUP):
    """Build the age-excluded reference, individual removals, and one group removal."""
    age_column = split["age_column"]
    base_train = split["X_train"].drop(columns=[age_column])
    base_test = split["X_test"].drop(columns=[age_column])
    variants = {
        "Reference (age excluded)": {
            "removed_features": [age_column],
            "X_train": base_train,
            "X_test": base_test,
        }
    }

    candidate_columns = {item["column"] for item in candidate_features}
    missing = candidate_columns - set(base_train.columns)
    if missing:
        raise ValueError(f"Candidate columns are missing from the data: {sorted(missing)}")

    for candidate in candidate_features:
        column = candidate["column"]
        variants[f"Without {candidate['name']} ({column})"] = {
            "removed_features": [age_column, column],
            "X_train": base_train.drop(columns=[column]),
            "X_test": base_test.drop(columns=[column]),
        }

    group_columns = list(proxy_group["columns"])
    missing_group_columns = set(group_columns) - set(base_train.columns)
    if missing_group_columns:
        raise ValueError(f"Proxy group columns are missing: {sorted(missing_group_columns)}")
    variants[f"Without {proxy_group['name']}"] = {
        "removed_features": [age_column, *group_columns],
        "X_train": base_train.drop(columns=group_columns),
        "X_test": base_test.drop(columns=group_columns),
    }
    return variants


def _evaluate(model, features, actual, actual_ages, good_label, bad_label):
    """Evaluate one model on the fixed test rows and group by held-out ages."""
    predicted = model.predict(features)
    good_index = list(model.named_steps["classifier"].classes_).index(good_label)
    good_probability = model.predict_proba(features)[:, good_index]
    scores = evaluate_predictions(actual, predicted, good_label, bad_label, good_probability)
    groups = age_groups(actual_ages)
    fairness = group_metrics(actual, predicted, groups, good_label, bad_label)
    return scores, fairness


def _compare_reference(saved, computed):
    """Fail if the loaded Model B does not reproduce the saved age-ablation result."""
    for name in [
        "accuracy", "balanced_accuracy", "roc_auc", "false_approvals",
        "false_rejections", "error_cost", "confusion_matrix",
    ]:
        if saved[name] != computed[name]:
            raise ValueError(f"Reference Model B metric {name!r} does not match its saved result.")


def _safe_records(frame):
    """Convert DataFrame values to JSON-friendly Python values and nulls."""
    records = []
    for row in frame.to_dict(orient="records"):
        clean = {}
        for key, value in row.items():
            if pd.isna(value):
                clean[key] = None
            elif hasattr(value, "item"):
                clean[key] = value.item()
            else:
                clean[key] = value
        records.append(clean)
    return records


def _write_outputs(dataset, split, variants, results, output_dir):
    """Save all proxy-ablation files in the dedicated output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir()

    metric_rows = []
    fairness_rows = []
    display_rows = []
    gap_rows = []
    for name, result in results.items():
        metric_rows.append({
            "experiment": name,
            "removed_features": ", ".join(variants[name]["removed_features"]),
            **result["metrics"],
        })
        fairness = result["fairness"].copy()
        fairness["group"] = pd.Categorical(
            fairness["group"], categories=["Under 25", "25 or older"], ordered=True
        )
        fairness = fairness.sort_values("group").reset_index(drop=True)
        fairness.insert(0, "experiment", name)
        fairness_rows.extend(fairness.to_dict(orient="records"))

        display = format_fairness_table(fairness)
        display.insert(0, "Experiment", name)
        display_rows.extend(display.to_dict(orient="records"))

        gaps = rate_gaps(fairness)
        gaps.insert(0, "experiment", name)
        gap_rows.extend(gaps.to_dict(orient="records"))

    metric_table = pd.DataFrame(metric_rows)
    fairness_frame = pd.DataFrame(fairness_rows)
    display_table = pd.DataFrame(display_rows)
    gaps_frame = pd.DataFrame(gap_rows)
    metric_table.to_csv(output_dir / "model_comparison.csv", index=False)
    fairness_frame.to_csv(output_dir / "fairness_metrics.csv", index=False)
    display_table.to_csv(output_dir / "fairness_table.csv", index=False)
    gaps_frame.to_csv(output_dir / "fairness_gaps.csv", index=False)

    candidate_rows = []
    descriptions = dataset["feature_descriptions"]
    for candidate in PROXY_CANDIDATES:
        column = candidate["column"]
        counts = dataset["features"][column].value_counts(dropna=False)
        candidate_rows.append({
            "candidate": candidate["name"],
            "column": column,
            "uci_description": descriptions.get(column, "Description not available."),
            "uci_code_meanings": "; ".join(
                f"{code}: {meaning}" for code, meaning in UCI_CATEGORY_MEANINGS[column].items()
            ),
            "observed_code_counts": "; ".join(
                f"{code}: {int(count)}" for code, count in counts.items()
            ),
            "experiment": f"Without {candidate['name']} ({candidate['column']})",
        })
    pd.DataFrame(candidate_rows).to_csv(output_dir / "candidate_features.csv", index=False)

    split_rows = [
        {"row_index": index, "split": "train"} for index in split["X_train"].index
    ] + [
        {"row_index": index, "split": "test"} for index in split["X_test"].index
    ]
    pd.DataFrame(split_rows).to_csv(output_dir / "split_indices.csv", index=False)

    result_json = {
        "target": {"name": dataset["target_name"], "good_label": 1, "bad_label": 2},
        "age_column": split["age_column"],
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratified": True,
            "train_rows": len(split["y_train"]),
            "test_rows": len(split["y_test"]),
            "train_indices": list(split["X_train"].index),
            "test_indices": list(split["X_test"].index),
        },
        "reference_model": "saved outputs/age_ablation/model_b_without_age.joblib",
        "models": metric_rows,
        "fairness": _safe_records(fairness_frame),
        "fairness_gaps": _safe_records(gaps_frame),
        "group_experiment": PROXY_GROUP,
    }
    (output_dir / "experiment_results.json").write_text(
        json.dumps(result_json, indent=2, default=lambda value: value.item() if hasattr(value, "item") else str(value)),
        encoding="utf-8",
    )

    for name, result in results.items():
        if result["new_model"] is not None:
            filename = result["model_filename"]
            dump(result["new_model"], models_dir / filename)

    ref = results["Reference (age excluded)"]["metrics"]
    delta_rows = []
    for row in metric_rows:
        if row["experiment"] == "Reference (age excluded)":
            continue
        delta_rows.append({
            "experiment": row["experiment"],
            "accuracy_change_percentage_points": (row["accuracy"] - ref["accuracy"]) * 100,
            "balanced_accuracy_change_percentage_points": (row["balanced_accuracy"] - ref["balanced_accuracy"]) * 100,
            "roc_auc_change": row["roc_auc"] - ref["roc_auc"],
            "cost_change": row["error_cost"] - ref["error_cost"],
        })
    delta_table = pd.DataFrame(delta_rows)
    delta_table.to_csv(output_dir / "performance_changes.csv", index=False)

    candidate_table = pd.read_csv(output_dir / "candidate_features.csv")
    changes = delta_table.set_index("experiment")
    gap_lookup = gaps_frame.pivot(
        index="experiment", columns="metric", values="absolute_gap_percentage_points"
    )
    checking = changes.loc["Without Checking account status (Attribute1)"]
    savings = changes.loc["Without Savings (Attribute6)"]
    group_name = f"Without {PROXY_GROUP['name']}"
    group_changes = changes.loc[group_name]
    group_gaps = gap_lookup.loc[group_name]
    reference_gaps = gap_lookup.loc["Reference (age excluded)"]
    report = [
        "# Proxy feature ablation experiment",
        "",
        "The reference is the already-saved age-excluded Model B. Each candidate is removed individually in a separate Logistic Regression model. The pre-defined group experiment removes savings, employment duration, and housing together after the individual experiments. Candidate choices come from UCI feature descriptions and category mappings, not test-set scores.",
        f"All models use the same stratified 80/20 split (random_state={RANDOM_STATE}), class 1 = Good / class 2 = Bad, the same Logistic Regression settings, and the existing preprocessing function. Preprocessing is fit on training rows only. Test rows are used only for final evaluation.",
        "",
        "## Candidate features and descriptions",
        "",
        _markdown_table(candidate_table),
        "",
        f"Group ablation: {PROXY_GROUP['name']} ({', '.join(PROXY_GROUP['columns'])}). {PROXY_GROUP['reason']}",
        "",
        "## Predictive performance",
        "",
        _markdown_table(metric_table[[
            "experiment", "accuracy", "balanced_accuracy", "roc_auc",
            "false_approvals", "false_rejections", "error_cost",
        ]]),
        "",
        "## Performance change from reference",
        "",
        _markdown_table(delta_table),
        "",
        "## Age-group fairness metrics",
        "",
        _markdown_table(display_table),
        "",
        "## Absolute age-group gaps",
        "",
        _markdown_table(gaps_frame),
        "",
        "## Interpretation",
        "",
        f"Among individual removals, excluding checking-account status had the largest predictive decline: accuracy changed by {checking['accuracy_change_percentage_points']:+.2f} percentage points, balanced accuracy by {checking['balanced_accuracy_change_percentage_points']:+.2f} points, ROC-AUC by {checking['roc_auc_change']:+.3f}, and cost by {checking['cost_change']:+.0f}. Excluding savings gave the largest accuracy and balanced-accuracy increases among the individual removals ({savings['accuracy_change_percentage_points']:+.2f} and {savings['balanced_accuracy_change_percentage_points']:+.2f} points; cost {savings['cost_change']:+.0f}); these are single-split results, not evidence that savings harms generalization.",
        f"The group removal changed accuracy by {group_changes['accuracy_change_percentage_points']:+.2f} points and cost by {group_changes['cost_change']:+.0f}. Its approval-rate gap fell from {reference_gaps['Approval rate']:.2f} to {group_gaps['Approval rate']:.2f} points, while its false-approval gap rose from {reference_gaps['False approval rate']:.2f} to {group_gaps['False approval rate']:.2f} points. The group experiment therefore does not show a uniform fairness improvement.",
        "",
        "Rates include numerator/denominator in `fairness_table.csv`. Approval denominator is all applicants in that age group; false approval denominator is actual Bad-risk applicants in that group; false rejection denominator is actual Good-risk applicants in that group. `candidate_features.csv` lists UCI code meanings and observed feature-column category counts.",
        "",
        "A feature removal that changes model scores or subgroup gaps does not prove the feature causes unfair outcomes. The dataset has only 1,000 historical cases and the under-25 test group is small. Remaining features may also encode age-related information. These results are exploratory and use one fixed split.",
        "",
    ]
    (output_dir / "proxy_ablation_report.md").write_text("\n".join(report), encoding="utf-8")


def run_proxy_ablation(dataset, output_dir=OUTPUT_DIR):
    """Run individual and group proxy ablations without changing prior outputs."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Proxy-ablation output folder is not empty: {output_dir}. "
            "Choose a new folder to avoid replacing prior results."
        )

    split = make_shared_split(dataset)
    variants = create_feature_variants(split)

    # Use Model B from the previous age-ablation experiment as the reference.
    reference_path = AGE_ABLATION_DIR / "model_b_without_age.joblib"
    reference_model = load(reference_path)
    reference_variant = variants["Reference (age excluded)"]
    if list(reference_model.feature_names_in_) != list(reference_variant["X_train"].columns):
        raise ValueError("Saved Model B inputs do not match the age-excluded reference features.")

    saved_age_results = json.loads(
        (AGE_ABLATION_DIR / "experiment_results.json").read_text(encoding="utf-8")
    )
    if list(split["X_test"].index) != saved_age_results["split"]["test_indices"]:
        raise ValueError("Recreated test rows do not match the saved age-ablation split.")
    if list(split["X_train"].index) != saved_age_results["split"]["train_indices"]:
        raise ValueError("Recreated training rows do not match the saved age-ablation split.")

    actual_ages = split["X_test"][split["age_column"]]
    results = {}
    for name, variant in variants.items():
        if name == "Reference (age excluded)":
            model = reference_model
            filename = None
        else:
            model = Pipeline([
                ("preprocessor", make_preprocessor(variant["X_train"])),
                ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ])
            model.fit(variant["X_train"], split["y_train"])
            filename = (
                name.lower().replace(" ", "_").replace("+", "and").replace("(", "").replace(")", "")
                + ".joblib"
            )

        scores, fairness = _evaluate(
            model,
            variant["X_test"],
            split["y_test"],
            actual_ages,
            dataset["good_label"],
            dataset["bad_label"],
        )
        results[name] = {
            "metrics": scores,
            "fairness": fairness,
            "new_model": None if filename is None else model,
            "model_filename": filename,
        }

    saved_reference = saved_age_results["models"]["Model B (without age)"]
    _compare_reference(saved_reference, results["Reference (age excluded)"]["metrics"])
    _write_outputs(dataset, split, variants, results, output_dir)

    metrics = pd.read_csv(output_dir / "model_comparison.csv")
    fairness = pd.read_csv(output_dir / "fairness_table.csv")
    gaps = pd.read_csv(output_dir / "fairness_gaps.csv")
    return {"metrics": metrics, "fairness": fairness, "gaps": gaps, "output_dir": output_dir}


def main():
    """Load UCI data, run the proxy ablation, and print its tables."""
    from .data_loader import load_credit_dataset

    results = run_proxy_ablation(load_credit_dataset())
    print("Predictive performance:")
    print(results["metrics"].to_string(index=False))
    print("\nFairness metrics:")
    print(results["fairness"].to_string(index=False))
    print("\nAge-group gaps:")
    print(results["gaps"].to_string(index=False))
    print(f"\nNew outputs: {results['output_dir']}")


if __name__ == "__main__":
    main()
