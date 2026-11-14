"""Explore age groups, data quality, feature differences, and possible proxies."""

import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, mannwhitneyu, spearmanr
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from .fairness import age_groups, format_fairness_table, group_metrics, rate_gaps
from .model import make_preprocessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
PLOT_DIR = PROJECT_ROOT / "outputs" / "plots"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
BASELINE_PLOTS = PROJECT_ROOT / "data" / "eda"


def wilson_interval(successes, total):
    """Return a 95% Wilson confidence interval for a proportion."""
    if total == 0:
        return float("nan"), float("nan")
    z = 1.96
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z**2 / (4 * total**2)
    ) / denominator
    return center - margin, center + margin


def cramers_v(first, second):
    """Measure association between two categorical columns (0 to 1)."""
    table = pd.crosstab(first, second)
    if min(table.shape) < 2 or table.values.sum() == 0:
        return float("nan")
    chi_square = chi2_contingency(table, correction=False)[0]
    sample_count = table.values.sum()
    denominator = sample_count * max(1, min(table.shape) - 1)
    return float(math.sqrt(chi_square / denominator))


def find_age_column(features, descriptions):
    """Find the age feature using the UCI descriptions, not column numbers."""
    for column in features:
        description = descriptions.get(str(column), "")
        if re.search(r"\bage\b", description, flags=re.IGNORECASE):
            return column
    raise ValueError("Could not find an age feature in the UCI feature descriptions.")


def write_age_fairness_report(dataset, test_features, actual, predicted):
    """Save age-group rates and their percentage-point gaps for test rows."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    age_column = find_age_column(dataset["features"], dataset["feature_descriptions"])
    groups = age_groups(test_features[age_column])
    metrics = group_metrics(actual, predicted, groups, dataset["good_label"], dataset["bad_label"])
    metrics["group"] = pd.Categorical(
        metrics["group"], categories=["Under 25", "25 or older"], ordered=True
    )
    metrics = metrics.sort_values("group").reset_index(drop=True)
    gaps = rate_gaps(metrics)
    display_table = format_fairness_table(metrics)

    metrics.to_csv(REPORT_DIR / "age_group_model_fairness.csv", index=False)
    gaps.to_csv(REPORT_DIR / "age_group_fairness_gaps.csv", index=False)
    display_table.to_csv(REPORT_DIR / "age_group_fairness_table.csv", index=False)

    def higher_group(column):
        under_25 = metrics.loc[metrics["group"] == "Under 25", column].iloc[0]
        older = metrics.loc[metrics["group"] == "25 or older", column].iloc[0]
        if pd.isna(under_25) or pd.isna(older):
            return "cannot be compared because a denominator is zero"
        if under_25 == older:
            return "is equal for both groups"
        group_name = "Under 25" if under_25 > older else "25 or older"
        return f"is higher for {group_name}"

    table_lines = [
        "| " + " | ".join(display_table.columns) + " |",
        "|" + "|".join(["---"] * len(display_table.columns)) + "|",
    ]
    for _, row in display_table.iterrows():
        table_lines.append("| " + " | ".join(str(value) for value in row) + " |")

    gap_lines = []
    for _, row in gaps.iterrows():
        value = row["absolute_gap_percentage_points"]
        gap_lines.append(
            f"- {row['metric']}: {value:.2f} percentage points."
            if value is not None and not pd.isna(value)
            else f"- {row['metric']}: N/A because one group has a zero denominator."
        )

    report = [
        "# Age-based fairness audit",
        "",
        "This audit uses predictions from the saved Logistic Regression model on the existing stratified 20% test split (seed 42). It does not fit or change the model.",
        "The target is not included in prediction inputs. The saved baseline was trained with all 20 original features, including the age feature, so this audit preserves that baseline model as requested.",
        "",
        "## Test-set comparison",
        "",
        *table_lines,
        "",
        "Rates show their numerator and denominator in parentheses. Approval rate = predicted-good applicants / all applicants in group. False approval rate = actual-bad applicants predicted good / actual-bad applicants in group. False rejection rate = actual-good applicants predicted bad / actual-good applicants in group. True positive rate = actual-good applicants predicted good / actual-good applicants in group.",
        "",
        "## Absolute rate gaps",
        "",
        *gap_lines,
        "",
        "## Interpretation",
        "",
        f"- Approval rate {higher_group('approval_rate')}; false approval rate {higher_group('false_approval_rate')}; false rejection rate {higher_group('false_rejection_rate')}; true positive rate {higher_group('true_positive_rate')}.",
        "- These are descriptive differences, not proof of discrimination. Only 39 under-25 applicants are in the held-out test set, so those subgroup rates are less stable. The German Credit dataset is small and reflects historical lending outcomes.",
        "",
    ]
    (REPORT_DIR / "age_fairness_audit.md").write_text("\n".join(report), encoding="utf-8")
    return metrics, gaps, display_table


def save_baseline_plots(dataset):
    """Save the original target and feature-by-risk charts under data/eda."""
    BASELINE_PLOTS.mkdir(parents=True, exist_ok=True)
    features = dataset["features"]
    descriptions = dataset["feature_descriptions"]
    labels = dataset["target"].map({dataset["good_label"]: "Good credit risk", dataset["bad_label"]: "Bad credit risk"})
    sns.set_theme(style="whitegrid")

    fig, axis = plt.subplots(figsize=(7, 5))
    sns.countplot(x=labels, order=["Good credit risk", "Bad credit risk"], color="#3973ac", ax=axis)
    axis.set(title="German Credit: target class distribution", xlabel="Credit risk", ylabel="Applicants")
    fig.tight_layout()
    fig.savefig(BASELINE_PLOTS / "target_distribution.png", dpi=160)
    plt.close(fig)

    feature_names = {
        "age": ["age"],
        "employment": ["employment"],
        "savings": ["savings", "saving"],
        "credit_history": ["credit history"],
        "loan_purpose": ["purpose"],
        "credit_amount": ["credit amount", "amount"],
        "duration": ["duration"],
    }
    for concept, terms in feature_names.items():
        for column in features:
            text = f"{column} {descriptions.get(str(column), '')}".lower()
            if not any(re.search(rf"\b{re.escape(term)}\b", text) for term in terms):
                continue
            frame = pd.DataFrame({"value": features[column], "Credit risk": labels})
            fig, axis = plt.subplots(figsize=(9, 5))
            if pd.api.types.is_numeric_dtype(features[column]) and features[column].nunique() > 12:
                sns.histplot(
                    data=frame, x="value", hue="Credit risk", bins=20,
                    element="step", stat="count", common_norm=False, ax=axis,
                )
            else:
                sns.countplot(
                    data=frame,
                    x="value",
                    hue="Credit risk",
                    order=frame["value"].value_counts().index,
                    ax=axis,
                )
                axis.tick_params(axis="x", rotation=35)
            axis.set_title(f"{concept.replace('_', ' ').title()} by credit-risk class")
            axis.set_xlabel(f"{column}: {descriptions.get(str(column), '')}".strip())
            fig.tight_layout()
            filename = f"{concept}_{str(column).lower().replace(' ', '_')}.png"
            fig.savefig(BASELINE_PLOTS / filename, dpi=150)
            plt.close(fig)


def save_age_group_plots(age, groups, risk, features, numeric_columns, categorical_columns, associations):
    """Save age, credit-risk, and age-group feature comparison charts."""
    sns.set_theme(style="whitegrid")
    group_risk_data = pd.DataFrame({"age_group": groups, "risk": risk})
    fig, axis = plt.subplots(figsize=(7, 4))
    sns.histplot(age, bins=18, ax=axis)
    axis.set(title="Applicant age distribution", xlabel="Age (years)", ylabel="Applicants")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "age_distribution.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4))
    sns.countplot(x=groups, order=["Under 25", "25 or older"], ax=axis)
    axis.set(title="Applicant age groups", xlabel="Age group", ylabel="Applicants")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "age_group_distribution.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4))
    sns.countplot(x=risk, order=["Good", "Bad"], ax=axis)
    axis.set(title="Credit risk labels", xlabel="Credit risk", ylabel="Applicants")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "credit_risk_distribution.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4))
    sns.countplot(
        data=group_risk_data,
        x="age_group",
        hue="risk",
        order=["Under 25", "25 or older"],
        hue_order=["Good", "Bad"],
        ax=axis,
    )
    axis.set(title="Credit risk by age group", xlabel="Age group", ylabel="Applicants")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "credit_risk_by_age_group.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, max(3, 0.38 * len(numeric_columns + categorical_columns))))
    association_frame = pd.DataFrame(associations).dropna()
    if not association_frame.empty:
        association_frame = association_frame.sort_values("association_strength", ascending=False)
        sns.heatmap(
            association_frame.set_index("feature"), annot=True, vmin=0, vmax=1,
            cmap="mako", ax=axis, cbar_kws={"label": "Association magnitude"},
        )
    axis.set_title("Feature association with age or age group (exploratory)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "feature_association_heatmap.png", dpi=150)
    plt.close(fig)

    for filename, columns, kind in [
        ("numerical_feature_comparisons.png", numeric_columns, "numeric"),
        ("categorical_feature_comparisons.png", categorical_columns, "categorical"),
    ]:
        fig, axes = plt.subplots(max(1, len(columns)), 1, figsize=(9, 4 * max(1, len(columns))))
        axes = np.atleast_1d(axes)
        for axis, column in zip(axes, columns):
            if kind == "numeric":
                sns.boxplot(x=groups, y=features[column], ax=axis)
            else:
                chart_data = pd.DataFrame({"group": groups, "value": features[column]})
                sns.histplot(data=chart_data, x="value", hue="group", multiple="fill", shrink=0.8, ax=axis)
                axis.tick_params(axis="x", rotation=35)
            axis.set_title(str(column))
        fig.tight_layout()
        fig.savefig(PLOT_DIR / filename, dpi=140)
        plt.close(fig)


def run_proxy_experiment(features, groups, age_column, target_name):
    """Check whether non-age, non-credit-target features predict age group."""
    predictors = features.drop(columns=[age_column])
    proxy_target = (groups == "Under 25").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        predictors, proxy_target, test_size=0.20, random_state=42, stratify=proxy_target
    )
    model = Pipeline([
        ("preprocessor", make_preprocessor(X_train)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    scoring = ["balanced_accuracy", "precision", "recall", "f1", "roc_auc"]
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cross_validation = cross_validate(model, X_train, y_train, cv=folds, scoring=scoring)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    dummy_predictions = dummy.predict(X_test)

    def scores(actual, predicted, probability=None):
        result = {
            "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
            "precision": float(precision_score(actual, predicted, zero_division=0)),
            "recall": float(recall_score(actual, predicted, zero_division=0)),
            "f1": float(f1_score(actual, predicted, zero_division=0)),
        }
        if probability is not None:
            result["roc_auc"] = float(roc_auc_score(actual, probability))
        return result

    return {
        "age_feature_excluded": str(age_column),
        "target_feature_excluded": target_name,
        "age_group_counts": {str(group): int(count) for group, count in groups.value_counts().items()},
        "test_n": len(y_test),
        "model_test": scores(y_test, predictions, probabilities),
        "majority_baseline_test": scores(y_test, dummy_predictions),
        "cross_validation_train_folds": {
            metric: {
                "mean": float(np.mean(cross_validation[f"test_{metric}"])),
                "std": float(np.std(cross_validation[f"test_{metric}"], ddof=1)),
            }
            for metric in scoring
        },
        "interpretation": (
            "Predictability suggests the other features contain information about age group. "
            "It does not prove causation or discrimination."
        ),
    }


def run_data_analysis(dataset, baseline):
    """Write data-quality, age-group, proxy, fairness, and baseline reports."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    features = dataset["features"].copy()
    target = dataset["target"]
    descriptions = dataset["feature_descriptions"]
    age_column = find_age_column(features, descriptions)
    age = features[age_column].astype(float)
    groups = age_groups(age)
    risk = target.map({dataset["good_label"]: "Good", dataset["bad_label"]: "Bad"})
    combined = features.copy()
    combined[dataset["target_name"]] = target
    row_count = len(features)
    risk_counts = risk.value_counts()

    # Save a full feature and data-quality summary.
    profile = {
        "uci_id": 144,
        "dataset": dataset["metadata"]["name"],
        "rows": row_count,
        "feature_count": len(features.columns),
        "columns": list(features.columns),
        "target": {
            "name": dataset["target_name"],
            "good_label": int(dataset["good_label"]),
            "bad_label": int(dataset["bad_label"]),
            "counts": {str(key): int(value) for key, value in risk_counts.items()},
            "percentages": {str(key): float(value / row_count * 100) for key, value in risk_counts.items()},
        },
        "numeric_features": int(len(features.select_dtypes(include="number").columns)),
        "categorical_features": int(len(features.select_dtypes(exclude="number").columns)),
        "missing_by_column": {str(key): int(value) for key, value in combined.isna().sum().items()},
        "total_missing": int(combined.isna().sum().sum()),
        "duplicate_rows": int(combined.duplicated().sum()),
        "unique_by_feature": {str(column): int(features[column].nunique(dropna=False)) for column in features},
        "dtypes": {str(column): str(dtype) for column, dtype in combined.dtypes.items()},
    }
    (REPORT_DIR / "dataset_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")

    dictionary = []
    sensitive_terms = ["age", "sex", "personal status", "employment", "saving", "account", "housing", "job", "worker"]
    for column in features:
        description = descriptions.get(str(column), "Description not supplied by UCI.")
        dictionary.append({
            "original_column": column,
            "description": description,
            "dtype": str(features[column].dtype),
            "kind": "numerical" if pd.api.types.is_numeric_dtype(features[column]) else "categorical",
            "examples": "; ".join(features[column].dropna().astype(str).unique()[:5]),
            "demographic_or_socioeconomic_implications": any(term in description.lower() for term in sensitive_terms),
        })
    pd.DataFrame(dictionary).to_csv(REPORT_DIR / "feature_dictionary.csv", index=False)

    # Summarize good and bad historical labels within the two age groups.
    age_outcomes = []
    for group_name in ["Under 25", "25 or older"]:
        group_risk = risk[groups == group_name]
        total = len(group_risk)
        good_count = int((group_risk == "Good").sum())
        bad_count = total - good_count
        low, high = wilson_interval(good_count, total)
        age_outcomes.append({
            "age_group": group_name,
            "applicants": total,
            "share_percent": 100 * total / row_count,
            "good": good_count,
            "bad": bad_count,
            "good_percent_within_group": 100 * good_count / total if total else np.nan,
            "good_wilson95_low_percent": 100 * low,
            "good_wilson95_high_percent": 100 * high,
            "bad_percent_within_group": 100 * bad_count / total if total else np.nan,
        })
    pd.DataFrame(age_outcomes).to_csv(REPORT_DIR / "age_group_outcomes.csv", index=False)

    # Compare demographic and financial features between age groups.
    topics = [
        "employment", "saving", "checking account", "housing", "job", "credit history",
        "purpose", "credit amount", "duration", "property", "residence",
    ]
    selected = [
        column for column in features
        if any(topic in descriptions.get(str(column), "").lower() for topic in topics)
    ]
    numeric_columns = [column for column in selected if pd.api.types.is_numeric_dtype(features[column])]
    categorical_columns = [column for column in selected if column not in numeric_columns]
    comparisons = []
    associations = []
    for column in numeric_columns:
        under_25 = features.loc[groups == "Under 25", column].dropna()
        older = features.loc[groups == "25 or older", column].dropna()
        association = spearmanr(age, features[column], nan_policy="omit").statistic
        comparisons.append({
            "feature": column,
            "description": descriptions.get(str(column), ""),
            "type": "numeric",
            "age_association_spearman": association,
            "under25_mean": under_25.mean(),
            "under25_median": under_25.median(),
            "25plus_mean": older.mean(),
            "25plus_median": older.median(),
            "mannwhitney_p_descriptive": mannwhitneyu(under_25, older).pvalue if len(under_25) and len(older) else np.nan,
        })
        associations.append({"feature": column, "association_strength": abs(association)})

    for column in categorical_columns:
        association = cramers_v(groups, features[column])
        frequencies = {
            group_name: {
                str(value): int(count)
                for value, count in features.loc[groups == group_name, column].value_counts().items()
            }
            for group_name in ["Under 25", "25 or older"]
        }
        comparisons.append({
            "feature": column,
            "description": descriptions.get(str(column), ""),
            "type": "categorical",
            "age_association_cramers_v": association,
            "category_frequencies": json.dumps(frequencies),
        })
        associations.append({"feature": column, "association_strength": association})

    pd.DataFrame(comparisons).to_csv(REPORT_DIR / "age_group_feature_comparison.csv", index=False)
    save_age_group_plots(age, groups, risk, features, numeric_columns, categorical_columns, associations)
    save_baseline_plots(dataset)

    # Reuse test-row indexes to align age with the baseline predictions.
    test_features = features.loc[baseline["y_test"].index]
    age_group_rates, fairness_gaps, fairness_table = write_age_fairness_report(
        dataset,
        test_features,
        baseline["y_test"],
        baseline["predictions"],
    )

    # A separate experiment asks if other columns can predict age group.
    proxy = run_proxy_experiment(features, groups, age_column, dataset["target_name"])
    (METRICS_DIR / "proxy_prediction_metrics.json").write_text(json.dumps(proxy, indent=2), encoding="utf-8")

    # Write concise reports with clear limitations and the original baseline scores.
    negative_numbers = all(
        (features[column].dropna() >= 0).all()
        for column in features.select_dtypes(include="number")
    )
    quality_lines = [
        "# Data quality report", "",
        f"Missing values: {profile['total_missing']}; duplicate complete rows: {profile['duplicate_rows']}.", "",
        "Numeric features contain no negative values." if negative_numbers else "Some numeric features contain negative values; check the UCI descriptions before cleaning.", "",
        "No rows or category codes were changed. Category codes should be interpreted using the UCI feature descriptions.",
        "",
        "Association and variation checks are exploratory; no feature was removed.",
    ]
    (REPORT_DIR / "data_quality_report.md").write_text("\n".join(quality_lines), encoding="utf-8")
    (REPORT_DIR / "dataset_profile.md").write_text(
        f"# Dataset profile\n\nUCI German Credit has {row_count} applicants and {features.shape[1]} features. "
        f"There are {profile['numeric_features']} numeric and {profile['categorical_features']} categorical features. "
        f"Missing values: {profile['total_missing']}; duplicate rows: {profile['duplicate_rows']}.\n",
        encoding="utf-8",
    )
    (REPORT_DIR / "proxy_analysis_report.md").write_text(
        f"# Proxy analysis\n\nAge group is predicted without the age column `{age_column}` or the credit-risk target `{dataset['target_name']}`. "
        "The model uses a stratified split and five-fold cross-validation on training rows.\n\n"
        f"Test scores: `{json.dumps(proxy['model_test'])}`. Majority baseline: `{json.dumps(proxy['majority_baseline_test'])}`.\n\n"
        "Predictability suggests the other features contain information about age group. It does not prove causation or discrimination.",
        encoding="utf-8",
    )
    age_counts = groups.value_counts().to_dict()
    associations.sort(key=lambda item: item["association_strength"] if pd.notna(item["association_strength"]) else -1, reverse=True)
    limitations = (
        "The dataset contains only 1,000 applicants and reflects an older lending context. Age groups are imbalanced, "
        "and the smaller under-25 group makes estimates less certain. Historical good/bad labels may reflect past lending "
        "and measurement choices. Repayment outcomes can be unavailable for people who were previously denied credit. "
        "Savings, housing, employment, and account status also reflect socioeconomic opportunity. All reported differences "
        "are descriptive, not causal, and do not establish discrimination or fairness."
    )
    (REPORT_DIR / "data_audit_report.md").write_text(
        f"# German Credit Data Audit\n\n"
        f"The dataset has {row_count} applicants, {features.shape[1]} features, and target labels 1 = good and 2 = bad credit risk. "
        "The personal-status field combines status and sex; the dataset does not provide separate gender or marital-status fields.\n\n"
        f"## Data quality\n\nMissing cells: {profile['total_missing']}; duplicate rows: {profile['duplicate_rows']}. No observations were changed.\n\n"
        f"## Age-group comparison\n\nUnder 25: {age_counts.get('Under 25', 0)} applicants. 25 or older: {age_counts.get('25 or older', 0)} applicants. "
        "Good/bad label rates and Wilson 95% intervals are in `age_group_outcomes.csv`. Baseline model rates are in `age_group_model_fairness.csv`.\n\n"
        f"## Feature and proxy investigation\n\nThe strongest observed age associations include {', '.join(item['feature'] for item in associations[:3])}. "
        "These are candidate proxy signals, not evidence of unfairness. See `age_group_feature_comparison.csv` and `proxy_analysis_report.md`.\n\n"
        f"## Limitations\n\n{limitations}\n",
        encoding="utf-8",
    )

    # Keep the original Stage 1 report and its metric definitions current.
    scores = baseline["scores"]
    logistic = scores["logistic_regression"]
    dummy = scores["dummy_all_good"]
    (PROJECT_ROOT / "docs" / "stage1_report.md").write_text(
        "# Immortals AI - Dataset Exploration and Baseline\n\n"
        f"Dataset: {dataset['metadata']['name']} (UCI ID 144), {row_count} applicants and {features.shape[1]} input features.\n\n"
        f"Target values: 1 = good credit risk and 2 = bad credit risk. Missing values: {profile['total_missing']}; "
        f"duplicate complete rows: {profile['duplicate_rows']}. See `../outputs/reports/dataset_profile.json` and `../outputs/reports/feature_dictionary.csv` for details.\n\n"
        "## Baseline model\n\n"
        "The data is split into 80% training and 20% test rows, stratified by target, with random seed 42. "
        "Numeric columns are median-imputed and standardized. Categories are most-frequent-imputed and one-hot encoded. "
        "The classifier is Logistic Regression with `max_iter=1000`. Good credit risk is the positive class.\n\n"
        "Financial error cost = (false approvals x 5) + (false rejections x 1).\n\n"
        f"| Model | Accuracy | Balanced accuracy | ROC-AUC | False approvals | False rejections | Cost |\n"
        f"|---|---:|---:|---:|---:|---:|---:|\n"
        f"| Logistic Regression | {logistic['accuracy']:.3f} | {logistic['balanced_accuracy']:.3f} | {logistic['roc_auc']:.3f} | {logistic['false_approvals']} | {logistic['false_rejections']} | {logistic['error_cost']} |\n"
        f"| All-good dummy | {dummy['accuracy']:.3f} | {dummy['balanced_accuracy']:.3f} | - | {dummy['false_approvals']} | {dummy['false_rejections']} | {dummy['error_cost']} |\n\n"
        "## Plots\n\n"
        "The target and selected feature charts are in `../data/eda/`. Age-group plots and feature comparisons are in `../outputs/plots/`. "
        "These plots show associations and distributions; they do not by themselves establish bias or unfairness.\n\n"
        "Full baseline metrics are in `../models/stage1_metrics.json`. The separate age-group, proxy, and data-quality analysis is in `../outputs/reports/data_audit_report.md`.\n",
        encoding="utf-8",
    )
    return {
        "profile": profile,
        "age_outcomes": age_outcomes,
        "proxy": proxy,
        "fairness": age_group_rates,
        "fairness_gaps": fairness_gaps,
        "fairness_table": fairness_table,
    }
