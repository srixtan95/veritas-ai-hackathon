"""Global and applicant-level explanations for the saved age-excluded model."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load
from scipy.special import expit

from .data_analysis import find_age_column


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_B_PATH = PROJECT_ROOT / "outputs" / "age_ablation" / "model_b_without_age.joblib"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "explainability"
DEFAULT_AGE_COLUMN = "Attribute13"
NEAR_THRESHOLD_BAND = 0.10


def _model_parts(model, good_label=1, age_column=DEFAULT_AGE_COLUMN):
    """Return fitted preprocessing/classifier pieces and Good-risk logit params."""
    if not hasattr(model, "named_steps") or not {"preprocessor", "classifier"} <= set(model.named_steps):
        raise ValueError("Expected the fitted Logistic Regression pipeline with preprocessor and classifier steps.")
    expected = list(model.feature_names_in_)
    if age_column in expected:
        raise ValueError(f"The explainability model must exclude age feature {age_column!r}.")
    classifier = model.named_steps["classifier"]
    classes = list(classifier.classes_)
    if good_label not in classes or len(classes) != 2:
        raise ValueError("Expected a fitted binary Logistic Regression model containing the Good-risk label.")
    good_position = classes.index(good_label)
    # sklearn's binary coef_ is the log odds of classes_[1] over classes_[0].
    orientation = 1.0 if good_position == 1 else -1.0
    coefficients = np.asarray(classifier.coef_[0], dtype=float) * orientation
    intercept = float(classifier.intercept_[0]) * orientation
    return model.named_steps["preprocessor"], classifier, expected, coefficients, intercept


def _validate_applicants(applicants, expected, age_column):
    if isinstance(applicants, pd.Series):
        applicants = applicants.to_frame().T
    elif isinstance(applicants, dict):
        applicants = pd.DataFrame([applicants])
    elif not isinstance(applicants, pd.DataFrame):
        raise TypeError("Applicant inputs must be a dict, pandas Series, or DataFrame.")
    if age_column in applicants.columns:
        raise ValueError(f"Age feature {age_column!r} cannot be supplied to the age-excluded model.")
    missing = [column for column in expected if column not in applicants.columns]
    extra = [column for column in applicants.columns if column not in expected]
    if missing or extra:
        raise ValueError(f"Applicant columns do not match Model B; missing={missing}, extra={extra}.")
    return applicants.loc[:, expected]


def _transformed_labels(preprocessor, expected):
    names = list(preprocessor.get_feature_names_out())
    numeric_columns = list(preprocessor.transformers_[0][2])
    categorical_columns = list(preprocessor.transformers_[1][2])
    labels = []
    for name in names:
        bare_name = name.split("__", 1)[-1]
        if name.startswith("numbers__"):
            labels.append((bare_name, bare_name))
            continue
        source = next((column for column in sorted(categorical_columns, key=len, reverse=True)
                       if bare_name.startswith(f"{column}_")), None)
        if source is None:
            raise ValueError(f"Could not map encoded feature {name!r} to an input feature.")
        category = bare_name[len(source) + 1:]
        labels.append((source, f"{source} = {category}"))
    if not labels:
        raise ValueError("The fitted preprocessor did not expose transformed feature names.")
    return labels


def global_feature_importance(model, features, good_label=1, age_column=DEFAULT_AGE_COLUMN):
    """Rank source features by mean absolute per-row contribution to Good-risk log odds.

    Numeric contributions use the fitted scaler output. Each one-hot category's
    contribution is evaluated on actual rows, then encoded columns are grouped
    under their original categorical feature. This avoids comparing raw LR
    coefficient magnitudes across different feature scales.
    """
    preprocessor, _, expected, coefficients, _ = _model_parts(model, good_label, age_column)
    if age_column in features.columns:
        raise ValueError(f"Age feature {age_column!r} cannot be used to explain Model B.")
    if list(features.columns) != expected:
        raise ValueError("Dataset columns must match the Model B training feature order.")
    transformed = np.asarray(preprocessor.transform(features), dtype=float)
    contribution = transformed * coefficients
    labels = _transformed_labels(preprocessor, expected)
    rows = []
    for source in expected:
        indices = [i for i, (feature, _) in enumerate(labels) if feature == source]
        per_row = contribution[:, indices].sum(axis=1)
        rows.append({
            "feature": source,
            "mean_absolute_log_odds_contribution": float(np.mean(np.abs(per_row))),
            "mean_log_odds_contribution": float(np.mean(per_row)),
            "direction": "towards Good risk" if np.mean(per_row) > 0 else (
                "towards Bad risk" if np.mean(per_row) < 0 else "neutral on average"
            ),
        })
    return pd.DataFrame(rows).sort_values(
        "mean_absolute_log_odds_contribution", ascending=False
    ).reset_index(drop=True)


def logistic_coefficients(model, good_label=1, age_column=DEFAULT_AGE_COLUMN):
    """Return the fitted coefficient for every transformed feature as Good log odds."""
    preprocessor, _, expected, coefficients, _ = _model_parts(model, good_label, age_column)
    labels = _transformed_labels(preprocessor, expected)
    return pd.DataFrame([
        {
            "source_feature": source,
            "transformed_feature": transformed,
            "coefficient_good_log_odds": float(coefficient),
            "direction": "towards Good risk" if coefficient > 0 else (
                "towards Bad risk" if coefficient < 0 else "no linear effect"
            ),
        }
        for (source, transformed), coefficient in zip(labels, coefficients)
    ])


def explain_applicant(model, applicant, good_label=1, bad_label=2,
                      age_column=DEFAULT_AGE_COLUMN, top_n=3,
                      near_threshold_band=NEAR_THRESHOLD_BAND):
    """Explain one prediction with additive transformed-feature log-odds terms."""
    preprocessor, classifier, expected, coefficients, intercept = _model_parts(
        model, good_label, age_column
    )
    applicant_frame = _validate_applicants(applicant, expected, age_column)
    transformed = np.asarray(preprocessor.transform(applicant_frame), dtype=float)
    terms = (transformed * coefficients)[0]
    good_log_odds = intercept + float(terms.sum())
    calculated_good_probability = float(expit(good_log_odds))
    model_probabilities = model.predict_proba(applicant_frame)[0]
    classes = list(classifier.classes_)
    model_good_probability = float(model_probabilities[classes.index(good_label)])
    if not np.isclose(calculated_good_probability, model_good_probability, rtol=1e-9, atol=1e-10):
        raise ValueError("Contribution calculation does not reproduce the pipeline probability.")

    labels = _transformed_labels(preprocessor, expected)
    contributions = []
    for (source, label), value in zip(labels, terms):
        if not np.isclose(value, 0.0):
            contributions.append({"feature": source, "transformed_feature": label,
                                  "good_log_odds_contribution": float(value)})
    towards_good = sorted((item for item in contributions if item["good_log_odds_contribution"] > 0),
                          key=lambda item: item["good_log_odds_contribution"], reverse=True)[:top_n]
    towards_bad = sorted((item for item in contributions if item["good_log_odds_contribution"] < 0),
                         key=lambda item: item["good_log_odds_contribution"])[:top_n]
    prediction = model.predict(applicant_frame)[0]
    predicted_name = "Good risk" if prediction == good_label else "Bad risk"
    near_threshold = abs(model_good_probability - 0.5) <= near_threshold_band
    summary = (
        f"The model predicts {predicted_name} (Good-risk probability {model_good_probability:.1%}). "
        f"The listed factors add to a Good-versus-Bad log-odds score of {good_log_odds:.3f}."
    )
    if near_threshold:
        summary += " The probabilities are close to 50%, so this prediction is relatively uncertain; a probability is not a guarantee."
    else:
        summary += " The probability is a model estimate, not a guarantee."
    return {
        "predicted_class": int(prediction) if isinstance(prediction, (int, np.integer)) else prediction,
        "predicted_class_name": predicted_name,
        "probabilities": {("Good risk" if label == good_label else "Bad risk"): float(probability)
                          for label, probability in zip(classes, model_probabilities)},
        "good_log_odds": good_log_odds,
        "intercept_good_log_odds": intercept,
        "contributions": contributions,
        "strongest_towards_good": towards_good,
        "strongest_towards_bad": towards_bad,
        "near_threshold": bool(near_threshold),
        "explanation": summary,
        "applicant": applicant_frame.iloc[0].to_dict(),
    }


def _write_chart(importance, output_dir, top_n=15):
    shown = importance.head(top_n).sort_values("mean_absolute_log_odds_contribution")
    colors = ["#26734d" if value >= 0 else "#b54747"
              for value in shown["mean_log_odds_contribution"]]
    fig, axis = plt.subplots(figsize=(10, max(5, len(shown) * 0.36)))
    axis.barh(shown["feature"], shown["mean_absolute_log_odds_contribution"], color=colors)
    axis.set_xlabel("Mean absolute contribution to Good-risk log odds")
    axis.set_title("Model B global feature influence")
    fig.tight_layout()
    fig.savefig(output_dir / "global_feature_importance.png", dpi=160)
    plt.close(fig)


def _markdown_case(case):
    def lines(items):
        if not items:
            return "- No feature contributions in this direction."
        return "\n".join(
            f"- {item['transformed_feature']}: {item['good_log_odds_contribution']:+.3f} log odds"
            for item in items
        )
    probabilities = case["probabilities"]
    return "\n".join([
        f"### {case['predicted_class_name']} case",
        "",
        f"- Predicted class: {case['predicted_class_name']} ({case['predicted_class']})",
        f"- Probabilities: Good {probabilities['Good risk']:.1%}; Bad {probabilities['Bad risk']:.1%}",
        f"- Near-threshold warning: {'yes' if case['near_threshold'] else 'no'}",
        f"- Explanation: {case['explanation']}",
        "",
        "Strongest factors towards Good risk:", lines(case["strongest_towards_good"]),
        "",
        "Strongest factors towards Bad risk:", lines(case["strongest_towards_bad"]),
    ])


def _markdown_table(frame):
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame.iterrows():
        values = []
        for value in row:
            values.append(f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def run_explainability(dataset, model_path=MODEL_B_PATH, output_dir=OUTPUT_DIR):
    """Reuse saved Model B and write an isolated report, chart, and demos."""
    model = load(model_path)
    features = dataset["features"]
    age_column = find_age_column(features, dataset.get("feature_descriptions", {}))
    expected = list(model.feature_names_in_)
    if age_column in expected:
        raise ValueError("The primary explainability model must exclude age.")
    if list(features.drop(columns=[age_column]).columns) != expected:
        raise ValueError("Dataset features do not match the saved age-excluded Model B inputs.")
    model_features = features.loc[:, expected]
    importance = global_feature_importance(model, model_features, dataset["good_label"], age_column)
    coefficients = logistic_coefficients(model, dataset["good_label"], age_column)
    predictions = model.predict(model_features)
    good_position = list(model.named_steps["classifier"].classes_).index(dataset["good_label"])
    probabilities = model.predict_proba(model_features)[:, good_position]
    candidates = {
        "Good risk": np.flatnonzero(predictions == dataset["good_label"]),
        "Bad risk": np.flatnonzero(predictions == dataset["bad_label"]),
    }
    if not len(candidates["Good risk"]) or not len(candidates["Bad risk"]):
        raise ValueError("Dataset did not produce both Good-risk and Bad-risk demonstration predictions.")
    good_i = candidates["Good risk"][np.argmax(probabilities[candidates["Good risk"]])]
    bad_i = candidates["Bad risk"][np.argmin(probabilities[candidates["Bad risk"]])]
    uncertain_i = int(np.argmin(np.abs(probabilities - 0.5)))
    selected = []
    for label, position in (("Good risk", int(good_i)), ("Bad risk", int(bad_i)), ("Relatively uncertain", uncertain_i)):
        case = explain_applicant(model, model_features.iloc[position], dataset["good_label"],
                                 dataset["bad_label"], age_column)
        case["demonstration_type"] = label
        selected.append(case)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    importance.to_csv(output_dir / "global_feature_importance.csv", index=False)
    coefficients.to_csv(output_dir / "logistic_regression_coefficients.csv", index=False)
    _write_chart(importance, output_dir)
    (output_dir / "demonstration_cases.json").write_text(
        json.dumps(selected, indent=2, default=lambda value: value.item() if hasattr(value, "item") else str(value)),
        encoding="utf-8",
    )
    top_features = importance.head(10)
    report = [
        "# Immortals AI explainability", "",
        "Primary model: saved age-excluded Model B (`outputs/age_ablation/model_b_without_age.joblib`). It is reused as-is; no prediction model is rebuilt. Age is excluded from model inputs and remains available separately to the fairness audit.",
        "",
        "## Global influence", "",
        "Importance is the dataset average of the absolute contribution to the Good-versus-Bad log odds. Each contribution is the fitted transformed value multiplied by its Logistic Regression coefficient; one-hot levels are combined under their original categorical feature for the global report. Numeric values therefore use the fitted scaler, and categorical values use their actual encoded activations. The signed mean indicates average direction. This is model influence, not causal effect.",
        "",
        _markdown_table(top_features), "",
        "Green bars have an average direction towards Good risk; red bars have an average direction towards Bad risk. The chart length ranks mean absolute log-odds contribution.", "",
        "The complete transformed Logistic Regression coefficient table is saved as `logistic_regression_coefficients.csv`. Positive coefficients increase Good-versus-Bad log odds; negative coefficients decrease them. For categorical inputs, each row is a one-hot encoded level.", "",
        "## Individual demonstration cases", "",
        "These are synthetic demonstrations selected from valid rows in the dataset, not real customer applications. For each applicant, transformed values are multiplied by the saved coefficients. Contributions sum with the intercept to the Good-risk log odds; applying the logistic function reproduces Model B's Good-risk probability.", "",
        *sum((["### " + case["demonstration_type"], "", _markdown_case(case).split("\n", 2)[2], ""] for case in selected), []),
        "## Limits", "",
        "Predicted probabilities are estimates, not guarantees. A case is flagged relatively uncertain when the Good-risk probability is within 10 percentage points of 50%. Explanations describe how this fitted model produced a score; they do not prove a prediction is fair, correct, or causally justified.", "",
    ]
    (output_dir / "explainability_report.md").write_text("\n".join(report), encoding="utf-8")
    return {"model": model, "importance": importance, "demonstrations": selected,
            "output_dir": output_dir}


def main():
    from .data_loader import load_credit_dataset

    results = run_explainability(load_credit_dataset())
    print("Global feature importance:")
    print(results["importance"].head(10).to_string(index=False))
    print(f"\nExplainability outputs: {results['output_dir']}")


if __name__ == "__main__":
    main()
