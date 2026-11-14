"""Reusable application services for the Immortals AI Streamlit prototype."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

from .explainability import DEFAULT_AGE_COLUMN, NEAR_THRESHOLD_BAND, explain_applicant


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_B_PATH = PROJECT_ROOT / "outputs" / "age_ablation" / "model_b_without_age.joblib"
AGE_FAIRNESS_PATH = PROJECT_ROOT / "outputs" / "age_ablation" / "fairness_metrics.csv"
PROXY_FAIRNESS_PATH = PROJECT_ROOT / "outputs" / "proxy_ablation" / "fairness_metrics.csv"
PROXY_PERFORMANCE_PATH = PROJECT_ROOT / "outputs" / "proxy_ablation" / "model_comparison.csv"
PROXY_CHANGES_PATH = PROJECT_ROOT / "outputs" / "proxy_ablation" / "performance_changes.csv"
REPEAT_FAIRNESS_PATH = PROJECT_ROOT / "outputs" / "repeated_validation" / "fairness_summary.csv"
REPEAT_GAPS_PATH = PROJECT_ROOT / "outputs" / "repeated_validation" / "fairness_gap_summary.csv"
REPEAT_PERFORMANCE_PATH = PROJECT_ROOT / "outputs" / "repeated_validation" / "performance_summary.csv"

FEATURE_LABELS = {
    "Attribute1": "Existing checking account status",
    "Attribute2": "Credit duration (months)",
    "Attribute3": "Credit history",
    "Attribute4": "Loan purpose",
    "Attribute5": "Credit amount",
    "Attribute6": "Savings account or bonds",
    "Attribute7": "Present employment duration",
    "Attribute8": "Installment rate (% of disposable income)",
    "Attribute9": "Personal status and sex",
    "Attribute10": "Other debtors or guarantors",
    "Attribute11": "Present residence duration (years)",
    "Attribute12": "Property",
    "Attribute14": "Other installment plans",
    "Attribute15": "Housing",
    "Attribute16": "Existing credits at this bank",
    "Attribute17": "Job category",
    "Attribute18": "People supported for maintenance",
    "Attribute19": "Telephone status",
    "Attribute20": "Foreign worker status",
}

FEATURE_HELP = {
    "Attribute1": {"A11": "Below 0 DM", "A12": "0 to below 200 DM", "A13": "200 DM or more / salary assignments", "A14": "No checking account"},
    "Attribute3": {"A30": "No credits or all paid", "A31": "All paid at this bank", "A32": "Existing credits paid duly", "A33": "Delayed payments", "A34": "Critical or other loans"},
    "Attribute4": {"A40": "Car (new)", "A41": "Car (used)", "A42": "Furniture/equipment", "A43": "Radio/TV", "A44": "Domestic appliances", "A45": "Repairs", "A46": "Education", "A48": "Retraining", "A49": "Business", "A410": "Others"},
    "Attribute6": {"A61": "Below 100 DM", "A62": "100 to below 500 DM", "A63": "500 to below 1,000 DM", "A64": "1,000 DM or more", "A65": "Unknown / no savings account"},
    "Attribute7": {"A71": "Unemployed", "A72": "Below 1 year", "A73": "1 to below 4 years", "A74": "4 to below 7 years", "A75": "7 years or more"},
    "Attribute9": {"A91": "Male: divorced/separated", "A92": "Female: divorced/separated/married", "A93": "Male: single", "A94": "Male: married/widowed"},
    "Attribute10": {"A101": "None", "A102": "Co-applicant", "A103": "Guarantor"},
    "Attribute12": {"A121": "Real estate", "A122": "Building society savings / life insurance", "A123": "Car or other", "A124": "Unknown / no property"},
    "Attribute14": {"A141": "Bank", "A142": "Stores", "A143": "None"},
    "Attribute15": {"A151": "Rent", "A152": "Own", "A153": "For free"},
    "Attribute17": {"A171": "Unemployed / unskilled non-resident", "A172": "Unskilled resident", "A173": "Skilled employee / official", "A174": "Management / self-employed / highly qualified"},
    "Attribute19": {"A191": "None", "A192": "Yes, registered under applicant's name"},
    "Attribute20": {"A201": "Yes", "A202": "No"},
}

NUMERIC_DEFAULTS = {
    "Attribute2": 24, "Attribute5": 5000, "Attribute8": 2,
    "Attribute11": 2, "Attribute16": 1, "Attribute18": 1,
}

FAIRNESS_FILES = {
    "age_fairness": AGE_FAIRNESS_PATH,
    "proxy_fairness": PROXY_FAIRNESS_PATH,
    "proxy_performance": PROXY_PERFORMANCE_PATH,
    "proxy_performance_changes": PROXY_CHANGES_PATH,
    "repeated_fairness": REPEAT_FAIRNESS_PATH,
    "repeated_fairness_gaps": REPEAT_GAPS_PATH,
    "repeated_performance": REPEAT_PERFORMANCE_PATH,
}


def load_credit_model(model_path=MODEL_B_PATH):
    """Load the already-fitted age-excluded Model B; never fit or retrain it."""
    model = load(model_path)
    if not hasattr(model, "feature_names_in_"):
        raise ValueError("Saved credit-risk model does not expose its expected input schema.")
    expected = list(model.feature_names_in_)
    if DEFAULT_AGE_COLUMN in expected:
        raise ValueError("Saved Model B must exclude age (Attribute13).")
    if len(expected) != 19:
        raise ValueError(f"Expected 19 Model B features, found {len(expected)}.")
    classifier = model.named_steps.get("classifier")
    if classifier is None or list(classifier.classes_) != [1, 2]:
        raise ValueError("Saved Model B must predict class 1 = Good risk and class 2 = Bad risk.")
    return model


def model_input_schema(model):
    """Describe Model B's numeric and categorical fields from fitted transformers."""
    expected = list(model.feature_names_in_)
    if DEFAULT_AGE_COLUMN in expected:
        raise ValueError("Age cannot be an input to Model B.")
    preprocessor = model.named_steps["preprocessor"]
    numeric_fields = list(preprocessor.transformers_[0][2])
    categorical_fields = list(preprocessor.transformers_[1][2])
    encoder = preprocessor.named_transformers_["categories"].named_steps["onehot"]
    category_map = dict(zip(categorical_fields, encoder.categories_))
    schema = []
    for field in expected:
        if field in numeric_fields:
            schema.append({"name": field, "label": FEATURE_LABELS.get(field, field),
                           "kind": "numeric", "default": NUMERIC_DEFAULTS.get(field, 0)})
        elif field in categorical_fields:
            schema.append({"name": field, "label": FEATURE_LABELS.get(field, field),
                           "kind": "categorical", "choices": [str(value) for value in category_map[field]],
                           "help": FEATURE_HELP.get(field, {})})
        else:
            raise ValueError(f"Model input {field!r} is not assigned to a fitted transformer.")
    return schema


def validate_applicant_input(model, values):
    """Validate a single applicant against Model B's exact 19-feature schema."""
    if not isinstance(values, dict):
        raise TypeError("Applicant values must be supplied as a feature-name mapping.")
    expected = list(model.feature_names_in_)
    if DEFAULT_AGE_COLUMN in values:
        raise ValueError("Age (Attribute13) must not be supplied to age-excluded Model B.")
    missing = [name for name in expected if name not in values]
    extra = [name for name in values if name not in expected]
    if missing or extra:
        raise ValueError(f"Applicant feature schema mismatch; missing={missing}; unexpected={extra}.")
    validated = {}
    for item in model_input_schema(model):
        name = item["name"]
        value = values[name]
        if item["kind"] == "numeric":
            if isinstance(value, bool):
                raise ValueError(f"{item['label']} must be a nonnegative whole number.")
            try:
                number = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{item['label']} must be a nonnegative whole number.") from error
            if not math.isfinite(number) or number < 0 or not number.is_integer():
                raise ValueError(f"{item['label']} must be a nonnegative whole number.")
            validated[name] = int(number)
        else:
            if value is None or value not in item["choices"]:
                raise ValueError(f"Choose a valid dataset category for {item['label']}.")
            validated[name] = value
    return validated


def assess_credit_risk(model, values, near_threshold_band=NEAR_THRESHOLD_BAND):
    """Predict and explain one applicant using the existing fitted Model B."""
    validated = validate_applicant_input(model, values)
    applicant = pd.DataFrame([validated], columns=list(model.feature_names_in_))
    prediction = int(model.predict(applicant)[0])
    probabilities_array = model.predict_proba(applicant)[0]
    classes = list(model.named_steps["classifier"].classes_)
    probabilities = {"Good risk": float(probabilities_array[classes.index(1)]),
                     "Bad risk": float(probabilities_array[classes.index(2)])}
    explanation = explain_applicant(model, applicant.iloc[0], good_label=1, bad_label=2,
                                    near_threshold_band=near_threshold_band)
    if prediction != explanation["predicted_class"]:
        raise ValueError("Prediction and individual explanation do not agree.")
    if not np.isclose(probabilities["Good risk"], explanation["probabilities"]["Good risk"]):
        raise ValueError("Prediction and individual explanation probabilities do not agree.")
    return {
        "predicted_class": prediction,
        "predicted_class_name": "Good risk" if prediction == 1 else "Bad risk",
        "probabilities": probabilities,
        "near_threshold": explanation["near_threshold"],
        "explanation": explanation,
        "features": validated,
        "disclaimer": "Credit-risk predictions are estimates, not guarantees, and are not loan approval decisions.",
    }


def load_research_results(paths=None):
    """Load saved fairness/validation outputs without running any experiments."""
    paths = FAIRNESS_FILES if paths is None else paths
    frames = {}
    missing = [f"{name}: {path}" for name, path in paths.items() if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("Required saved research output(s) are missing: " + "; ".join(missing))
    for name, path in paths.items():
        frames[name] = pd.read_csv(path)
        if frames[name].empty:
            raise ValueError(f"Saved research output {path} is empty.")
    required = {
        "age_fairness": {"model", "group", "approval_rate", "false_approval_rate", "false_rejection_rate"},
        "proxy_fairness": {"experiment", "group", "approval_rate", "false_approval_rate", "false_rejection_rate"},
        "proxy_performance": {"experiment", "accuracy", "balanced_accuracy", "roc_auc", "error_cost"},
        "repeated_fairness": {"model", "group", "metric", "mean", "empirical_2_5_percentile", "empirical_97_5_percentile"},
        "repeated_fairness_gaps": {"model", "metric", "mean", "empirical_2_5_percentile", "empirical_97_5_percentile"},
        "repeated_performance": {"model", "metric", "mean", "empirical_2_5_percentile", "empirical_97_5_percentile"},
    }
    for name, columns in required.items():
        absent = columns - set(frames[name].columns)
        if absent:
            raise ValueError(f"Saved research output {paths[name]} is missing columns: {sorted(absent)}")
    return frames
