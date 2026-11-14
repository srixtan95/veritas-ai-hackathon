"""Train and evaluate the Logistic Regression credit-risk baseline."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"


def make_preprocessor(features):
    """Impute and scale numbers, and impute and encode categories."""
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = [column for column in features if column not in numeric_columns]

    numeric_steps = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_steps = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numbers", numeric_steps, numeric_columns),
        ("categories", categorical_steps, categorical_columns),
    ])


def evaluate_predictions(actual, predicted, good_label, bad_label, good_probability=None):
    """Calculate standard scores, error counts, ROC-AUC, and financial cost."""
    matrix = confusion_matrix(actual, predicted, labels=[good_label, bad_label])
    false_rejections = int(matrix[0, 1])
    false_approvals = int(matrix[1, 0])
    scores = {
        "accuracy": float(accuracy_score(actual, predicted)),
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": {
            "rows": ["actual_good", "actual_bad"],
            "columns": ["predicted_good", "predicted_bad"],
        },
        "precision_good": float(precision_score(actual, predicted, pos_label=good_label, zero_division=0)),
        "recall_good": float(recall_score(actual, predicted, pos_label=good_label, zero_division=0)),
        "f1_good": float(f1_score(actual, predicted, pos_label=good_label, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "false_approvals": false_approvals,
        "false_rejections": false_rejections,
        "error_cost": 5 * false_approvals + false_rejections,
    }
    if good_probability is not None:
        actual_good = (np.asarray(actual) == good_label).astype(int)
        scores["roc_auc"] = float(roc_auc_score(actual_good, good_probability))
    return scores


def predict_applicants(features):
    """Use the saved model to predict risk labels and good-risk probabilities."""
    model = load(MODEL_DIR / "logistic_regression_pipeline.joblib")
    good_index = list(model.named_steps["classifier"].classes_).index(1)
    return pd.DataFrame(
        {
            "predicted_risk": model.predict(features),
            "good_risk_probability": model.predict_proba(features)[:, good_index],
        },
        index=features.index,
    )


def predict_saved_test_set(dataset):
    """Recreate the original test split and score it with the saved model.

    This function does not fit a model. The saved baseline expects all 20
    original features, including age; the target stays in a separate Series.
    """
    _, X_test, _, y_test = train_test_split(
        dataset["features"],
        dataset["target"],
        test_size=0.20,
        random_state=42,
        stratify=dataset["target"],
    )
    model = load(MODEL_DIR / "logistic_regression_pipeline.joblib")
    expected_columns = list(model.feature_names_in_)
    if list(X_test.columns) != expected_columns:
        raise ValueError("The loaded dataset columns do not match the saved model inputs.")
    predictions = model.predict(X_test)
    return X_test, y_test, predictions


def train_baseline(dataset):
    """Fit the original stratified baseline and save its artifacts and scores."""
    X_train, X_test, y_train, y_test = train_test_split(
        dataset["features"],
        dataset["target"],
        test_size=0.20,
        random_state=42,
        stratify=dataset["target"],
    )
    model = Pipeline([
        ("preprocessor", make_preprocessor(X_train)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    good_index = list(model.named_steps["classifier"].classes_).index(dataset["good_label"])
    good_probability = model.predict_proba(X_test)[:, good_index]
    dummy_predictions = pd.Series(dataset["good_label"], index=y_test.index)

    scores = {
        "dummy_all_good": evaluate_predictions(
            y_test, dummy_predictions, dataset["good_label"], dataset["bad_label"]
        ),
        "logistic_regression": evaluate_predictions(
            y_test, predictions, dataset["good_label"], dataset["bad_label"], good_probability
        ),
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dump(model, MODEL_DIR / "logistic_regression_pipeline.joblib")
    dump(model.named_steps["preprocessor"], MODEL_DIR / "preprocessor.joblib")
    payload = {
        "dataset": dataset["metadata"],
        "target": {
            "name": dataset["target_name"],
            "good_label": int(dataset["good_label"]),
            "bad_label": int(dataset["bad_label"]),
        },
        "split": {
            "test_size": 0.2,
            "random_state": 42,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
        },
        "metrics": scores,
    }
    (MODEL_DIR / "stage1_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "predictions": predictions,
        "good_probability": good_probability,
        "scores": scores,
        "train_counts": y_train.value_counts().to_dict(),
        "test_counts": y_test.value_counts().to_dict(),
    }
