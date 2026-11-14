"""Download UCI German Credit data and identify the good and bad labels."""

import re

import pandas as pd
from ucimlrepo import fetch_ucirepo


def load_credit_dataset():
    """Return the dataset, target labels, and UCI feature descriptions."""
    repository = fetch_ucirepo(id=144)
    features = repository.data.features.copy()
    target_table = repository.data.targets.copy()

    if target_table.shape[1] != 1:
        raise ValueError("Expected one target column in UCI dataset 144.")

    target_name = str(target_table.columns[0])
    target = target_table.iloc[:, 0].copy()
    variables = repository.variables.copy()
    target_rows = variables[variables["name"].astype(str) == target_name]
    target_description = " ".join(
        target_rows.get("description", pd.Series(dtype=str)).fillna("").astype(str)
    )

    meanings = {}
    for label in pd.unique(target.dropna()):
        text = str(label).lower()
        if "good" in text:
            meanings[label] = "good"
        elif "bad" in text:
            meanings[label] = "bad"
        else:
            match = re.search(
                rf"(?<!\w){re.escape(str(label))}\s*(?:=|:|-)\s*(good|bad)\b",
                target_description,
                flags=re.IGNORECASE,
            )
            if match:
                meanings[label] = match.group(1).lower()

    good_labels = [label for label, meaning in meanings.items() if meaning == "good"]
    bad_labels = [label for label, meaning in meanings.items() if meaning == "bad"]
    if len(good_labels) != 1 or len(bad_labels) != 1:
        raise ValueError(
            "Could not map the target labels to good and bad credit risk. "
            f"Labels were {list(pd.unique(target.dropna()))!r}; "
            f"UCI description was {target_description!r}."
        )

    descriptions = {}
    if "description" in variables.columns:
        descriptions = variables.set_index("name")["description"].fillna("").astype(str).to_dict()

    return {
        "features": features,
        "target": target,
        "target_name": target_name,
        "good_label": good_labels[0],
        "bad_label": bad_labels[0],
        "metadata": {
            "uci_id": repository.metadata.uci_id,
            "name": repository.metadata.name,
            "target_description": target_description,
            "target_labels": [str(label) for label in pd.unique(target.dropna())],
        },
        "feature_descriptions": descriptions,
    }
