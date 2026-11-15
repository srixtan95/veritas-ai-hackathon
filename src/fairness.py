"""Calculate age-group prediction rates and compare their gaps."""

import pandas as pd


def age_groups(ages):
    """Split known ages at 25, leaving missing ages without a group."""
    age_values = pd.Series(ages)
    return pd.Series(
        [
            pd.NA if pd.isna(age) else "Under 25" if age < 25 else "25 or older"
            for age in age_values
        ],
        index=age_values.index,
        name="age_group",
    )


def group_metrics(actual, predicted, groups, good_label=1, bad_label=2):
    """Calculate approval and error rates with their numerators and denominators."""
    rows = []
    data = pd.DataFrame({
        "actual": pd.Series(actual).reset_index(drop=True),
        "predicted": pd.Series(predicted).reset_index(drop=True),
        "group": pd.Series(groups).reset_index(drop=True),
    }).dropna()

    for group, applicants in data.groupby("group", sort=False):
        actual_good = applicants["actual"] == good_label
        actual_bad = applicants["actual"] == bad_label
        predicted_good = applicants["predicted"] == good_label

        group_size = len(applicants)
        good_count = int(actual_good.sum())
        bad_count = int(actual_bad.sum())
        approval_count = int(predicted_good.sum())
        false_approval_count = int((actual_bad & predicted_good).sum())
        false_rejection_count = int((actual_good & ~predicted_good).sum())
        true_positive_count = int((actual_good & predicted_good).sum())

        rows.append({
            "group": group,
            "group_size": group_size,
            "approval_count": approval_count,
            "approval_denominator": group_size,
            "approval_rate": approval_count / group_size if group_size else None,
            "false_approval_count": false_approval_count,
            "false_approval_denominator": bad_count,
            "false_approval_rate": false_approval_count / bad_count if bad_count else None,
            "false_rejection_count": false_rejection_count,
            "false_rejection_denominator": good_count,
            "false_rejection_rate": false_rejection_count / good_count if good_count else None,
            "true_positive_count": true_positive_count,
            "true_positive_denominator": good_count,
            "true_positive_rate": true_positive_count / good_count if good_count else None,
            "good_count": good_count,
            "bad_count": bad_count,
        })
    return pd.DataFrame(rows)


def rate_gaps(metrics):
    """Return absolute differences between groups in percentage points."""
    gap_rows = []
    for column, label in [
        ("approval_rate", "Approval rate"),
        ("false_approval_rate", "False approval rate"),
        ("false_rejection_rate", "False rejection rate"),
    ]:
        values = metrics[column].tolist()
        if len(values) < 2 or any(value is None or pd.isna(value) for value in values):
            gap = None
        else:
            gap = abs(values[0] - values[1]) * 100
        gap_rows.append({"metric": label, "absolute_gap_percentage_points": gap})
    return pd.DataFrame(gap_rows)


def format_fairness_table(metrics):
    """Format each rate as a percentage with its numerator and denominator."""
    columns = [
        ("approval_rate", "approval_count", "approval_denominator", "Approval rate"),
        ("false_approval_rate", "false_approval_count", "false_approval_denominator", "False approval rate"),
        ("false_rejection_rate", "false_rejection_count", "false_rejection_denominator", "False rejection rate"),
        ("true_positive_rate", "true_positive_count", "true_positive_denominator", "True positive rate"),
    ]
    rows = []
    for _, row in metrics.iterrows():
        formatted = {"Age group": row["group"], "Group size": int(row["group_size"])}
        for rate, numerator, denominator, title in columns:
            value = row[rate]
            count = int(row[numerator])
            total = int(row[denominator])
            if value is None or pd.isna(value):
                formatted[title] = f"N/A ({count}/{total})"
            else:
                formatted[title] = f"{value * 100:.1f}% ({count}/{total})"
        rows.append(formatted)
    return pd.DataFrame(rows)
