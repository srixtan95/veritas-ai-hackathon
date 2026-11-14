"""Run the loan-risk baseline and the age fairness research."""

import sys

from src.data_analysis import run_data_analysis, write_age_fairness_report
from src.data_loader import load_credit_dataset
from src.model import predict_saved_test_set, train_baseline


def main():
    """Run the full project or audit the saved model without retraining."""
    dataset = load_credit_dataset()

    if "--fairness-only" in sys.argv:
        test_features, actual, predicted = predict_saved_test_set(dataset)
        metrics, gaps, table = write_age_fairness_report(
            dataset, test_features, actual, predicted
        )
        print("Age-based fairness metrics (rate includes numerator / denominator):")
        print(table.to_string(index=False))
        print("\nAbsolute rate gaps:")
        for _, row in gaps.iterrows():
            value = row["absolute_gap_percentage_points"]
            gap_text = "N/A" if value is None or value != value else f"{value:.2f} percentage points"
            print(f"{row['metric']}: {gap_text}")
        print("\nFull report: outputs/reports/age_fairness_audit.md")
        return

    baseline = train_baseline(dataset)
    results = run_data_analysis(dataset, baseline)

    print("Baseline model results:")
    for model_name, scores in baseline["scores"].items():
        print(
            f"{model_name}: accuracy={scores['accuracy']:.3f}, "
            f"false approvals={scores['false_approvals']}, "
            f"false rejections={scores['false_rejections']}, "
            f"error cost={scores['error_cost']}"
        )
    print("\nAge-group counts:")
    for row in results["age_outcomes"]:
        print(f"{row['age_group']}: {row['applicants']}")
    print("\nReports: outputs/reports/")
    print("Charts: outputs/plots/ and data/eda/")


if __name__ == "__main__":
    main()
