# Repeated stratified validation

We evaluated eight predeclared Logistic Regression feature configurations using 10 repeats of 5-fold stratified cross-validation (random_state=42). All configurations use the same folds. Every repeat combines its 5 held-out folds into one complete OOF prediction per dataset row. Preprocessing is fitted separately on each training fold only. The existing preprocessor, Logistic Regression (`max_iter=1000`, `random_state=42`), target mapping (1 = Good, 2 = Bad), and `predict` threshold were preserved.

All held-out predictions, actual labels and ages are retained in `oof_predictions.csv` for audit; the fold membership is in `fold_assignments.csv`.

The report gives the mean, sample standard deviation and empirical 2.5th–97.5th percentile interval across 10 repeat-level OOF evaluations. These percentiles describe variation under repeated partitions of this dataset; they are not population confidence intervals. The repeated evaluations are dependent because they reuse rows and overlap in training data.

## Predictive performance

| model | metric | mean | standard_deviation | empirical_2_5_percentile | empirical_97_5_percentile | repeats |
|---|---|---|---|---|---|---|
| A: All features | accuracy | 0.752 | 0.005 | 0.745 | 0.761 | 10.000 |
| A: All features | balanced_accuracy | 0.674 | 0.008 | 0.664 | 0.688 | 10.000 |
| A: All features | roc_auc | 0.784 | 0.007 | 0.775 | 0.795 | 10.000 |
| A: All features | false_approvals | 156.200 | 4.803 | 148.225 | 161.775 | 10.000 |
| A: All features | false_rejections | 92.100 | 2.998 | 89.000 | 97.000 | 10.000 |
| A: All features | error_cost | 873.100 | 23.849 | 831.925 | 902.200 | 10.000 |
| A: All features | error_cost_per_100 | 87.310 | 2.385 | 83.193 | 90.220 | 10.000 |
| B: Age excluded | accuracy | 0.752 | 0.004 | 0.747 | 0.759 | 10.000 |
| B: Age excluded | balanced_accuracy | 0.672 | 0.007 | 0.664 | 0.683 | 10.000 |
| B: Age excluded | roc_auc | 0.784 | 0.007 | 0.775 | 0.794 | 10.000 |
| B: Age excluded | false_approvals | 158.300 | 3.974 | 151.225 | 162.775 | 10.000 |
| B: Age excluded | false_rejections | 89.800 | 2.348 | 85.675 | 93.550 | 10.000 |
| B: Age excluded | error_cost | 881.300 | 19.630 | 846.575 | 903.875 | 10.000 |
| B: Age excluded | error_cost_per_100 | 88.130 | 1.963 | 84.657 | 90.388 | 10.000 |
| C: Age + checking excluded | accuracy | 0.737 | 0.005 | 0.730 | 0.746 | 10.000 |
| C: Age + checking excluded | balanced_accuracy | 0.638 | 0.005 | 0.633 | 0.647 | 10.000 |
| C: Age + checking excluded | roc_auc | 0.746 | 0.007 | 0.737 | 0.757 | 10.000 |
| C: Age + checking excluded | false_approvals | 182.800 | 2.251 | 179.225 | 185.775 | 10.000 |
| C: Age + checking excluded | false_rejections | 80.300 | 4.498 | 73.225 | 87.650 | 10.000 |
| C: Age + checking excluded | error_cost | 994.300 | 12.508 | 974.900 | 1009.325 | 10.000 |
| C: Age + checking excluded | error_cost_per_100 | 99.430 | 1.251 | 97.490 | 100.933 | 10.000 |
| D: Age + purpose excluded | accuracy | 0.751 | 0.008 | 0.739 | 0.760 | 10.000 |
| D: Age + purpose excluded | balanced_accuracy | 0.666 | 0.011 | 0.649 | 0.682 | 10.000 |
| D: Age + purpose excluded | roc_auc | 0.774 | 0.006 | 0.767 | 0.783 | 10.000 |
| D: Age + purpose excluded | false_approvals | 164.300 | 6.482 | 153.675 | 173.325 | 10.000 |
| D: Age + purpose excluded | false_rejections | 84.300 | 4.596 | 78.000 | 90.325 | 10.000 |
| D: Age + purpose excluded | error_cost | 905.800 | 32.666 | 855.475 | 954.175 | 10.000 |
| D: Age + purpose excluded | error_cost_per_100 | 90.580 | 3.267 | 85.547 | 95.418 | 10.000 |
| E: Age + savings excluded | accuracy | 0.752 | 0.007 | 0.744 | 0.766 | 10.000 |
| E: Age + savings excluded | balanced_accuracy | 0.669 | 0.011 | 0.657 | 0.690 | 10.000 |
| E: Age + savings excluded | roc_auc | 0.780 | 0.007 | 0.769 | 0.791 | 10.000 |
| E: Age + savings excluded | false_approvals | 161.700 | 6.001 | 149.475 | 167.775 | 10.000 |
| E: Age + savings excluded | false_rejections | 86.400 | 3.307 | 79.575 | 89.000 | 10.000 |
| E: Age + savings excluded | error_cost | 894.900 | 30.755 | 834.050 | 927.100 | 10.000 |
| E: Age + savings excluded | error_cost_per_100 | 89.490 | 3.076 | 83.405 | 92.710 | 10.000 |
| F: Age + employment excluded | accuracy | 0.752 | 0.002 | 0.749 | 0.754 | 10.000 |
| F: Age + employment excluded | balanced_accuracy | 0.671 | 0.003 | 0.667 | 0.677 | 10.000 |
| F: Age + employment excluded | roc_auc | 0.783 | 0.007 | 0.773 | 0.792 | 10.000 |
| F: Age + employment excluded | false_approvals | 159.200 | 2.781 | 154.450 | 162.775 | 10.000 |
| F: Age + employment excluded | false_rejections | 88.800 | 2.530 | 85.000 | 91.775 | 10.000 |
| F: Age + employment excluded | error_cost | 884.800 | 12.118 | 863.800 | 899.100 | 10.000 |
| F: Age + employment excluded | error_cost_per_100 | 88.480 | 1.212 | 86.380 | 89.910 | 10.000 |
| G: Age + housing excluded | accuracy | 0.754 | 0.006 | 0.744 | 0.760 | 10.000 |
| G: Age + housing excluded | balanced_accuracy | 0.674 | 0.008 | 0.660 | 0.684 | 10.000 |
| G: Age + housing excluded | roc_auc | 0.784 | 0.007 | 0.774 | 0.793 | 10.000 |
| G: Age + housing excluded | false_approvals | 157.300 | 4.596 | 150.675 | 165.100 | 10.000 |
| G: Age + housing excluded | false_rejections | 89.200 | 2.860 | 84.225 | 92.000 | 10.000 |
| G: Age + housing excluded | error_cost | 875.700 | 23.711 | 844.600 | 916.725 | 10.000 |
| G: Age + housing excluded | error_cost_per_100 | 87.570 | 2.371 | 84.460 | 91.672 | 10.000 |
| H: Age + savings + employment + housing excluded | accuracy | 0.754 | 0.007 | 0.745 | 0.767 | 10.000 |
| H: Age + savings + employment + housing excluded | balanced_accuracy | 0.669 | 0.009 | 0.656 | 0.685 | 10.000 |
| H: Age + savings + employment + housing excluded | roc_auc | 0.776 | 0.008 | 0.766 | 0.787 | 10.000 |
| H: Age + savings + employment + housing excluded | false_approvals | 162.900 | 4.771 | 156.125 | 170.325 | 10.000 |
| H: Age + savings + employment + housing excluded | false_rejections | 83.600 | 4.351 | 75.675 | 89.550 | 10.000 |
| H: Age + savings + employment + housing excluded | error_cost | 898.100 | 25.666 | 857.950 | 936.400 | 10.000 |
| H: Age + savings + employment + housing excluded | error_cost_per_100 | 89.810 | 2.567 | 85.795 | 93.640 | 10.000 |

Cost is 5 × false approvals + false rejections. Cost per 100 applicants is that cost divided by dataset size and multiplied by 100.

## Fairness rates by age group

| model | group | metric | denominator_per_repeat | mean_numerator_per_repeat | mean | standard_deviation | empirical_2_5_percentile | empirical_97_5_percentile | repeats |
|---|---|---|---|---|---|---|---|---|---|
| A: All features | Under 25 | approval_rate | 149.000 | 93.900 | 0.630 | 0.015 | 0.606 | 0.649 | 10.000 |
| A: All features | Under 25 | false_approval_rate | 61.000 | 29.800 | 0.489 | 0.030 | 0.459 | 0.537 | 10.000 |
| A: All features | Under 25 | false_rejection_rate | 88.000 | 23.900 | 0.272 | 0.016 | 0.253 | 0.302 | 10.000 |
| A: All features | Under 25 | true_positive_rate | 88.000 | 64.100 | 0.728 | 0.016 | 0.698 | 0.747 | 10.000 |
| A: All features | 25 or older | approval_rate | 851.000 | 670.200 | 0.788 | 0.006 | 0.779 | 0.797 | 10.000 |
| A: All features | 25 or older | false_approval_rate | 239.000 | 126.400 | 0.529 | 0.019 | 0.499 | 0.550 | 10.000 |
| A: All features | 25 or older | false_rejection_rate | 612.000 | 68.200 | 0.111 | 0.005 | 0.106 | 0.119 | 10.000 |
| A: All features | 25 or older | true_positive_rate | 612.000 | 543.800 | 0.889 | 0.005 | 0.881 | 0.894 | 10.000 |
| B: Age excluded | Under 25 | approval_rate | 149.000 | 98.000 | 0.658 | 0.015 | 0.634 | 0.678 | 10.000 |
| B: Age excluded | Under 25 | false_approval_rate | 61.000 | 31.600 | 0.518 | 0.026 | 0.492 | 0.557 | 10.000 |
| B: Age excluded | Under 25 | false_rejection_rate | 88.000 | 21.600 | 0.245 | 0.014 | 0.227 | 0.270 | 10.000 |
| B: Age excluded | Under 25 | true_positive_rate | 88.000 | 66.400 | 0.755 | 0.014 | 0.730 | 0.773 | 10.000 |
| B: Age excluded | 25 or older | approval_rate | 851.000 | 670.500 | 0.788 | 0.005 | 0.780 | 0.792 | 10.000 |
| B: Age excluded | 25 or older | false_approval_rate | 239.000 | 126.700 | 0.530 | 0.015 | 0.506 | 0.544 | 10.000 |
| B: Age excluded | 25 or older | false_rejection_rate | 612.000 | 68.200 | 0.111 | 0.004 | 0.105 | 0.119 | 10.000 |
| B: Age excluded | 25 or older | true_positive_rate | 612.000 | 543.800 | 0.889 | 0.004 | 0.881 | 0.895 | 10.000 |
| C: Age + checking excluded | Under 25 | approval_rate | 149.000 | 108.800 | 0.730 | 0.018 | 0.705 | 0.755 | 10.000 |
| C: Age + checking excluded | Under 25 | false_approval_rate | 61.000 | 35.500 | 0.582 | 0.022 | 0.545 | 0.607 | 10.000 |
| C: Age + checking excluded | Under 25 | false_rejection_rate | 88.000 | 14.700 | 0.167 | 0.024 | 0.130 | 0.202 | 10.000 |
| C: Age + checking excluded | Under 25 | true_positive_rate | 88.000 | 73.300 | 0.833 | 0.024 | 0.798 | 0.870 | 10.000 |
| C: Age + checking excluded | 25 or older | approval_rate | 851.000 | 693.700 | 0.815 | 0.006 | 0.806 | 0.823 | 10.000 |
| C: Age + checking excluded | 25 or older | false_approval_rate | 239.000 | 147.300 | 0.616 | 0.012 | 0.599 | 0.634 | 10.000 |
| C: Age + checking excluded | 25 or older | false_rejection_rate | 612.000 | 65.600 | 0.107 | 0.008 | 0.093 | 0.117 | 10.000 |
| C: Age + checking excluded | 25 or older | true_positive_rate | 612.000 | 546.400 | 0.893 | 0.008 | 0.883 | 0.907 | 10.000 |
| D: Age + purpose excluded | Under 25 | approval_rate | 149.000 | 98.600 | 0.662 | 0.012 | 0.646 | 0.682 | 10.000 |
| D: Age + purpose excluded | Under 25 | false_approval_rate | 61.000 | 30.200 | 0.495 | 0.028 | 0.463 | 0.550 | 10.000 |
| D: Age + purpose excluded | Under 25 | false_rejection_rate | 88.000 | 19.600 | 0.223 | 0.017 | 0.196 | 0.247 | 10.000 |
| D: Age + purpose excluded | Under 25 | true_positive_rate | 88.000 | 68.400 | 0.777 | 0.017 | 0.753 | 0.804 | 10.000 |
| D: Age + purpose excluded | 25 or older | approval_rate | 851.000 | 681.400 | 0.801 | 0.008 | 0.787 | 0.812 | 10.000 |
| D: Age + purpose excluded | 25 or older | false_approval_rate | 239.000 | 134.100 | 0.561 | 0.025 | 0.518 | 0.590 | 10.000 |
| D: Age + purpose excluded | 25 or older | false_rejection_rate | 612.000 | 64.700 | 0.106 | 0.006 | 0.097 | 0.114 | 10.000 |
| D: Age + purpose excluded | 25 or older | true_positive_rate | 612.000 | 547.300 | 0.894 | 0.006 | 0.886 | 0.903 | 10.000 |
| E: Age + savings excluded | Under 25 | approval_rate | 149.000 | 99.600 | 0.668 | 0.024 | 0.638 | 0.702 | 10.000 |
| E: Age + savings excluded | Under 25 | false_approval_rate | 61.000 | 31.300 | 0.513 | 0.043 | 0.450 | 0.583 | 10.000 |
| E: Age + savings excluded | Under 25 | false_rejection_rate | 88.000 | 19.700 | 0.224 | 0.023 | 0.184 | 0.250 | 10.000 |
| E: Age + savings excluded | Under 25 | true_positive_rate | 88.000 | 68.300 | 0.776 | 0.023 | 0.750 | 0.816 | 10.000 |
| E: Age + savings excluded | 25 or older | approval_rate | 851.000 | 675.700 | 0.794 | 0.006 | 0.784 | 0.803 | 10.000 |
| E: Age + savings excluded | 25 or older | false_approval_rate | 239.000 | 130.400 | 0.546 | 0.021 | 0.503 | 0.569 | 10.000 |
| E: Age + savings excluded | 25 or older | false_rejection_rate | 612.000 | 66.700 | 0.109 | 0.007 | 0.094 | 0.117 | 10.000 |
| E: Age + savings excluded | 25 or older | true_positive_rate | 612.000 | 545.300 | 0.891 | 0.007 | 0.883 | 0.906 | 10.000 |
| F: Age + employment excluded | Under 25 | approval_rate | 149.000 | 100.200 | 0.672 | 0.022 | 0.646 | 0.707 | 10.000 |
| F: Age + employment excluded | Under 25 | false_approval_rate | 61.000 | 32.600 | 0.534 | 0.026 | 0.508 | 0.574 | 10.000 |
| F: Age + employment excluded | Under 25 | false_rejection_rate | 88.000 | 20.400 | 0.232 | 0.023 | 0.196 | 0.259 | 10.000 |
| F: Age + employment excluded | Under 25 | true_positive_rate | 88.000 | 67.600 | 0.768 | 0.023 | 0.741 | 0.804 | 10.000 |
| F: Age + employment excluded | 25 or older | approval_rate | 851.000 | 670.200 | 0.788 | 0.006 | 0.781 | 0.797 | 10.000 |
| F: Age + employment excluded | 25 or older | false_approval_rate | 239.000 | 126.600 | 0.530 | 0.010 | 0.517 | 0.546 | 10.000 |
| F: Age + employment excluded | 25 or older | false_rejection_rate | 612.000 | 68.400 | 0.112 | 0.005 | 0.105 | 0.119 | 10.000 |
| F: Age + employment excluded | 25 or older | true_positive_rate | 612.000 | 543.600 | 0.888 | 0.005 | 0.881 | 0.895 | 10.000 |
| G: Age + housing excluded | Under 25 | approval_rate | 149.000 | 100.800 | 0.677 | 0.018 | 0.653 | 0.707 | 10.000 |
| G: Age + housing excluded | Under 25 | false_approval_rate | 61.000 | 32.700 | 0.536 | 0.021 | 0.508 | 0.570 | 10.000 |
| G: Age + housing excluded | Under 25 | false_rejection_rate | 88.000 | 19.900 | 0.226 | 0.029 | 0.176 | 0.250 | 10.000 |
| G: Age + housing excluded | Under 25 | true_positive_rate | 88.000 | 68.100 | 0.774 | 0.029 | 0.750 | 0.824 | 10.000 |
| G: Age + housing excluded | 25 or older | approval_rate | 851.000 | 667.300 | 0.784 | 0.006 | 0.774 | 0.791 | 10.000 |
| G: Age + housing excluded | 25 or older | false_approval_rate | 239.000 | 124.600 | 0.521 | 0.020 | 0.486 | 0.549 | 10.000 |
| G: Age + housing excluded | 25 or older | false_rejection_rate | 612.000 | 69.300 | 0.113 | 0.004 | 0.107 | 0.119 | 10.000 |
| G: Age + housing excluded | 25 or older | true_positive_rate | 612.000 | 542.700 | 0.887 | 0.004 | 0.881 | 0.893 | 10.000 |
| H: Age + savings + employment + housing excluded | Under 25 | approval_rate | 149.000 | 107.700 | 0.723 | 0.021 | 0.698 | 0.752 | 10.000 |
| H: Age + savings + employment + housing excluded | Under 25 | false_approval_rate | 61.000 | 34.300 | 0.562 | 0.027 | 0.525 | 0.590 | 10.000 |
| H: Age + savings + employment + housing excluded | Under 25 | false_rejection_rate | 88.000 | 14.600 | 0.166 | 0.023 | 0.128 | 0.193 | 10.000 |
| H: Age + savings + employment + housing excluded | Under 25 | true_positive_rate | 88.000 | 73.400 | 0.834 | 0.023 | 0.807 | 0.872 | 10.000 |
| H: Age + savings + employment + housing excluded | 25 or older | approval_rate | 851.000 | 671.600 | 0.789 | 0.006 | 0.779 | 0.795 | 10.000 |
| H: Age + savings + employment + housing excluded | 25 or older | false_approval_rate | 239.000 | 128.600 | 0.538 | 0.015 | 0.517 | 0.562 | 10.000 |
| H: Age + savings + employment + housing excluded | 25 or older | false_rejection_rate | 612.000 | 69.000 | 0.113 | 0.007 | 0.101 | 0.123 | 10.000 |
| H: Age + savings + employment + housing excluded | 25 or older | true_positive_rate | 612.000 | 543.000 | 0.887 | 0.007 | 0.877 | 0.899 | 10.000 |

Rates use the fixed full-dataset subgroup denominators in `subgroup_denominators.csv`; group membership is based on actual age, not model inputs. The table includes the mean numerator count per repeat as well as the denominator.

## Absolute age-group gaps

| model | metric | mean | standard_deviation | empirical_2_5_percentile | empirical_97_5_percentile | repeats |
|---|---|---|---|---|---|---|
| A: All features | approval_rate_gap_pp | 15.734 | 1.475 | 14.016 | 18.416 | 10.000 |
| A: All features | false_approval_rate_gap_pp | 4.310 | 3.097 | 1.134 | 9.140 | 10.000 |
| A: All features | false_rejection_rate_gap_pp | 16.015 | 1.700 | 13.671 | 18.916 | 10.000 |
| A: All features | true_positive_rate_gap_pp | 16.015 | 1.700 | 13.671 | 18.916 | 10.000 |
| B: Age excluded | approval_rate_gap_pp | 13.018 | 1.463 | 11.181 | 15.174 | 10.000 |
| B: Age excluded | false_approval_rate_gap_pp | 2.455 | 1.740 | 0.215 | 5.025 | 10.000 |
| B: Age excluded | false_rejection_rate_gap_pp | 13.402 | 1.699 | 10.783 | 16.196 | 10.000 |
| B: Age excluded | true_positive_rate_gap_pp | 13.402 | 1.699 | 10.783 | 16.196 | 10.000 |
| C: Age + checking excluded | approval_rate_gap_pp | 8.496 | 2.117 | 5.685 | 11.453 | 10.000 |
| C: Age + checking excluded | false_approval_rate_gap_pp | 3.435 | 3.068 | 0.108 | 8.755 | 10.000 |
| C: Age + checking excluded | false_rejection_rate_gap_pp | 5.986 | 2.791 | 1.700 | 10.460 | 10.000 |
| C: Age + checking excluded | true_positive_rate_gap_pp | 5.986 | 2.791 | 1.700 | 10.460 | 10.000 |
| D: Age + purpose excluded | approval_rate_gap_pp | 13.896 | 1.194 | 12.108 | 15.693 | 10.000 |
| D: Age + purpose excluded | false_approval_rate_gap_pp | 6.601 | 3.273 | 2.394 | 11.165 | 10.000 |
| D: Age + purpose excluded | false_rejection_rate_gap_pp | 11.701 | 1.534 | 9.345 | 13.923 | 10.000 |
| D: Age + purpose excluded | true_positive_rate_gap_pp | 11.701 | 1.534 | 9.345 | 13.923 | 10.000 |
| E: Age + savings excluded | approval_rate_gap_pp | 12.555 | 2.415 | 9.151 | 15.989 | 10.000 |
| E: Age + savings excluded | false_approval_rate_gap_pp | 4.300 | 3.264 | 0.191 | 9.906 | 10.000 |
| E: Age + savings excluded | false_rejection_rate_gap_pp | 11.488 | 2.829 | 6.820 | 15.556 | 10.000 |
| E: Age + savings excluded | true_positive_rate_gap_pp | 11.488 | 2.829 | 6.820 | 15.556 | 10.000 |
| F: Age + employment excluded | approval_rate_gap_pp | 11.506 | 2.408 | 7.455 | 14.495 | 10.000 |
| F: Age + employment excluded | false_approval_rate_gap_pp | 2.187 | 1.743 | 0.347 | 4.982 | 10.000 |
| F: Age + employment excluded | false_rejection_rate_gap_pp | 12.005 | 2.621 | 7.683 | 14.790 | 10.000 |
| F: Age + employment excluded | true_positive_rate_gap_pp | 12.005 | 2.621 | 7.683 | 14.790 | 10.000 |
| G: Age + housing excluded | approval_rate_gap_pp | 10.763 | 2.040 | 7.494 | 13.752 | 10.000 |
| G: Age + housing excluded | false_approval_rate_gap_pp | 2.400 | 2.705 | 0.210 | 8.051 | 10.000 |
| G: Age + housing excluded | false_rejection_rate_gap_pp | 11.290 | 3.078 | 6.135 | 14.269 | 10.000 |
| G: Age + housing excluded | true_positive_rate_gap_pp | 11.290 | 3.078 | 6.135 | 14.269 | 10.000 |
| H: Age + savings + employment + housing excluded | approval_rate_gap_pp | 6.637 | 2.235 | 3.472 | 9.376 | 10.000 |
| H: Age + savings + employment + housing excluded | false_approval_rate_gap_pp | 2.641 | 1.873 | 0.218 | 5.076 | 10.000 |
| H: Age + savings + employment + housing excluded | false_rejection_rate_gap_pp | 5.316 | 2.608 | 0.828 | 8.223 | 10.000 |
| H: Age + savings + employment + housing excluded | true_positive_rate_gap_pp | 5.316 | 2.608 | 0.828 | 8.223 | 10.000 |

## Paired changes against age-excluded Model B

| model | metric | mean_paired_delta | standard_deviation | empirical_2_5_percentile | empirical_97_5_percentile | repeats_delta_below_zero | repeats_delta_equal_zero | repeats_delta_above_zero |
|---|---|---|---|---|---|---|---|---|
| A: All features | accuracy | -0.000 | 0.004 | -0.005 | 0.006 | 6.000 | 1.000 | 3.000 |
| A: All features | balanced_accuracy | 0.002 | 0.004 | -0.004 | 0.009 | 4.000 | 0.000 | 6.000 |
| A: All features | roc_auc | -0.000 | 0.001 | -0.002 | 0.002 | 5.000 | 0.000 | 5.000 |
| A: All features | error_cost_per_100 | -0.820 | 1.155 | -2.610 | 0.953 | 8.000 | 0.000 | 2.000 |
| A: All features | approval_rate_gap_pp | 2.716 | 1.220 | 1.278 | 4.600 | 0.000 | 0.000 | 10.000 |
| A: All features | false_approval_rate_gap_pp | 1.855 | 2.573 | -2.344 | 5.250 | 2.000 | 1.000 | 7.000 |
| A: All features | false_rejection_rate_gap_pp | 2.614 | 1.425 | 0.846 | 4.799 | 0.000 | 0.000 | 10.000 |
| A: All features | true_positive_rate_gap_pp | 2.614 | 1.425 | 0.846 | 4.799 | 0.000 | 0.000 | 10.000 |
| C: Age + checking excluded | accuracy | -0.015 | 0.005 | -0.025 | -0.010 | 10.000 | 0.000 | 0.000 |
| C: Age + checking excluded | balanced_accuracy | -0.034 | 0.007 | -0.046 | -0.026 | 10.000 | 0.000 | 0.000 |
| C: Age + checking excluded | roc_auc | -0.038 | 0.003 | -0.042 | -0.035 | 10.000 | 0.000 | 0.000 |
| C: Age + checking excluded | error_cost_per_100 | 11.300 | 2.002 | 9.090 | 14.960 | 0.000 | 0.000 | 10.000 |
| C: Age + checking excluded | approval_rate_gap_pp | -4.522 | 2.372 | -8.033 | -0.707 | 10.000 | 0.000 | 0.000 |
| C: Age + checking excluded | false_approval_rate_gap_pp | 0.980 | 3.163 | -3.024 | 5.611 | 4.000 | 0.000 | 6.000 |
| C: Age + checking excluded | false_rejection_rate_gap_pp | -7.416 | 3.019 | -12.408 | -2.836 | 10.000 | 0.000 | 0.000 |
| C: Age + checking excluded | true_positive_rate_gap_pp | -7.416 | 3.019 | -12.408 | -2.836 | 10.000 | 0.000 | 0.000 |
| D: Age + purpose excluded | accuracy | -0.001 | 0.005 | -0.008 | 0.008 | 5.000 | 1.000 | 4.000 |
| D: Age + purpose excluded | balanced_accuracy | -0.006 | 0.008 | -0.017 | 0.008 | 9.000 | 0.000 | 1.000 |
| D: Age + purpose excluded | roc_auc | -0.010 | 0.003 | -0.014 | -0.006 | 10.000 | 0.000 | 0.000 |
| D: Age + purpose excluded | error_cost_per_100 | 2.450 | 2.332 | -1.758 | 5.610 | 1.000 | 0.000 | 9.000 |
| D: Age + purpose excluded | approval_rate_gap_pp | 0.878 | 1.810 | -1.987 | 3.015 | 2.000 | 0.000 | 8.000 |
| D: Age + purpose excluded | false_approval_rate_gap_pp | 4.146 | 3.706 | -1.704 | 9.503 | 1.000 | 0.000 | 9.000 |
| D: Age + purpose excluded | false_rejection_rate_gap_pp | -1.701 | 1.448 | -4.321 | 0.218 | 9.000 | 0.000 | 1.000 |
| D: Age + purpose excluded | true_positive_rate_gap_pp | -1.701 | 1.448 | -4.321 | 0.218 | 9.000 | 0.000 | 1.000 |
| E: Age + savings excluded | accuracy | 0.000 | 0.005 | -0.007 | 0.007 | 5.000 | 1.000 | 4.000 |
| E: Age + savings excluded | balanced_accuracy | -0.003 | 0.007 | -0.012 | 0.008 | 7.000 | 1.000 | 2.000 |
| E: Age + savings excluded | roc_auc | -0.005 | 0.002 | -0.007 | -0.002 | 10.000 | 0.000 | 0.000 |
| E: Age + savings excluded | error_cost_per_100 | 1.360 | 2.053 | -2.108 | 3.900 | 2.000 | 0.000 | 8.000 |
| E: Age + savings excluded | approval_rate_gap_pp | -0.463 | 2.516 | -4.542 | 3.208 | 6.000 | 0.000 | 4.000 |
| E: Age + savings excluded | false_approval_rate_gap_pp | 1.845 | 3.374 | -2.929 | 7.026 | 3.000 | 0.000 | 7.000 |
| E: Age + savings excluded | false_rejection_rate_gap_pp | -1.914 | 2.410 | -5.640 | 2.114 | 8.000 | 0.000 | 2.000 |
| E: Age + savings excluded | true_positive_rate_gap_pp | -1.914 | 2.410 | -5.640 | 2.114 | 8.000 | 0.000 | 2.000 |
| F: Age + employment excluded | accuracy | 0.000 | 0.004 | -0.005 | 0.005 | 5.000 | 0.000 | 5.000 |
| F: Age + employment excluded | balanced_accuracy | -0.001 | 0.005 | -0.009 | 0.006 | 5.000 | 0.000 | 5.000 |
| F: Age + employment excluded | roc_auc | -0.001 | 0.001 | -0.003 | 0.001 | 9.000 | 0.000 | 1.000 |
| F: Age + employment excluded | error_cost_per_100 | 0.350 | 1.556 | -1.753 | 2.962 | 5.000 | 0.000 | 5.000 |
| F: Age + employment excluded | approval_rate_gap_pp | -1.512 | 1.738 | -4.882 | 0.607 | 8.000 | 0.000 | 2.000 |
| F: Age + employment excluded | false_approval_rate_gap_pp | -0.268 | 2.469 | -3.701 | 3.410 | 6.000 | 0.000 | 4.000 |
| F: Age + employment excluded | false_rejection_rate_gap_pp | -1.396 | 1.960 | -4.888 | 1.334 | 7.000 | 1.000 | 2.000 |
| F: Age + employment excluded | true_positive_rate_gap_pp | -1.396 | 1.960 | -4.888 | 1.334 | 7.000 | 1.000 | 2.000 |
| G: Age + housing excluded | accuracy | 0.002 | 0.005 | -0.004 | 0.009 | 4.000 | 0.000 | 6.000 |
| G: Age + housing excluded | balanced_accuracy | 0.002 | 0.006 | -0.005 | 0.010 | 4.000 | 0.000 | 6.000 |
| G: Age + housing excluded | roc_auc | -0.001 | 0.001 | -0.002 | 0.002 | 7.000 | 0.000 | 3.000 |
| G: Age + housing excluded | error_cost_per_100 | -0.560 | 1.578 | -2.500 | 1.510 | 6.000 | 0.000 | 4.000 |
| G: Age + housing excluded | approval_rate_gap_pp | -2.255 | 2.200 | -5.075 | 1.378 | 8.000 | 1.000 | 1.000 |
| G: Age + housing excluded | false_approval_rate_gap_pp | -0.055 | 3.797 | -3.668 | 7.577 | 6.000 | 0.000 | 4.000 |
| G: Age + housing excluded | false_rejection_rate_gap_pp | -2.112 | 2.512 | -6.397 | 0.973 | 7.000 | 0.000 | 3.000 |
| G: Age + housing excluded | true_positive_rate_gap_pp | -2.112 | 2.512 | -6.397 | 0.973 | 7.000 | 0.000 | 3.000 |
| H: Age + savings + employment + housing excluded | accuracy | 0.002 | 0.005 | -0.005 | 0.009 | 3.000 | 1.000 | 6.000 |
| H: Age + savings + employment + housing excluded | balanced_accuracy | -0.003 | 0.008 | -0.013 | 0.009 | 7.000 | 0.000 | 3.000 |
| H: Age + savings + employment + housing excluded | roc_auc | -0.008 | 0.003 | -0.012 | -0.005 | 10.000 | 0.000 | 0.000 |
| H: Age + savings + employment + housing excluded | error_cost_per_100 | 1.680 | 2.297 | -1.930 | 4.410 | 3.000 | 0.000 | 7.000 |
| H: Age + savings + employment + housing excluded | approval_rate_gap_pp | -6.381 | 1.988 | -8.840 | -3.590 | 10.000 | 0.000 | 0.000 |
| H: Age + savings + employment + housing excluded | false_approval_rate_gap_pp | 0.187 | 3.137 | -4.222 | 3.846 | 4.000 | 0.000 | 6.000 |
| H: Age + savings + employment + housing excluded | false_rejection_rate_gap_pp | -8.085 | 2.373 | -11.439 | -4.801 | 10.000 | 0.000 | 0.000 |
| H: Age + savings + employment + housing excluded | true_positive_rate_gap_pp | -8.085 | 2.373 | -11.439 | -4.801 | 10.000 | 0.000 | 0.000 |

## Interpretation

- Removing age changed accuracy in 9 of 10 repeats, but direction was mixed: Model A minus B accuracy was lower in 6 repeats, higher in 3, and equal in 1. AUC direction was lower in 5 and higher in 5. Removing age increased cost per 100 by an average of 0.82; that direction occurred in 8 repeats, but its paired empirical interval crosses zero.
- Removing checking status narrowed the approval-rate gap in 10 of 10 repeats; its mean gap change versus Model B was -4.52 percentage points. This removal also raised cost per 100 by 11.30 and lowered accuracy by 0.015 in every repeat.
- The combined removal's false-approval gap change versus Model B averaged +0.19 percentage points and was positive in 6 of 10 repeats. This does not reproduce the very large fixed-test split gap as a stable pattern.
- More stable fairness patterns: including age (Model A versus B) had a larger approval-rate gap and false-rejection gap in all 10 repeats. Removing checking status lowered both of those gaps in all 10 repeats, but did not consistently lower the false-approval gap.
- Stability should be judged from the paired sign counts and empirical ranges alongside absolute performance and all fairness gaps. Approval and error-rate gaps can move in different directions, so no single metric establishes an overall improvement.

The under-25 subgroup is smaller than the 25+ subgroup. Even with OOF predictions for all 1000 rows, its rates use fewer actual good- and bad-risk cases. Repeated folds do not create new independent applicants. This historical dataset, one model family, fixed default threshold, and lack of external validation limit conclusions. Do not infer discrimination or causality from feature-removal differences.
