import math
import unittest

from src.affordability import (
    assess_affordability,
    calculate_emi,
    principal_for_emi,
)


class AffordabilityTests(unittest.TestCase):
    def test_standard_emi_matches_annuity_formula(self):
        principal, annual_rate, months = 100000.0, 12.0, 12
        monthly_rate = annual_rate / 100 / 12
        expected = principal * monthly_rate * (1 + monthly_rate) ** months / (
            (1 + monthly_rate) ** months - 1
        )
        self.assertTrue(math.isclose(calculate_emi(principal, annual_rate, months), expected))

    def test_zero_interest_emi_and_inverse_principal(self):
        self.assertEqual(calculate_emi(120000, 0, 12), 10000)
        self.assertEqual(principal_for_emi(10000, 0, 12), 120000)

    def test_inverse_emi_recovers_principal_at_positive_rate(self):
        principal = 275000.0
        emi = calculate_emi(principal, 8.5, 48)
        self.assertTrue(math.isclose(principal_for_emi(emi, 8.5, 48), principal, rel_tol=1e-12))

    def test_invalid_inputs_are_rejected(self):
        base = dict(
            monthly_income=50000,
            existing_monthly_loan_repayments=5000,
            other_recurring_monthly_obligations=3000,
            annual_interest_rate=10,
            tenure_months=24,
        )
        invalid_cases = [
            {"monthly_income": -1},
            {"existing_monthly_loan_repayments": -1},
            {"other_recurring_monthly_obligations": -1},
            {"annual_interest_rate": -0.1},
            {"annual_interest_rate": 100.1},
            {"tenure_months": 0},
            {"tenure_months": 12.5},
            {"proposed_loan_amount": -1},
            {"proposed_loan_amount": float("nan")},
            {"affordability_percentage": -1},
            {"affordability_percentage": 101},
        ]
        for override in invalid_cases:
            with self.subTest(override=override), self.assertRaises(ValueError):
                assess_affordability(**(base | override))

    def test_zero_disposable_income_has_zero_affordable_emi(self):
        result = assess_affordability(20000, 12000, 8000, 0, 12, proposed_loan_amount=1)
        self.assertEqual(result["disposable_income_after_existing_obligations"], 0)
        self.assertEqual(result["maximum_illustrative_affordable_emi"], 0)
        self.assertTrue(result["proposed_emi_exceeds_illustrative_affordable_emi"])

    def test_proposed_loan_within_limit_and_remaining_income(self):
        result = assess_affordability(50000, 5000, 5000, 0, 10, proposed_loan_amount=100000)
        self.assertFalse(result["proposed_emi_exceeds_illustrative_affordable_emi"])
        self.assertEqual(result["assessment"], "within_illustrative_limit")
        self.assertEqual(result["proposed_monthly_emi"], 10000)
        self.assertEqual(result["remaining_monthly_income_after_proposed_loan"], 30000)

    def test_proposed_loan_above_limit_and_total_costs(self):
        result = assess_affordability(50000, 5000, 5000, 0, 10, proposed_loan_amount=500000)
        self.assertTrue(result["proposed_emi_exceeds_illustrative_affordable_emi"])
        self.assertEqual(result["assessment"], "above_illustrative_limit")
        self.assertEqual(result["proposed_total_repayment"], 500000)
        self.assertEqual(result["proposed_total_interest"], 0)
        self.assertEqual(result["remaining_monthly_income_after_proposed_loan"], -10000)

    def test_affordability_percentage_is_configurable_and_disclosed(self):
        base = dict(monthly_income=60000, existing_monthly_loan_repayments=10000,
                    other_recurring_monthly_obligations=10000,
                    annual_interest_rate=12, tenure_months=36)
        default = assess_affordability(**base)
        half = assess_affordability(**base, affordability_percentage=15)
        self.assertEqual(default["affordability_percentage"], 30)
        self.assertEqual(default["maximum_illustrative_affordable_emi"], 12000)
        self.assertEqual(half["maximum_illustrative_affordable_emi"], 6000)
        self.assertIn("not a validated lending standard", default["affordability_assumption"])


if __name__ == "__main__":
    unittest.main()
