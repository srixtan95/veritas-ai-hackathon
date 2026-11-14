"""Standalone rule-based affordability calculations, separate from credit risk."""

import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "affordability"
DEFAULT_AFFORDABILITY_PERCENTAGE = 30.0
MAX_ANNUAL_INTEREST_PERCENTAGE = 100.0


def _finite_number(name, value):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number.")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number.") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number.")
    return number


def _validate_inputs(monthly_income, existing_monthly_loan_repayments,
                     other_recurring_monthly_obligations, annual_interest_rate,
                     tenure_months, proposed_loan_amount,
                     affordability_percentage):
    income = _finite_number("Monthly income", monthly_income)
    existing = _finite_number("Existing monthly loan repayments", existing_monthly_loan_repayments)
    other = _finite_number("Other recurring monthly obligations", other_recurring_monthly_obligations)
    annual_rate = _finite_number("Proposed annual interest rate", annual_interest_rate)
    percentage = _finite_number("Affordability percentage", affordability_percentage)
    if income < 0:
        raise ValueError("Monthly income cannot be negative.")
    if existing < 0:
        raise ValueError("Existing monthly loan repayments cannot be negative.")
    if other < 0:
        raise ValueError("Other recurring monthly obligations cannot be negative.")
    if not 0 <= annual_rate <= MAX_ANNUAL_INTEREST_PERCENTAGE:
        raise ValueError("Annual interest rate must be between 0 and 100 percent, inclusive.")
    if not 0 <= percentage <= 100:
        raise ValueError("Affordability percentage must be between 0 and 100, inclusive.")
    if isinstance(tenure_months, bool) or not isinstance(tenure_months, int) or tenure_months <= 0:
        raise ValueError("Loan tenure must be a positive whole number of months.")
    proposed = None
    if proposed_loan_amount is not None:
        proposed = _finite_number("Proposed loan amount", proposed_loan_amount)
        if proposed <= 0:
            raise ValueError("Proposed loan amount must be greater than zero.")
    return income, existing, other, annual_rate, tenure_months, proposed, percentage


def calculate_emi(principal, annual_interest_rate, tenure_months):
    """Calculate fixed monthly EMI from principal, nominal annual rate (%), and months."""
    principal = _finite_number("Principal", principal)
    annual_interest_rate = _finite_number("Annual interest rate", annual_interest_rate)
    if principal < 0:
        raise ValueError("Principal cannot be negative.")
    if not 0 <= annual_interest_rate <= MAX_ANNUAL_INTEREST_PERCENTAGE:
        raise ValueError("Annual interest rate must be between 0 and 100 percent, inclusive.")
    if isinstance(tenure_months, bool) or not isinstance(tenure_months, int) or tenure_months <= 0:
        raise ValueError("Loan tenure must be a positive whole number of months.")
    if annual_interest_rate == 0:
        return principal / tenure_months
    monthly_rate = annual_interest_rate / 100.0 / 12.0
    # Algebraically equivalent EMI formula, arranged to avoid large powers.
    denominator = -math.expm1(-tenure_months * math.log1p(monthly_rate))
    return principal * monthly_rate / denominator


def principal_for_emi(monthly_emi, annual_interest_rate, tenure_months):
    """Invert the EMI formula to find the principal supported by a monthly EMI."""
    monthly_emi = _finite_number("Monthly EMI", monthly_emi)
    annual_interest_rate = _finite_number("Annual interest rate", annual_interest_rate)
    if monthly_emi < 0:
        raise ValueError("Monthly EMI cannot be negative.")
    if not 0 <= annual_interest_rate <= MAX_ANNUAL_INTEREST_PERCENTAGE:
        raise ValueError("Annual interest rate must be between 0 and 100 percent, inclusive.")
    if isinstance(tenure_months, bool) or not isinstance(tenure_months, int) or tenure_months <= 0:
        raise ValueError("Loan tenure must be a positive whole number of months.")
    if annual_interest_rate == 0:
        return monthly_emi * tenure_months
    monthly_rate = annual_interest_rate / 100.0 / 12.0
    present_value_factor = -math.expm1(-tenure_months * math.log1p(monthly_rate)) / monthly_rate
    return monthly_emi * present_value_factor


def assess_affordability(monthly_income, existing_monthly_loan_repayments,
                         other_recurring_monthly_obligations, annual_interest_rate,
                         tenure_months, proposed_loan_amount=None,
                         affordability_percentage=DEFAULT_AFFORDABILITY_PERCENTAGE):
    """Return an illustrative affordability estimate without using credit-risk data.

    The affordable EMI is the configured percentage of nonnegative disposable
    income. This is a prototype assumption, not a validated lending standard.
    All monetary values use the same caller-supplied currency unit.
    """
    (income, existing, other, annual_rate, months, proposed,
     percentage) = _validate_inputs(
        monthly_income, existing_monthly_loan_repayments,
        other_recurring_monthly_obligations, annual_interest_rate,
        tenure_months, proposed_loan_amount, affordability_percentage,
    )
    total_obligations = existing + other
    disposable = income - total_obligations
    affordable_emi = max(disposable, 0.0) * percentage / 100.0
    max_principal = principal_for_emi(affordable_emi, annual_rate, months)

    result = {
        "monthly_income": income,
        "existing_monthly_loan_repayments": existing,
        "other_recurring_monthly_obligations": other,
        "total_monthly_obligations": total_obligations,
        "disposable_income_after_existing_obligations": disposable,
        "affordability_percentage": percentage,
        "affordability_assumption": (
            f"Illustrative maximum EMI = {percentage:g}% of nonnegative disposable income; "
            "this is a prototype assumption, not a validated lending standard."
        ),
        "proposed_annual_interest_rate_percentage": annual_rate,
        "proposed_tenure_months": months,
        "maximum_illustrative_affordable_emi": affordable_emi,
        "indicative_maximum_loan_principal": max_principal,
        "proposed_loan_amount": proposed,
        "proposed_monthly_emi": None,
        "proposed_total_repayment": None,
        "proposed_total_interest": None,
        "proposed_emi_exceeds_illustrative_affordable_emi": None,
        "remaining_monthly_income_after_proposed_loan": None,
        "assessment": "not_assessed" if proposed is None else None,
        "assessment_message": (
            "No proposed amount supplied; calculated the illustrative EMI limit and indicative principal only."
            if proposed is None else None
        ),
        "disclaimer": (
            "This calculation is an illustrative budgeting estimate, not a guarantee of repayment ability, "
            "a creditworthiness decision, or an approval or rejection. It is separate from the "
            "Immortals AI credit-risk prediction model."
        ),
    }
    if proposed is not None:
        emi = calculate_emi(proposed, annual_rate, months)
        total_repayment = emi * months
        exceeds = emi > affordable_emi + 1e-10
        remaining = disposable - emi
        result.update({
            "proposed_monthly_emi": emi,
            "proposed_total_repayment": total_repayment,
            "proposed_total_interest": total_repayment - proposed,
            "proposed_emi_exceeds_illustrative_affordable_emi": exceeds,
            "remaining_monthly_income_after_proposed_loan": remaining,
            "assessment": "above_illustrative_limit" if exceeds else "within_illustrative_limit",
            "assessment_message": (
                "The proposed EMI exceeds the illustrative affordability limit. This is a budgeting signal, "
                "not an automatic credit rejection."
                if exceeds else
                "The proposed EMI is within the illustrative affordability limit. This does not guarantee "
                "repayment ability or imply credit approval."
            ),
        })
    return result


def write_demonstration(output_dir=OUTPUT_DIR):
    """Write a clearly synthetic sample assessment and assumptions report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    example = assess_affordability(
        monthly_income=60000,
        existing_monthly_loan_repayments=8000,
        other_recurring_monthly_obligations=12000,
        annual_interest_rate=12,
        tenure_months=36,
        proposed_loan_amount=700000,
    )
    (output_dir / "demonstration_assessment.json").write_text(
        json.dumps({"synthetic_demonstration": True, "assessment": example}, indent=2),
        encoding="utf-8",
    )
    report = [
        "# Affordability engine demonstration", "",
        "This example uses synthetic values and is not a real applicant assessment. All monetary figures are in one unspecified currency unit.", "",
        "## Example result", "",
        f"- Monthly income: {example['monthly_income']:,.2f}",
        f"- Existing loan repayments and other obligations: {example['total_monthly_obligations']:,.2f} per month",
        f"- Disposable income: {example['disposable_income_after_existing_obligations']:,.2f} per month",
        f"- Affordability assumption: {example['affordability_percentage']:.1f}% of nonnegative disposable income",
        f"- Maximum illustrative affordable EMI: {example['maximum_illustrative_affordable_emi']:,.2f}",
        f"- Indicative maximum principal: {example['indicative_maximum_loan_principal']:,.2f}",
        f"- Proposed loan EMI: {example['proposed_monthly_emi']:,.2f}",
        f"- Proposed total repayment / interest: {example['proposed_total_repayment']:,.2f} / {example['proposed_total_interest']:,.2f}",
        f"- Assessment: {example['assessment']}",
        f"- Remaining income after obligations and proposed EMI: {example['remaining_monthly_income_after_proposed_loan']:,.2f} per month", "",
        "## Formula and assumptions", "",
        "For a positive annual rate, monthly rate r = annual percentage / 100 / 12. EMI is P × r × (1+r)^n / ((1+r)^n − 1), evaluated in a numerically stable equivalent form. At zero interest, EMI = P / n. The inverse principal is EMI × (1 − (1+r)^−n) / r; at zero interest, principal = EMI × n.", "",
        "The default illustrative affordability percentage is 30%, configurable when calling `assess_affordability`. It has not been validated as a lending standard. Negative disposable income is preserved in the result, but the affordable EMI is floored at zero. Interest rates are nominal annual percentages from 0% through 100%; tenure must be a positive whole number of months.", "",
        "This rule-based budget estimate does not use the German Credit dataset or the machine-learning model. It is not equivalent to creditworthiness, does not guarantee repayment ability, and does not approve or reject a loan.", "",
    ]
    (output_dir / "affordability_report.md").write_text("\n".join(report), encoding="utf-8")
    return example


def main():
    result = write_demonstration()
    print("Synthetic affordability demonstration:")
    print(json.dumps(result, indent=2))
    print(f"\nReports written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
