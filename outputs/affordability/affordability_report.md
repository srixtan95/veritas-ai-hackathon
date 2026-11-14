# Affordability engine demonstration

This example uses synthetic values and is not a real applicant assessment. All monetary figures are in one unspecified currency unit.

## Example result

- Monthly income: 60,000.00
- Existing loan repayments and other obligations: 20,000.00 per month
- Disposable income: 40,000.00 per month
- Affordability assumption: 30.0% of nonnegative disposable income
- Maximum illustrative affordable EMI: 12,000.00
- Indicative maximum principal: 361,290.06
- Proposed loan EMI: 23,250.02
- Proposed total repayment / interest: 837,000.61 / 137,000.61
- Assessment: above_illustrative_limit
- Remaining income after obligations and proposed EMI: 16,749.98 per month

## Formula and assumptions

For a positive annual rate, monthly rate r = annual percentage / 100 / 12. EMI is P × r × (1+r)^n / ((1+r)^n − 1), evaluated in a numerically stable equivalent form. At zero interest, EMI = P / n. The inverse principal is EMI × (1 − (1+r)^−n) / r; at zero interest, principal = EMI × n.

The default illustrative affordability percentage is 30%, configurable when calling `assess_affordability`. It has not been validated as a lending standard. Negative disposable income is preserved in the result, but the affordable EMI is floored at zero. Interest rates are nominal annual percentages from 0% through 100%; tenure must be a positive whole number of months.

This rule-based budget estimate does not use the German Credit dataset or the machine-learning model. It is not equivalent to creditworthiness, does not guarantee repayment ability, and does not approve or reject a loan.
