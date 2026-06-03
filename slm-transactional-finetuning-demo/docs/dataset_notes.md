# Dataset notes

## Original dataset columns

The raw CSV is expected to contain these columns:

| Column | Description |
| --- | --- |
| `loan_id` | Unique identifier for the historical loan application. |
| `no_of_dependents` | Number of dependents reported by the applicant. |
| `education` | Applicant education category. |
| `self_employed` | Whether the applicant is self-employed. |
| `income_annum` | Annual income reported for the applicant. |
| `loan_amount` | Requested loan amount. |
| `loan_term` | Requested loan term in months. |
| `cibil_score` | Applicant credit score. |
| `residential_assets_value` | Value of residential assets. |
| `commercial_assets_value` | Value of commercial assets. |
| `luxury_assets_value` | Value of luxury assets. |
| `bank_asset_value` | Value of bank assets. |
| `loan_status` | Historical business decision for the application. |

## Target column

The target column is `loan_status`. It represents the historical business decision and is converted into the `Decision` line in the output text.

The preparation script does not override this decision. If a row is marked `Approved`, the generated decision is `approve`. Any other status is treated as `reject` for this first demo version.

## Input fields

All non-target business fields are summarized in natural language. The generated input includes:

- dependents
- education
- self-employment status
- annual income
- requested loan amount
- loan term
- CIBIL score
- residential asset value
- commercial asset value
- luxury asset value
- bank asset value

## Derived explanatory signals

The generated `Reason`, `Recommended action`, and `Customer message` use deterministic derived signals from the application row. These signals are used only to make the generated text richer and more consistent with the applicant profile; they do not change the historical decision from `loan_status`.

| Signal | Definition | Categories |
| --- | --- | --- |
| `total_assets` | `residential_assets_value + commercial_assets_value + luxury_assets_value + bank_asset_value` | Numeric total asset value. |
| `loan_to_income_ratio` | `loan_amount / income_annum` | Used to classify affordability. |
| `loan_to_assets_ratio` | `loan_amount / total_assets` | Used to classify asset coverage. |
| `credit_profile` | Based on `cibil_score`. | `strong` if score is at least 750; `acceptable` from 650 through 749; `moderate` from 550 through 649; `weak` below 550. |
| `affordability_profile` | Based on `loan_to_income_ratio`. | `comfortable` at 3.0 or below; `stretched` above 3.0 and up to 5.0; `high_risk` above 5.0. |
| `asset_coverage` | Based on `loan_to_assets_ratio`. | `strong` at 0.5 or below; `moderate` above 0.5 and up to 0.8; `weak` above 0.8. |

The generated reasons combine multiple signals instead of relying on a single generic statement. For example, an approved row with a strong credit profile can mention credit strength, affordability, and asset coverage together. A rejected row can explain that a weak credit profile, high loan-to-income ratio, limited asset coverage, or the combined historical pattern drove the generated explanation.

## Output format

Each generated JSONL record has three top-level fields:

```json
{
  "instruction": "Analyze the loan application based on the company's historical credit decisions.",
  "input": "Natural language summary of the application.",
  "output": "Structured text response."
}
```

The `output` field is a structured text block, not nested JSON:

```text
Decision: approve
Reason: The application was approved because the applicant has a strong credit score, comfortable loan affordability, and strong asset coverage.
Recommended action: Proceed with approval and standard documentation review.
Customer message: Your loan application was approved based on your credit profile and the overall financial analysis of the application.
```

## Limitations

This project uses a public dataset for demonstration. Public data is useful for showing the workflow, but it has limitations:

- it may not reflect a real company's underwriting policy;
- it may not contain all features used in real credit decisions;
- generated reasons are deterministic demo text, not audited business explanations;
- the historical target may include biases or simplified assumptions;
- the dataset should not be used to make real lending decisions.

The project should be treated as an educational example of dataset transformation, not a production credit decisioning system.
