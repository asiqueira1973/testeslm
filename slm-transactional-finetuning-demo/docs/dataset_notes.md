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
Reason: The applicant has a strong credit profile, compatible income, and relevant asset values.
Recommended action: Proceed with loan approval.
Customer message: Your loan application was approved based on the analysis of your financial profile and credit history.
```

## Limitations

This project uses a public dataset for demonstration. Public data is useful for showing the workflow, but it has limitations:

- it may not reflect a real company's underwriting policy;
- it may not contain all features used in real credit decisions;
- generated reasons are deterministic demo text, not audited business explanations;
- the historical target may include biases or simplified assumptions;
- the dataset should not be used to make real lending decisions.

The project should be treated as an educational example of dataset transformation, not a production credit decisioning system.
