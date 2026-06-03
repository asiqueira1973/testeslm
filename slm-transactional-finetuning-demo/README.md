# SLM Transactional Fine-Tuning Demo

## Project goal

This project demonstrates how historical transactional business data can be converted into an instruction-tuning dataset for Small Language Model (SLM) fine-tuning.

The first version focuses only on repository setup and dataset preparation. It does not fine-tune a model. The goal is to make the data transformation simple, reproducible, and easy to run locally or in Google Colab.

## Dataset description

The demo uses a loan approval dataset where each row represents one historical loan application. The applicant and loan request fields are treated as the model input, and the historical `loan_status` field is treated as the business decision to be learned.

Expected columns:

- `loan_id`
- `no_of_dependents`
- `education`
- `self_employed`
- `income_annum`
- `loan_amount`
- `loan_term`
- `cibil_score`
- `residential_assets_value`
- `commercial_assets_value`
- `luxury_assets_value`
- `bank_asset_value`
- `loan_status`

## Setup instructions

Create and activate a Python environment, then install the minimal dependencies:

```bash
pip install -r requirements.txt
```

The project intentionally uses only:

- `pandas`
- `scikit-learn`

## Add the raw CSV

Place the source CSV at:

```text
data/raw/loan_approval_dataset.csv
```

The preparation script expects the file to contain all required columns listed above. Column names may contain leading or trailing spaces; the script strips whitespace before validation.

## Run the preparation script

From the project root, run:

```bash
python src/prepare_dataset.py --input data/raw/loan_approval_dataset.csv --output-dir data/processed --test-size 0.2 --sample-size 20
```

The script logs:

- number of rows loaded
- train and test split sizes
- generated output file paths

## Generated files

The script writes three JSONL files:

- `data/processed/train.jsonl`
- `data/processed/test.jsonl`
- `data/processed/sample.jsonl`

Each JSONL line has this structure:

```json
{"instruction": "...", "input": "...", "output": "..."}
```

The `instruction` is fixed:

```text
Analyze the loan application based on the company's historical credit decisions.
```

The `input` is a natural language summary of the loan application. The `output` is a structured text response with:

- `Decision`
- `Reason`
- `Recommended action`
- `Customer message`

The historical decision always comes from `loan_status`. Simple deterministic rules use credit score, income, requested loan amount, and asset values to generate supporting explanation text, but those rules never override the original historical decision.

## Why this matters

Many companies already have transactional business history: applications, orders, claims, support tickets, approvals, rejections, reviews, and outcomes. This project shows the first step in turning that history into an instruction dataset that can later be used to fine-tune an SLM to reproduce or explain business decisions in a controlled demo setting.
