"""Prepare loan approval transactions as an instruction-tuning JSONL dataset.

This script converts historical loan application rows into records with the
following fields: instruction, input, and output. The historical decision always
comes from the source loan_status column; helper text is generated with simple,
deterministic rules for demonstration purposes.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split


EXPECTED_COLUMNS = [
    "loan_id",
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
    "loan_status",
]

ASSET_COLUMNS = [
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]

INSTRUCTION = "Analyze the loan application based on the company's historical credit decisions."
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert a loan approval CSV into instruction-tuning JSONL files."
    )
    parser.add_argument(
        "--input",
        default="data/raw/loan_approval_dataset.csv",
        help="Path to the raw loan approval CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory where JSONL files will be written.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of rows to reserve for the test split.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20,
        help="Number of training examples to write to sample.jsonl.",
    )
    return parser.parse_args()


def clean_column_names(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the dataframe with whitespace stripped from column names."""
    cleaned = dataframe.copy()
    cleaned.columns = [column.strip() for column in cleaned.columns]
    return cleaned


def validate_columns(dataframe: pd.DataFrame) -> None:
    """Raise a clear error if the input dataframe does not have every required column."""
    missing_columns = [column for column in EXPECTED_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        expected = ", ".join(EXPECTED_COLUMNS)
        raise ValueError(
            f"Missing required column(s): {missing}. Expected columns are: {expected}"
        )


def normalize_loan_status(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the dataframe with normalized loan_status text."""
    normalized = dataframe.copy()
    normalized["loan_status"] = normalized["loan_status"].astype(str).str.strip()
    return normalized


def format_input(row: pd.Series) -> str:
    """Create a natural language summary of a single loan application row."""
    return (
        f"Applicant has {row['no_of_dependents']} dependents, is {row['education']}, "
        f"self-employed status is {row['self_employed']}, annual income is {row['income_annum']}, "
        f"requested loan amount is {row['loan_amount']}, loan term is {row['loan_term']} months, "
        f"CIBIL score is {row['cibil_score']}, residential assets value is "
        f"{row['residential_assets_value']}, commercial assets value is "
        f"{row['commercial_assets_value']}, luxury assets value is {row['luxury_assets_value']}, "
        f"and bank asset value is {row['bank_asset_value']}."
    )


def safe_number(value: object) -> float:
    """Convert numeric-looking values to float, using 0.0 when conversion fails."""
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return 0.0
    return float(number)


def build_reason(row: pd.Series, decision: str) -> str:
    """Create a deterministic business reason using simple observable signals."""
    cibil_score = safe_number(row["cibil_score"])
    income = safe_number(row["income_annum"])
    loan_amount = safe_number(row["loan_amount"])
    total_assets = sum(safe_number(row[column]) for column in ASSET_COLUMNS)
    loan_to_income = loan_amount / income if income > 0 else float("inf")

    credit_phrase = "strong credit profile" if cibil_score >= 700 else "weaker credit profile"
    income_phrase = "compatible income" if loan_to_income <= 4 else "high loan amount compared with income"
    asset_phrase = "relevant asset values" if total_assets >= loan_amount else "limited asset coverage"

    if decision == "approve":
        return f"The applicant has a {credit_phrase}, {income_phrase}, and {asset_phrase}."

    concerns = []
    if cibil_score < 700:
        concerns.append("credit score below the preferred range")
    if loan_to_income > 4:
        concerns.append("requested loan amount is high compared with annual income")
    if total_assets < loan_amount:
        concerns.append("asset values provide limited coverage for the requested loan")

    if not concerns:
        concerns.append("the historical business decision marked this application as rejected")

    return "The application was rejected because " + "; ".join(concerns) + "."


def build_output(row: pd.Series) -> str:
    """Create the structured output text for one row."""
    status = str(row["loan_status"]).strip().lower()
    decision = "approve" if status == "approved" else "reject"
    reason = build_reason(row, decision)

    if decision == "approve":
        recommended_action = "Proceed with loan approval."
        customer_message = (
            "Your loan application was approved based on the analysis of your financial "
            "profile and credit history."
        )
    else:
        recommended_action = "Do not proceed with loan approval at this time."
        customer_message = (
            "Your loan application was not approved based on the analysis of your financial "
            "profile and credit history."
        )

    return (
        f"Decision: {decision}\n"
        f"Reason: {reason}\n"
        f"Recommended action: {recommended_action}\n"
        f"Customer message: {customer_message}"
    )


def row_to_example(row: pd.Series) -> dict[str, str]:
    """Convert one dataframe row into an instruction-tuning example."""
    return {
        "instruction": INSTRUCTION,
        "input": format_input(row),
        "output": build_output(row),
    }


def dataframe_to_examples(dataframe: pd.DataFrame) -> list[dict[str, str]]:
    """Convert every dataframe row into an instruction-tuning example."""
    return [row_to_example(row) for _, row in dataframe.iterrows()]


def write_jsonl(path: Path, records: Iterable[dict[str, str]]) -> None:
    """Write records to a JSONL file."""
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_dataset(input_path: Path, output_dir: Path, test_size: float, sample_size: int) -> None:
    """Load, validate, transform, split, and write the instruction dataset."""
    dataframe = pd.read_csv(input_path)
    dataframe = clean_column_names(dataframe)
    validate_columns(dataframe)
    dataframe = normalize_loan_status(dataframe)

    logging.info("Loaded %s rows from %s", len(dataframe), input_path)

    train_dataframe, test_dataframe = train_test_split(
        dataframe,
        test_size=test_size,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    train_examples = dataframe_to_examples(train_dataframe)
    test_examples = dataframe_to_examples(test_dataframe)
    sample_examples = train_examples[:sample_size]

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    test_path = output_dir / "test.jsonl"
    sample_path = output_dir / "sample.jsonl"

    write_jsonl(train_path, train_examples)
    write_jsonl(test_path, test_examples)
    write_jsonl(sample_path, sample_examples)

    logging.info("Train split size: %s", len(train_examples))
    logging.info("Test split size: %s", len(test_examples))
    logging.info("Wrote train file to %s", train_path)
    logging.info("Wrote test file to %s", test_path)
    logging.info("Wrote sample file to %s", sample_path)


def main() -> None:
    """Run the dataset preparation command."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    try:
        prepare_dataset(
            input_path=Path(args.input),
            output_dir=Path(args.output_dir),
            test_size=args.test_size,
            sample_size=args.sample_size,
        )
    except Exception as error:
        raise SystemExit(f"Dataset preparation failed: {error}") from error


if __name__ == "__main__":
    main()
