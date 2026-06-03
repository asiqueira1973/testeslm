"""Prepare loan approval transactions as an instruction-tuning JSONL dataset.

This script converts historical loan application rows into records with the
following fields: instruction, input, and output. The historical decision always
comes from the source loan_status column; helper text is generated with richer,
deterministic rules for demonstration purposes.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
from pathlib import Path
from typing import Iterable


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
SHORT_LOAN_TERM_MONTHS = 36


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


def read_csv_rows(input_path: Path) -> list[dict[str, str]]:
    """Read the raw CSV and strip whitespace from headers and string values."""
    with input_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("Input CSV does not contain a header row.")

        cleaned_fieldnames = [field.strip() for field in reader.fieldnames]
        rows = []
        for raw_row in reader:
            cleaned_row = {}
            for original, cleaned in zip(reader.fieldnames, cleaned_fieldnames):
                value = raw_row.get(original, "")
                cleaned_row[cleaned] = value.strip() if isinstance(value, str) else str(value)
            rows.append(cleaned_row)

    validate_columns(cleaned_fieldnames)
    return normalize_loan_status(rows)


def validate_columns(columns: Iterable[str]) -> None:
    """Raise a clear error if the input columns do not include every required column."""
    column_set = set(columns)
    missing_columns = [column for column in EXPECTED_COLUMNS if column not in column_set]
    if missing_columns:
        missing = ", ".join(missing_columns)
        expected = ", ".join(EXPECTED_COLUMNS)
        raise ValueError(
            f"Missing required column(s): {missing}. Expected columns are: {expected}"
        )


def normalize_loan_status(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return rows with normalized loan_status text."""
    normalized = []
    for row in rows:
        normalized_row = row.copy()
        normalized_row["loan_status"] = str(normalized_row["loan_status"]).strip()
        normalized.append(normalized_row)
    return normalized


def format_input(row: dict[str, str]) -> str:
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
    try:
        return float(str(value).strip())
    except ValueError:
        return 0.0


def format_ratio(value: float) -> str:
    """Format a ratio for stable, human-readable output text."""
    if value == float("inf"):
        return "unavailable"
    return f"{value:.2f}"


def classify_credit_profile(cibil_score: float) -> str:
    """Classify credit strength from CIBIL score."""
    if cibil_score >= 750:
        return "strong"
    if cibil_score >= 650:
        return "acceptable"
    if cibil_score >= 550:
        return "moderate"
    return "weak"


def classify_affordability_profile(loan_to_income_ratio: float) -> str:
    """Classify affordability from loan amount relative to annual income."""
    if loan_to_income_ratio <= 3:
        return "comfortable"
    if loan_to_income_ratio <= 5:
        return "stretched"
    return "high_risk"


def classify_asset_coverage(loan_to_assets_ratio: float) -> str:
    """Classify asset coverage from loan amount relative to total assets."""
    if loan_to_assets_ratio <= 0.5:
        return "strong"
    if loan_to_assets_ratio <= 0.8:
        return "moderate"
    return "weak"


def derive_signals(row: dict[str, str]) -> dict[str, float | str | bool]:
    """Derive deterministic explanatory signals from raw application fields."""
    cibil_score = safe_number(row["cibil_score"])
    income = safe_number(row["income_annum"])
    loan_amount = safe_number(row["loan_amount"])
    loan_term = safe_number(row["loan_term"])
    total_assets = sum(safe_number(row[column]) for column in ASSET_COLUMNS)
    loan_to_income_ratio = loan_amount / income if income > 0 else float("inf")
    loan_to_assets_ratio = loan_amount / total_assets if total_assets > 0 else float("inf")

    return {
        "cibil_score": cibil_score,
        "income": income,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "total_assets": total_assets,
        "loan_to_income_ratio": loan_to_income_ratio,
        "loan_to_assets_ratio": loan_to_assets_ratio,
        "credit_profile": classify_credit_profile(cibil_score),
        "affordability_profile": classify_affordability_profile(loan_to_income_ratio),
        "asset_coverage": classify_asset_coverage(loan_to_assets_ratio),
        "has_short_term": loan_term <= SHORT_LOAN_TERM_MONTHS,
    }


def credit_description(profile: str) -> str:
    """Return a readable credit profile phrase."""
    descriptions = {
        "strong": "a strong credit score",
        "acceptable": "an acceptable credit score",
        "moderate": "a moderate credit score",
        "weak": "a weaker credit score",
    }
    return descriptions[profile]


def affordability_description(profile: str) -> str:
    """Return a readable affordability phrase."""
    descriptions = {
        "comfortable": "comfortable loan affordability",
        "stretched": "stretched but still measurable affordability",
        "high_risk": "a high requested amount relative to annual income",
    }
    return descriptions[profile]


def asset_description(profile: str) -> str:
    """Return a readable asset coverage phrase."""
    descriptions = {
        "strong": "strong asset coverage",
        "moderate": "moderate asset coverage",
        "weak": "limited asset coverage",
    }
    return descriptions[profile]


def support_factors(signals: dict[str, float | str | bool]) -> list[str]:
    """Return positive support factors in deterministic priority order."""
    factors = []
    if signals["credit_profile"] in {"strong", "acceptable"}:
        factors.append(credit_description(str(signals["credit_profile"])))
    if signals["affordability_profile"] == "comfortable":
        factors.append("income that is compatible with the requested loan amount")
    elif signals["affordability_profile"] == "stretched":
        factors.append("affordability that remains within the historical review range")
    if signals["asset_coverage"] in {"strong", "moderate"}:
        factors.append(asset_description(str(signals["asset_coverage"])))
    if signals["has_short_term"]:
        factors.append("a shorter requested loan term")
    return factors


def concern_factors(signals: dict[str, float | str | bool]) -> list[str]:
    """Return risk factors in deterministic priority order."""
    factors = []
    if signals["credit_profile"] == "weak":
        factors.append("a low credit score, which indicates higher repayment risk")
    elif signals["credit_profile"] == "moderate":
        factors.append("a moderate credit score that does not provide strong credit support")
    if signals["affordability_profile"] == "high_risk":
        factors.append("the requested loan amount is high relative to the applicant's annual income")
    elif signals["affordability_profile"] == "stretched":
        factors.append("loan affordability is stretched compared with annual income")
    if signals["asset_coverage"] == "weak":
        factors.append("available assets provide limited coverage for the requested amount")
    return factors


def join_phrases(phrases: list[str]) -> str:
    """Join explanatory phrases in a natural deterministic form."""
    if not phrases:
        return "overall financial analysis"
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return " and ".join(phrases)
    return ", ".join(phrases[:-1]) + ", and " + phrases[-1]


def strongest_signal(signals: dict[str, float | str | bool]) -> str:
    """Choose the strongest approval signal in deterministic priority order."""
    if signals["credit_profile"] == "strong":
        return "credit"
    if signals["asset_coverage"] == "strong":
        return "assets"
    if signals["affordability_profile"] == "comfortable":
        return "income"
    if signals["credit_profile"] == "acceptable":
        return "credit"
    return "overall"


def weakest_signal(signals: dict[str, float | str | bool]) -> str:
    """Choose the weakest rejection signal in deterministic priority order."""
    if signals["affordability_profile"] == "high_risk":
        return "affordability"
    if signals["credit_profile"] == "weak":
        return "credit"
    if signals["asset_coverage"] == "weak":
        return "assets"
    if signals["credit_profile"] == "moderate":
        return "credit"
    return "overall"


def build_approved_reason(signals: dict[str, float | str | bool]) -> str:
    """Create a multi-signal reason for an approved historical decision."""
    credit = str(signals["credit_profile"])
    affordability = str(signals["affordability_profile"])
    assets = str(signals["asset_coverage"])
    support = support_factors(signals)
    concerns = concern_factors(signals)

    if credit == "strong":
        return (
            "The application was approved because the applicant has "
            f"{credit_description(credit)}, {affordability_description(affordability)}, "
            f"and {asset_description(assets)}."
        )
    if credit == "weak":
        fallback_support = support or [
            "short loan term, income profile, or available assets"
            if signals["has_short_term"]
            else "income profile or available assets"
        ]
        return (
            "The application was approved despite a weaker credit score because the requested "
            f"amount is supported by {join_phrases(fallback_support)}."
        )
    if concerns:
        return (
            "The application was approved after balancing "
            f"{join_phrases(support)} against {join_phrases(concerns)}, consistent with the "
            "company's historical approval pattern."
        )
    return (
        "The application was approved because "
        f"{join_phrases(support)} collectively support the requested loan."
    )


def build_rejected_reason(signals: dict[str, float | str | bool]) -> str:
    """Create a multi-signal reason for a rejected historical decision."""
    credit = str(signals["credit_profile"])
    affordability = str(signals["affordability_profile"])
    assets = str(signals["asset_coverage"])
    concerns = concern_factors(signals)

    if credit == "weak":
        additional_concerns = [factor for factor in concerns[1:]]
        if additional_concerns:
            return (
                "The application was rejected because the applicant has a low credit score, "
                "which indicates higher repayment risk, and "
                f"{join_phrases(additional_concerns)}."
            )
        return (
            "The application was rejected because the applicant has a low credit score, "
            "which indicates higher repayment risk."
        )
    if affordability == "high_risk":
        return (
            "The application was rejected because the requested loan amount is high relative "
            f"to the applicant's annual income, with a loan-to-income ratio of "
            f"{format_ratio(float(signals['loan_to_income_ratio']))}."
        )
    if assets == "strong" and concerns:
        return (
            "The application was rejected despite relevant asset values because the credit "
            "profile and affordability indicators did not meet the company's historical "
            "approval pattern."
        )
    if concerns:
        return (
            "The application was rejected because "
            f"{join_phrases(concerns)} did not align with the company's historical approval pattern."
        )
    return (
        "The application was rejected even though no single indicator is severely weak; the "
        "combined credit, affordability, and asset profile matched historical rejected cases."
    )


def build_reason(row: dict[str, str], decision: str) -> str:
    """Create a deterministic business reason using derived observable signals."""
    signals = derive_signals(row)
    if decision == "approve":
        return build_approved_reason(signals)
    return build_rejected_reason(signals)


def rejection_focus(signals: dict[str, float | str | bool]) -> str:
    """Choose a deterministic customer/recommendation focus for rejected rows."""
    if signals["affordability_profile"] == "high_risk":
        return "affordability"
    if signals["credit_profile"] == "weak" and signals["affordability_profile"] == "stretched":
        return "credit_affordability"
    if signals["credit_profile"] == "weak":
        return "credit"
    if signals["asset_coverage"] == "weak":
        return "assets"
    return "overall"


def build_recommended_action(decision: str, signals: dict[str, float | str | bool]) -> str:
    """Create a deterministic recommended action aligned to the historical decision."""
    if decision == "approve":
        signal = strongest_signal(signals)
        actions = {
            "credit": "Proceed with approval and standard documentation review.",
            "assets": "Proceed with approval while documenting the asset support for the file.",
            "income": "Proceed with approval and confirm routine income verification steps.",
            "overall": "Proceed with approval based on the overall historical decision pattern.",
        }
        return actions[signal]

    focus = rejection_focus(signals)
    actions = {
        "credit": "Do not proceed with approval at this time; prioritize credit-strengthening guidance.",
        "credit_affordability": "Do not proceed with approval at this time; review both credit indicators and loan affordability before reapplication.",
        "affordability": "Do not proceed with approval at this time; review affordability or a lower requested amount.",
        "assets": "Do not proceed with approval at this time; reassess collateral or asset coverage if the applicant reapplies.",
        "overall": "Do not proceed with approval at this time; retain the application for historical review context.",
    }
    return actions[focus]


def build_customer_message(decision: str, signals: dict[str, float | str | bool]) -> str:
    """Create a varied deterministic customer-facing message."""
    if decision == "approve":
        signal = strongest_signal(signals)
        messages = {
            "credit": (
                "Your loan application was approved based on your credit profile and the "
                "overall financial analysis of the application."
            ),
            "assets": (
                "Your loan application was approved because the review found sufficient "
                "asset coverage along with supportive financial indicators."
            ),
            "income": (
                "Your loan application was approved based on income compatibility with the "
                "requested loan amount and the company's historical review pattern."
            ),
            "overall": (
                "Your loan application was approved after reviewing your credit indicators, "
                "income profile, asset coverage, and requested loan terms together."
            ),
        }
        return messages[signal]

    focus = rejection_focus(signals)
    messages = {
        "credit": (
            "Your loan application was not approved based on the current credit indicators "
            "and overall financial profile."
        ),
        "credit_affordability": (
            "Your loan application was not approved after reviewing the current credit "
            "indicators together with the loan affordability profile."
        ),
        "affordability": (
            "Your loan application was not approved because the current loan affordability "
            "profile did not align with the historical approval pattern."
        ),
        "assets": (
            "Your loan application was not approved based on the current asset coverage and "
            "related financial indicators."
        ),
        "overall": (
            "Your loan application was not approved after reviewing the current financial "
            "profile, credit indicators, and requested loan terms together."
        ),
    }
    return messages[focus]


def build_output(row: dict[str, str]) -> str:
    """Create the structured output text for one row."""
    status = str(row["loan_status"]).strip().lower()
    decision = "approve" if status == "approved" else "reject"
    signals = derive_signals(row)
    reason = build_approved_reason(signals) if decision == "approve" else build_rejected_reason(signals)
    recommended_action = build_recommended_action(decision, signals)
    customer_message = build_customer_message(decision, signals)

    return (
        f"Decision: {decision}\n"
        f"Reason: {reason}\n"
        f"Recommended action: {recommended_action}\n"
        f"Customer message: {customer_message}"
    )


def row_to_example(row: dict[str, str]) -> dict[str, str]:
    """Convert one dataframe row into an instruction-tuning example."""
    return {
        "instruction": INSTRUCTION,
        "input": format_input(row),
        "output": build_output(row),
    }


def rows_to_examples(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Convert every row into an instruction-tuning example."""
    return [row_to_example(row) for row in rows]


def write_jsonl(path: Path, records: Iterable[dict[str, str]]) -> None:
    """Write records to a JSONL file."""
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def split_rows(
    rows: list[dict[str, str]], test_size: float, random_state: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Create deterministic train and test splits without changing row contents."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1.")

    row_indices = list(range(len(rows)))
    random.Random(random_state).shuffle(row_indices)
    test_count = math.ceil(len(rows) * test_size)
    test_indices = row_indices[:test_count]
    train_indices = row_indices[test_count:]

    train_rows = [rows[index] for index in train_indices]
    test_rows = [rows[index] for index in test_indices]
    return train_rows, test_rows


def prepare_dataset(input_path: Path, output_dir: Path, test_size: float, sample_size: int) -> None:
    """Load, validate, transform, split, and write the instruction dataset."""
    rows = read_csv_rows(input_path)

    logging.info("Loaded %s rows from %s", len(rows), input_path)

    train_rows, test_rows = split_rows(rows, test_size, RANDOM_STATE)

    train_examples = rows_to_examples(train_rows)
    test_examples = rows_to_examples(test_rows)
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
