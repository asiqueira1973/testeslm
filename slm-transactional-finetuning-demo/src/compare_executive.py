"""Generate an executive comparison across base Gemma, LoRA Gemma, and manual public LLM outputs."""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path
from typing import Any


DEFAULT_MODEL_NAME = "google/gemma-2-2b-it"


DEFAULT_CASES_FILE = "data/evaluation/portuguese_cases.jsonl"
DEFAULT_PUBLIC_OUTPUTS_FILE = "data/evaluation/public_llm_outputs.jsonl"
DEFAULT_ADAPTER_DIR = "models/loan-approval-gemma-lora"
DEFAULT_OUTPUT_FILE = "outputs/executive_comparison.md"
REQUIRED_CASE_FIELDS = {
    "case_id",
    "portuguese_question",
    "normalized_instruction",
    "normalized_input",
    "expected_output",
}
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the executive comparison workflow."""
    parser = argparse.ArgumentParser(
        description="Compare base Gemma, fine-tuned Gemma LoRA, and optional manual public LLM outputs."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Base model on Hugging Face.")
    parser.add_argument("--cases-file", default=DEFAULT_CASES_FILE, help="Portuguese business cases JSONL file.")
    parser.add_argument("--adapter-dir", default=DEFAULT_ADAPTER_DIR, help="Fine-tuned LoRA adapter directory.")
    parser.add_argument(
        "--public-outputs-file",
        default=DEFAULT_PUBLIC_OUTPUTS_FILE,
        help="Optional JSONL file containing manually pasted public LLM outputs.",
    )
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE, help="Markdown evidence report output path.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum generated tokens per model response.")
    return parser.parse_args()


def configure_logging() -> None:
    """Configure readable console logging."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load records from a JSONL file, ignoring blank lines."""
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path} on line {line_number}: {error}") from error
    return records


def load_cases(cases_file: str) -> list[dict[str, str]]:
    """Load and validate Portuguese executive comparison cases."""
    path = Path(cases_file)
    cases = load_jsonl(path)
    if not cases:
        raise ValueError(f"No cases found in {path}")

    validated_cases = []
    for index, case in enumerate(cases, start=1):
        missing_fields = REQUIRED_CASE_FIELDS.difference(case)
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"Case {index} in {path} is missing required field(s): {missing}")
        validated_cases.append({field: str(case[field]) for field in REQUIRED_CASE_FIELDS})
    return validated_cases


def load_public_outputs(public_outputs_file: str) -> dict[str, dict[str, str]]:
    """Load optional manually pasted public LLM outputs keyed by case_id."""
    path = Path(public_outputs_file)
    if not path.exists():
        LOGGER.info("Public LLM outputs file %s not found; continuing without public outputs", path)
        return {}

    public_outputs = {}
    for record in load_jsonl(path):
        case_id = str(record.get("case_id", "")).strip()
        output = str(record.get("public_llm_output", "")).strip()
        if not case_id or not output:
            continue
        public_outputs[case_id] = {
            "public_llm_name": str(record.get("public_llm_name", "Public LLM/manual")).strip()
            or "Public LLM/manual",
            "public_llm_output": output,
        }
    LOGGER.info("Loaded %s manual public LLM output(s) from %s", len(public_outputs), path)
    return public_outputs


def run_model_outputs(
    cases: list[dict[str, str]],
    model_name: str,
    max_new_tokens: int,
    adapter_dir: str | None = None,
) -> list[str]:
    """Run deterministic inference for all cases with the selected model setup."""
    import torch

    from inference import generate_response, load_model

    model, tokenizer = load_model(model_name, adapter_dir)
    outputs = []
    for case in cases:
        LOGGER.info("Generating response for %s", case["case_id"])
        outputs.append(
            generate_response(
                model=model,
                tokenizer=tokenizer,
                instruction=case["normalized_instruction"],
                input_text=case["normalized_input"],
                max_new_tokens=max_new_tokens,
            )
        )
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs


def markdown_block(text: str) -> str:
    """Wrap text in a Markdown code fence suitable for business evidence."""
    clean_text = text.strip() or "Not provided."
    return f"```text\n{clean_text}\n```"


def build_observation(has_tuned_output: bool, has_public_output: bool) -> str:
    """Return a short, honest observation prompt for each comparison case."""
    public_note = "Manual public LLM output is included for side-by-side review." if has_public_output else "No manual public LLM output was provided for this case."
    tuned_note = (
        "Review whether the fine-tuned response follows the trained operational fields and decision style."
        if has_tuned_output
        else "Fine-tuned output was not generated because the adapter directory was unavailable."
    )
    return f"{tuned_note} {public_note} This comparison demonstrates response-format specialization, not general model superiority."


def write_report(
    output_file: str,
    cases: list[dict[str, str]],
    base_outputs: list[str],
    tuned_outputs: list[str] | None,
    public_outputs: dict[str, dict[str, str]],
    model_name: str,
    adapter_dir: str,
    public_outputs_file: str,
) -> None:
    """Write the executive Markdown comparison report."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Executive Comparison: Base Gemma vs. Fine-Tuned Gemma LoRA vs. Public LLM",
        "",
        "This report compares how the same Portuguese business cases are handled by:",
        "",
        "1. The base Gemma instruction model.",
        "2. The same Gemma model with the loan-approval LoRA adapter, when available.",
        "3. A manually pasted public LLM response, when provided by the user.",
        "",
        "The purpose is to show behavior specialization and trained operational response format. It does not claim that the fine-tuned SLM is generally better than public LLMs.",
        "",
        "## Run configuration",
        "",
        f"- Base model: `{model_name}`",
        f"- LoRA adapter directory: `{adapter_dir}`",
        f"- Manual public LLM file: `{public_outputs_file}`",
        "",
    ]

    if tuned_outputs is None:
        lines.extend(
            [
                "> Fine-tuned model output was skipped because the LoRA adapter directory was not found.",
                "> Run the fine-tuning workflow first or pass `--adapter-dir` pointing to the trained adapter.",
                "",
            ]
        )

    if not public_outputs:
        lines.extend(
            [
                "> No manual public LLM outputs were loaded. Copy `data/evaluation/public_llm_outputs.template.jsonl` to `data/evaluation/public_llm_outputs.jsonl` and paste responses to include this column.",
                "",
            ]
        )

    for index, case in enumerate(cases, start=1):
        case_id = case["case_id"]
        public_record = public_outputs.get(case_id)
        tuned_output = tuned_outputs[index - 1] if tuned_outputs is not None else "Fine-tuned output not generated."
        public_heading = "Public LLM output"
        public_output = "Not provided."
        if public_record:
            public_heading = f"Public LLM output ({public_record['public_llm_name']})"
            public_output = public_record["public_llm_output"]

        lines.extend(
            [
                f"## Case {index}: `{case_id}`",
                "",
                "### Portuguese question",
                markdown_block(case["portuguese_question"]),
                "",
                "### Normalized input sent to the model",
                markdown_block(case["normalized_input"]),
                "",
                "### Expected historical output",
                markdown_block(case["expected_output"]),
                "",
                "### Base model output",
                markdown_block(base_outputs[index - 1]),
                "",
                "### Fine-tuned model output",
                markdown_block(tuned_output),
                "",
                f"### {public_heading}",
                markdown_block(public_output),
                "",
                "### Observation",
                build_observation(tuned_outputs is not None, public_record is not None),
                "",
            ]
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Wrote executive comparison report to %s", output_path)


def main() -> None:
    """CLI entrypoint."""
    configure_logging()
    args = parse_args()
    cases = load_cases(args.cases_file)
    public_outputs = load_public_outputs(args.public_outputs_file)

    LOGGER.info("Running base model inference for %s case(s)", len(cases))
    base_outputs = run_model_outputs(cases, args.model_name, args.max_new_tokens)

    tuned_outputs = None
    if Path(args.adapter_dir).exists():
        LOGGER.info("Running fine-tuned LoRA inference for %s case(s)", len(cases))
        tuned_outputs = run_model_outputs(cases, args.model_name, args.max_new_tokens, args.adapter_dir)
    else:
        LOGGER.warning("Adapter directory %s does not exist; skipping fine-tuned inference", args.adapter_dir)

    write_report(
        output_file=args.output_file,
        cases=cases,
        base_outputs=base_outputs,
        tuned_outputs=tuned_outputs,
        public_outputs=public_outputs,
        model_name=args.model_name,
        adapter_dir=args.adapter_dir,
        public_outputs_file=args.public_outputs_file,
    )


if __name__ == "__main__":
    main()
