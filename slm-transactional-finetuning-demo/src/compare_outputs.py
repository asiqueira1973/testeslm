"""Compare base-model and LoRA-adapter outputs for sample loan examples."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from inference import DEFAULT_MODEL_NAME, generate_response, load_model


DEFAULT_SAMPLE_FILE = "data/processed/sample.jsonl"
DEFAULT_ADAPTER_DIR = "models/loan-approval-gemma-lora"
DEFAULT_OUTPUT_FILE = "outputs/comparison.md"
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for output comparison."""
    parser = argparse.ArgumentParser(description="Compare base and fine-tuned outputs.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Base model on Hugging Face.")
    parser.add_argument("--sample-file", default=DEFAULT_SAMPLE_FILE, help="Sample JSONL examples.")
    parser.add_argument("--adapter-dir", default=DEFAULT_ADAPTER_DIR, help="LoRA adapter directory.")
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE, help="Markdown comparison output.")
    parser.add_argument("--num-examples", type=int, default=3, help="Number of examples to compare.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum generated tokens.")
    return parser.parse_args()


def configure_logging() -> None:
    """Configure readable console logging."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def load_examples(sample_file: str, num_examples: int) -> list[dict[str, str]]:
    """Load a small set of JSONL examples."""
    examples = []
    with Path(sample_file).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                examples.append(json.loads(line))
            if len(examples) >= num_examples:
                break
    return examples


def markdown_block(text: str) -> str:
    """Wrap text in a Markdown code fence."""
    return f"```text\n{text.strip()}\n```"


def write_comparison(
    output_file: str,
    examples: list[dict[str, str]],
    base_outputs: list[str],
    tuned_outputs: list[str] | None,
    adapter_dir: str,
) -> None:
    """Write comparison results to a Markdown file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# Base vs. LoRA Fine-Tuned Output Comparison", ""]
    if tuned_outputs is None:
        lines.extend(
            [
                f"Fine-tuned comparison skipped because `{adapter_dir}` does not exist.",
                "Run `python src/train_lora.py` first to create the adapter.",
                "",
            ]
        )

    for index, example in enumerate(examples, start=1):
        lines.extend(
            [
                f"## Example {index}",
                "",
                "### Instruction",
                markdown_block(example["instruction"]),
                "",
                "### Input",
                markdown_block(example["input"]),
                "",
                "### Expected historical output",
                markdown_block(example["output"]),
                "",
                "### Base model output",
                markdown_block(base_outputs[index - 1]),
                "",
            ]
        )
        if tuned_outputs is not None:
            lines.extend(["### Fine-tuned LoRA output", markdown_block(tuned_outputs[index - 1]), ""])

    output_path.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Wrote comparison to %s", output_path)


def main() -> None:
    """CLI entrypoint."""
    configure_logging()
    args = parse_args()
    examples = load_examples(args.sample_file, args.num_examples)

    LOGGER.info("Running base model inference for %s examples", len(examples))
    base_model, base_tokenizer = load_model(args.model_name)
    base_outputs = [
        generate_response(
            base_model,
            base_tokenizer,
            example["instruction"],
            example["input"],
            args.max_new_tokens,
        )
        for example in examples
    ]

    tuned_outputs = None
    if Path(args.adapter_dir).exists():
        LOGGER.info("Running fine-tuned adapter inference from %s", args.adapter_dir)
        tuned_model, tuned_tokenizer = load_model(args.model_name, args.adapter_dir)
        tuned_outputs = [
            generate_response(
                tuned_model,
                tuned_tokenizer,
                example["instruction"],
                example["input"],
                args.max_new_tokens,
            )
            for example in examples
        ]
    else:
        LOGGER.warning("Adapter directory %s does not exist; skipping fine-tuned comparison", args.adapter_dir)

    write_comparison(args.output_file, examples, base_outputs, tuned_outputs, args.adapter_dir)


if __name__ == "__main__":
    main()
