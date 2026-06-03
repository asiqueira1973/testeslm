"""Fine-tune Gemma on the loan approval instruction dataset with LoRA.

This script is intended for GPU notebook or workstation environments such as
Google Colab, Kaggle Notebook, or a local CUDA machine. Gemma models may require
accepting terms on Hugging Face and logging in before model download.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


DEFAULT_MODEL_NAME = "google/gemma-2-2b-it"
DEFAULT_TRAIN_FILE = "data/processed/train.jsonl"
DEFAULT_EVAL_FILE = "data/processed/test.jsonl"
DEFAULT_OUTPUT_DIR = "models/loan-approval-gemma-lora"


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for LoRA training."""
    parser = argparse.ArgumentParser(description="Fine-tune Gemma with LoRA for loan approvals.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Base model on Hugging Face.")
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE, help="Training JSONL file.")
    parser.add_argument("--eval-file", default=DEFAULT_EVAL_FILE, help="Evaluation JSONL file.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for LoRA adapter.")
    parser.add_argument("--num-train-epochs", type=float, default=1, help="Number of training epochs.")
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=1,
        help="Per-device training batch size.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps.",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--max-seq-length", type=int, default=512, help="Maximum sequence length.")
    return parser.parse_args()


def configure_logging() -> None:
    """Configure readable console logging."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def resolve_torch_dtype() -> torch.dtype:
    """Choose a practical dtype for the current machine."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_jsonl_dataset(train_file: str, eval_file: str):
    """Load train and eval splits from JSONL files."""
    LOGGER.info("Loading training data from %s", train_file)
    LOGGER.info("Loading evaluation data from %s", eval_file)
    dataset = load_dataset("json", data_files={"train": train_file, "eval": eval_file})
    LOGGER.info("Loaded %s training rows and %s evaluation rows", len(dataset["train"]), len(dataset["eval"]))
    return dataset


def format_prompt(example: dict[str, str], tokenizer: AutoTokenizer) -> str:
    """Format one dataset example as a single supervised fine-tuning text."""
    user_content = f"{example['instruction']}\n\nInput:\n{example['input']}"
    assistant_content = example["output"]

    if tokenizer.chat_template:
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)

    return f"Instruction:\n{example['instruction']}\n\nInput:\n{example['input']}\n\nResponse:\n{assistant_content}"


def add_training_text(dataset, tokenizer: AutoTokenizer):
    """Add the text field consumed by TRL's SFTTrainer."""
    LOGGER.info("Formatting examples for supervised fine-tuning")
    return dataset.map(lambda example: {"text": format_prompt(example, tokenizer)})


def build_lora_config() -> LoraConfig:
    """Create the LoRA adapter configuration for Gemma-style decoder layers."""
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


def train(args: argparse.Namespace) -> None:
    """Run LoRA supervised fine-tuning and save the adapter."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading tokenizer for %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = add_training_text(load_jsonl_dataset(args.train_file, args.eval_file), tokenizer)

    torch_dtype = resolve_torch_dtype()
    LOGGER.info("Loading base model %s with dtype %s", args.model_name, torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.config.use_cache = False

    LOGGER.info("Configuring LoRA adapter")
    peft_config = build_lora_config()

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=torch_dtype == torch.bfloat16,
        fp16=torch_dtype == torch.float16,
        max_length=args.max_seq_length,
        dataset_text_field="text",
        packing=False,
        report_to="none",
    )

    LOGGER.info("Starting LoRA fine-tuning")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()

    LOGGER.info("Saving LoRA adapter to %s", output_dir)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    LOGGER.info("Training complete")


def main() -> None:
    """CLI entrypoint."""
    configure_logging()
    train(parse_args())


if __name__ == "__main__":
    main()
