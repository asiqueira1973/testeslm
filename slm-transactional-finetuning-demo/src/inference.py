"""Run inference with the base Gemma model and an optional LoRA adapter."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_NAME = "google/gemma-2-2b-it"
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for inference."""
    parser = argparse.ArgumentParser(description="Generate a loan approval response.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Base model on Hugging Face.")
    parser.add_argument("--adapter-dir", default=None, help="Optional LoRA adapter directory.")
    parser.add_argument("--instruction", required=True, help="Instruction for the model.")
    parser.add_argument("--input", dest="input_text", required=True, help="Loan application input text.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum generated tokens.")
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


def build_prompt(instruction: str, input_text: str, tokenizer: AutoTokenizer) -> str:
    """Build an inference prompt matching the training format."""
    user_content = f"{instruction}\n\nInput:\n{input_text}"
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"Instruction:\n{instruction}\n\nInput:\n{input_text}\n\nResponse:\n"


def load_model(model_name: str, adapter_dir: str | None = None):
    """Load the base model and, when present, attach a LoRA adapter."""
    torch_dtype = resolve_torch_dtype()
    LOGGER.info("Loading tokenizer for %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    LOGGER.info("Loading base model %s", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if adapter_dir:
        adapter_path = Path(adapter_dir)
        if adapter_path.exists():
            LOGGER.info("Loading LoRA adapter from %s", adapter_path)
            model = PeftModel.from_pretrained(model, str(adapter_path))
        else:
            LOGGER.warning("Adapter directory %s does not exist; using the base model only", adapter_path)

    model.eval()
    return model, tokenizer


def generate_response(
    model,
    tokenizer: AutoTokenizer,
    instruction: str,
    input_text: str,
    max_new_tokens: int = 256,
) -> str:
    """Generate and return the model response text."""
    prompt = build_prompt(instruction, input_text, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = generated[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def main() -> None:
    """CLI entrypoint."""
    configure_logging()
    args = parse_args()
    model, tokenizer = load_model(args.model_name, args.adapter_dir)
    response = generate_response(
        model=model,
        tokenizer=tokenizer,
        instruction=args.instruction,
        input_text=args.input_text,
        max_new_tokens=args.max_new_tokens,
    )
    print(response)


if __name__ == "__main__":
    main()
