# Fine-tuning Gemma with LoRA

This note explains the first fine-tuning implementation for the loan approval SLM demo. The implementation is intentionally small and runnable in a GPU notebook or workstation environment rather than assuming training will run inside Codex.

## What LoRA is

LoRA, or Low-Rank Adaptation, is a parameter-efficient fine-tuning technique. Instead of updating every weight in the base language model, LoRA freezes the base model and trains small adapter matrices inside selected transformer layers. At inference time, those adapter weights can be loaded on top of the original model.

## Why LoRA is used in this demo

LoRA is a good fit for this project because it keeps the example practical:

- It trains far fewer parameters than full fine-tuning.
- It produces a small adapter directory instead of a full model copy.
- It is easier to run in Colab, Kaggle Notebook, or a local GPU environment.
- It keeps the base model reusable while letting the demo specialize behavior for loan approval examples.

The adapter is saved to:

```text
models/loan-approval-gemma-lora/
```

Do not commit generated model or adapter files to Git.

## Why `google/gemma-2-2b-it` was selected

The default base model is:

```text
google/gemma-2-2b-it
```

Gemma was selected to keep this implementation aligned with the article context, where Google Gemma is mentioned as an example base model for SLM fine-tuning. The 2B instruction-tuned Gemma variant is also small enough to be a realistic first demo target compared with larger models.

## Hardware expectations

Actual training should preferably run in an environment with a GPU. Good options include:

- Google Colab with a GPU runtime
- Kaggle Notebook with GPU enabled
- A local machine or cloud VM with a CUDA-compatible NVIDIA GPU

The script chooses `bfloat16` when supported, `float16` on other CUDA GPUs, and `float32` on CPU. CPU execution is useful for checking code paths but is generally not practical for training a 2B-parameter model.

The default settings are intentionally conservative for a demo:

- 1 epoch
- per-device training batch size of 1
- gradient accumulation of 4
- maximum sequence length of 512

If your GPU has more memory, you can increase batch size or sequence length. If you run out of memory, reduce `--max-seq-length` or keep batch size at 1.

## Hugging Face access for Gemma

Gemma models may require accepting Google/Hugging Face license terms before download. You may also need to log in with a Hugging Face token in the environment where training or inference runs.

Typical setup:

```bash
huggingface-cli login
```

Then open the model page on Hugging Face, accept any required terms, and rerun the script.

## Install dependencies

From the project root, install the project dependencies:

```bash
pip install -r requirements.txt
```

The ML dependencies are intentionally limited to PyTorch, Hugging Face Transformers/Datasets, PEFT, TRL, Accelerate, and Safetensors.

## Run LoRA training

From the project root:

```bash
python src/train_lora.py \
  --train-file data/processed/train.jsonl \
  --eval-file data/processed/test.jsonl \
  --output-dir models/loan-approval-gemma-lora
```

The training script will:

1. Load `data/processed/train.jsonl` and `data/processed/test.jsonl`.
2. Format each row into one supervised fine-tuning text.
3. Load `google/gemma-2-2b-it` by default.
4. Configure PEFT LoRA adapters for Gemma-style attention and MLP projection layers.
5. Train with TRL `SFTTrainer`.
6. Save the LoRA adapter to `models/loan-approval-gemma-lora/`.

Common optional arguments:

```bash
python src/train_lora.py \
  --model-name google/gemma-2-2b-it \
  --num-train-epochs 1 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --learning-rate 2e-4 \
  --max-seq-length 512
```

## Run inference

Run inference with the fine-tuned adapter:

```bash
python src/inference.py \
  --adapter-dir models/loan-approval-gemma-lora \
  --instruction "Analyze the loan application based on the company's historical credit decisions." \
  --input "Applicant has 2 dependents, is Graduate, self-employed status is No, annual income is 9600000, requested loan amount is 29900000, loan term is 12 months, CIBIL score is 778, residential assets value is 2400000, commercial assets value is 17600000, luxury assets value is 22700000, and bank asset value is 8000000."
```

Run inference with only the base model by omitting `--adapter-dir`:

```bash
python src/inference.py \
  --instruction "Analyze the loan application based on the company's historical credit decisions." \
  --input "Applicant has 2 dependents, is Graduate, self-employed status is No, annual income is 9600000, requested loan amount is 29900000, loan term is 12 months, CIBIL score is 778, residential assets value is 2400000, commercial assets value is 17600000, luxury assets value is 22700000, and bank asset value is 8000000."
```

If `--adapter-dir` points to a missing directory, the inference script logs a warning and uses the base model only. This keeps the project usable before training has been run.

## Compare base vs fine-tuned outputs

After training, generate a small Markdown comparison from `data/processed/sample.jsonl`:

```bash
python src/compare_outputs.py
```

The comparison is written to:

```text
outputs/comparison.md
```

If `models/loan-approval-gemma-lora/` does not exist, the script still runs base-model inference and writes a note explaining that the fine-tuned comparison was skipped.
