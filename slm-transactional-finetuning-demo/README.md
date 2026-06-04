# SLM Transactional Fine-Tuning Demo

## Project goal

This project demonstrates how historical transactional business data can be converted into an instruction-tuning dataset for Small Language Model (SLM) fine-tuning.

The first version focuses on repository setup, dataset preparation, and a small LoRA fine-tuning path. The goal is to make the data transformation and first training workflow simple, reproducible, and easy to run locally or in Google Colab.

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

Create and activate a Python environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

The project uses `pandas` and `scikit-learn` for dataset preparation, plus PyTorch and Hugging Face libraries for LoRA fine-tuning.

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


## Step 2: Fine-tune Gemma with LoRA

After generating `data/processed/train.jsonl` and `data/processed/test.jsonl`, you can fine-tune a LoRA adapter on top of Gemma.

This step should preferably run in a GPU environment such as Google Colab, Kaggle Notebook, or a local CUDA workstation. Gemma models may require accepting the model license/terms on Hugging Face and logging in with a Hugging Face token before the model can be downloaded.

Install dependencies:

```bash
pip install -r requirements.txt
```

Log in to Hugging Face if required:

```bash
huggingface-cli login
```

Train the LoRA adapter:

```bash
python src/train_lora.py \
  --train-file data/processed/train.jsonl \
  --eval-file data/processed/test.jsonl \
  --output-dir models/loan-approval-gemma-lora
```

Run inference with the trained adapter:

```bash
python src/inference.py \
  --adapter-dir models/loan-approval-gemma-lora \
  --instruction "Analyze the loan application based on the company's historical credit decisions." \
  --input "Applicant has 2 dependents, is Graduate, self-employed status is No, annual income is 9600000, requested loan amount is 29900000, loan term is 12 months, CIBIL score is 778, residential assets value is 2400000, commercial assets value is 17600000, luxury assets value is 22700000, and bank asset value is 8000000."
```

You can also compare base-model outputs with fine-tuned outputs:

```bash
python src/compare_outputs.py
```

The comparison script writes `outputs/comparison.md`. If the LoRA adapter has not been trained yet, it skips the fine-tuned comparison and explains why in the output file.

For more detail, see `docs/fine_tuning_notes.md`.

## Running the fine-tuning demo in Google Colab

A Colab-friendly walkthrough is available at:

```text
notebooks/01_finetune_gemma_lora_colab.ipynb
```

Open the notebook in Google Colab, replace the placeholder repository URL in the first cell if needed, and run the cells from top to bottom. The notebook assumes you have cloned this GitHub repository into the Colab runtime, checks for GPU availability, installs `requirements.txt`, logs in to Hugging Face, verifies the prepared JSONL files, runs the existing LoRA training and inference scripts, and displays the generated comparison report.

Gemma models may require accepting the model license/terms on Hugging Face before download. The LoRA adapter is generated locally in the notebook environment under `models/loan-approval-gemma-lora`; trained model files and generated outputs are ignored by Git and should not be committed.

## Why this matters

Many companies already have transactional business history: applications, orders, claims, support tickets, approvals, rejections, reviews, and outcomes. This project shows the first step in turning that history into an instruction dataset that can later be used to fine-tune an SLM to reproduce or explain business decisions in a controlled demo setting.

## Executive comparison workflow

After fine-tuning `google/gemma-2-2b-it` with the loan-approval LoRA adapter, you can generate an article-ready evidence report that compares three response styles on the same Portuguese business cases:

1. the base Gemma instruction model,
2. the fine-tuned Gemma model with the LoRA adapter, and
3. an optional public LLM answer pasted manually by the user.

The workflow does **not** retrain the model and does **not** require OpenAI, Claude, or Gemini API keys. Public LLM comparison is intentionally manual so the report stays transparent about what was generated locally and what was pasted from an external model.

The Portuguese cases are stored at:

```text
data/evaluation/portuguese_cases.jsonl
```

Each case includes the original Portuguese executive question, the normalized instruction and input sent to Gemma, and the expected historical operational output.

To add public LLM responses, copy the template and paste one answer per `case_id`:

```bash
cp data/evaluation/public_llm_outputs.template.jsonl data/evaluation/public_llm_outputs.jsonl
```

Then edit `data/evaluation/public_llm_outputs.jsonl` and replace the placeholder `public_llm_output` values with manually collected GPT, Claude, Gemini, or other public LLM answers. If this file does not exist, the comparison still runs and marks public LLM output as not provided.

Run the executive comparison from the project root:

```bash
python src/compare_executive.py \
  --adapter-dir models/loan-approval-gemma-lora \
  --output-file outputs/executive_comparison.md
```

The script loads the Portuguese cases, runs deterministic inference with the base Gemma model, runs inference with the LoRA adapter when the adapter directory is available, reads optional manual public LLM outputs, and writes:

```text
outputs/executive_comparison.md
```

The generated Markdown includes the Portuguese question, normalized model input, expected historical output, base model output, fine-tuned model output, optional public LLM output, and a short observation. Use the report to discuss behavior specialization and operational response format; do not present it as proof that the fine-tuned SLM is generally more intelligent or universally better than public LLMs.
