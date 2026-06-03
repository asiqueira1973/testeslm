# Experiment design

## Hypothesis

Historical transactional business data can be transformed into an instruction-tuning dataset that teaches a Small Language Model to produce structured, business-readable responses from tabular records.

For this first step, the objective is not to train a model. The objective is to prove that historical rows can be converted into deterministic instruction examples with a consistent instruction, natural language input, and structured output.

## Business scenario

A lending company receives loan applications and records each historical decision as either approved or rejected. Each application includes applicant information, requested loan terms, credit score, income, and asset values.

The company wants to explore whether its historical decisions can become training examples for an SLM. A prepared example should read like a business task:

1. Analyze the application.
2. Use the applicant and loan request fields as context.
3. Return the historical decision with a concise reason, recommended action, and customer-facing message.

## Why loan approval is a useful proxy

Loan approval data is a good proxy for historical transactional decision data because it has the same pattern found in many business processes:

- A row represents a completed transaction or case.
- Several input fields describe the customer, request, or situation.
- One target field records the historical business decision.
- The decision can be expressed as a structured response for downstream users.

This mirrors other business workflows such as insurance claim decisions, customer support resolutions, fraud review outcomes, quote approvals, and account risk reviews.

## What this first step proves

This repository setup and dataset preparation step proves that:

- a raw transactional CSV can be validated before use;
- column names and decision labels can be normalized;
- each row can be converted into an instruction, input, and output example;
- train, test, and sample JSONL files can be generated reproducibly;
- explanation text can be created with simple deterministic rules without changing the historical decision.

The output is suitable as a first demonstration dataset for instruction tuning, review, and iteration.

## What comes next

The next phase will fine-tune a small open model using parameter-efficient methods such as LoRA or QLoRA. That future phase should include:

- selecting a small open base model;
- loading the prepared JSONL dataset;
- formatting examples for the selected model template;
- running LoRA or QLoRA fine-tuning;
- evaluating outputs on the held-out test set;
- comparing generated decisions and explanations with the historical records.

Fine-tuning is intentionally out of scope for this first repository setup ticket.
