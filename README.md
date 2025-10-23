# Teaching-DNR

Prototype tooling for evaluating large language models on deliberative normative reasoning datasets.

## Prerequisites

- Python 3.9 or newer.
- Access to the Hugging Face Hub for downloading models and datasets (network connectivity is required; gated models such as Llama-3 may also require accepting the license on the Hub).
- Hardware capable of running the desired models (LLMs with 3B–4B parameters typically need at least 16 GB of GPU memory or a large CPU RAM footprint).

Install dependencies in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Running an evaluation

The `run_evaluation.py` CLI evaluates an auto-regressive Hugging Face model on the ETHICS dataset or Social Chemistry 101. Example commands:

```bash
# LLAMA 3B instruct-style model on the ETHICS commonsense split
python run_evaluation.py \
  --task ethics \
  --subset commonsense \
  --split validation \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --torch-dtype bfloat16 \
  --device auto \
  --output results_llama_ethics.json

# Qwen 4B chat model on Social Chemistry 101
python run_evaluation.py \
  --task social_chemistry \
  --split validation \
  --model Qwen/Qwen2-4B-Instruct \
  --torch-dtype float16 \
  --device auto \
  --output results_qwen_social_chem.json
```

Each command will automatically download the required dataset split via the `datasets` library and the specified model weights via `transformers`. For chat-tuned checkpoints, the CLI uses the tokenizer's chat template by default; toggle `--no-chat-template` to disable that behaviour.

The summary accuracy is printed to stdout. When `--output` is provided, detailed per-example records (prompt, gold label, prediction, raw model output) are saved as JSON for further analysis.

## Notes and limitations

- Running the evaluations in this repository requires internet access for the initial downloads; this is not possible inside the restricted Codex CLI sandbox without explicit approval.
- Model and dataset downloads can be time-consuming and storage-intensive. Consider configuring `HF_HOME` if you want to control cache locations.
- Accuracy is computed by matching the model's response against known answer strings. Depending on the task, you might want to extend the match logic or add more nuanced metrics (e.g., scoring rationales).
