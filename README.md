# Teaching-DNR

Teaching-DNR is a research codebase for training small open-weight language models on safety classification with explicit deliberative normative reasoning.

The core task is binary prompt classification:

- `harmful`
- `unharmful`

The main question is whether a model improves on adversarially framed safety prompts when it is trained not only to output the final label, but also to produce a separate `normative_reasoning` field and, in some settings, to optimize that field with a local LLM judge.

This repository implements the following pipeline:

- supervised fine-tuning on WildGuardMix prompt-harm labels
- GRPO with direct label-accuracy reward
- GRPO with judge-based reward over normative reasoning
- held-out evaluation on WildGuardTest-style data and an additional adversarial DNR dataset
- offline judge quality control for the reasoning judge

## Main Idea

The project compares three training families on the same harmful/unharmful classification setting:

1. `SFT`
   Direct supervised fine-tuning on the gold label.

2. `GRPO + label reward`
   Reinforcement learning where reward is based on final label correctness.

3. `GRPO + judge reward`
   Reinforcement learning where reward depends on the quality of the model's `normative_reasoning`, scored by a local judge (`CompassJudger-1-1.5B-Instruct`) with optional label-accuracy combination.

The structured output format used in the normative setup is:

```xml
<think>...</think>
<normative_reasoning>...</normative_reasoning>
<answer>harmful|unharmful</answer>
```

## Repository Layout

### Training

- [fine_tune/fine_tune_sft_label.py](fine_tune/fine_tune_sft_label.py)
  SFT baseline for label classification.

- [fine_tune/fine_tune_grpo_label.py](fine_tune/fine_tune_grpo_label.py)
  GRPO with direct label-accuracy reward.

- [fine_tune/fine_tune_grpo_judge.py](fine_tune/fine_tune_grpo_judge.py)
  GRPO with judge-based reward modes:
  `rubric_only`, `rubric_plus_accuracy`, `rubric_with_gold_direction`.

- [fine_tune/reward_funcs_judge.py](fine_tune/reward_funcs_judge.py)
  Judge prompting, parsing, and reward computation for CompassJudger.

- [fine_tune/shared.py](fine_tune/shared.py)
  Shared prompts, dataset loading, format validation, and utility functions.

### Evaluation

- [eval/check.py](eval/check.py)
  Main evaluation script for checkpoints or base models.

- [eval/testdata.py](eval/testdata.py)
  Dataset selection and evaluation data loading.

- [eval/compare_outputs.py](eval/compare_outputs.py)
  Comparison utilities for produced evaluation outputs.

### Judge Audit

- [eval/judge_audit/real_prompts_synthetic_reasoning_v1.jsonl](eval/judge_audit/real_prompts_synthetic_reasoning_v1.jsonl)
  Final controlled audit set: 10 real adversarial prompts × 4 synthetic reasoning variants.

- [eval/judge_audit/score_audit_set_compass.py](eval/judge_audit/score_audit_set_compass.py)
  Scores the audit set with CompassJudger.

- [eval/judge_audit/score_audit_set_openai.py](eval/judge_audit/score_audit_set_openai.py)
  Scores the audit set with OpenAI reference judges.

- [eval/judge_audit/analyze_variant_scores.py](eval/judge_audit/analyze_variant_scores.py)
  Summarizes average judge scores by reasoning variant.

- [eval/judge_audit/analyze_rank_order.py](eval/judge_audit/analyze_rank_order.py)
  Evaluates whether judge rankings match the intended ordering of reasoning quality.

- [eval/judge_audit/judge_quality_control_report.txt](eval/judge_audit/judge_quality_control_report.txt)
  Narrative write-up of the final judge audit.

### Supporting Material

- [scripts](scripts)
  Cluster job scripts used on the HPC environment.

## Environment

The project was run primarily in a Conda environment named `tdnr`.

Main dependency files:

- [environment.yml](environment.yml)
- [environment_full.yml](environment_full.yml)
- [requirements.txt](requirements.txt)
- [requirements_full.txt](requirements_full.txt)

Typical setup:

```bash
conda env create -f environment.yml
conda activate tdnr
```

You will also need:

- access to the Hugging Face Hub
- a working `wandb` login for training runs
- a CUDA-capable GPU for most fine-tuning and judge-reward experiments

## Typical Workflow

### 1. Train a baseline model

For SFT:

```bash
python -m fine_tune.fine_tune_sft_label
```

For GRPO with label reward:

```bash
python -m fine_tune.fine_tune_grpo_label
```

For GRPO with judge reward:

```bash
python -m fine_tune.fine_tune_grpo_judge
```

Important: the training scripts currently rely on in-file configuration constants rather than a stable CLI. Before running, check the active values in the script, especially:

- `MODEL_ID`
- `TRAIN_SAMPLES`
- `NORMATIVE`
- `REWARD_MODE`
- LoRA settings
- batch size / accumulation / prompt length

### 2. Evaluate a checkpoint

```bash
python -m eval.check path/to/checkpoint-or-adapter
```

Outputs are written either to:

- [eval/check_outputs](/home/s77estef/code/Teaching-DNR/eval/check_outputs)
  for base-model or ad hoc evaluation

or

- the run-local `eval/` directory inside a checkpoint/run directory

depending on how the script is called.

### 3. Run judge quality control

Compass judge on the final audit set:

```bash
python -m eval.judge_audit.score_audit_set_compass \
  --audit-set eval/judge_audit/real_prompts_synthetic_reasoning_v1.jsonl \
  --output eval/judge_audit/results/real_prompts_synthetic_compass_gold_direction.jsonl \
  --mode gold_direction
```

Then analyze score behavior:

```bash
python -m eval.judge_audit.analyze_variant_scores \
  --dataset eval/judge_audit/real_prompts_synthetic_reasoning_v1.jsonl \
  --scores eval/judge_audit/results/real_prompts_synthetic_compass_gold_direction.jsonl \
  --mode gold_direction \
  --summary-output eval/judge_audit/results/real_prompts_synthetic_compass_gold_direction_summary.json \
  --table-output eval/judge_audit/results/real_prompts_synthetic_compass_gold_direction_table.csv
```

And rank-order analysis:

```bash
python -m eval.judge_audit.analyze_rank_order \
  --dataset eval/judge_audit/real_prompts_synthetic_reasoning_v1.jsonl \
  --scores eval/judge_audit/results/real_prompts_synthetic_compass_gold_direction.jsonl \
  --mode gold_direction \
  --summary-output eval/judge_audit/results/real_prompts_synthetic_compass_gold_direction_rank_summary.json
```

The same pattern applies to OpenAI-based judge runs.

