# Judge Audit Workflow

This folder builds a fixed offline audit set for the GRPO normative-reasoning judge, scores it with both `CompassJudger` and an OpenAI reference judge, and compares the results.

The audit workflow is adversarial-only by design. Candidate harvesting and the fixed review set both keep only examples where `adversarial = True`.

## Files

- `build_audit_set.py`: harvests examples from eval reports and checkpoint batch logs, assigns heuristic buckets, and writes a fixed `audit_set_v1.jsonl`
- `score_audit_set_compass.py`: scores the fixed set with the current local judge
- `score_audit_set_openai.py`: scores the fixed set with an OpenAI model through the Responses API
- `analyze_judge_agreement.py`: computes agreement statistics and exports the biggest disagreements

## Recommended workflow

1. Build the candidate and fixed audit set.

```bash
python -m eval.judge_audit.build_audit_set
```

This writes:

- `audit_set_v1_candidates.jsonl`
- `audit_set_v1.jsonl`
- `audit_set_v1_review.csv`
- `audit_set_v1_summary.json`

Review `audit_set_v1_review.csv` and trim or replace examples if needed. If you make manual changes, save them back into `audit_set_v1.jsonl` and keep that file fixed for the experiment.

2. Score with Compass.

```bash
python -m eval.judge_audit.score_audit_set_compass --mode gold_direction
```

3. Score with OpenAI.

```bash
export OPENAI_API_KEY=...
python -m eval.judge_audit.score_audit_set_openai --mode gold_direction --model gpt-5.2
```

For a ChatGPT-style alias, use `gpt-5.2-chat-latest`, but prefer a pinned model for reproducibility.

4. Analyze agreement.

```bash
python -m eval.judge_audit.analyze_judge_agreement --mode gold_direction
```

This writes:

- `results/agreement_report_v1.json`
- `results/disagreements_v1.csv`

## Notes

- The bucket assignments are heuristic. They are intended to speed up manual curation, not replace it.
- The OpenAI scorer intentionally runs offline from training. Do not use it inside GRPO reward computation.
- For a complete audit, run the pipeline twice:
  - `--mode rubric_only`
  - `--mode gold_direction`
