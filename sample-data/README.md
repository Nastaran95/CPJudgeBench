# Sample data

A single-problem snapshot (`domino_tiling`) showing the inputs and outputs of the
full CPJudgeBench pipeline. It lets you inspect the data format and reproduce the
run end-to-end without the complete dataset.

> The complete benchmark dataset (all problems, all generator/judge models) will be
> shared in a future release.

## Contents

```
domino_tiling-problem.jsonl   # pipeline input: one benchmark problem record
domino_tiling/
  instances/
    <model>.json              # instances proposed by one generator LLM
    all_instances.json        # deduplicated union of all per-model instances
  candidates/
    candidate-models.json                    # plain {llm: {language: {label: {ok, code}}}} payload fed to judges
    candidate-models-data-generation.json     # full generation metadata (attempts, fp/fn, solution-space sizes)
    judge-results.csv                         # one row per (judge, candidate): predicted vs. claimed label
    judge-summary.csv                         # accuracy aggregated by judge / language / approach / label
    sat-meta-eval-results.csv                 # SAT-only baseline vs. full-space ground truth, per candidate
    sat-meta-eval-summary.csv                 # SAT-only baseline confusion matrix / precision / recall / F1
```

## Reproduce

```bash
python main.py \
  --problems-file sample-data/domino_tiling-problem.jsonl \
  --problem-ids domino_tiling \
  --data-generation \
  --judge --judge-approach reference_free reference_based \
  --sat-meta-eval
```

Outputs are written under `data-storage/domino_tiling/` (gitignored); the files
here are a committed copy of one such run.
