# CPJudgeBench

Benchmark for evaluating **LLM-as-judge** systems on constraint programming (CP) model correctness.

The pipeline (1) generates candidate CP models in multiple languages, (2) validates them against a reference model via solution-space enumeration, and (3) asks judge LLMs to classify or score those candidates.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file with your OpenRouter key:

```
OPENROUTER_API_KEY=...
```

You also need **CPMpy** (Python) and **MiniZinc** on `PATH` for model execution.

## Quick start

Run all steps on one sample problem (falls back to built-in `domino_tiling` if no problem file is found):

```bash
python main.py --num-problems 1
```

Or use the bundled sample input:

```bash
python main.py \
  --problems-file sample-data/domino_tiling-problem.jsonl \
  --problem-ids domino_tiling \
  --data-generation \
  --judge --judge-approach reference_free
```

## Pipeline steps

| Flag | What it does |
|------|----------------|
| `--generate-instances` | LLM-generated benchmark instances |
| `--data-generation` | Generate + validate candidate models per correctness label |
| `--judge` | LLM-as-judge evaluation (`--judge-approach` selects variant) |
<!-- | `--pairwise-judge` | Pairwise comparison of candidates |
| `--sat-meta-eval` | SAT-only meta-evaluation baseline | -->

Judge approaches: `reference_free`, `reference_based`, `score_reference_free`, `score_reference_based`, `binary_reference_free`, `binary_reference_based`.

Configure generator and judge models in `src/config.py`.

## Output layout

```
data-storage/<problem_id>/
  instances/           # generated instances
  candidates/
    candidate-models.json                    # plain payload for judges
    candidate-models-data-generation.json    # full generation metadata
    judge-results.csv
    judge-summary.csv
```

Progress/resume state is tracked under `logs/`.

## Sample data

A trimmed snapshot for the `domino_tiling` problem lives in [`sample-data/`](sample-data/). See [`sample-data/README.md`](sample-data/README.md) for file descriptions.

**The complete benchmark dataset will be shared in a future release.**

## Project structure

```
main.py              # CLI entry point
src/
  main.py            # argument parsing and pipeline orchestration
  config.py          # experiment context and model lists
  data_generation.py # candidate generation + validation loop
  judge.py           # LLM-as-judge evaluation
  executors.py       # run models and enumerate solution spaces
  generation.py      # LLM prompts for instances and candidates
  llm.py             # OpenRouter client
  meta_eval.py       # SAT-only baseline
  pairwise_judge.py  # pairwise judge variant
preprocess_dcp_open.py   # optional preprocessing of DCP-Bench-Open problems
```


## Problem input

Problems are JSONL records with fields: `id`, `description`, `model`, `example_instance`, `decision_variables`, `instances`. Place your full problem file at `extra_files/dcp-bench-open.jsonl` (gitignored) or pass `--problems-file`.
