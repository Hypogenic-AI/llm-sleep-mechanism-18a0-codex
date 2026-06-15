# Code Walkthrough

## Overview

The project has two executable scripts:

- `src/sleep_nudge_experiment.py`: validates datasets, builds prompts, calls real OpenAI models, labels responses with an LLM judge, and audits WildChat sleep-phrase matches.
- `src/analyze_results.py`: converts cached JSONL files into labeled CSVs, computes rates and statistical tests, writes figures, and produces error-analysis examples.

## Data Flow

Raw resources -> synthetic prompt builder -> OpenAI generations -> LLM judge -> labeled CSVs -> statistics/figures/report.

Key files:

- Inputs: `datasets/`, `prompts/synthetic_cases.jsonl`
- Raw outputs: `results/model_outputs/*.jsonl`
- Judgments: `results/judgments/*.jsonl`
- Analysis: `results/summary_tables/*.csv`, `figures/*.png`, `results/analysis_summary.json`

## Important Functions

### `build_synthetic_cases()`

Creates the 54 synthetic factorial cases by crossing profile, context length, task urgency, and paraphrase variant.

### `generate_model_outputs(...)`

Calls the OpenAI Responses API for each `(case, model)` pair and appends cached records to `results/model_outputs/synthetic_generations.jsonl`. Completed records are skipped unless `--force` is passed.

### `judge_items(...)`

Uses `gpt-4.1-mini-2025-04-14` as a JSON judge with a fixed sleep-nudge rubric. It writes cached judgments to `results/judgments/`.

### `summarize_groups(...)`

Computes rates, bootstrap confidence intervals, hard-disengagement rates, answer rates, and tone means by condition.

### `run_pairwise_tests(...)`

Runs the preregistered Fisher/chi-square comparisons and applies Holm correction. Logistic regression is skipped when positive events are too rare for a stable fit.

## Reproduction

```bash
source .venv/bin/activate
python src/sleep_nudge_experiment.py --mode all
python src/analyze_results.py
```

The first command requires `OPENAI_API_KEY` for any missing API outputs. The second command is offline if cached outputs are present.
