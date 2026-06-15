# LLM "Go to Sleep" Mechanism Study

This workspace tests whether LLMs show a unified assistant-initiated "go to sleep" response after late-night or prolonged conversations, or whether the behavior depends on perceived user state. The study uses real OpenAI API calls, cached outputs, LLM-judge labels, and statistical summaries.

## Key Findings

- Bedtime nudges were rare in the controlled experiment: 2/108 synthetic outputs (1.85%).
- Both nudges came from `gpt-4.1-mini-2025-04-14` in the fatigued profile; `gpt-5.4-mini-2026-03-17` produced none.
- No response hard-refused, disengaged, or replaced the requested answer with sleep advice.
- The dependent/vulnerable profile produced zero bedtime nudges in this prompt-only design.
- WildChat sleep-phrase matches were mostly not assistant-initiated bedtime nudges: 19/25 no nudge, 5/25 soft nudges, 1/25 ordinary sleep-topic answer.

See [REPORT.md](REPORT.md) for the full methodology, results, and limitations.

## Reproduce

Use the isolated environment:

```bash
source .venv/bin/activate
```

Install dependencies if needed:

```bash
uv pip install -r requirements.txt
```

Run the full experiment and analysis:

```bash
python src/sleep_nudge_experiment.py --mode all
python src/analyze_results.py
```

The experiment script skips cached API outputs by default. Delete or move files under `results/model_outputs/` and `results/judgments/`, or pass `--force`, only if you intentionally want new API calls.

## File Structure

- `planning.md`: preregistered motivation, hypothesis decomposition, and analysis plan.
- `src/sleep_nudge_experiment.py`: dataset validation, prompt construction, real API generation, LLM judging, and WildChat audit.
- `src/analyze_results.py`: summary tables, Fisher tests, bootstrap intervals, plots, and examples.
- `prompts/synthetic_cases.jsonl`: exact synthetic prompts.
- `results/model_outputs/`: cached model generations and WildChat audit items.
- `results/judgments/`: cached LLM-judge outputs.
- `results/summary_tables/`: labeled CSVs and statistical summaries.
- `figures/`: generated plots.
- `REPORT.md`: primary research report.
