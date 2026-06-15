# Cloned Repositories

All repositories were cloned with shallow history where possible.

| Repository | URL | Location | Purpose | Notes |
|---|---|---|---|---|
| refusal_direction | https://github.com/andyrdt/refusal_direction | `code/refusal_direction/` | Reproduce refusal-direction extraction, ablation, and evaluation | Requires Hugging Face token for gated models and Together/OpenAI-style evaluators for some metrics |
| persona_vectors | https://github.com/safety-research/persona_vectors | `code/persona_vectors/` | Generate and apply persona vectors for traits such as sycophancy, apathy, politeness, and hallucination | Includes `dataset.zip`, trait data, `generate_vec.py`, `activation_steer.py`, and training scripts |
| sorry_bench | https://github.com/SORRY-Bench/sorry-bench | `code/sorry_bench/` | Benchmark LLM safety refusal with fine-grained unsafe-topic taxonomy and linguistic mutations | Dataset is gated; repo includes generation and judgment scripts |
| do_not_answer | https://github.com/libr-ai/do-not-answer | `code/do_not_answer/` | Safety prompt dataset, model responses, and evaluator resources | Includes English and Chinese data files plus response annotations |
| xstest | https://github.com/paul-rottger/xstest | `code/xstest/` | XSTest prompts and evaluation code for exaggerated safety | `xstest_prompts.csv` is directly usable |
| or_bench | https://github.com/justincui03/or-bench | `code/or_bench/` | OR-Bench over-refusal generation/evaluation pipeline | Dataset downloaded separately from Hugging Face under `datasets/` |
| false_refusal_mitigation | https://github.com/mainlp/False-Refusal-Mitigation | `code/false_refusal_mitigation/` | Single-vector false-refusal ablation pipeline | Built on refusal_direction; includes `demo.ipynb` and `pipeline.run_pipeline` |
| realpref | https://github.com/GG14127/RealPref | `code/realpref/` | Long-horizon personalized preference-following benchmark | Includes `test_data/`, generation scripts, test scripts, and judging scripts |

## Key Entry Points

- `code/refusal_direction/`: `python3 -m pipeline.run_pipeline --model_path {model_path}`
- `code/false_refusal_mitigation/`: `python -m pipeline.run_pipeline --config_path configs/cfg.yaml`
- `code/persona_vectors/`: `python generate_vec.py ...`, `python -m eval.eval_persona ...`
- `code/sorry_bench/`: `gen_model_answer.py`, `gen_model_answer_vllm.py`, `gen_judgment_safety.py`, `gen_judgment_safety_vllm.py`
- `code/xstest/`: `xstest_prompts.csv`; evaluation scripts are under `evaluation/`
- `code/or_bench/`: generation and moderation scripts described in README; `plot.py` for summaries
- `code/realpref/`: `prepare_data.py`, `test.py`, `judge.py`, plus `run_generation.sh`, `run_test.sh`, `run_judge.sh`

## Practical Constraints

- Several repos require GPU memory, gated model access, and API keys. I did not install their full dependency stacks because many include heavy model-serving dependencies such as vLLM, FastChat, or custom evaluation harnesses.
- The current workspace `.venv` contains lightweight resource-gathering dependencies only. Use repo-specific environments for full model experiments if needed.

