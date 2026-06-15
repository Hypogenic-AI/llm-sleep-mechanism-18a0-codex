# Resources Catalog

## Summary

Resources gathered for the project:

- Papers downloaded: 17
- Datasets downloaded or sampled: 7
- Repositories cloned: 8

## Papers

| Title | Year | File | Key Info |
|---|---:|---|---|
| Refusal in Language Models Is Mediated by a Single Direction | 2024 | `papers/2406.11717_refusal_in_language_models_is_mediated_by_a_single_direction.pdf` | Core unified refusal-direction evidence |
| There Is More to Refusal in Large Language Models than a Single Direction | 2026 | `papers/2602.02132_there_is_more_to_refusal_in_large_language_models_than_a_single_direct.pdf` | Multi-direction counterpoint |
| Over-Refusal and Representation Subspaces | 2026 | `papers/2603.27518_over_refusal_and_representation_subspaces_a_mechanistic_analysis_of_ta.pdf` | Harmful refusal vs over-refusal geometry |
| Refusal Direction is Universal Across Safety-Aligned Languages | 2025 | `papers/2505.17306_refusal_direction_is_universal_across_safety_aligned_languages.pdf` | Cross-lingual refusal universality |
| Surgical, Cheap, and Flexible | 2024 | `papers/2410.03415_surgical_cheap_and_flexible_mitigating_false_refusal_in_language_model.pdf` | False-refusal vector ablation |
| No for Some, Yes for Others | 2025 | `papers/2509.08075_no_for_some_yes_for_others_persona_prompts_and_other_sources_of_false.pdf` | Sociodemographic persona effects on false refusal |
| XSTest | 2023 | `papers/2308.01263_xstest_a_test_suite_for_identifying_exaggerated_safety_behaviours_in_l.pdf` | Exaggerated safety benchmark |
| OR-Bench | 2024 | `papers/2405.20947_or_bench_an_over_refusal_benchmark_for_large_language_models.pdf` | Large over-refusal benchmark |
| SORRY-Bench | 2024 | `papers/2406.14598_sorry_bench_systematically_evaluating_large_language_model_safety_refu.pdf` | Fine-grained safety refusal evaluation |
| Do-Not-Answer | 2023 | `papers/2308.13387_do_not_answer_a_dataset_for_evaluating_safeguards_in_llms.pdf` | Safety prompt/action labels |
| Persona Vectors | 2025 | `papers/2507.21509_persona_vectors_monitoring_and_controlling_character_traits_in_languag.pdf` | Trait vectors for assistant persona shifts |
| Interaction Context Often Increases Sycophancy in LLMs | 2025 | `papers/2509.12517_interaction_context_often_increases_sycophancy_in_llms.pdf` | User context and memory effects |
| Personalization Increases Affective Alignment | 2026 | `papers/2603.00024_personalization_increases_affective_alignment_but_has_role_dependent_e.pdf` | Personalization and sycophancy |
| Towards Realistic Personalization | 2026 | `papers/2603.04191_towards_realistic_personalization_evaluating_long_horizon_preference_f.pdf` | RealPref benchmark |
| Enabling Personalized Long-term Interactions | 2025 | `papers/2510.07925_enabling_personalized_long_term_interactions_in_llm_based_agents_throu.pdf` | Persistent memory/user profiles |
| User Interaction Patterns and Breakdowns | 2023 | `papers/2309.13879_user_interaction_patterns_and_breakdowns_in_conversing_with_llm_powere.pdf` | Human-AI interaction breakdowns |
| When Chatbots Accommodate | 2026 | `papers/2606.04431_when_chatbots_accommodate_what_ai_companions_optimize_for_in_vulnerabl.pdf` | Vulnerable extended conversations |

See `papers/README.md` for detailed descriptions and source links.

## Datasets

| Name | Source | Size | Task | Location | Notes |
|---|---|---:|---|---|---|
| XSTest | `Paul/XSTest` | 450 | False-refusal evaluation | `datasets/xstest/` | Safe prompts plus unsafe contrasts |
| Do-Not-Answer | `LibrAI/do-not-answer` | 939 | Safety refusal/action classification | `datasets/do_not_answer/` | Includes responses from several models |
| OR-Bench 80K | `bench-llm/or-bench` | 80,359 | Over-refusal evaluation | `datasets/or_bench_80k/` | Full large prompt set |
| OR-Bench Hard 1K | `bench-llm/or-bench` | 1,319 | Hard over-refusal evaluation | `datasets/or_bench_hard_1k/` | Recommended first OR-Bench split |
| OR-Bench Toxic | `bench-llm/or-bench` | 655 | Toxic contrast prompts | `datasets/or_bench_toxic/` | Prevents indiscriminate compliance |
| PersonaMem-v2 Benchmark Text | `bowen-upenn/PersonaMem-v2` | 5,000 | Long-context personalization | `datasets/personamem_v2_benchmark_text/` | User profiles, preferences, context links |
| WildChat bounded sample | `allenai/WildChat` | 200 sample, 25 sleep matches | Real-world conversation audit | `datasets/wildchat_sample/` | Scanned 14,782 conversations |

See `datasets/README.md` for loading and re-download instructions.

## Code Repositories

| Name | URL | Purpose | Location | Notes |
|---|---|---|---|---|
| refusal_direction | https://github.com/andyrdt/refusal_direction | Refusal-vector extraction and ablation | `code/refusal_direction/` | Core mechanistic baseline |
| persona_vectors | https://github.com/safety-research/persona_vectors | Trait-vector extraction/steering | `code/persona_vectors/` | Useful for bedtime/persona vector comparison |
| sorry_bench | https://github.com/SORRY-Bench/sorry-bench | Safety refusal evaluation | `code/sorry_bench/` | Dataset gated |
| do_not_answer | https://github.com/libr-ai/do-not-answer | Safety dataset/evaluator | `code/do_not_answer/` | Includes local CSV data |
| xstest | https://github.com/paul-rottger/xstest | Exaggerated safety prompts/eval | `code/xstest/` | Contains `xstest_prompts.csv` |
| or_bench | https://github.com/justincui03/or-bench | Over-refusal benchmark pipeline | `code/or_bench/` | Dataset also downloaded locally |
| false_refusal_mitigation | https://github.com/mainlp/False-Refusal-Mitigation | False-refusal vector ablation | `code/false_refusal_mitigation/` | Built on refusal_direction |
| realpref | https://github.com/GG14127/RealPref | Long-horizon personalization benchmark | `code/realpref/` | Includes `test_data/` |

See `code/README.md` for entry points and requirements.

## Resource Gathering Notes

### Search Strategy

I tried the local paper-finder service first, then used manual academic search over arXiv, ACL Anthology/OpenReview, Hugging Face datasets, and GitHub. Search terms focused on refusal direction, over-refusal, false refusal, persona prompts, user context, personalization, sycophancy, long-horizon interaction, and bedtime/sleep phrase behavior.

### Selection Criteria

I prioritized papers and resources that support one of three experiment needs:

- Mechanistic tests of whether a response is represented by a shared direction.
- Behavioral benchmarks for refusal, false refusal, and over-refusal.
- User-context datasets for testing persona, vulnerability, memory, and long-interaction effects.

### Challenges Encountered

- The paper-finder client stalled even though the backend service was running, so manual search was used.
- SORRY-Bench and LMSYS-Chat-1M are gated on Hugging Face without current credentials.
- WildChat streaming completed and wrote files, but Python/pyarrow aborted during interpreter shutdown. The written JSON and summary files validate successfully.

### Gaps and Workarounds

- There is no direct academic benchmark for assistant-initiated "go to sleep" behavior. The closest resources are false-refusal, over-refusal, sycophancy, vulnerable-conversation, and long-context personalization work.
- WildChat sleep-phrase matches include ordinary sleep-related conversations, so downstream labeling must separate assistant-initiated bedtime nudges from user-requested sleep advice.
- SORRY-Bench data was not downloaded due gating; use its cloned code and request Hugging Face access if needed.

## Recommendations for Experiment Design

1. **Primary datasets**: Use WildChat sample for phrase taxonomy, XSTest/OR-Bench/Do-Not-Answer for refusal classifier calibration, and PersonaMem-v2/RealPref for user-conditioned long-context prompts.
2. **Baselines**: Keyword classifier, LLM-as-judge classifier, refusal-vector projection, false-refusal-vector projection, and logistic mixed-effects models.
3. **Metrics**: Bedtime-nudge rate, hard refusal rate, soft-boundary rate, context-length effect, persona effect size, and activation-vector similarity.
4. **Code to reuse**: Start with `code/refusal_direction/` and `code/false_refusal_mitigation/` for mechanistic baselines; use `code/persona_vectors/` for trait-vector comparisons; use `code/xstest/` and `code/or_bench/` for prompt/evaluation examples.

## Research Execution Outputs

The completed automated research run used the behavioral benchmark path recommended above rather than an open-weight activation-vector path. The final executed artifacts are:

| Artifact | Location | Notes |
|---|---|---|
| Preregistered plan | `planning.md` | Motivation, novelty, factorial design, and statistical analysis plan |
| Synthetic prompt set | `prompts/synthetic_cases.jsonl` | 54 factorial cases: 3 profiles x 3 context lengths x 2 urgency levels x 3 variants |
| Real model outputs | `results/model_outputs/synthetic_generations.jsonl` | 108 real OpenAI API generations across two models |
| Synthetic judgments | `results/judgments/synthetic_judgments.jsonl` | 108 LLM-judge labels |
| WildChat audit | `results/model_outputs/wildchat_sleep_audit_items.jsonl` and `results/judgments/wildchat_sleep_audit_judgments.jsonl` | 25 sleep-phrase matches with LLM-judge labels |
| Analysis tables | `results/summary_tables/` | Group rates, Fisher tests, labeled outputs, and audit counts |
| Figures | `figures/` | Bedtime-nudge rates by profile, context, and model |
| Cost estimate | `results/cost_estimate.json` | Token counts and uncached pricing estimate |
| Final report | `REPORT.md` | Primary deliverable with results and limitations |

Main result: bedtime nudges were rare (2/108 synthetic outputs), soft, limited to the fatigued profile in `gpt-4.1-mini`, and never hard disengagements. The dependent/vulnerable profile produced zero bedtime nudges in this prompt-only experiment.
