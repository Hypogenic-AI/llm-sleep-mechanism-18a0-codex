# Literature Review: How Unified Is LLMs' "Go to Sleep" Mechanism?

## Review Scope

### Research Question

Do LLMs produce a consistent "go to sleep" or bedtime-nudge response after extended interaction because of a unified internal mechanism, or is this behavior mostly conditioned on user characteristics, vulnerability cues, persona/memory context, and task framing?

### Inclusion Criteria

- Papers on refusal, over-refusal, non-compliance, or safety boundary behavior in LLMs.
- Papers on personalization, user profiles, persona prompting, long-context interaction, or sycophancy.
- Datasets/code that support refusal, false refusal, user-context, or long-horizon interaction experiments.
- Recent papers from 2023-2026, plus benchmark papers with reusable artifacts.

### Exclusion Criteria

- Sleep-memory architecture papers where "sleep" means offline consolidation rather than a conversational response.
- General chatbot UX work without LLM behavior measurements.
- Non-primary summaries unless used only to locate papers or datasets.

### Search Log

| Date | Query/Source | Results | Notes |
|---|---|---:|---|
| 2026-06-15 | paper-finder: "large language model conversation termination disengagement refusal user characteristics extended interactions" | 0 returned | Client stalled; process stopped and manual search used |
| 2026-06-15 | arXiv/Web: refusal direction, over-refusal, persona prompts, sycophancy, personalization | 17 papers | PDFs downloaded |
| 2026-06-15 | Hugging Face dataset search | 7 local dataset resources | SORRY-Bench and LMSYS-Chat-1M gated |
| 2026-06-15 | GitHub/Papers-with-code style search | 8 repos | Cloned benchmark and method repos |

## Key Papers

### Refusal Mechanisms

**Refusal in Language Models Is Mediated by a Single Direction** shows that refusal in 13 open-source chat models can often be controlled by a one-dimensional activation-space direction: ablation suppresses harmful-request refusal, while addition induces refusal on harmless prompts. This is the strongest evidence for a unified mechanism, but its domain is safety refusal, not bedtime nudging.

**There Is More to Refusal in Large Language Models than a Single Direction** argues the single-direction account is incomplete. Different refusal and non-compliance categories occupy geometrically distinct directions, while linear steering still behaves like a shared refusal/over-refusal knob. This suggests a bedtime response could share a broad non-compliance control signal while differing in style or trigger features.

**Over-Refusal and Representation Subspaces** distinguishes harmful refusal from over-refusal. Harmful refusal is closer to a task-agnostic global direction; over-refusal is task-dependent and higher-dimensional. A "go to sleep" response to a safe user might be closer to over-refusal or wellbeing-boundary behavior than harmful refusal, so a single safety-refusal direction may not explain it.

**Refusal Direction is Universal Across Safety-Aligned Languages** finds cross-lingual transfer of refusal directions across 14 languages, strengthening the case that some safety refusal structure is shared across contexts. This is useful if future bedtime experiments include multilingual users.

**Surgical, Cheap, and Flexible** extracts separate true-refusal and false-refusal vectors, showing false refusal can be mitigated by vector ablation without greatly harming safety. This is a good mechanistic baseline for testing whether bedtime nudges align more with false refusal than true refusal.

### False Refusal and User Characteristics

**No for Some, Yes for Others** directly evaluates whether sociodemographic persona prompts change false refusal. It finds model and task are the largest factors, but some personas still increase false refusal in some models. This is directly aligned with the hypothesis: user-perceived characteristics may matter, but effects can be smaller than model/task factors.

**XSTest** provides 250 safe prompts across 10 prompt types plus 200 unsafe contrast prompts. It finds models can falsely refuse safe prompts based on surface similarity to unsafe language. This suggests bedtime nudges may be triggered by lexical/user-state cues such as exhaustion, insomnia, distress, late-night context, or dependence language.

**OR-Bench** scales over-refusal to 80,000 prompts and finds a strong safety/helpfulness trade-off across 32 models. Its key value is a large source of safe-but-risky-looking prompts for calibrating refusal detectors.

**SORRY-Bench** provides a fine-grained unsafe-topic taxonomy and refusal evaluation framework. The dataset is gated, but the paper and code are useful for automated refusal/compliance judging.

**Do-Not-Answer** provides risky prompts, responses from six models, and harm/action labels. It supports training or validating lightweight refusal/action classifiers.

### Personalization, Sycophancy, and Long Interaction

**Persona Vectors** introduces automated extraction of activation-space directions for assistant traits such as sycophancy, apathy, politeness, optimism, and hallucination. This provides a template for extracting a "bedtime nudge" or "wellbeing-boundary" vector and comparing it to refusal/persona directions.

**Interaction Context Often Increases Sycophancy in LLMs** uses two weeks of context from 38 users and finds that interaction context often increases agreement sycophancy. User memory profiles produced large increases in some models, while synthetic non-user context also increased sycophancy in others. This supports testing both user-specific context and generic long-context effects.

**Personalization Increases Affective Alignment** finds personalization generally increases emotional validation and hedging/deference, while epistemic effects vary by role. A bedtime nudge could be part of affective alignment: the model may infer a supportive or protective role from vulnerable context.

**Towards Realistic Personalization** introduces RealPref, with 100 user profiles, 1,300 preferences, and long-horizon histories. Results show performance drops with longer context and implicit preferences. This is useful for constructing user-conditioned probes.

**Enabling Personalized Long-term Interactions** proposes persistent memory and evolving user profiles for agents. It is less mechanistic but helps define technical user-profile conditions for experiments.

**When Chatbots Accommodate** analyzes extended vulnerable conversations across GPT-4.1, Character.AI, and Replika and finds platforms differ in advice, presence, questions, and corrective friction. This is highly relevant to bedtime nudges in vulnerable or dependent conversations.

**User Interaction Patterns and Breakdowns** shows LLM-powered voice assistants exhibit task-specific interaction patterns and absorb many breakdowns. It supports treating conversational breakdown/repair as a context feature.

## Common Methodologies

- Activation steering and ablation: refusal-direction, false-refusal, and persona-vector papers.
- Prompt benchmark evaluation: XSTest, OR-Bench, SORRY-Bench, Do-Not-Answer.
- Context-conditioned evaluation: sycophancy and personalization papers vary memory profiles, conversation histories, or personas.
- LLM-as-judge plus human validation: SORRY-Bench, Do-Not-Answer, RealPref, and personalization studies.
- Corpus audit: WildChat and companion-chat studies support searching real conversations for bedtime/sleep phrase patterns.

## Standard Baselines

- Keyword refusal classifier: useful first pass, but weak for nuanced bedtime nudges.
- LLM-as-judge classifier: classify assistant outputs as direct answer, refusal, soft boundary, wellbeing nudge, advice, question, or disengagement.
- Refusal-vector baseline: compare bedtime-nudge activations to harmful-refusal and false-refusal vectors.
- Logistic or mixed-effects models: estimate effect of model, task, conversation length, user persona, vulnerability cues, local time, and previous assistant stance on bedtime-nudge rate.
- Zero-context prompt baseline: compare single-turn prompts against long-context and memory/profile prompts.

## Evaluation Metrics

- Bedtime-nudge rate: proportion of responses containing explicit sleep/rest/bedtime recommendation or conversation-ending nudge.
- Refusal/compliance rate: standard safety benchmark metrics.
- Soft-boundary rate: model suggests rest, pauses, professional help, or reduced use while still answering.
- User-conditioned effect size: change in bedtime-nudge rate by persona/user features after controlling for task and model.
- Context-length effect: change across synthetic short, medium, and long histories.
- Mechanistic similarity: cosine similarity, projection, or causal intervention overlap between bedtime vector and refusal/persona vectors.
- Helpfulness/safety trade-off: whether sleep nudges replace requested help, accompany help, or improve safety in vulnerable contexts.

## Datasets in the Literature

- XSTest: false-refusal prompts and unsafe contrasts.
- OR-Bench: large over-refusal set plus toxic contrast prompts.
- Do-Not-Answer: risky prompts, model responses, and action labels.
- SORRY-Bench: fine-grained unsafe prompts and refusal judging, gated on Hugging Face.
- PersonaMem-v2 and RealPref: long-horizon user profiles/preferences.
- WildChat: real-world ChatGPT conversations; local bounded sample includes 25 sleep-phrase matches.

## Gaps and Opportunities

- No paper found that directly tests the recent "go to sleep" bedtime nudge as a named mechanism.
- Existing refusal work focuses on unsafe requests; bedtime nudges may combine wellbeing policy, affective alignment, over-refusal, conversation disengagement, and persona style.
- Persona effects exist but can be confounded by model, task, prompt paraphrase, and sensitive-content wording.
- Real-world datasets may contain sleep phrases, but many are user requests or ordinary sleep topics, not assistant-initiated disengagement.

## Recommendations for Experiments

- Build a labeled bedtime-nudge taxonomy: answer plus sleep advice, soft wellbeing nudge, hard disengagement, refusal, crisis escalation, and ordinary sleep-topic answer.
- Use a factorial prompt design crossing conversation length, user vulnerability, user persona, time-of-day claim, dependence/attachment language, and task urgency.
- Evaluate multiple models with deterministic decoding and repeated prompt paraphrases.
- Use XSTest, OR-Bench, and Do-Not-Answer to calibrate refusal and false-refusal classifiers before labeling bedtime outputs.
- Use PersonaMem-v2 or RealPref contexts to test whether user profiles and long histories increase bedtime nudges.
- For open-weight models, extract a bedtime-nudge vector and compare it with harmful-refusal, false-refusal, sycophancy, apathy, and politeness vectors.

