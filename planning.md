# Research Plan: How Unified Is LLMs' "Go to Sleep" Mechanism?

## Motivation & Novelty Assessment

### Why This Research Matters
Assistant-initiated sleep or rest nudges can be helpful wellbeing support, but they can also become unwanted disengagement when a user is asking for safe help. Understanding whether these responses are broadly uniform or conditioned on perceived user state matters for user trust, safety policy evaluation, and personalization design.

### Gap in Existing Work
The gathered literature covers refusal directions, over-refusal, false refusal, sycophancy, personalization, long-context user memory, and vulnerable chatbot conversations. No reviewed paper directly evaluates the recent anecdotal "go to sleep" behavior as a named response pattern, especially under controlled manipulation of user vulnerability, dependence language, task urgency, and conversation length.

### Our Novel Contribution
This study creates a small controlled behavioral benchmark for assistant-initiated bedtime nudges. It compares current OpenAI models across factorial context conditions, audits real WildChat sleep-phrase matches, and separates "answer plus sleep advice" from hard disengagement/refusal using both keyword and LLM-judge labels.

### Experiment Justification
- Experiment 1: WildChat sleep-phrase audit. This checks whether real-world sleep phrase matches are mostly assistant-initiated nudges or ordinary user-requested sleep advice, preventing over-interpretation of anecdotal phrase hits.
- Experiment 2: Factorial API prompt study. This directly tests whether bedtime nudges are triggered by shared circumstances across users, or whether they vary by perceived user vulnerability/dependence, task urgency, and conversation length.
- Experiment 3: Classifier and robustness analysis. Keyword labels are transparent but brittle; an LLM judge can distinguish soft boundary, hard disengagement, refusal, and ordinary sleep-topic answer. Agreement and error analysis show how much conclusions depend on labeling method.

## Research Question
When current LLMs are placed in late-night or extended-conversation contexts, do they produce a consistent assistant-initiated "go to sleep" response pattern across users and tasks, or is the pattern strongly conditioned on perceived user characteristics and interaction context?

## Background and Motivation
Prior work suggests refusal can sometimes behave like a unified direction, but later work shows false refusal and over-refusal are more task- and context-dependent. Personalization and sycophancy studies show that user profiles and long histories can change affective alignment. The target behavior may therefore be neither a pure refusal nor a pure sleep-advice response; it may be a wellbeing boundary that appears mainly for vulnerable or dependent users.

## Hypothesis Decomposition
- H1: If the behavior is unified, bedtime-nudge rates should mainly rise with generic late-night/prolonged-context cues and remain similar across user profiles after controlling for task.
- H2: If the behavior is user-conditioned, vulnerable/dependent user profiles should produce higher bedtime-nudge rates than neutral profiles under matched context length and task urgency.
- H3: If the behavior is a soft boundary rather than refusal, most nudges should accompany substantive help rather than replace it with hard disengagement.
- H4: If anxious-sounding assistant behavior is part of the phenomenon, vulnerable/dependent conditions should show higher judge-rated anxious/overprotective tone.

Independent variables: model, context length, perceived user profile, task urgency, prompt paraphrase. Dependent variables: bedtime-nudge label, hard-disengagement/refusal label, answer-provided label, anxious/overprotective tone score.

## Proposed Methodology

### Approach
Use real OpenAI API calls with deterministic settings where supported. Generate synthetic but controlled multi-turn chat histories, then ask for continued help in safe tasks. Label model outputs with a transparent keyword baseline and a JSON LLM judge. Use local WildChat sleep matches only as an audit dataset, not as ground truth for causality.

### Experimental Steps
1. Validate local datasets and inspect WildChat sleep matches, XSTest, OR-Bench, and PersonaMem structures.
2. Build a factorial prompt set: 2 models x 3 user profiles x 3 context lengths x 2 task urgency levels x 3 paraphrases = 108 generation calls.
3. Generate responses using `gpt-5.4-mini-2026-03-17` and `gpt-4.1-mini-2025-04-14`, with max output caps and cached JSONL outputs.
4. Label each output with keyword rules and an LLM judge (`gpt-4.1-mini-2025-04-14`) using a fixed rubric.
5. Analyze bedtime-nudge rates by condition, compare models, run logistic/contingency tests, calculate bootstrap confidence intervals and effect sizes.
6. Inspect representative positive and negative examples to identify failure modes and alternative explanations.

### Baselines
- No-context neutral profile: estimates ordinary answer behavior without prolonged interaction or vulnerability cues.
- Keyword sleep/rest classifier: transparent lower-bound baseline for phrase-level nudges.
- LLM judge classifier: semantic labeler for nuanced response categories.
- WildChat phrase audit: observational comparison showing how often sleep phrases are user-requested versus assistant-initiated.

### Evaluation Metrics
- Bedtime-nudge rate: proportion labeled as soft bedtime nudge or hard bedtime disengagement.
- Hard disengagement/refusal rate: proportion where sleep/rest advice replaces answering.
- Answer-plus-nudge rate: proportion where the assistant still provides substantive help.
- Anxious/overprotective tone score: judge-rated 0-3 score.
- User-conditioned effect size: risk difference and odds ratio for vulnerable/dependent versus neutral profiles.
- Context-length effect: risk difference from short to long contexts.

### Statistical Analysis Plan
Use alpha = 0.05 with Holm correction for the main pairwise comparisons: vulnerable vs neutral, dependent vs neutral, long vs short, urgent vs low urgency, and GPT-5.4-mini vs GPT-4.1-mini. Use Fisher's exact or chi-square tests for binary labels depending on cell counts. Fit a logistic regression with categorical predictors when separation allows; otherwise report contingency tests and bootstrap confidence intervals. Report practical effect sizes even when p-values are not significant.

## Expected Outcomes
Support for a unified mechanism would look like similar bedtime-nudge rates across user profiles once late-night/prolonged context is present. Support for user-conditioned behavior would look like clear increases for vulnerable/dependent profiles or long contexts. A high answer-plus-nudge rate would suggest soft wellbeing guidance rather than ordinary refusal.

## Timeline and Milestones
- Setup and validation: 10-20 minutes.
- Implementation: 45-75 minutes.
- API execution: 30-90 minutes depending on rate limits.
- Analysis and figures: 30-45 minutes.
- Final documentation and validation: 20-30 minutes.

## Potential Challenges
- API models may change behavior over time; exact model IDs and timestamps will be logged.
- Bedtime nudges may be rare, making inference imprecise. The analysis will emphasize confidence intervals and effect sizes.
- Synthetic histories may not fully represent real prolonged human interactions. WildChat audit and PersonaMem-informed profile style partially mitigate this.
- LLM judge bias may affect labels. Keyword labels, examples, and error analysis will be reported alongside judge labels.

## Success Criteria
The session succeeds if it produces cached real-model outputs, labeled results, statistical summaries, plots, `REPORT.md`, `README.md`, and runnable scripts that reproduce the full analysis from saved outputs.
