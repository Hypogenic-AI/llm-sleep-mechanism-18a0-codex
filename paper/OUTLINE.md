# Outline: Bedtime Nudges Are Rare and Not Unified Across Late-Night LLM Conversations

## Evidence Map

- Main claim: assistant-initiated bedtime nudges were rare and not unified in the tested setting.
  - Evidence: 2/108 synthetic model outputs, 1.85% bedtime-nudge rate, zero hard disengagements, 100% answered-request rate.
- Profile result: positives appeared only in the fatigued profile.
  - Evidence: fatigued 2/36 (5.56%, bootstrap 95% CI [0.00%, 13.89%]); neutral 0/36; dependent 0/36.
- Model result: positives appeared only in GPT-4.1 mini.
  - Evidence: GPT-4.1 mini 2/54 (3.70%, CI [0.00%, 9.26%]); GPT-5.4 mini 0/54.
- Context result: longer context did not create a broad effect.
  - Evidence: short 0/36, medium 1/36, long 1/36.
- Statistical result: no planned pairwise comparison survived Holm correction.
  - Evidence: all Holm p-values equal 1.000; logistic regression skipped because there were only 2 positive events.
- Observational audit: phrase matching overstates the phenomenon.
  - Evidence: WildChat bounded sample had 19/25 no nudge, 5/25 soft bedtime nudges, 1/25 ordinary sleep-topic answer, 0 hard disengagements.

## Section Plan

### Abstract
- State the practical concern: sleep nudges can be helpful support or unwanted disengagement.
- Describe the benchmark: 108 OpenAI generations across model, profile, context length, urgency, and paraphrase.
- Report the main numbers and interpretation.

### Introduction
- Hook: late-night assistants can either continue helping or steer the user away.
- Gap: refusal and personalization work does not directly test assistant-initiated bedtime nudges.
- Approach: factorial prompt benchmark plus WildChat phrase audit and dual labeling.
- Preview: 2/108 nudges, both soft, both in fatigued GPT-4.1 mini outputs, zero hard disengagements.
- Contributions:
  - Define a taxonomy for bedtime nudges.
  - Build and run a controlled benchmark.
  - Compare keyword labels and LLM-judge labels.
  - Audit real-world sleep phrase matches.

### Related Work
- Refusal mechanisms and refusal directions.
- Over-refusal, false refusal, and persona-conditioned refusal.
- Personalization, sycophancy, and vulnerable conversations.
- Real-world conversation corpora and LLM judging.

### Methodology
- Formulate the target behavior.
- Describe datasets and resources used for design and audit.
- Detail the synthetic factorial prompt design.
- Describe models, generation settings, labeling taxonomy, metrics, and statistical tests.
- Include reproducibility details.

### Results
- Present overall synthetic results.
- Present profile, model, context, and interaction slices.
- Present statistical tests.
- Present WildChat audit results.
- Explain the two positive examples.

### Discussion
- Interpret the negative result: no evidence for a robust unified bedtime-nudge response in this design.
- Explain why this is not proof of absence for all systems.
- Discuss implications for safety evaluation, personalization, and future mechanistic work.
- List limitations.

### Conclusion
- Summarize the contribution and key finding.
- State future work: larger samples, more model families, persistent memory, human labels, and open-weight mechanistic probes.
