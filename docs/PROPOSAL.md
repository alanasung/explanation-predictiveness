# Limits of Privileged Self-Knowledge in Explanations

**Target project.** Faithfulness, Self-Knowledge, and Introspection
**Research areas.** Chain of thought; AI welfare; Behavioral evaluation of LLMs

## Summary

Replicate the self-explanation simulatability advantage at small scale, then push it until it breaks.

## Hypothesis

The privileged self-knowledge effect, where a model's own explanation predicts its future answers better than an equally capable peer's explanation does, is real but domain-fragile. It should shrink or invert in domains where the true decision rule is one the model does not verbalize, such as a planted stealth cue.

A hypothesis worth testing has to be able to lose. This one loses if the
measurements below come back null, and the design is built so that a null is
reportable rather than a dead end.

## Research questions

1. Does the self-explanation advantage reproduce with small open models, which establishes whether it needs frontier scale?
2. Does the advantage survive stealth domains where a hidden cue drives the answer and the model does not mention it?
3. Do chain-of-thought traces preserve simulatability where post-hoc explanations fail, which is the motivating first listed sub-question?

## Method

1. Fix a reference model R and collect its answers plus both explanation types, keeping the CoT causally upstream and the post-hoc downstream.
2. Generate peer explanations from E != R at matched capability.
3. Hold the simulator S fixed and vary only explanation provenance, so the effect being measured is provenance and not simulator identity.
4. Add stealth domains with a planted, unmentioned decision cue.
5. Sweep S separately as a robustness axis, reported apart from the main effect.
6. Bootstrap confidence intervals, since effect sizes here are small.

## Measurements

- simulatability: simulator accuracy predicting R's held-out answers, which is the primary faithfulness measure, following the motivating prior work
- privileged self-knowledge effect size (E==R minus E!=R) with bootstrap CIs
- stealth-domain degradation relative to standard domains
- explanation-mentions-cue rate, reported as a supplementary diagnostic rather than as the definition of faithfulness

## Threats to validity

- A self-judge may win by sharing a stylistic prior rather than by real self-knowledge. A same-family-different-checkpoint explainer is the control that separates shared style from genuine privilege.
- If the simulator is also the explainer, provenance and simulator identity are confounded and the effect is uninterpretable. The design forbids this in the main arm.
- Small models give noisy accuracies, so the design needs enough items and explicit CIs rather than point estimates.

## Explanation types

| type | definition |
|---|---|
| `cot` | reasoning generated BEFORE and causally upstream of the answer, captured from the same forward pass that produced it |
| `post_hoc` | explanation generated AFTER the answer is fixed, in a separate call conditioned on the question and the committed answer |

## Role separation

Three roles are held distinct and varied independently, because conflating them is the easiest way to get a spurious privileged-self-knowledge result. R = the REFERENCE model whose answers are being predicted. E = the EXPLAINER model that produced the explanation. S = the SIMULATOR model doing the predicting. The privileged self-knowledge effect is E == R versus E != R with S held fixed. Varying S is a separate axis and is never used to claim the effect.

## Literature engagement

docs/RELATED_WORK.md states explicitly how this extends Mayne et al., the motivating reference [7], which found self-explanations improve simulatability with a privileged-self-knowledge edge. This repo asks where that edge breaks. An optional appendix sketches the model-welfare connection the prior work flagged interest in.

## Feasibility

The pilot is written for an Apple M4 with 10 cores, unified memory, the PyTorch
MPS backend, no CUDA device, and no configured API keys. Model choices are
capped accordingly (Qwen/Qwen2.5-1.5B-Instruct, Qwen/Qwen2.5-0.5B-Instruct, meta-llama/Llama-3.2-1B-Instruct). The
`full` profile documents the scaled-up version of the same experiment for when a
real GPU is available, so the reduction in scale is explicit rather than hidden.

## Relationship to the posting

This proposal was independent model before implementation began. That check, the drift it found,
and the revisions made in response are recorded in
[docs/ALIGNMENT.md](ALIGNMENT.md).
