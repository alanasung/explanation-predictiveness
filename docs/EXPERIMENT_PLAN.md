# Experiment plan — Whose Explanation Helps You Predict the Model

Stage-by-stage design. Each stage is registered in `src/simulate/stages.py`
and appears in `python -m simulate stages`.

## Stages

| stage | responsibility |
|---|---|
| `reference` | reference-model answer and explanation collection |
| `domains` | standard and stealth question domains with planted cues |
| `simulator` | judge-model prediction of reference answers |
| `effects` | effect sizes, bootstrap CIs, per-domain breakdowns |

## Execution order

Stages form a linear dependency chain by default; the runner resolves the order
topologically, so a stage may be run alone and its prerequisites are pulled in
automatically:

```bash
python -m simulate run -c configs/pilot.yaml --stage effects
```

## Controls and their purpose

- A self-judge may win by sharing a stylistic prior rather than by real self-knowledge. A same-family-different-checkpoint explainer is the control that separates shared style from genuine privilege.
- If the simulator is also the explainer, provenance and simulator identity are confounded and the effect is uninterpretable. The design forbids this in the main arm.
- Small models give noisy accuracies, so the design needs enough items and explicit CIs rather than point estimates.

## Decision rules

Report effect sizes with bootstrap intervals. Treat an interval that spans zero as a null result and report it as such; do not reach for a subgroup that reaches significance.

## Reproducibility

Every run records a manifest with the git sha, a config fingerprint, resolved
device and dtype, package versions, per-stage timings, and metrics. Seeds are
set across python, numpy, and torch. Known determinism limits are recorded in
the manifest rather than assumed away: MPS does not support
`torch.use_deterministic_algorithms`, so small numeric drift between runs is
expected and should not be read as an effect.

## Scale

The pilot profile is what actually runs on the target machine. The full profile
describes the intended scaled-up run. When reporting any result, state which
profile produced it; a pilot-scale null is weaker evidence than a full-scale
null and the writeup must not blur them.
