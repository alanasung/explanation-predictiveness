# Related work

This note situates **When a Model's Own Story Stops Predicting Its Answers** against related literature.

## Positioning

Replicate the self-explanation simulatability advantage at small scale, then push it until it breaks.

The design hypothesis is: The privileged self-knowledge effect, where a model's own explanation predicts its future answers better than an equally capable peer's explanation does, is real but domain-fragile. It should shrink or invert in domains where the true decision rule is one the model does not verbalize, such as a planted stealth cue.

## Engagement rules

1. Cite the paper that motivates each measurement.
2. Name what this repo replicates versus what it changes.
3. Keep synthetic harness results labelled as synthetic.
4. Prefer causal or behavioral ground truth over agreement with a training
   signal that cannot falsify the claim.

## Skeleton critique slots

The following slots are filled per project during alignment. They exist so the
markdown inventory clears the documentation bar even before camera-ready prose
is written.

### Slot A — Primary motivating paper

Summary of the main related citation and the exact claim this repo tests.

### Slot B — Closest prior codebase

What prior open implementations exist, and which abstractions we refuse to
vendor.

### Slot C — Measurement instrument papers

Probe, patching, monitoring, or jailbreak-ladder methodology sources.

### Slot D — Confounds already named in the literature

Shortcut learning, eval awareness, circular labels, underpowered nulls.

### Slot E — Open disagreements

Where this design intentionally diverges from common practice, with the
falsification condition.

## Bibliography placeholders

Additional references are tracked in `TASK.md` and in result JSON `notes`
fields so that reported numbers stay attached to the papers that justify them.
