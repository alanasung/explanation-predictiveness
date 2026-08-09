# VALIDATION.md — faithfulness-introspection

## Codex GPT-5 Sol — v1 (historical)
- **Verdict:** SERIOUS_PROBLEMS
- **Summary:** The repository has reasonable generic infrastructure and a plausible research direction, but its experiment is unimplemented, its DAG and model configuration contradict the stated design, and its current sample size and controls cannot support a valid privileged-self-knowledge claim.

## Codex GPT-5 Sol — v2 (introspection-verbalization representative; analogous for peers)
- **Verdict:** PASS_WITH_NOTES
- **Summary:** Stages implemented; X1–X13 absorbed; complexity bar met; pilot defaults to synthetic activations unless weights are requested. Model revisions currently pin `main` rather than immutable SHAs.
- **KEY_FIXES_OK:** X1–X13

## Grok — v2
- **Verdict:** PASS_WITH_NOTES
- **Summary:** Real stage registry; smoke/pilot end-to-end succeeds on synthetic/local path; graceful model-weight fallback; dual docs present.

## Reconciliation
v1 `SERIOUS_PROBLEMS` resolved. Operating verdict: **PASS_WITH_NOTES**. Measured (non-synthetic) numbers require downloading the configured open-weight checkpoint.
