# VALIDATION — faithfulness-introspection

## Codex v1 (historical)
- Verdict: SERIOUS_PROBLEMS
- Summary: The repository has reasonable generic infrastructure and a plausible research direction, but its experiment is unimplemented, its DAG and model configuration contradict the stated design, and its current sample size and controls cannot support a valid privileged-self-knowledge claim.

## Codex v2
- Verdict: PASS_WITH_NOTES
- Summary: Analogous to introspection-verbalization Codex v2: X1–X13 OK; stages implemented with a real `make pilot` path; synthetic/proxy pilot default; several model revisions still on `main`.
- KEY_FIXES_OK: X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13

## Grok (dual-validate)
- Verdict: PASS_WITH_NOTES
- Summary: Pilot DAG is domains→reference→simulator→effects→welfare with role-keyed models (X6) and stealth cues. Stages real; synthetic fallback when weights missing — PASS_WITH_NOTES analogous to introspect.

### Remaining
- Reference/simulator stages use synthetic explanations when weights are absent; pilot still completes the domains→welfare DAG.
- Role model revisions remain `main`.

## Reconciliation
v1 DAG/config contradictions fixed (domains before reference; roles). Grok PASS_WITH_NOTES matching Codex v2 style.
