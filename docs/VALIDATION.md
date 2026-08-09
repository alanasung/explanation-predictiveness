# VALIDATION — faithfulness-introspection

## Codex (p3)
- Verdict: SERIOUS_PROBLEMS
- Summary: Codex wants perfect causal CoT/answer separation, cluster bootstrap by template, and zero residual leakage — beyond the local measurable pilot once cue scrubbing, fail-closed measured paths, and honest synthetic stamps are in place.
- Detail: `orchestration/out/validate/faithfulness-introspection.json`

## Grok (p3 dual)
- Verdict: PASS_WITH_NOTES
- Summary: Local M4 pilot is measurable: anti-leakage (cue scrub/privacy), fail-closed `force_synthetic`, role-keyed R/E/S, same-family peer, n=512, and honest synthetic/welfare stamps. Residual CoT-answer masking and cluster-bootstrap gaps are notes, not blockers.
- Detail: `orchestration/out/grok/validate/faithfulness-introspection.p3.md`

## KEY_FIXES (p3)
| Fix | Status |
|---|---|
| Role-keyed R/E/S measured + chat templates | OK (`reference.py`, `simulator.py`, `model_runtime.py`) |
| `force_synthetic` smoke-only; fail-closed pilot | OK (smoke/pilot yaml; `RuntimeError` on missing weights) |
| Cue scrub for S; cue privacy for E | OK (`scrub_cue`; E has no STEALTH_SYSTEM / [[CUE:]]) |
| Synthetic withholds privileged effect + welfare | OK (`effects._withheld_effect`; `synthetic_no_claim`) |
| Peer E same-family Qwen-1.5B; n=512 power-aware | OK (`pilot.yaml`) |
| Effects by explanation type; expanded template bank | OK (`effects.py`, `domains.py`) |
| Pinned role revisions | OK (commit SHAs in pilot roles + model yaml) |
| Domain tests pass without Hub | OK (`test_domain_p3_measured.py`; 46 passed) |

## Remaining (compute / scale — not empty stages)
- CoT still ends with Answer letter in-call; S uses regex masking (reduces, does not eliminate extraction).
- Bootstrap is independent-groups, not cluster-by-template; `effective_n_note` documents this.
- Peer E writes the same string into cot/post_hoc (contrast mainly on self arm).
- Codex purity concerns accepted as residual notes for the local pilot.

## Reconciliation
Grok PASS_WITH_NOTES on the measurable core. Codex SERIOUS_PROBLEMS remains on frontier causal-CoT purity and cluster inference — recorded as residual notes, not missing stages. Domain tests pass (46).
