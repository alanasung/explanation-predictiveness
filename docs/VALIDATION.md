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

## P5 rigor pass (measured prior work-critical paths)

- Live / measured paths preferred; synthetic remains smoke-only with honesty stamps.
- Claim gating tightened where proxies previously looked like evidence.
- Domain tests green without Hub downloads.

## P6 rigor pass

| Fix | Status |
|---|---|
| Fail-closed withhold when soft `synthetic_item` rate exceeds threshold | OK (`simulator.py` / `effects.py`) |
| Cluster bootstrap by `template_id` (`inference=cluster_template`) | OK (`effects.cluster_bootstrap_diff`) |
| Stronger answer masking: strip `Answer:` lines before S sees CoT | OK (`mask_answer_letters`) |
| Domain P6 tests Hub-free | OK (`test_domain_p6_failclosed.py`) |

Residual: masking reduces but cannot prove zero extraction; peer contrast still limited on smoke n.

## P7 rigor pass (stats + claim contracts)

| Fix | Status |
|---|---|
| TOST on privileged-effect CI; `privileged_claim_ok` only if significant or clears band | OK (`effects.py`) |
| Welfare inherits `privileged_claim_ok` gate | OK (`welfare.py`) |
| Leakage audit after Answer masking; `leakage_claim_ok` gates simulatability headlines | OK (`simulator.py`) |
| Peer distinctness rate + CI; peer–self contrast requires floor on measured peer | OK (`reference.py`, wired into `effects.py`) |
| Hub-free domain P7 tests | OK (`test_domain_p7_stats_claims.py`) |

Residual: masking audit is heuristic (not proof of zero leakage); licensed corpora still out of scope.

## Codex problem-statement fit

Model: `gpt-5.6-sol` · gates: `match` + `validate` · 2026-08-09
Artifacts: `orchestration/out/match/faithfulness-introspection.md`, `orchestration/out/validate/faithfulness-introspection.md`

- **Match verdict:** `MINOR_DRIFT` · methods `mixed_proxy`
- **Validate overall:** `SERIOUS_PROBLEMS` · fit `ALIGNED` · feasibility `RUNNABLE_IF_SHRUNK`
- **Match summary:** Directly aligned with the mentor's stealth-simulatability idea and largely executable locally, but the pilot's S==R assignment confounds its central privileged-self-knowledge comparison.
- **Validate summary:** Excellent topical fit and substantial implementation effort, but invalid CoT matching, no held-out questions, the wrong stealth estimand, and an impractically unbatched runtime make the current pilot scientifically unreliable.

### Top drift / missing (match)
- Pilot sets S to the exact R checkpoint, so the self arm has S==E while the peer arm has S!=E; explanation provenance is confounded with simulator-explainer identity.
- Peer CoT is generated after seeing R's answer, whereas self CoT is causally upstream, so the CoT provenance comparison is not matched.
- Generic hooks, probes, monitor infrastructure, and ablations are not used by the registered faithfulness stages and cannot support mechanistic claims.
- The welfare index is an authored interpretation of simulatability metrics rather than a validated sentience or welfare measure.

### Blocking (validate)
- `src/simulate/simulate/reference.py`: Peer cot is generated after E is shown R's committed answer, so it is a post-hoc rationalization and cannot be compared with R's causally upstream CoT.
- `src/simulate/simulate/domains.py`: The 512-item pilot repeats only 24 base questions and never applies the configured train/validation/test split, so it does not evaluate prediction on held-out new questions.
- `src/simulate/simulate/effects.py`: stealth_domain_degradation measures standard-self accuracy minus stealth-self accuracy, not shrinkage of the privileged self-minus-peer effect.
- `configs/experiment/pilot.yaml`: The allegedly capability-matched peer is Qwen 1.5B while R is Qwen 0.5B, confounding provenance with model capability.
