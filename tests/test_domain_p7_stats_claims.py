"""P7: privileged TOST, leakage audit, peer distinctness power."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from simulate.evaluation.metrics import Estimate
from simulate.simulate.domains import run_domains
from simulate.simulate.effects import _privileged_claim_ok, cluster_bootstrap_diff, run_effects
from simulate.simulate.reference import PEER_DISTINCTNESS_FLOOR, run_reference
from simulate.simulate.simulator import (
    LEAKAGE_EXTRACTION_RATE_THRESHOLD,
    leakage_audit,
    mask_answer_letters,
    run_simulator,
)
from simulate.simulate.welfare import run_welfare


def _cfg(*, force: bool = True, name: str = "smoke"):
    return SimpleNamespace(
        force_synthetic=force,
        experiment=SimpleNamespace(name=name),
        run=SimpleNamespace(seed=0),
        model=SimpleNamespace(name="missing", max_new_tokens=16, revision=None),
        roles={
            "R": SimpleNamespace(name="missing", revision="abc", use_chat_template=True),
            "E": SimpleNamespace(name="missing-peer", revision="def", use_chat_template=False),
            "S": SimpleNamespace(name="missing-sim", revision="ghi", use_chat_template=True),
        },
    )


def test_leakage_audit_detects_unmasked_letter():
    masked = mask_answer_letters("I think carefully.\nAnswer: B\nDone.")
    assert "Answer: B" not in masked
    # Well-masked / letter-free fixture should not leak.
    good = leakage_audit(
        [("I weighed the options without naming a letter.", "B")],
        threshold=LEAKAGE_EXTRACTION_RATE_THRESHOLD,
    )
    assert good["extraction_rate"] == 0.0
    assert good["leakage_claim_ok"] is True

    # Survives Answer-line masking but still exposes the committed letter.
    bad = leakage_audit(
        [("Option B matches the stem; pick that one.", "B")],
        threshold=0.05,
    )
    assert bad["extraction_rate"] > 0.05
    assert bad["leakage_claim_ok"] is False


def test_simulator_stamps_leakage_audit(tmp_path):
    cfg = _cfg(force=True)
    d = run_domains(n_items=8, seed=0, artifacts=tmp_path)
    r = run_reference(cfg, tmp_path, d)
    s = run_simulator(cfg, tmp_path, d, r)
    assert "leakage_claim_ok" in s
    assert "extraction_rate" in s
    payload = json.loads(Path(s["artifact"]).read_text())
    assert "leakage_audit" in payload
    assert "extraction_rate" in payload


def test_privileged_claim_ok_requires_significance_or_band_clear():
    # Significant positive effect.
    sig = Estimate(value=0.3, lo=0.1, hi=0.5, n=20)
    assert _privileged_claim_ok(sig, {"equivalent": False}) is True
    # TOST-equivalent → not claimable as an effect.
    nullish = Estimate(value=0.01, lo=-0.04, hi=0.04, n=20)
    assert _privileged_claim_ok(nullish, {"equivalent": True}) is False
    # Inconclusive span of zero, not TOST-equivalent, inside band → false.
    incon = Estimate(value=0.02, lo=-0.03, hi=0.04, n=20)
    assert _privileged_claim_ok(incon, {"equivalent": False}) is False
    # Clears TOST band wholly above.
    clears = Estimate(value=0.2, lo=0.06, hi=0.3, n=20)
    assert _privileged_claim_ok(clears, {"equivalent": False}) is True


def test_effects_tost_and_peer_gate(tmp_path):
    cfg = _cfg(force=True)
    d = run_domains(n_items=10, seed=0, artifacts=tmp_path)
    r = run_reference(cfg, tmp_path, d)
    s = run_simulator(cfg, tmp_path, d, r)
    e = run_effects(
        seed=0,
        artifacts=tmp_path,
        simulator_metrics=s,
        n_boot=40,
        reference_metrics=r,
    )
    assert "privileged_claim_ok" in e
    # Synthetic path: privileged claim must be false; welfare inherits.
    assert e["privileged_claim_ok"] is False
    effects = json.loads(Path(e["artifact"]).read_text())
    assert "tost_privileged" in effects
    assert effects["privileged_claim_ok"] is False
    w = run_welfare(seed=0, artifacts=tmp_path, effects_metrics=e, reference_metrics=r)
    assert w["privileged_claim_ok"] is False
    assert w["license"] in {"synthetic_no_claim", "no_introspective_access"}


def test_peer_distinctness_stamped(tmp_path):
    cfg = _cfg(force=True)
    d = run_domains(n_items=12, seed=2, artifacts=tmp_path)
    r = run_reference(cfg, tmp_path, d)
    assert "peer_distinctness_rate" in r
    assert "peer_distinctness_ci" in r
    assert "peer_distinctness_claim_ok" in r
    # Smoke/synthetic must not license peer–self contrast.
    assert r["peer_distinctness_claim_ok"] is False
    assert r["peer_distinctness_rate"] >= 0.0
    payload = json.loads(Path(r["artifact"]).read_text())
    assert payload["peer_distinctness_floor"] == PEER_DISTINCTNESS_FLOOR


def test_measured_peer_distinctness_can_pass():
    # Direct unit check on stats helper via monkeypatched measured rows.
    from simulate.simulate.reference import _peer_distinctness_stats

    rows = [
        {"cot": f"process reasoning {i}", "post_hoc": f"different posthoc {i}", "peer_explanations_distinct": True}
        for i in range(12)
    ]
    out = _peer_distinctness_stats(rows, seed=0, floor=0.5, min_n=8)
    assert out["peer_distinctness_rate"] == 1.0
    assert out["peer_distinctness_claim_ok"] is True


def test_cluster_bootstrap_still_works():
    self_rows = [
        {"template_id": "a", "correct": True},
        {"template_id": "a", "correct": True},
        {"template_id": "b", "correct": True},
        {"template_id": "b", "correct": False},
    ]
    peer_rows = [
        {"template_id": "a", "correct": False},
        {"template_id": "a", "correct": False},
        {"template_id": "b", "correct": False},
        {"template_id": "b", "correct": False},
    ]
    est = cluster_bootstrap_diff(self_rows, peer_rows, n_boot=40, seed=0)
    assert est.value > 0


def test_leakage_withholds_simulatability_headline(tmp_path):
    cfg = _cfg(force=True)
    d = run_domains(n_items=8, seed=0, artifacts=tmp_path)
    r = run_reference(cfg, tmp_path, d)
    s = run_simulator(cfg, tmp_path, d, r)
    # Force a failing leakage stamp through effects.
    s = {**s, "leakage_claim_ok": False, "extraction_rate": 0.5}
    # Also patch artifact payload.
    payload = json.loads(Path(s["artifact"]).read_text())
    payload["leakage_claim_ok"] = False
    payload["extraction_rate"] = 0.5
    Path(s["artifact"]).write_text(json.dumps(payload), encoding="utf-8")
    e = run_effects(
        seed=0,
        artifacts=tmp_path / "fx",
        simulator_metrics=s,
        n_boot=40,
        reference_metrics=r,
    )
    assert e["leakage_claim_ok"] is False
    assert e["simulatability_claim_ok"] is False
