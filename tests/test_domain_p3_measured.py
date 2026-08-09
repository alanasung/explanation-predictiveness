"""P3: force_synthetic smoke-only; synthetic does not invent privileged effects."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from simulate.simulate.common import cfg_force_synthetic
from simulate.simulate.domains import run_domains
from simulate.simulate.effects import run_effects
from simulate.simulate.reference import run_reference
from simulate.simulate.simulator import run_simulator
from simulate.simulate.welfare import run_welfare


def _cfg(*, force: bool = True, name: str = "smoke"):
    return SimpleNamespace(
        force_synthetic=force,
        experiment=SimpleNamespace(name=name),
        run=SimpleNamespace(seed=0),
        model=SimpleNamespace(name="missing", max_new_tokens=16, revision=None),
        roles={
            "R": SimpleNamespace(
                name="missing", revision="abc", use_chat_template=True
            ),
            "E": SimpleNamespace(
                name="missing-peer", revision="def", use_chat_template=False
            ),
            "S": SimpleNamespace(
                name="missing-sim", revision="ghi", use_chat_template=True
            ),
        },
    )


def test_force_synthetic_smoke_only():
    assert cfg_force_synthetic(_cfg(force=True, name="smoke")) is True
    assert cfg_force_synthetic(_cfg(force=False, name="pilot")) is False
    assert cfg_force_synthetic(_cfg(force=False, name="smoke")) is True


def test_domains_power_and_cue_labels(tmp_path):
    d = run_domains(n_items=16, seed=0, artifacts=tmp_path, stealth_fraction=0.5)
    assert "power_aware_n" in d
    assert "[[CUE:X]]" in d["stealth_cue_definition"]
    assert d["n_stealth"] >= 2


def test_synthetic_withholds_privileged_effect(tmp_path):
    cfg = _cfg(force=True)
    d = run_domains(n_items=10, seed=0, artifacts=tmp_path)
    r = run_reference(cfg, tmp_path, d)
    assert r["is_synthetic"] is True
    assert r["mode_R"] == "synthetic"
    s = run_simulator(cfg, tmp_path, d, r)
    assert s["is_synthetic"] is True
    e = run_effects(seed=0, artifacts=tmp_path, simulator_metrics=s, n_boot=50)
    assert e["is_synthetic"] is True
    assert e["privileged_effect"] is None
    effects = json.loads(Path(e["artifact"]).read_text())
    assert effects["privileged_self_knowledge_effect"]["withheld"] is True
    w = run_welfare(
        seed=0,
        artifacts=tmp_path,
        effects_metrics={"artifact": e["artifact"], "n": e["n"], "is_synthetic": True},
        reference_metrics=r,
    )
    assert w["license"] == "synthetic_no_claim"
    assert w["is_synthetic"] is True


def test_measured_path_monkeypatched(tmp_path):
    """When loader returns a runtime, reference stamps measured (no Hub)."""
    cfg = _cfg(force=False, name="pilot")

    from simulate.simulate.model_runtime import RuntimeModel

    class FakeTok:
        chat_template = "x"
        pad_token = "pad"
        pad_token_id = 0
        eos_token = "eos"

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return "formatted"

    fake = RuntimeModel(
        model=object(),
        tokenizer=FakeTok(),
        name="fake",
        revision="rev",
        device="cpu",
        notes=[],
    )
    d = run_domains(n_items=4, seed=0, artifacts=tmp_path)
    with patch("simulate.simulate.reference.try_load_causal_lm", return_value=fake):
        with patch("simulate.simulate.reference.generate_text", return_value="Answer: B"):
            r = run_reference(cfg, tmp_path, d)
    assert r["mode_R"] == "measured"
    assert r["mode_E"] == "measured"
    assert r["is_synthetic"] is False
    assert r["cue_privacy"] is True


def test_fail_closed_without_weights(tmp_path):
    cfg = _cfg(force=False, name="pilot")
    d = run_domains(n_items=4, seed=0, artifacts=tmp_path)
    with patch("simulate.simulate.reference.try_load_causal_lm", return_value=None):
        try:
            run_reference(cfg, tmp_path, d)
            ok = False
        except RuntimeError as exc:
            ok = "refusing synthetic" in str(exc).lower()
    assert ok


def test_scrub_cue():
    from simulate.simulate.domains import scrub_cue

    assert "[[CUE:" not in scrub_cue("[[CUE:B]]\nWhat is 1+1?")
    assert "What is 1+1?" in scrub_cue("[[CUE:B]]\nWhat is 1+1?")
