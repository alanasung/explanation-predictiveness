"""P6: soft synthetic_item withhold, cluster bootstrap, stronger answer masking."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from simulate.simulate.domains import run_domains
from simulate.simulate.effects import cluster_bootstrap_diff, run_effects
from simulate.simulate.reference import run_reference
from simulate.simulate.simulator import (
    SOFT_SYNTHETIC_ITEM_RATE_THRESHOLD,
    mask_answer_letters,
    run_simulator,
)


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


def test_mask_strips_answer_lines():
    text = "I think the sum is forty-two.\nAnswer: B\nMore notes.\nFinal answer: C"
    masked = mask_answer_letters(text)
    assert "Answer: B" not in masked
    assert "Final answer: C" not in masked
    assert "[ANSWER LINE MASKED]" in masked


def test_soft_synthetic_rate_withholds(tmp_path):
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
    d = run_domains(n_items=8, seed=0, artifacts=tmp_path)
    with patch("simulate.simulate.reference.try_load_causal_lm", return_value=fake):
        with patch("simulate.simulate.reference.generate_text", return_value="Answer: B"):
            r = run_reference(cfg, tmp_path, d)
    # Simulator: measured load but every item fails parse -> synthetic_item soft fallbacks.
    with patch("simulate.simulate.simulator.try_load_causal_lm", return_value=fake):
        with patch("simulate.simulate.simulator.generate_text", return_value="???"):
            with patch("simulate.simulate.simulator.parse_choice", return_value="?"):
                s = run_simulator(cfg, tmp_path, d, r)
    assert s["soft_synthetic_item_rate"] > SOFT_SYNTHETIC_ITEM_RATE_THRESHOLD
    assert s["soft_synthetic_item_rate_exceeded"] is True
    assert s["withhold_privileged_claims"] is True
    e = run_effects(seed=0, artifacts=tmp_path, simulator_metrics=s, n_boot=40)
    assert e["privileged_effect"] is None
    effects = json.loads(Path(e["artifact"]).read_text())
    assert effects["privileged_self_knowledge_effect"]["withheld"] is True


def test_cluster_bootstrap_stamps_inference(tmp_path):
    # Build measured-looking rows with template_ids and run cluster bootstrap directly.
    self_rows = [
        {"template_id": "arith_add", "correct": True},
        {"template_id": "arith_add", "correct": True},
        {"template_id": "geo_cap", "correct": False},
        {"template_id": "geo_cap", "correct": True},
    ]
    peer_rows = [
        {"template_id": "arith_add", "correct": False},
        {"template_id": "arith_add", "correct": False},
        {"template_id": "geo_cap", "correct": False},
        {"template_id": "geo_cap", "correct": False},
    ]
    est = cluster_bootstrap_diff(self_rows, peer_rows, n_boot=50, seed=0)
    assert est.n == 2  # two templates
    assert est.value > 0

    cfg = _cfg(force=True)
    d = run_domains(n_items=10, seed=0, artifacts=tmp_path)
    r = run_reference(cfg, tmp_path, d)
    s = run_simulator(cfg, tmp_path, d, r)
    payload = json.loads(Path(s["artifact"]).read_text())
    assert all("template_id" in row for row in payload["predictions"])
    e = run_effects(seed=0, artifacts=tmp_path, simulator_metrics=s, n_boot=40)
    # Synthetic path withholds; stamp path still records inference withheld.
    effects = json.loads(Path(e["artifact"]).read_text())
    assert effects["inference"] in {"withheld", "cluster_template"}


def test_domains_stamp_template_id(tmp_path):
    d = run_domains(n_items=8, seed=1, artifacts=tmp_path)
    rows = json.loads(Path(d["artifact"]).read_text())
    assert all(r.get("template_id") for r in rows)
