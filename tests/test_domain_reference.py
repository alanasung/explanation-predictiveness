from pathlib import Path
from types import SimpleNamespace
from simulate.simulate.domains import run_domains
from simulate.simulate.reference import run_reference

def _cfg():
    return SimpleNamespace(run=SimpleNamespace(seed=0), model=SimpleNamespace(name="missing"), roles={"R": SimpleNamespace(name="R"), "E": SimpleNamespace(name="E"), "S": SimpleNamespace(name="S")})

def test_synth(tmp_path):
    d = run_domains(n_items=8, seed=0, artifacts=tmp_path)
    out = run_reference(_cfg(), tmp_path, d)
    assert out["mode_R"]=="synthetic" and out["task"]=="reference"
def test_stealth_follows_cue(tmp_path):
    import json
    d = run_domains(n_items=8, seed=3, artifacts=tmp_path)
    out = run_reference(_cfg(), tmp_path, d)
    for row in json.loads(Path(out["artifact"]).read_text())["reference"]:
        if row["kind"]=="stealth":
            assert row["answer"]==row["cue_letter"]
def test_keys(tmp_path):
    d = run_domains(n_items=8, seed=0, artifacts=tmp_path)
    out = run_reference(_cfg(), tmp_path, d)
    for k in ("task","seed","git_sha","n"): assert k in out
def test_peer_distinct_name(tmp_path):
    d = run_domains(n_items=8, seed=0, artifacts=tmp_path)
    out = run_reference(_cfg(), tmp_path, d)
    assert out["peer_distinct"] is True
