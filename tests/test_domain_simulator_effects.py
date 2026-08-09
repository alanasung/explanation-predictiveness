from pathlib import Path
from types import SimpleNamespace
from simulate.simulate.domains import run_domains
from simulate.simulate.reference import run_reference
from simulate.simulate.simulator import run_simulator
from simulate.simulate.effects import run_effects
from simulate.simulate.welfare import run_welfare, introspective_access_index

def _pipe(tmp_path):
    cfg = SimpleNamespace(
        force_synthetic=True,
        experiment=SimpleNamespace(name="smoke"),
        run=SimpleNamespace(seed=0),
        model=SimpleNamespace(name="m", max_new_tokens=16),
        roles={"R": SimpleNamespace(name="R", revision=None), "E": SimpleNamespace(name="E", revision=None), "S": SimpleNamespace(name="S", revision=None)},
    )
    d = run_domains(n_items=10, seed=0, artifacts=tmp_path)
    r = run_reference(cfg, tmp_path, d)
    s = run_simulator(cfg, tmp_path, d, r)
    return s, r

def test_sim(tmp_path):
    s,_ = _pipe(tmp_path)
    assert s["primary_metric"]=="simulatability"
    assert 0<=s["overall_simulatability"]<=1
def test_effects(tmp_path):
    s,_ = _pipe(tmp_path)
    e = run_effects(seed=0, artifacts=tmp_path, simulator_metrics=s, n_boot=100)
    assert e["primary_metric"]=="simulatability" and "privileged_effect" in e
    # Synthetic must withhold invented privileged-self-knowledge claims.
    assert e["is_synthetic"] is True
    assert e["privileged_effect"] is None
def test_welfare(tmp_path):
    s,r = _pipe(tmp_path)
    e = run_effects(seed=0, artifacts=tmp_path, simulator_metrics=s, n_boot=100)
    # wrap
    out = run_welfare(seed=0, artifacts=tmp_path, effects_metrics={"artifact": e["artifact"], "n": e["n"], "is_synthetic": True}, reference_metrics=r)
    assert "introspective_access_index" in out
    assert out["license"] == "synthetic_no_claim"
def test_index_clamp():
    assert introspective_access_index(2,-1,0)==1.0
def test_arms(tmp_path):
    import json
    s,_ = _pipe(tmp_path)
    rows = json.loads(Path(s["artifact"]).read_text())["predictions"]
    assert {r["arm"] for r in rows}=={"self","peer"}
