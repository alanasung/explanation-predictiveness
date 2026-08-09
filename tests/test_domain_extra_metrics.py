
from simulate.simulate.domains import build_items
from simulate.simulate.welfare import _license, introspective_access_index
from simulate.simulate.common import cfg_n_items, cfg_seed
from types import SimpleNamespace

def test_stealth_fraction_exact():
    items = build_items(100, seed=0, stealth_fraction=0.4)
    assert sum(1 for i in items if i.kind=="stealth")==40

def test_license_null():
    code, _ = _license(-0.1, 0.0, [-0.2, 0.0])
    assert code=="no_introspective_access"

def test_license_domain_limited():
    code, _ = _license(0.2, 0.25, [0.05, 0.35])
    assert code=="domain_limited_access"

def test_license_positive():
    code, _ = _license(0.3, 0.0, [0.1, 0.5])
    assert code=="positive_but_not_welfare_license"

def test_cfg_helpers():
    cfg = SimpleNamespace(run=SimpleNamespace(seed=9), data=SimpleNamespace(n_items=64))
    assert cfg_seed(cfg)==9 and cfg_n_items(cfg)==64

def test_item_ids_unique():
    ids = [i.item_id for i in build_items(30, 0)]
    assert len(ids)==len(set(ids))

def test_choices_present():
    for it in build_items(10, 0):
        assert len(it.choices)==4 and it.correct in "ABCD"

def test_index_monotonic():
    a = introspective_access_index(0.2, 0.0, 0.0)
    b = introspective_access_index(0.2, 0.3, 0.0)
    assert a > b

def test_prompt_contains_question():
    for it in build_items(6, 1):
        assert it.question in it.prompt

def test_domains_git_sha_key(tmp_path):
    from simulate.simulate.domains import run_domains
    out = run_domains(n_items=8, seed=0, artifacts=tmp_path)
    assert "git_sha" in out

def test_effects_ci_order(tmp_path):
    from types import SimpleNamespace
    from simulate.simulate.domains import run_domains
    from simulate.simulate.reference import run_reference
    from simulate.simulate.simulator import run_simulator
    from simulate.simulate.effects import run_effects
    cfg = SimpleNamespace(force_synthetic=True, experiment=SimpleNamespace(name="smoke"), run=SimpleNamespace(seed=0), model=SimpleNamespace(name="m", max_new_tokens=16), roles={"R": SimpleNamespace(name="R"), "E": SimpleNamespace(name="E"), "S": SimpleNamespace(name="S")})
    d=run_domains(n_items=8, seed=0, artifacts=tmp_path)
    r=run_reference(cfg, tmp_path, d)
    s=run_simulator(cfg, tmp_path, d, r)
    e=run_effects(seed=0, artifacts=tmp_path, simulator_metrics=s, n_boot=50)
    lo, hi = e["simulatability_ci"]
    assert lo <= e["simulatability"] <= hi

def test_simulator_n_multiple_of_items(tmp_path):
    from types import SimpleNamespace
    from simulate.simulate.domains import run_domains
    from simulate.simulate.reference import run_reference
    from simulate.simulate.simulator import run_simulator
    cfg = SimpleNamespace(force_synthetic=True, experiment=SimpleNamespace(name="smoke"), run=SimpleNamespace(seed=0), model=SimpleNamespace(name="m", max_new_tokens=16), roles={"R": SimpleNamespace(name="R"), "E": SimpleNamespace(name="E"), "S": SimpleNamespace(name="S")})
    d=run_domains(n_items=8, seed=0, artifacts=tmp_path)
    r=run_reference(cfg, tmp_path, d)
    s=run_simulator(cfg, tmp_path, d, r)
    assert s["n"] == 8 * 2 * 2  # items * arms * expl types
