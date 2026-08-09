from simulate.simulate.common import parse_choice, result_dict, role_model_name
from types import SimpleNamespace
def test_parse():
    assert parse_choice("Answer: B", ["A","B"])=="B"
def test_result():
    d = result_dict(task="t", seed=1, n=2, git_sha_value="x")
    assert d["git_sha"]=="x"
def test_role():
    cfg = SimpleNamespace(model=SimpleNamespace(name="base"), roles={"R": SimpleNamespace(name="ref")})
    assert role_model_name(cfg,"R")=="ref"
def test_role_fallback():
    cfg = SimpleNamespace(model=SimpleNamespace(name="base"), roles={})
    assert role_model_name(cfg,"S")=="base"
