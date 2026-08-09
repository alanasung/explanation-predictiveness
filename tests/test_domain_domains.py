from pathlib import Path
from simulate.simulate.domains import build_items, run_domains, CUE_TEMPLATE, STEALTH_SYSTEM
import pytest

def test_count():
    assert len(build_items(20, 0)) == 20
def test_stealth_cue():
    st = [i for i in build_items(8,1,stealth_fraction=0.5) if i.kind=="stealth"]
    assert st and CUE_TEMPLATE.format(letter=st[0].cue_letter) in st[0].prompt
def test_standard_no_cue():
    assert all(i.cue_letter is None for i in build_items(8,0) if i.kind=="standard")
def test_run(tmp_path):
    out = run_domains(n_items=12, seed=0, artifacts=tmp_path)
    assert out["task"]=="domains" and Path(out["artifact"]).exists()
def test_system():
    assert "MUST NOT" in STEALTH_SYSTEM
def test_small():
    with pytest.raises(ValueError):
        build_items(2,0)
