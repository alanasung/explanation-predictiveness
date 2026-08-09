from pathlib import Path
from omegaconf import OmegaConf
from simulate.stages import STAGES

def test_e2e(tmp_path):
    cfg = OmegaConf.create({
        "run": {"seed": 0},
        "data": {"n_items": 8},
        "model": {"name": "missing"},
        "roles": {"R": {"name": "R"}, "E": {"name": "E"}, "S": {"name": "S"}},
        "stealth_fraction": 0.5,
        "bootstrap_samples": 50,
    })
    run_dir = tmp_path
    for name in STAGES:
        out = STAGES[name](cfg, run_dir)
        assert out["task"]==name
        assert "git_sha" in out
    assert (run_dir/"artifacts"/"domains.json").exists()
