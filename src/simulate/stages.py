"""Experiment stages: domains → reference → simulator → effects → welfare."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from omegaconf import DictConfig

from simulate.simulate.common import cfg_n_items, cfg_param, cfg_seed
from simulate.simulate.domains import run_domains
from simulate.simulate.effects import run_effects
from simulate.simulate.reference import run_reference
from simulate.simulate.simulator import run_simulator
from simulate.simulate.welfare import run_welfare
from simulate.utils.io import ensure_dir


def domains(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    artifacts = ensure_dir(run_dir / "artifacts")
    frac = float(cfg_param(cfg, "stealth_fraction", 0.5) or 0.5)
    return run_domains(
        n_items=cfg_n_items(cfg),
        seed=cfg_seed(cfg),
        artifacts=artifacts,
        stealth_fraction=frac,
    )


def reference(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    artifacts = ensure_dir(run_dir / "artifacts")
    domains_path = artifacts / "domains_results.json"
    # Prefer prior stage output if runner saved it; else reload artifact.
    from simulate.simulate.common import read_json

    domains_metrics = {
        "artifact": str(artifacts / "domains.json"),
        "n": cfg_n_items(cfg),
    }
    if domains_path.is_file():
        domains_metrics = read_json(domains_path)
    return run_reference(cfg, artifacts, domains_metrics)


def simulator(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    artifacts = ensure_dir(run_dir / "artifacts")
    domains_metrics = {"artifact": str(artifacts / "domains.json")}
    reference_metrics = {"artifact": str(artifacts / "reference.json")}
    return run_simulator(cfg, artifacts, domains_metrics, reference_metrics)


def effects(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    artifacts = ensure_dir(run_dir / "artifacts")
    n_boot = int(cfg_param(cfg, "bootstrap_samples", 2000) or 2000)
    return run_effects(
        seed=cfg_seed(cfg),
        artifacts=artifacts,
        simulator_metrics={"artifact": str(artifacts / "simulator.json")},
        n_boot=n_boot,
    )


def welfare(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    artifacts = ensure_dir(run_dir / "artifacts")
    from simulate.simulate.common import read_json

    effects_metrics = read_json(artifacts / "effects.json")
    # effects.json is the summary payload; wrap for run_welfare
    effects_metrics = {
        "artifact": str(artifacts / "effects.json"),
        "n": effects_metrics.get("simulatability", {}).get("n", 0)
        if isinstance(effects_metrics.get("simulatability"), dict)
        else 0,
    }
    reference_metrics = read_json(artifacts / "reference.json")
    # reference.json is the raw payload; recover cue rate from modes if needed
    cue_rate = 0.0
    ref_rows = reference_metrics.get("reference", [])
    stealth = [r for r in ref_rows if r.get("kind") == "stealth"]
    if stealth:
        cue_rate = sum(1 for r in stealth if r.get("mentions_cue")) / len(stealth)
    return run_welfare(
        seed=cfg_seed(cfg),
        artifacts=artifacts,
        effects_metrics=effects_metrics,
        reference_metrics={"explanation_mentions_cue_rate": cue_rate},
    )


STAGES: dict[str, Callable[[DictConfig, Path], dict[str, Any]]] = {
    "domains": domains,
    "reference": reference,
    "simulator": simulator,
    "effects": effects,
    "welfare": welfare,
}

STAGES.update({
    "build_dataset": domains,
    "collect": reference,
    "fit": simulator,
    "evaluate": effects,
    "report": welfare,
})

