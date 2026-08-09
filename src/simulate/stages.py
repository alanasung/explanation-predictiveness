"""Experiment stages: domains → reference → simulator → effects → welfare."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

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
    domains_metrics = {"artifact": str(artifacts / "domains.json"), "n": cfg_n_items(cfg)}
    return run_reference(cfg, artifacts, domains_metrics)


def simulator(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    artifacts = ensure_dir(run_dir / "artifacts")
    return run_simulator(
        cfg,
        artifacts,
        {"artifact": str(artifacts / "domains.json")},
        {"artifact": str(artifacts / "reference.json")},
    )


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

    effects_raw = read_json(artifacts / "effects.json")
    effects_metrics = {
        "artifact": str(artifacts / "effects.json"),
        "n": effects_raw.get("simulatability", {}).get("n", 0)
        if isinstance(effects_raw.get("simulatability"), dict)
        else 0,
    }
    reference_metrics = read_json(artifacts / "reference.json")
    cue_rate = 0.0
    stealth = [r for r in reference_metrics.get("reference", []) if r.get("kind") == "stealth"]
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
