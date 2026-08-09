"""Effect sizes, bootstrap CIs, and per-domain breakdowns."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..evaluation.metrics import Estimate, bootstrap_diff, bootstrap_mean
from .common import cfg_seed, dump_json_text, read_json, result_dict


def _acc(rows: list[dict[str, Any]]) -> list[float]:
    return [1.0 if r["correct"] else 0.0 for r in rows]


def _subset(
    rows: list[dict[str, Any]],
    *,
    arm: str | None = None,
    kind: str | None = None,
    explanation_type: str | None = None,
) -> list[dict[str, Any]]:
    out = rows
    if arm is not None:
        out = [r for r in out if r["arm"] == arm]
    if kind is not None:
        out = [r for r in out if r["kind"] == kind]
    if explanation_type is not None:
        out = [r for r in out if r["explanation_type"] == explanation_type]
    return out


def _est_dict(est: Estimate) -> dict[str, float | int | bool]:
    return est.to_dict()  # type: ignore[return-value]


def run_effects(
    *,
    seed: int,
    artifacts: Path,
    simulator_metrics: dict[str, Any],
    n_boot: int = 2000,
) -> dict[str, Any]:
    payload = read_json(Path(simulator_metrics["artifact"]))
    rows: list[dict[str, Any]] = payload["predictions"]

    overall = bootstrap_mean(_acc(rows), n_boot=n_boot, seed=seed)
    self_rows = _subset(rows, arm="self")
    peer_rows = _subset(rows, arm="peer")
    priv = bootstrap_diff(_acc(self_rows), _acc(peer_rows), n_boot=n_boot, seed=seed)

    by_domain: dict[str, Any] = {}
    for kind in ("standard", "stealth"):
        s = _subset(rows, arm="self", kind=kind)
        p = _subset(rows, arm="peer", kind=kind)
        by_domain[kind] = {
            "self": _est_dict(bootstrap_mean(_acc(s), n_boot=n_boot, seed=seed)),
            "peer": _est_dict(bootstrap_mean(_acc(p), n_boot=n_boot, seed=seed)),
            "privileged_effect": _est_dict(
                bootstrap_diff(_acc(s), _acc(p), n_boot=n_boot, seed=seed)
            ),
        }

    stealth_self = float(by_domain["stealth"]["self"]["value"])
    standard_self = float(by_domain["standard"]["self"]["value"])
    stealth_degradation = standard_self - stealth_self

    summary = {
        "primary_metric": "simulatability",
        "simulatability": _est_dict(overall),
        "privileged_self_knowledge_effect": _est_dict(priv),
        "stealth_domain_degradation": stealth_degradation,
        "by_domain": by_domain,
    }
    path = dump_json_text(artifacts / "effects.json", summary)
    return result_dict(
        task="effects",
        seed=seed,
        n=len(rows),
        artifact=str(path),
        primary_metric="simulatability",
        simulatability=float(overall.value),
        simulatability_ci=[float(overall.lo), float(overall.hi)],
        privileged_effect=float(priv.value),
        privileged_effect_ci=[float(priv.lo), float(priv.hi)],
        stealth_domain_degradation=stealth_degradation,
    )
