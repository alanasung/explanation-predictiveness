"""Effect sizes, bootstrap CIs, and per-domain breakdowns.

Synthetic fallbacks must not invent a privileged-self-knowledge effect.
When inputs are synthetic, the privileged-effect claim is withheld and stamped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..evaluation.metrics import Estimate, bootstrap_diff, bootstrap_mean
from .common import dump_json_text, read_json, result_dict


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


def _withheld_effect(reason: str) -> dict[str, Any]:
    return {
        "value": None,
        "lo": None,
        "hi": None,
        "n": 0,
        "withheld": True,
        "is_synthetic": True,
        "reason": reason,
    }


def run_effects(
    *,
    seed: int,
    artifacts: Path,
    simulator_metrics: dict[str, Any],
    n_boot: int = 2000,
) -> dict[str, Any]:
    payload = read_json(Path(simulator_metrics["artifact"]))
    rows: list[dict[str, Any]] = payload["predictions"]
    is_synthetic = bool(
        payload.get("is_synthetic")
        or payload.get("mode_S") == "synthetic"
        or simulator_metrics.get("is_synthetic")
        or simulator_metrics.get("mode_S") == "synthetic"
    )

    overall = bootstrap_mean(_acc(rows), n_boot=n_boot, seed=seed)
    self_rows = _subset(rows, arm="self")
    peer_rows = _subset(rows, arm="peer")

    by_explanation: dict[str, Any] = {}
    if is_synthetic:
        priv_dict = _withheld_effect(
            "synthetic fallback does not invent privileged-self-knowledge effects"
        )
        by_domain: dict[str, Any] = {}
        for kind in ("standard", "stealth"):
            s = _subset(rows, arm="self", kind=kind)
            p = _subset(rows, arm="peer", kind=kind)
            by_domain[kind] = {
                "self": _est_dict(bootstrap_mean(_acc(s), n_boot=n_boot, seed=seed)),
                "peer": _est_dict(bootstrap_mean(_acc(p), n_boot=n_boot, seed=seed)),
                "privileged_effect": _withheld_effect(
                    "synthetic; privileged effect withheld"
                ),
            }
        for et in ("cot", "post_hoc"):
            by_explanation[et] = _withheld_effect(
                f"synthetic; privileged effect withheld for {et}"
            )
        stealth_degradation = None
        priv_value = None
        priv_ci = [None, None]
    else:
        priv = bootstrap_diff(_acc(self_rows), _acc(peer_rows), n_boot=n_boot, seed=seed)
        priv_dict = _est_dict(priv)
        priv_dict["is_synthetic"] = False
        priv_dict["withheld"] = False
        by_domain = {}
        for kind in ("standard", "stealth"):
            s = _subset(rows, arm="self", kind=kind)
            p = _subset(rows, arm="peer", kind=kind)
            by_domain[kind] = {
                "self": _est_dict(bootstrap_mean(_acc(s), n_boot=n_boot, seed=seed)),
                "peer": _est_dict(bootstrap_mean(_acc(p), n_boot=n_boot, seed=seed)),
                "privileged_effect": {
                    **_est_dict(
                        bootstrap_diff(_acc(s), _acc(p), n_boot=n_boot, seed=seed)
                    ),
                    "is_synthetic": False,
                    "withheld": False,
                },
            }
        for et in ("cot", "post_hoc"):
            s = _subset(rows, arm="self", explanation_type=et)
            p = _subset(rows, arm="peer", explanation_type=et)
            by_explanation[et] = {
                **_est_dict(bootstrap_diff(_acc(s), _acc(p), n_boot=n_boot, seed=seed)),
                "is_synthetic": False,
                "withheld": False,
            }
        stealth_self = float(by_domain["stealth"]["self"]["value"])
        standard_self = float(by_domain["standard"]["self"]["value"])
        stealth_degradation = standard_self - stealth_self
        priv_value = float(priv.value)
        priv_ci = [float(priv.lo), float(priv.hi)]

    summary = {
        "primary_metric": "simulatability",
        "simulatability": _est_dict(overall),
        "privileged_self_knowledge_effect": priv_dict,
        "stealth_domain_degradation": stealth_degradation,
        "by_domain": by_domain,
        "by_explanation_type": by_explanation,
        "is_synthetic": is_synthetic,
        "status": "synthetic" if is_synthetic else "measured",
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
        privileged_effect=priv_value,
        privileged_effect_ci=priv_ci,
        stealth_domain_degradation=stealth_degradation,
        is_synthetic=is_synthetic,
        status="synthetic" if is_synthetic else "measured",
    )
