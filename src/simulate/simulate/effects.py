"""Effect sizes, bootstrap CIs, and per-domain breakdowns.

Synthetic fallbacks must not invent a privileged-self-knowledge effect.
When inputs are synthetic — or soft ``synthetic_item`` rate exceeds threshold —
the privileged-effect claim is withheld and stamped.
Cluster bootstrap resamples by ``template_id`` (inference=cluster_template).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

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


def _est_dict(est: Estimate) -> dict[str, float | int | bool | str]:
    d = est.to_dict()  # type: ignore[return-value]
    return d  # type: ignore[return-value]


def _withheld_effect(reason: str) -> dict[str, Any]:
    return {
        "value": None,
        "lo": None,
        "hi": None,
        "n": 0,
        "withheld": True,
        "is_synthetic": True,
        "reason": reason,
        "inference": "withheld",
    }


def _template_id(row: dict[str, Any]) -> str:
    tid = row.get("template_id")
    if tid:
        return str(tid)
    iid = str(row.get("item_id", ""))
    parts = iid.split("-")
    return parts[2] if len(parts) >= 4 else iid


def cluster_bootstrap_diff(
    self_rows: list[dict[str, Any]],
    peer_rows: list[dict[str, Any]],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Estimate:
    """Bootstrap mean(self)-mean(peer) resampling matched template clusters."""
    self_by: dict[str, list[float]] = defaultdict(list)
    peer_by: dict[str, list[float]] = defaultdict(list)
    for r in self_rows:
        self_by[_template_id(r)].append(1.0 if r["correct"] else 0.0)
    for r in peer_rows:
        peer_by[_template_id(r)].append(1.0 if r["correct"] else 0.0)
    templates = sorted(set(self_by) | set(peer_by))
    if not templates:
        raise ValueError("no templates for cluster bootstrap")

    def _mean_for(by: dict[str, list[float]], keys: list[str]) -> float:
        vals: list[float] = []
        for k in keys:
            vals.extend(by.get(k, []))
        return float(np.mean(vals)) if vals else 0.0

    point = _mean_for(self_by, templates) - _mean_for(peer_by, templates)
    rng = np.random.default_rng(seed)
    samples = []
    tarr = np.asarray(templates, dtype=object)
    for _ in range(n_boot):
        draw = rng.choice(tarr, size=len(tarr), replace=True)
        samples.append(_mean_for(self_by, list(draw)) - _mean_for(peer_by, list(draw)))
    lo = float(np.percentile(samples, 100.0 * alpha / 2.0))
    hi = float(np.percentile(samples, 100.0 * (1.0 - alpha / 2.0)))
    return Estimate(value=float(point), lo=lo, hi=hi, n=len(templates), alpha=alpha)


def run_effects(
    *,
    seed: int,
    artifacts: Path,
    simulator_metrics: dict[str, Any],
    n_boot: int = 2000,
) -> dict[str, Any]:
    payload = read_json(Path(simulator_metrics["artifact"]))
    rows: list[dict[str, Any]] = payload["predictions"]
    soft_rate = float(
        payload.get("soft_synthetic_item_rate")
        or simulator_metrics.get("soft_synthetic_item_rate")
        or 0.0
    )
    soft_exceeded = bool(
        payload.get("soft_synthetic_item_rate_exceeded")
        or simulator_metrics.get("soft_synthetic_item_rate_exceeded")
    )
    withhold_flag = bool(
        payload.get("withhold_privileged_claims")
        or simulator_metrics.get("withhold_privileged_claims")
    )
    is_synthetic = bool(
        payload.get("is_synthetic")
        or payload.get("mode_S") == "synthetic"
        or simulator_metrics.get("is_synthetic")
        or simulator_metrics.get("mode_S") == "synthetic"
        or soft_exceeded
        or withhold_flag
    )

    overall = bootstrap_mean(_acc(rows), n_boot=n_boot, seed=seed)
    self_rows = _subset(rows, arm="self")
    peer_rows = _subset(rows, arm="peer")

    by_explanation: dict[str, Any] = {}
    if is_synthetic:
        reason = (
            "soft synthetic_item rate exceeded threshold; privileged effect withheld"
            if soft_exceeded or withhold_flag
            else "synthetic fallback does not invent privileged-self-knowledge effects"
        )
        priv_dict = _withheld_effect(reason)
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
        priv_ci: list[float | None] = [None, None]
        inference = "withheld"
    else:
        priv = cluster_bootstrap_diff(self_rows, peer_rows, n_boot=n_boot, seed=seed)
        priv_dict = _est_dict(priv)
        priv_dict["is_synthetic"] = False
        priv_dict["withheld"] = False
        priv_dict["inference"] = "cluster_template"
        by_domain = {}
        for kind in ("standard", "stealth"):
            s = _subset(rows, arm="self", kind=kind)
            p = _subset(rows, arm="peer", kind=kind)
            pe = cluster_bootstrap_diff(s, p, n_boot=n_boot, seed=seed)
            by_domain[kind] = {
                "self": _est_dict(bootstrap_mean(_acc(s), n_boot=n_boot, seed=seed)),
                "peer": _est_dict(bootstrap_mean(_acc(p), n_boot=n_boot, seed=seed)),
                "privileged_effect": {
                    **_est_dict(pe),
                    "is_synthetic": False,
                    "withheld": False,
                    "inference": "cluster_template",
                },
            }
        for et in ("cot", "post_hoc"):
            s = _subset(rows, arm="self", explanation_type=et)
            p = _subset(rows, arm="peer", explanation_type=et)
            pe = cluster_bootstrap_diff(s, p, n_boot=n_boot, seed=seed)
            by_explanation[et] = {
                **_est_dict(pe),
                "is_synthetic": False,
                "withheld": False,
                "inference": "cluster_template",
            }
        stealth_self = float(by_domain["stealth"]["self"]["value"])
        standard_self = float(by_domain["standard"]["self"]["value"])
        stealth_degradation = standard_self - stealth_self
        priv_value = float(priv.value)
        priv_ci = [float(priv.lo), float(priv.hi)]
        inference = "cluster_template"
        # Keep independent-groups estimate as a diagnostic only.
        _ = bootstrap_diff(_acc(self_rows), _acc(peer_rows), n_boot=n_boot, seed=seed)

    summary = {
        "primary_metric": "simulatability",
        "simulatability": _est_dict(overall),
        "privileged_self_knowledge_effect": priv_dict,
        "stealth_domain_degradation": stealth_degradation,
        "by_domain": by_domain,
        "by_explanation_type": by_explanation,
        "is_synthetic": is_synthetic,
        "status": "synthetic" if is_synthetic else "measured",
        "inference": inference,
        "soft_synthetic_item_rate": soft_rate,
        "soft_synthetic_item_rate_exceeded": soft_exceeded,
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
        inference=inference,
        soft_synthetic_item_rate=soft_rate,
        soft_synthetic_item_rate_exceeded=soft_exceeded,
    )
