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

from ..evaluation.metrics import (
    Estimate,
    bootstrap_diff,
    bootstrap_mean,
    minimum_detectable_effect,
    tost_equivalence,
)
from .common import dump_json_text, read_json, result_dict

# Pre-registered TOST band for privileged-effect null / practical significance.
_PRIV_TOST_LOW = -0.05
_PRIV_TOST_HIGH = 0.05
_PEER_DISTINCTNESS_FLOOR = 0.5


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


def _est_dict(est: Estimate) -> dict[str, Any]:
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


def _per_template_diffs(
    self_rows: list[dict[str, Any]],
    peer_rows: list[dict[str, Any]],
) -> list[float]:
    self_by: dict[str, list[float]] = defaultdict(list)
    peer_by: dict[str, list[float]] = defaultdict(list)
    for r in self_rows:
        self_by[_template_id(r)].append(1.0 if r["correct"] else 0.0)
    for r in peer_rows:
        peer_by[_template_id(r)].append(1.0 if r["correct"] else 0.0)
    diffs: list[float] = []
    for tid in sorted(set(self_by) | set(peer_by)):
        s = float(np.mean(self_by[tid])) if self_by.get(tid) else 0.0
        p = float(np.mean(peer_by[tid])) if peer_by.get(tid) else 0.0
        diffs.append(s - p)
    return diffs


def _privileged_claim_ok(priv: Estimate, tost: dict[str, Any]) -> bool:
    """True only if effect is significantly nonzero or clears the TOST band."""
    if tost.get("equivalent"):
        return False
    if priv.excludes_zero:
        return True
    # Clears TOST band: CI wholly outside [-ε, ε].
    return bool(priv.lo > _PRIV_TOST_HIGH or priv.hi < _PRIV_TOST_LOW)


def run_effects(
    *,
    seed: int,
    artifacts: Path,
    simulator_metrics: dict[str, Any],
    n_boot: int = 2000,
    reference_metrics: dict[str, Any] | None = None,
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
    leakage_claim_ok = payload.get("leakage_claim_ok")
    if leakage_claim_ok is None:
        leakage_claim_ok = simulator_metrics.get("leakage_claim_ok")
    if leakage_claim_ok is None:
        leakage_claim_ok = True
    leakage_claim_ok = bool(leakage_claim_ok)

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
        priv_dict = {
            **_withheld_effect(reason),
            "privileged_claim_ok": False,
        }
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
        tost_priv: dict[str, Any] = {
            "equivalent": False,
            "note": "withheld; TOST not applicable",
            "band": [_PRIV_TOST_LOW, _PRIV_TOST_HIGH],
        }
        privileged_claim_ok = False
        mde_priv = None
    else:
        priv = cluster_bootstrap_diff(self_rows, peer_rows, n_boot=n_boot, seed=seed)
        diffs = _per_template_diffs(self_rows, peer_rows)
        if len(diffs) >= 2:
            tost_priv = tost_equivalence(
                diffs,
                low=_PRIV_TOST_LOW,
                high=_PRIV_TOST_HIGH,
                n_boot=min(n_boot, 2000),
                seed=seed,
            )
            mde_priv = float(
                minimum_detectable_effect(
                    len(diffs),
                    sigma=float(np.std(diffs, ddof=1) or 1.0),
                )
            )
        else:
            tost_priv = {
                "equivalent": False,
                "note": "insufficient templates for TOST",
                "band": [_PRIV_TOST_LOW, _PRIV_TOST_HIGH],
            }
            mde_priv = None
        privileged_claim_ok = _privileged_claim_ok(priv, tost_priv)
        priv_dict = _est_dict(priv)
        priv_dict["is_synthetic"] = False
        priv_dict["withheld"] = False
        priv_dict["inference"] = "cluster_template"
        priv_dict["tost"] = tost_priv
        priv_dict["mde"] = mde_priv
        priv_dict["privileged_claim_ok"] = privileged_claim_ok
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

    # Peer distinctness gate for peer–self contrast headlines.
    peer_distinctness_rate = None
    peer_distinctness_ci: list[float | None] = [None, None]
    peer_distinctness_claim_ok = False
    if reference_metrics is not None:
        peer_distinctness_rate = reference_metrics.get("peer_distinctness_rate")
        peer_distinctness_ci = list(
            reference_metrics.get("peer_distinctness_ci") or [None, None]
        )
        peer_distinctness_claim_ok = bool(
            reference_metrics.get("peer_distinctness_claim_ok", False)
        )
    if not peer_distinctness_claim_ok and not is_synthetic:
        # Fail-closed: without powered distinctness, withhold privileged contrast.
        privileged_claim_ok = False

    # Leakage gate: withhold simulatability headlines when extraction exceeds τ.
    simulatability_claim_ok = bool(leakage_claim_ok) and not is_synthetic
    if not leakage_claim_ok:
        overall_dict = {
            **_est_dict(overall),
            "withheld_headline": True,
            "reason": "leakage_claim_ok=false; extraction_rate exceeded threshold",
        }
    else:
        overall_dict = _est_dict(overall)

    if is_synthetic:
        privileged_claim_ok = False
        simulatability_claim_ok = False

    summary = {
        "primary_metric": "simulatability",
        "simulatability": overall_dict,
        "simulatability_claim_ok": simulatability_claim_ok,
        "leakage_claim_ok": leakage_claim_ok,
        "privileged_self_knowledge_effect": priv_dict,
        "privileged_claim_ok": privileged_claim_ok,
        "tost_privileged": tost_priv,
        "mde_privileged": mde_priv,
        "stealth_domain_degradation": stealth_degradation,
        "by_domain": by_domain,
        "by_explanation_type": by_explanation,
        "peer_distinctness_rate": peer_distinctness_rate,
        "peer_distinctness_ci": peer_distinctness_ci,
        "peer_distinctness_claim_ok": peer_distinctness_claim_ok,
        "peer_distinctness_floor": _PEER_DISTINCTNESS_FLOOR,
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
        # Diagnostic point estimate always emitted; headlines gated by *_claim_ok.
        simulatability=float(overall.value),
        simulatability_ci=[float(overall.lo), float(overall.hi)],
        simulatability_claim_ok=simulatability_claim_ok,
        leakage_claim_ok=leakage_claim_ok,
        privileged_effect=priv_value,
        privileged_effect_ci=priv_ci,
        privileged_claim_ok=privileged_claim_ok,
        mde_privileged=mde_priv,
        stealth_domain_degradation=stealth_degradation,
        peer_distinctness_claim_ok=peer_distinctness_claim_ok,
        is_synthetic=is_synthetic,
        status="synthetic" if is_synthetic else "measured",
        inference=inference,
        soft_synthetic_item_rate=soft_rate,
        soft_synthetic_item_rate_exceeded=soft_exceeded,
    )
