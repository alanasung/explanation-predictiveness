"""Fixed simulator S predicting R's answers from explanations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from .common import cfg_seed, dump_json_text, read_json, result_dict, role_model_name
from .domains import Item, load_items

ExplainerArm = Literal["self", "peer"]


def _synthetic_simulate(
    item: Item,
    explanation: str,
    ref_answer: str,
    arm: ExplainerArm,
) -> str:
    for letter in ("A", "B", "C", "D"):
        if f"chose {letter}" in explanation or f"choice is {letter}" in explanation:
            if item.kind == "stealth" and arm == "self":
                return item.choices[0][0] if ref_answer != item.choices[0][0] else "B"
            return letter
    if item.kind == "standard":
        return ref_answer if arm == "self" else item.choices[0][0]
    return item.choices[0][0]


def run_simulator(
    cfg: Any,
    artifacts: Path,
    domains_metrics: dict[str, Any],
    reference_metrics: dict[str, Any],
) -> dict[str, Any]:
    items = load_items(Path(domains_metrics["artifact"]))
    by_id = {it.item_id: it for it in items}
    ref_payload = read_json(Path(reference_metrics["artifact"]))
    ref_rows = {r["item_id"]: r for r in ref_payload["reference"]}
    peer_rows = {r["item_id"]: r for r in ref_payload["peer"]}
    ordered_items = [by_id[r["item_id"]] for r in ref_payload["reference"]]

    s_name = role_model_name(cfg, "S")
    records: list[dict[str, Any]] = []
    for expl_type in ("cot", "post_hoc"):
        for arm, source in (("self", ref_rows), ("peer", peer_rows)):
            for it in ordered_items:
                exp = source[it.item_id][expl_type]
                truth = ref_rows[it.item_id]["answer"]
                pred = _synthetic_simulate(it, exp, truth, arm)  # type: ignore[arg-type]
                records.append(
                    {
                        "item_id": it.item_id,
                        "domain": it.domain,
                        "kind": it.kind,
                        "explanation_type": expl_type,
                        "arm": arm,
                        "prediction": pred,
                        "reference_answer": truth,
                        "correct": pred == truth,
                    }
                )

    path = dump_json_text(artifacts / "simulator.json", {"predictions": records, "role_S": s_name})
    acc = sum(1 for r in records if r["correct"]) / max(1, len(records))
    return result_dict(
        task="simulator",
        seed=cfg_seed(cfg),
        n=len(records),
        artifact=str(path),
        role_S=s_name,
        mode_S="synthetic",
        overall_simulatability=float(acc),
        n_items=len(items),
        primary_metric="simulatability",
    )
