"""Shared helpers for the faithfulness domain (v2 spine)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..utils.git import git_sha
from ..utils.io import ensure_dir, load_json, save_json


def result_dict(
    *,
    task: str,
    seed: int,
    n: int,
    git_sha_value: str | None = None,
    **metrics: Any,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "task": task,
        "seed": int(seed),
        "git_sha": git_sha_value if git_sha_value is not None else git_sha(),
        "n": int(n),
    }
    out.update(metrics)
    return out


def write_json(path: Path, payload: Any) -> Path:
    ensure_dir(path.parent)
    save_json(path, payload)
    return path


def read_json(path: Path) -> Any:
    return load_json(path)


def cfg_seed(cfg: Any) -> int:
    return int(getattr(getattr(cfg, "run", cfg), "seed", 0))


def cfg_n_items(cfg: Any) -> int:
    return int(getattr(getattr(cfg, "data", cfg), "n_items", 128))


def cfg_param(cfg: Any, key: str, default: Any = None) -> Any:
    exp = getattr(cfg, "experiment", None)
    if exp is not None and hasattr(exp, key):
        return getattr(exp, key)
    # Hydra often parks free keys under cfg directly or under a params-like node.
    if hasattr(cfg, key):
        return getattr(cfg, key)
    params = getattr(cfg, "params", None)
    if params is not None and hasattr(params, key):
        return getattr(params, key)
    if isinstance(params, dict):
        return params.get(key, default)
    return default


def role_model_name(cfg: Any, role: str) -> str:
    roles = getattr(cfg, "roles", None)
    if roles is not None:
        # DictConfig or dict
        try:
            entry = roles[role]
            name = getattr(entry, "name", None) or (entry.get("name") if isinstance(entry, dict) else None)
            if name:
                return str(name)
        except Exception:  # noqa: BLE001
            pass
    return str(getattr(cfg.model, "name", "synthetic"))


def parse_choice(text: str, choices: list[str] | None = None) -> str:
    cleaned = (text or "").strip().upper()
    if not cleaned:
        return "?"
    for marker in ("ANSWER:", "FINAL:", "CHOICE:"):
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[1].strip()
            break
    token = cleaned.split()[0].strip(".,);:]")
    if choices:
        letters = {c[0].upper() for c in choices}
        if token[:1] in letters:
            return token[:1]
    if token[:1].isalpha():
        return token[:1]
    return token[:16]


def dump_json_text(path: Path, payload: Any) -> Path:
    """Fallback writer that does not require OmegaConf-serializable objects."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
