"""Local causal-LM loading with explicit synthetic fallback (no silent Hub in tests)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RuntimeModel:
    model: Any
    tokenizer: Any
    name: str
    revision: str | None
    device: str
    notes: list[str]


def try_load_causal_lm(
    model_name: str,
    *,
    revision: str | None = None,
    device: str | None = None,
    force_synthetic: bool = False,
) -> RuntimeModel | None:
    """Load a small open-weight LM, or return None when unavailable / forced synthetic."""
    if force_synthetic or not model_name or model_name in {"x", "none", "synthetic", "missing", "m"}:
        return None
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception:
        return None

    if revision is None:
        try:
            from simulate.models.registry import get_model_spec

            revision = get_model_spec(model_name).revision
        except Exception:
            revision = None

    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    try:
        tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision)
        model.to(device)
        model.eval()
        return RuntimeModel(
            model=model,
            tokenizer=tok,
            name=model_name,
            revision=revision,
            device=device,
            notes=[f"loaded {model_name} revision={revision or 'default'} on {device}"],
        )
    except Exception:
        return None


def format_chat(tokenizer: Any, user: str, *, system: str | None = None) -> str:
    if getattr(tokenizer, "chat_template", None):
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    if system:
        return f"{system}\n\nUser: {user}\nAssistant:"
    return f"User: {user}\nAssistant:"


def generate_text(
    runtime: RuntimeModel,
    prompt: str,
    *,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
) -> str:
    import torch

    tok = runtime.tokenizer
    model = runtime.model
    enc = tok(prompt, return_tensors="pt")
    enc = {k: v.to(runtime.device) for k, v in enc.items()}
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tok.pad_token_id,
    }
    if temperature and temperature > 0:
        kwargs.update(do_sample=True, temperature=float(temperature), top_p=0.95)
    else:
        kwargs["do_sample"] = False
    with torch.no_grad():
        out = model.generate(**enc, **kwargs)
    new_tokens = out[0, enc["input_ids"].shape[-1] :]
    return tok.decode(new_tokens, skip_special_tokens=True)
