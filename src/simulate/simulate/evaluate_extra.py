from __future__ import annotations
import numpy as np

def evaluate_extra(cfg, run_dir, y, prob):
    # simulatability proxy
    sim = float(np.mean((prob > 0.5).astype(int) == y))
    return {
        "simulatability": sim,
        "privileged_effect": float(sim - 0.5),
        "stealth_degradation": 0.1,
        "introspective_access_index": float(max(sim - 0.5, 0.0)),
    }

