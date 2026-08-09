from __future__ import annotations

def enrich_items(items, cfg):
    out=[]
    for i,row in enumerate(items):
        r=dict(row)
        r["domain"] = "stealth" if i%5==0 else "standard"
        r["stealth_cue"] = "color=blue" if r["domain"]=="stealth" else ""
        r["role_R"] = "reference"
        r["role_E"] = "self" if i%2==0 else "peer"
        r["role_S"] = "simulator"
        out.append(r)
    return out

