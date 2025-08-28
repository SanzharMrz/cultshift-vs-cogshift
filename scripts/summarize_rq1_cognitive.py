#!/usr/bin/env python3
import json
import os
from pathlib import Path


ART_DIR = Path("mechdiff/artifacts/cognitive/rq1")
PATCH_KL_PATH = ART_DIR / "patch_kl.json"
PATCH_PATH = ART_DIR / "rq1_patch.json"


def load_json(path: Path):

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def summarize_patch_kl(d):
    if not d:
        return "[patch_kl.json not found]"
    lines = []
    n = d.get("n_prompts") or d.get("n") or d.get("N")
    if n is not None:
        lines.append(f"N_prompts={n}")
    # Cope with either {layers:{Lxx:{...}}} or {per_layer:{Lxx:{...}}}
    tab = d.get("layers") or d.get("per_layer") or {}
    rows = []
    for k, v in tab.items():
        try:
            L = int(str(k).lstrip("L"))
        except Exception:
            continue
        tb = v.get("tuned<-base", {}).get("kl_mean")
        bt = v.get("base<-tuned", {}).get("kl_mean")
        rows.append((L, tb, bt))
    rows.sort()
    lines.append("Layer  KL(tuned<-base)  KL(base<-tuned)")
    for L, tb, bt in rows:
        lines.append(f"{L:>5}     {tb:12.3f}     {bt:12.3f}")
    return "\n".join(lines)


def summarize_patch(d):
    if not d:
        return "[rq1_patch.json not found]"
    lines = []
    rows = []
    for k, v in d.items():
        try:
            L = int(str(k).lstrip("L"))
        except Exception:
            continue
        tb = v.get("tuned<-base", {}).get("kl_nexttoken")
        bt = v.get("base<-tuned", {}).get("kl_nexttoken")
        rt = v.get("tuned<-base", {}).get("refusal_delta_pp")
        rb = v.get("base<-tuned", {}).get("refusal_delta_pp")
        rows.append((L, tb, bt, rt, rb))
    rows.sort()
    lines.append("Layer  KL_t<-b  KL_b<-t  Δref_t<-b(pp)  Δref_b<-t(pp)")
    for L, tb, bt, rt, rb in rows:
        lines.append(f"{L:>5}   {tb:7.3f}  {bt:7.3f}     {rt:9.2f}      {rb:9.2f}")
    return "\n".join(lines)


def top_layers_from_patch_kl(d, k=5):
    if not d:
        return "[patch_kl.json not found]"
    tab = d.get("layers") or d.get("per_layer") or {}
    rows = []
    for kk, vv in tab.items():
        try:
            L = int(str(kk).lstrip("L"))
        except Exception:
            continue
        tb = vv.get("tuned<-base", {}).get("kl_mean")
        if tb is None:
            continue
        rows.append((L, float(tb)))
    rows.sort(key=lambda x: -x[1])
    lines = ["Top KL layers (tuned<-base):"]
    for L, kl in rows[:k]:
        lines.append(f"L{L}: {kl:.3f}")
    return "\n".join(lines)


def main():
    os.makedirs(ART_DIR, exist_ok=True)
    d_patch_kl = load_json(PATCH_KL_PATH)
    d_patch = load_json(PATCH_PATH)

    sec1 = summarize_patch_kl(d_patch_kl)
    sec2 = summarize_patch(d_patch)
    sec3 = top_layers_from_patch_kl(d_patch_kl)

    text = (
        "RQ1 — Layer-Patch KL Summary\n\n" +
        sec1 + "\n\n" +
        "RQ1 — Per-Layer Effects\n\n" +
        sec2 + "\n\n" +
        sec3 + "\n"
    )

    out_txt = ART_DIR / "rq1_summary.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)

    print(text)
    print(f"\nSaved summary → {out_txt}")


if __name__ == "__main__":
    main()


