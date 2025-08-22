import argparse
import importlib.util
import json
import os
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM

from mechdiff.config import DEFAULTS
from mechdiff.utils.tokenizers import load_tokenizers, shared_vocab_maps
from mechdiff.utils.activations import collect_last_token_resids
from mechdiff.utils.cka import linear_cka


def load_pair(pair_path: str):
    spec = importlib.util.spec_from_file_location("pair_cfg", pair_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod.PAIR


def standardize(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = X.mean(0, keepdim=True)
    sd = X.std(0, keepdim=True) + 1e-6
    return (X - mu) / sd, mu, sd


def fit_ridge(X: torch.Tensor, Y: torch.Tensor, lam: float) -> torch.Tensor:
    XtX = X.T @ X
    d = XtX.shape[0]
    return torch.linalg.solve(XtX + lam * torch.eye(d, device=X.device), X.T @ Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--n_samples", type=int, default=5000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--lam_grid", default="1e-6,1e-5,1e-4,1e-3,1e-2")
    args = ap.parse_args()

    pair = load_pair(args.pair)
    base_id = pair["base_id"]
    tuned_id = pair["tuned_id"]

    tok_b, tok_t = load_tokenizers(base_id, tuned_id)

    device = args.device
    dtype = torch.bfloat16 if device.startswith("cuda") else None
    base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=dtype).to(device)
    tuned = AutoModelForCausalLM.from_pretrained(tuned_id, torch_dtype=dtype).to(device)

    # For CLT, reuse the same 40 freeform prompts many times to reach N (simple bootstrap)
    prompts = []
    freeform = pair.get("datasets", {}).get("freeform_file")
    if freeform and os.path.exists(os.path.join("mechdiff", freeform)):
        with open(os.path.join("mechdiff", freeform), "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    t = obj.get("text") or obj.get("prompt") or ""
                except Exception:
                    t = line.strip()
                if t:
                    prompts.append(t)
    if not prompts:
        print("No prompts; exiting.")
        return
    if len(prompts) < args.n_samples:
        reps = (args.n_samples + len(prompts) - 1) // len(prompts)
        prompts = (prompts * reps)[: args.n_samples]
    else:
        prompts = prompts[: args.n_samples]

    L = args.layer
    # Collect features
    Hb = collect_last_token_resids(base, tok_b, prompts, L, device=device)  # (N,d)
    Ht = collect_last_token_resids(tuned, tok_t, prompts, L, device=device)  # (N,d)

    # Train/val split
    N = Hb.shape[0]
    n_val = max(1, int(0.2 * N))
    Hb_tr, Hb_val = Hb[:-n_val], Hb[-n_val:]
    Ht_tr, Ht_val = Ht[:-n_val], Ht[-n_val:]

    Hb_tr_z, mu_b, sd_b = standardize(Hb_tr)
    Ht_tr_z, mu_t, sd_t = standardize(Ht_tr)
    Hb_val_z = (Hb_val - mu_b) / sd_b
    Ht_val_z = (Ht_val - mu_t) / sd_t

    # Ridge grid search
    lams = [float(x) for x in args.lam_grid.split(",") if x]
    best = {"lam": None, "r2": -1.0}
    for lam in lams:
        W = fit_ridge(Hb_tr_z, Ht_tr_z, lam)
        pred = Hb_val_z @ W
        # R² on val
        ss_res = ((pred - Ht_val_z) ** 2).sum().item()
        ss_tot = ((Ht_val_z - Ht_val_z.mean(0, keepdim=True)) ** 2).sum().item()
        r2 = 1.0 - (ss_res / max(1e-8, ss_tot))
        if r2 > best["r2"]:
            best = {"lam": lam, "r2": r2}

    # Final model and CKA(mapped, tuned)
    W = fit_ridge(Hb_tr_z, Ht_tr_z, best["lam"])
    mapped = ((Hb - mu_b) / sd_b) @ W * sd_t + mu_t  # de-standardize into tuned space
    cka_mapped = linear_cka(mapped, Ht)

    os.makedirs("mechdiff/artifacts", exist_ok=True)
    out = {
        "layer": L,
        "n": int(N),
        "val_r2": round(best["r2"], 4),
        "lam": best["lam"],
        "cka_mapped_vs_tuned": round(float(cka_mapped), 4),
    }
    with open("mechdiff/artifacts/rq2_clt.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("Saved mechdiff/artifacts/rq2_clt.json")


if __name__ == "__main__":
    main()


