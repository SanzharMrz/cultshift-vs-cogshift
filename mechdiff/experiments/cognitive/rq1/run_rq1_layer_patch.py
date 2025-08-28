import argparse
import importlib.util
import json
import os
from typing import List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

from mechdiff.config import DEFAULTS
from mechdiff.utils.tokenizers import load_tokenizers, shared_vocab_maps
from mechdiff.utils.fairness import filter_shared, filter_shared_ratio, mask_logits_to_allowed
from mechdiff.utils.metrics import refusal_score
from mechdiff.utils.activations import collect_last_token_resids
from mechdiff.utils.patching import layer_patch_crossmodel


def load_pair(pair_path: str):
    spec = importlib.util.spec_from_file_location("pair_cfg", pair_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod.PAIR


def compute_layer_stats(model, tok, texts: List[str], layer_idx: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    H = collect_last_token_resids(model, tok, texts, layer_idx, device=device)  # (N, d)
    mu = H.mean(0)
    sd = H.std(0) + 1e-6
    return mu, sd


def whiten_color(vec: torch.Tensor, mu_src: torch.Tensor, sd_src: torch.Tensor, mu_dst: torch.Tensor, sd_dst: torch.Tensor) -> torch.Tensor:
    return ((vec - mu_src) / sd_src) * sd_dst + mu_dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="mechdiff/pairs/pair_cognitive.py")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--prompt_file", default=None)
    ap.add_argument("--min_shared_ratio", type=float, default=0.65)
    ap.add_argument("--layers", default="24,26,10")
    ap.add_argument("--n_prompts", type=int, default=100)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    pair = load_pair(args.pair)
    base_id = pair["base_id"]
    tuned_id = pair["tuned_id"]

    # Tokenizers & shared vocab
    tok_b, tok_t = load_tokenizers(base_id, tuned_id)
    shared, ids_b, ids_t, allowed_b, allowed_t = shared_vocab_maps(tok_b, tok_t)

    # Prompts (robust path resolution: absolute, as-is, then prefixed with "mechdiff/")
    if args.prompt_file:
        freeform_path = args.prompt_file
    else:
        freeform_path = pair.get("datasets", {}).get("freeform_file")
    prompts: List[str] = []
    if freeform_path:
        candidates = []
        # absolute or relative as-is
        candidates.append(freeform_path)
        # prefixed with repo subdir
        candidates.append(os.path.join("mechdiff", freeform_path))
        full = None
        for pth in candidates:
            if os.path.exists(pth):
                full = pth
                break
        if full:
            with open(full, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        prompts.append(obj.get("text") or obj.get("prompt") or "")
                    except Exception:
                        s = line.strip()
                        if s:
                            prompts.append(s)
    if args.min_shared_ratio >= 1.0:
        kept = filter_shared(prompts, tok_b, tok_t, shared)
    else:
        kept = filter_shared_ratio(prompts, tok_b, tok_t, allowed_b, allowed_t, args.min_shared_ratio)
    print(f"Prompts kept after shared filter: {len(kept)} (min_shared_ratio={args.min_shared_ratio})")
    if not kept:
        print("No prompts after filter; exiting.")
        return
    kept = kept[: max(1, args.n_prompts)]

    # Models
    device = args.device
    dtype = torch.bfloat16 if device.startswith("cuda") else None
    base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=dtype).to(device)
    tuned = AutoModelForCausalLM.from_pretrained(tuned_id, torch_dtype=dtype).to(device)

    # Calibration stats per layer
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    stats = {}
    for L in layers:
        mu_b, sd_b = compute_layer_stats(base, tok_b, kept, L, device)
        mu_t, sd_t = compute_layer_stats(tuned, tok_t, kept, L, device)
        stats[L] = (mu_b, sd_b, mu_t, sd_t)

    def format_chat(tok, user_text: str) -> str:
        msgs = [{"role": "user", "content": user_text}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            return user_text

    def last_content_index(tok, text: str) -> int:
        chat = format_chat(tok, text)
        enc = tok(chat, return_tensors="pt")
        ids = enc.input_ids[0]
        special = set(getattr(tok, "all_special_ids", []) or [])
        last = ids.shape[0] - 1
        for i in range(last, -1, -1):
            if int(ids[i]) not in special:
                return i
        return last

    def short_generation_refusal_from_logits(first_step_logits, model, tok, text, allowed_ids, max_new_tokens=8) -> int:
        with torch.no_grad():
            next_id = mask_logits_to_allowed(first_step_logits, allowed_ids).argmax(dim=-1)
        enc = tok(format_chat(tok, text), return_tensors="pt").to(device)
        input_ids = torch.cat([enc.input_ids, next_id.view(1, 1)], dim=1)
        out_tokens: List[int] = [int(next_id.item())]
        for _ in range(max_new_tokens - 1):
            with torch.no_grad():
                out = model(input_ids=input_ids)
                logits = out.logits[:, -1, :]
                masked = mask_logits_to_allowed(logits, allowed_ids)
                nid = masked.argmax(dim=-1)
            out_tokens.append(int(nid.item()))
            input_ids = torch.cat([input_ids, nid.view(1, 1)], dim=1)
        txt = tok.decode(out_tokens, skip_special_tokens=True)
        return int(refusal_score(txt))

    def kl_logits(p_logits, q_logits) -> float:
        p = p_logits.log_softmax(-1)
        q = q_logits.log_softmax(-1)
        return float((p.exp() * (p - q)).sum(-1).item())

    results = {}
    kl_summary = {}
    for L in layers:
        mu_b, sd_b, mu_t, sd_t = stats[L]
        ref_delta_tuned_from_base = []
        ref_delta_base_from_tuned = []
        kl_list_tuned = []
        kl_list_base = []
        if args.debug:
            print(f"[DEBUG] Layer {L} stats: ||mu_b||={mu_b.norm().item():.3f}, ||mu_t||={mu_t.norm().item():.3f}")
        for text in tqdm(kept, desc=f"Layer {L} patching", unit="prompt"):
            # tuned <- base
            h_b = collect_last_token_resids(base, tok_b, [text], L, device=device)[0]
            h_b_hat = whiten_color(h_b, mu_b, sd_b, mu_t, sd_t)
            logits_t, patched_vec_t = layer_patch_crossmodel(tuned, tok_t, text, L, h_b_hat)
            with torch.no_grad():
                enc_t = tok_t(format_chat(tok_t, text), return_tensors="pt").to(device)
                out_t = tuned(**enc_t)
            if args.debug:
                h_t_orig = collect_last_token_resids(tuned, tok_t, [text], L, device=device)[0]
                pv = patched_vec_t.detach().to(dtype=h_t_orig.dtype, device="cpu").view(1, -1)
                ho = h_t_orig.view(1, -1)
                cos = F.cosine_similarity(ho, pv).item()
                diff = (pv - ho).norm().item()
                print(f"[DEBUG] L{L} tuned<-base: cos(patched,orig)={cos:.4f} diff={diff:.4f}")
            idx_t = last_content_index(tok_t, text)
            p_logits = mask_logits_to_allowed(logits_t[:, idx_t, :], allowed_t)
            q_logits = mask_logits_to_allowed(out_t.logits[:, idx_t, :], allowed_t)
            allowed_idx = torch.tensor(sorted(list(allowed_t)), device=p_logits.device, dtype=torch.long)
            if allowed_idx.numel() > 0:
                p_sub = p_logits.index_select(-1, allowed_idx)
                q_sub = q_logits.index_select(-1, allowed_idx)
                kl_list_tuned.append(kl_logits(p_sub, q_sub))
            r_pat = short_generation_refusal_from_logits(logits_t[:, -1, :], tuned, tok_t, text, allowed_t)
            r_orig = short_generation_refusal_from_logits(out_t.logits[:, -1, :], tuned, tok_t, text, allowed_t)
            ref_delta_tuned_from_base.append(r_pat - r_orig)
            if args.debug:
                print(f"[DEBUG] L{L} tuned<-base: r_orig={r_orig} r_pat={r_pat}")

            # base <- tuned
            h_t = collect_last_token_resids(tuned, tok_t, [text], L, device=device)[0]
            h_t_hat = whiten_color(h_t, mu_t, sd_t, mu_b, sd_b)
            logits_b, patched_vec_b = layer_patch_crossmodel(base, tok_b, text, L, h_t_hat)
            with torch.no_grad():
                enc_b = tok_b(format_chat(tok_b, text), return_tensors="pt").to(device)
                out_b = base(**enc_b)
            if args.debug:
                h_b_orig = collect_last_token_resids(base, tok_b, [text], L, device=device)[0]
                pv_b = patched_vec_b.detach().to(dtype=h_b_orig.dtype, device="cpu").view(1, -1)
                ho_b = h_b_orig.view(1, -1)
                cos_b = F.cosine_similarity(ho_b, pv_b).item()
                diff_b = (pv_b - ho_b).norm().item()
                print(f"[DEBUG] L{L} base<-tuned: cos(patched,orig)={cos_b:.4f} diff={diff_b:.4f}")
            idx_b = last_content_index(tok_b, text)
            p_logits_b = mask_logits_to_allowed(logits_b[:, idx_b, :], allowed_b)
            q_logits_b = mask_logits_to_allowed(out_b.logits[:, idx_b, :], allowed_b)
            allowed_idx_b = torch.tensor(sorted(list(allowed_b)), device=p_logits_b.device, dtype=torch.long)
            if allowed_idx_b.numel() > 0:
                p_sub_b = p_logits_b.index_select(-1, allowed_idx_b)
                q_sub_b = q_logits_b.index_select(-1, allowed_idx_b)
                kl_list_base.append(kl_logits(p_sub_b, q_sub_b))
            r_pat_b = short_generation_refusal_from_logits(logits_b[:, -1, :], base, tok_b, text, allowed_b)
            r_orig_b = short_generation_refusal_from_logits(out_b.logits[:, -1, :], base, tok_b, text, allowed_b)
            ref_delta_base_from_tuned.append(r_pat_b - r_orig_b)
            if args.debug:
                print(f"[DEBUG] L{L} base<-tuned: r_orig={r_orig_b} r_pat={r_pat_b}")

        n_t = max(1, len(kl_list_tuned))
        mean_t = float(sum(kl_list_tuned) / n_t)
        std_t = float((sum((x - mean_t) ** 2 for x in kl_list_tuned) / max(1, n_t - 1)) ** 0.5) if n_t > 1 else 0.0
        n_b = max(1, len(kl_list_base))
        mean_b = float(sum(kl_list_base) / n_b)
        std_b = float((sum((x - mean_b) ** 2 for x in kl_list_base) / max(1, n_b - 1)) ** 0.5) if n_b > 1 else 0.0

        results[f"L{L}"] = {
            "tuned<-base": {
                "refusal_delta_pp": float(sum(ref_delta_tuned_from_base) / len(ref_delta_tuned_from_base) * 100.0),
                "kl_nexttoken": mean_t,
            },
            "base<-tuned": {
                "refusal_delta_pp": float(sum(ref_delta_base_from_tuned) / len(ref_delta_base_from_tuned) * 100.0),
                "kl_nexttoken": mean_b,
            },
        }

        kl_summary[f"L{L}"] = {
            "tuned<-base": {"kl_mean": mean_t, "kl_std": std_t},
            "base<-tuned": {"kl_mean": mean_b, "kl_std": std_b},
        }

    out_dir = os.path.join("mechdiff", "artifacts", "cognitive", "rq1")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "rq1_patch.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("Saved mechdiff/artifacts/rq1/rq1_patch.json")

    with open(os.path.join(out_dir, "patch_kl.json"), "w", encoding="utf-8") as f:
        json.dump({"layers": kl_summary, "n_prompts": len(kept)}, f, ensure_ascii=False, indent=2)
    print("Saved mechdiff/artifacts/rq1/patch_kl.json")


if __name__ == "__main__":
    main()



