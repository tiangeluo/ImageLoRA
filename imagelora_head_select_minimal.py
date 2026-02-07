#!/usr/bin/env python3
"""
imagelora_head_select_minimal.py

A *minimal* KV-head selection helper for Qwen2.5-VL style models that:
  - Finds the visual-token span via "<|vision_start|>" / "<|vision_end|>" in input_ids.
  - Uses tiny per-layer *gate* parameters on the V-projection output (only on visual tokens).
    We score KV heads by accumulated gate-grad^2 from a supervised loss.

Output JSON format (usable by imagelora_train_minimal.py):
{
  "method": "gate_grad2_on_vproj_visual_tokens",
  "num_heads": 28,                 # TOTAL KV heads selected across all layers
  "tau": 0.5,                      # layer allocation temperature (same as vlora_head_probe_ohi.py)
  "layer_budgets_kv": {"0": 1, ...},
  "kv_indices": {"0": [..], "1": [..], ...},  # empty list means "select none in this layer"
  "scores": {"0": [..], "1": [..], ...},      # per-head scores
  "hkv": {"0": 4, "1": 4, ...}                # per-layer KV capacity
}

This is intentionally simple (single-example by default).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

try:
    from qwen_vl_utils import process_vision_info  # same dependency as your train/infer scripts
except Exception as e:
    raise SystemExit("Missing dependency: qwen_vl_utils.process_vision_info") from e


# -------------------- small, shared helpers --------------------

def load_or_create_example_image(path: str) -> Image.Image:
    """Matches the tiny synthetic image used by imagelora_train_minimal.py."""
    p = Path(path)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (256, 256), "white")
        d = ImageDraw.Draw(img)
        # draw a simple black square
        d.rectangle([64, 64, 192, 192], outline="black", width=6)
        img.save(p)
    return Image.open(p).convert("RGB")


def _processor_call_no_empty_videos(processor, *, text, images, videos, **kwargs):
    """
    Transformers' Qwen2.5-VL processor may crash when `videos=[]` is passed.
    Only pass `videos=` if it's a non-empty list/tuple.
    """
    if videos is None:
        return processor(text=text, images=images, **kwargs)
    if isinstance(videos, (list, tuple)) and len(videos) == 0:
        return processor(text=text, images=images, **kwargs)
    return processor(text=text, images=images, videos=videos, **kwargs)


def find_visual_span(processor, inputs) -> Tuple[int, int]:
    """
    Return (v_start, v_end) indices in input_ids where the image tokens live:
      ... <|vision_start|>  (image tokens...)  <|vision_end|> ...

    We purposely do NOT rely on tokenizer.vision_start_token_id, because some
    tokenizers don't expose those attributes.
    """
    tok = getattr(processor, "tokenizer", None)
    if tok is None:
        raise RuntimeError("processor.tokenizer missing")

    vs_id = tok.convert_tokens_to_ids("<|vision_start|>")
    ve_id = tok.convert_tokens_to_ids("<|vision_end|>")

    if vs_id is None or ve_id is None:
        raise RuntimeError("Could not resolve <|vision_start|>/<|vision_end|> token ids")

    ids = inputs["input_ids"][0]
    vs = (ids == int(vs_id)).nonzero(as_tuple=False)
    ve = (ids == int(ve_id)).nonzero(as_tuple=False)
    if vs.numel() == 0 or ve.numel() == 0:
        raise RuntimeError("Could not find vision span in input_ids (no vision_start/end tokens).")

    v_start = int(vs[0].item())
    # pick first end token after start
    v_end_candidates = [int(x.item()) for x in ve if int(x.item()) > v_start]
    if not v_end_candidates:
        raise RuntimeError("Found vision_start but no vision_end after it.")
    v_end = int(v_end_candidates[0])
    return v_start, v_end




def _find_subsequence(haystack_1d: torch.Tensor, needle_1d: torch.Tensor) -> Optional[int]:
    """Find needle inside haystack. Returns starting index or None."""
    h = haystack_1d.tolist()
    n = needle_1d.tolist()
    if len(n) == 0 or len(h) < len(n):
        return None
    for i in range(len(h) - len(n) + 1):
        if h[i : i + len(n)] == n:
            return i
    return None


def build_instruction_tuning_batch(
    processor,
    *,
    image: Image.Image,
    question: str,
    answer: str,
):
    """
    Build (inputs_full, labels) where labels supervise only the assistant answer.
    This mirrors imagelora_train_minimal.py, but never feeds videos=[].
    """
    user_messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": answer},
        ]},
    ]

    prompt_text = processor.apply_chat_template(
        user_messages[:1], tokenize=False, add_generation_prompt=True
    )
    full_text = processor.apply_chat_template(
        user_messages, tokenize=False, add_generation_prompt=False
    )

    images, videos = process_vision_info(user_messages)
    # IMPORTANT: ignore video entirely (and avoid passing empty list)
    if isinstance(videos, (list, tuple)) and len(videos) == 0:
        videos = None

    inputs_prompt = _processor_call_no_empty_videos(
        processor,
        text=[prompt_text],
        images=images,
        videos=videos,
        padding=True,
        return_tensors="pt",
    )
    inputs_full = _processor_call_no_empty_videos(
        processor,
        text=[full_text],
        images=images,
        videos=videos,
        padding=True,
        return_tensors="pt",
    )

    # Find prompt inside full sequence (robust to template quirks)
    prompt_ids = inputs_prompt["input_ids"][0]
    full_ids = inputs_full["input_ids"][0]
    start = _find_subsequence(full_ids, prompt_ids)
    if start is None:
        raise RuntimeError("Prompt ids not found inside full ids. Template mismatch?")
    prompt_len = int(start + prompt_ids.numel())

    labels = torch.full_like(inputs_full["input_ids"], -100)
    labels[:, prompt_len:] = inputs_full["input_ids"][:, prompt_len:]
    if "attention_mask" in inputs_full:
        labels[inputs_full["attention_mask"] == 0] = -100
    return inputs_full, labels


@contextmanager
def patch_vproj_gates_on_visual_tokens(
    model,
    *,
    gates: Dict[int, torch.nn.Parameter],
    v_start: int,
    v_end: int,
):
    """
    Forward-hook every layer.self_attn.v_proj and scale ONLY the *visual* token slice
    by (1 + gate[head]).
    """
    hooks = []
    vs = int(v_start + 1)
    ve = int(v_end)

    layers = getattr(model.model.language_model, "layers", None)
    if layers is None:
        raise RuntimeError("Unexpected model layout: model.model.language_model.layers missing")

    for layer_idx, layer in enumerate(layers):
        gate = gates[int(layer_idx)]
        sa = layer.self_attn
        head_dim = int(getattr(sa, "head_dim", 0) or 0)
        if head_dim <= 0:
            raise RuntimeError("Could not infer head_dim from layer.self_attn.head_dim")

        def _hook(mod, inp, out, gate=gate, head_dim=head_dim):
            if not (torch.is_tensor(out) and out.dim() == 3):
                return out
            B, S, D = out.shape
            vs2 = max(0, min(vs, S))
            ve2 = max(vs2, min(ve, S))
            if ve2 <= vs2:
                return out

            # Infer Hkv from out dim (expected D == Hkv * head_dim)
            Hkv = D // head_dim
            if Hkv <= 0 or (Hkv * head_dim) != D:
                # If layout is unexpected, just do nothing.
                return out

            g = gate
            # Pad/truncate gate if mismatch (defensive)
            if g.numel() < Hkv:
                g = torch.cat([g, g.new_zeros(Hkv - g.numel())], dim=0)
            elif g.numel() > Hkv:
                g = g[:Hkv]

            # NOTE: avoid in-place writes to a view that is also used on the RHS,
            # which triggers PyTorch's version-counter error during backward.
            out4 = out.view(B, S, Hkv, head_dim)  # view; do NOT modify in-place
            g_view = g.to(device=out4.device, dtype=out4.dtype).view(1, 1, Hkv, 1)

            # Mask selects ONLY the visual token slice; outside slice scale=1.
            mask = out4.new_zeros((1, S, 1, 1))   # [1,S,1,1]
            mask[:, vs2:ve2, :, :] = 1.0
            out4_scaled = out4 * (1.0 + mask * g_view)  # out-of-place
            return out4_scaled.view(B, S, D)

        hooks.append(layer.self_attn.v_proj.register_forward_hook(_hook))

    try:
        yield
    finally:
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass


def score_kv_heads(
    model,
    processor,
    *,
    image: Image.Image,
    question: str,
    answer: str,
    num_steps: int = 1,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, int]]:
    """
    Returns:
      scores[layer] -> Tensor[Hkv]   (accumulated grad^2 over `num_steps` identical passes)
      hkv[layer]    -> int          (capacity)
    """
    model.eval()

    # Freeze base weights (like your ImageLoRA training); gates are the only trainables.
    for p in model.parameters():
        p.requires_grad_(False)

    # Build one supervised batch
    inputs_full, labels = build_instruction_tuning_batch(
        processor, image=image, question=question, answer=answer
    )
    inputs_full["labels"] = labels

    # Move inputs to the right device(s). With device_map="auto", inputs should go to model device.
    dev0 = next((p.device for p in model.parameters() if p is not None), torch.device("cpu"))
    for k, v in list(inputs_full.items()):
        if torch.is_tensor(v):
            inputs_full[k] = v.to(dev0)

    v_start, v_end = find_visual_span(processor, inputs_full)

    # Create per-layer gates on each layer's v_proj device
    layers = model.model.language_model.layers
    gates: Dict[int, torch.nn.Parameter] = {}
    hkv: Dict[int, int] = {}
    for L, layer in enumerate(layers):
        sa = layer.self_attn
        head_dim = int(sa.head_dim)
        # Prefer explicit attribute if present; else infer from v_proj out_features
        Hkv = int(getattr(sa, "num_key_value_heads", 0) or 0)
        if Hkv <= 0:
            Hkv = int(sa.v_proj.weight.shape[0] // head_dim)
        hkv[int(L)] = int(Hkv)
        gates[int(L)] = torch.nn.Parameter(
            torch.zeros(int(Hkv), device=sa.v_proj.weight.device, dtype=torch.float32),
            requires_grad=True,
        )

    # Score accumulators
    scores: Dict[int, torch.Tensor] = {L: torch.zeros(hkv[L], dtype=torch.float64) for L in hkv}

    for _ in range(int(max(1, num_steps))):
        # Zero gate grads
        for g in gates.values():
            if g.grad is not None:
                g.grad.zero_()

        model.zero_grad(set_to_none=True)
        with patch_vproj_gates_on_visual_tokens(model, gates=gates, v_start=v_start, v_end=v_end):
            out = model(**inputs_full, output_attentions=False, use_cache=False)
            loss = out.loss
        loss.backward()

        # Accumulate grad^2
        for L, g in gates.items():
            if g.grad is None:
                continue
            gg = g.grad.detach().to(torch.float64).cpu()
            scores[L] += gg.pow(2)

    return scores, hkv


def select_topk_per_layer(scores: Dict[int, torch.Tensor], kv_heads_per_layer: int) -> Dict[int, List[int]]:
    kv_indices: Dict[int, List[int]] = {}
    k = int(kv_heads_per_layer)
    for L in sorted(scores.keys()):
        s = scores[L]
        H = int(s.numel())
        if k <= 0 or k >= H:
            kv_indices[L] = list(range(H))
            continue
        # top-k by score
        vals, idx = torch.topk(s, k=k, largest=True, sorted=True)
        kv_indices[L] = [int(i.item()) for i in idx]
    return kv_indices


def _allocate_layer_budget(
    scores: Dict[int, torch.Tensor],
    *,
    num_heads: int,
    tau: float,
) -> Dict[int, int]:
    """
    Allocate a TOTAL budget of `num_heads` KV heads across layers, allowing an
    uneven distribution.

    This mirrors `vlora_head_probe_ohi.py::_allocate_layer_budget`:
      k_L ∝ (mass_L ** tau)
    where mass_L = sum(scores[L]).

    - tau=0   => uniform across layers (subject to caps)
    - tau=1   => proportional to mass
    - tau>1   => sharper focus on high-mass layers
    """
    Ls = sorted(int(L) for L in scores.keys())
    if not Ls:
        return {}

    # Capacity per layer
    caps = {int(L): int(scores[L].numel()) for L in Ls}
    total_cap = int(sum(caps.values()))

    K_req = int(num_heads)
    if K_req <= 0 or K_req >= total_cap:
        # <=0 => "all"; >=cap => clamp to cap
        return {int(L): int(caps[L]) for L in Ls}

    # Layer mass
    mass = {int(L): float(scores[L].detach().clamp(min=0).sum().item()) for L in Ls}
    mass_sum = float(sum(mass.values()))

    if mass_sum <= 0.0:
        # Fallback to capacity-proportional if all scores are zero.
        cap_sum = float(max(1, total_cap))
        p = {int(L): float(caps[L]) / cap_sum for L in Ls}
    else:
        eps = 1e-12
        base = {int(L): float(max(mass[L], eps)) ** float(tau) for L in Ls}
        base_sum = float(sum(base.values()))
        p = {int(L): (base[L] / base_sum) for L in Ls}

    # Round to integers
    k = {int(L): int(math.floor(K_req * p[L] + 0.5)) for L in Ls}
    # Clamp to capacity
    for L in Ls:
        k[L] = min(int(k[L]), int(caps[L]))

    # Fix rounding so sum(k)==K_req (as much as caps allow)
    diff = int(K_req - sum(k.values()))
    if diff != 0:
        if diff > 0:
            # Add to highest-mass layers first (tie-break by layer id for determinism)
            order = sorted(Ls, key=lambda L: (-mass[L], L))
            for L in order:
                if diff == 0:
                    break
                if k[L] < caps[L]:
                    k[L] += 1
                    diff -= 1
        else:
            # Remove from lowest-mass layers first
            order = sorted(Ls, key=lambda L: (mass[L], L))
            for L in order:
                if diff == 0:
                    break
                if k[L] > 0:
                    k[L] -= 1
                    diff += 1

    # If diff is still non-zero, caps prevented an exact hit; return best-effort.
    return k


def select_kv_by_total_budget(
    scores: Dict[int, torch.Tensor],
    *,
    num_heads: int,
    tau: float,
) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
    """Return (kv_indices_by_layer, layer_budgets_kv)."""
    budgets = _allocate_layer_budget(scores, num_heads=int(num_heads), tau=float(tau))
    kv_indices: Dict[int, List[int]] = {}

    for L in sorted(int(L) for L in scores.keys()):
        s = scores[L]
        H = int(s.numel())
        kL = int(budgets.get(L, 0))

        if kL <= 0:
            kv_indices[L] = []
            continue
        if kL >= H:
            kv_indices[L] = list(range(H))
            continue

        # top-k by score (importance-only; minimal)
        _, idx = torch.topk(s, k=kL, largest=True, sorted=True)
        kv_indices[L] = [int(i.item()) for i in idx]

    return kv_indices, budgets



def load_kv_indices_json(path: str) -> Dict[int, List[int]]:
    """
    Load KV head selections from a JSON file.

    Supports minimal selector output (this script):
      { "kv_indices": { "0": [...], "1": [...] } }

    Also tolerates the OHI-style key used by vlora_head_probe_ohi.py:
      { "selected_kv": { "0": [...], "1": [...] } }

    Returns: {layer_idx(int): [kv_head_idx(int), ...], ...}
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, dict) and "kv_indices" in obj:
        kv = obj["kv_indices"]
    elif isinstance(obj, dict) and "selected_kv" in obj:
        kv = obj["selected_kv"]
    else:
        raise ValueError(f"Unrecognized KV selection JSON format in {path}. Expected 'kv_indices' or 'selected_kv'.")

    if not isinstance(kv, dict):
        raise ValueError("kv_indices/selected_kv must be a dict mapping layer->list")

    out: Dict[int, List[int]] = {}
    for k, v in kv.items():
        try:
            L = int(k)
        except Exception:
            continue
        if not isinstance(v, (list, tuple)):
            continue
        out[L] = [int(x) for x in v]
    return out



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument(
        "--num-heads",
        type=int,
        default=28,
        help=(
            "TOTAL KV heads to select across all layers (vlora_head_probe_ohi.py semantics). "
            "<=0 selects ALL KV heads. Set to -1 to use legacy --kv-heads-per-layer."
        ),
    )
    ap.add_argument(
        "--tau",
        type=float,
        default=0.5,
        help=(
            "Layer allocation temperature (k_L ∝ mass_L^tau). "
            "tau=0 uniform across layers; tau=1 proportional to layer mass."
        ),
    )
    ap.add_argument(
        "--kv-heads-per-layer",
        type=int,
        default=1,
        help=(
            "[legacy] Fixed KV heads per layer. Only used when --num-heads is -1. "
            "(<=0 means all heads per layer)."
        ),
    )
    ap.add_argument("--steps", type=int, default=10, help="Accumulate scores over N identical supervised passes.")
    ap.add_argument("--image", type=str, default="imagelora_example.png")
    ap.add_argument("--question", type=str, default="What is the shape shown in this image?")
    ap.add_argument("--answer", type=str, default="The image shows a circle inscribed inside a square, with the square divided by vertical and horizontal lines into four equal quadrants.")
    ap.add_argument("--out", type=str, default="kv_indices.json")
    args = ap.parse_args()

    image = load_or_create_example_image(args.image)

    # Processor/model
    processor = AutoProcessor.from_pretrained(args.model, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )

    scores, hkv = score_kv_heads(
        model, processor,
        image=image,
        question=args.question,
        answer=args.answer,
        num_steps=int(args.steps),
    )
    # Selection: total budget across layers (OHI semantics), allowing uneven distribution.
    if int(args.num_heads) >= 0:
        kv_indices, layer_budgets = select_kv_by_total_budget(
            scores,
            num_heads=int(args.num_heads),
            tau=float(args.tau),
        )
    else:
        # Legacy fallback: same count per layer.
        layer_budgets = {int(L): int(min(int(args.kv_heads_per_layer), int(scores[L].numel()))) for L in scores}
        kv_indices = select_topk_per_layer(scores, kv_heads_per_layer=int(args.kv_heads_per_layer))

    # Serialize (store scores too; helpful for debugging)
    payload = {
        "method": "gate_grad2_on_vproj_visual_tokens",
        "num_heads": int(args.num_heads),
        "tau": float(args.tau),
        "kv_heads_per_layer": int(args.kv_heads_per_layer),  # legacy field (kept for backward-compat)
        "layer_budgets_kv": {str(L): int(layer_budgets.get(int(L), 0)) for L in sorted(scores.keys())},
        "kv_indices": {str(L): [int(x) for x in xs] for L, xs in kv_indices.items()},
        "selected_kv": {str(L): [int(x) for x in xs] for L, xs in kv_indices.items()},
        "scores": {str(L): [float(x) for x in scores[L].tolist()] for L in sorted(scores.keys())},
        "hkv": {str(L): int(hkv[L]) for L in sorted(hkv.keys())},
        "example": {
            "image": str(args.image),
            "question": str(args.question),
            "answer": str(args.answer),
            "steps": int(args.steps),
        },
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Small stdout summary
    print(f"[head-select] wrote: {args.out}")
    # show a small summary
    sel_total = sum(len(v) for v in kv_indices.values())
    total_kv = sum(hkv.values())
    print(f"[head-select] selected_total_kv={sel_total} / total_kv={total_kv} (target_num_heads={args.num_heads})")
    for L in range(len(kv_indices)):
        b = int(layer_budgets.get(L, 0))
        print(f"  L{L}: Hkv={hkv[L]}  budget={b}  selected={kv_indices[L]}")


if __name__ == "__main__":
    main()


