#!/usr/bin/env python3
"""
ImageLoRA minimal training script (Qwen2.5-VL).

This file is meant to be a *copy-pastable* reference implementation showing how to
train ImageLoRA for a VLM model with the smallest amount of code.

What this script demonstrates
- Injection is VISUAL-ONLY: the delta is applied only to the visual token span
  between <|vision_start|> and <|vision_end|> in the input sequence. Therefore, the model never affect text-only reasoning performance.
- ImageLoRA modifies ONLY the Value projection (v_proj) inside self-attention.
- A is shared per layer: A has shape [hidden_size, r].
- B is selective per layer per KV head: B has shape [Hsel, r, head_dim], where
  Hsel is the number of selected KV heads in that layer.
- Per-layer sqrt normalization: scale by 1/sqrt(Hsel).

Notes
- This demo trains on a single synthetic example by default. Replace the
  `build_instruction_tuning_batch()` call with your real dataset loader.
"""

from __future__ import annotations

import argparse
import math
import os
import random
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# Optional: load KV head selections produced by imagelora_head_select_minimal.py
try:
    from imagelora_head_select_minimal import load_kv_indices_json
except Exception:
    load_kv_indices_json = None  # allows running without the selector


# --------------------------
# ImageLoRA module (per layer)
# --------------------------

class ImageLoRALayer(nn.Module):
    """
    ImageLoRA parameters for a single transformer layer (V-only).

    A: [hidden_size, r] (shared in this layer)
    B: [Hsel, r, head_dim] (one B per selected KV head)
    kv: [Hsel] selected KV head indices (0-based)
    """
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        kv_indices: List[int],
        r: int,
        alpha: float,
        device=None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        kv_indices = sorted(set(int(i) for i in kv_indices))
        self.kv = torch.tensor(kv_indices, dtype=torch.long, device=device)

        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(1, self.r)

        hsel = max(1, len(kv_indices))
        self.norm = 1.0 / math.sqrt(hsel)

        self.A = nn.Parameter(torch.zeros(hidden_size, self.r, device=device, dtype=dtype))
        self.B = nn.Parameter(torch.zeros(hsel, self.r, head_dim, device=device, dtype=dtype))
        self.gamma = nn.Parameter(torch.tensor(0.01, device=device, dtype=dtype))

        nn.init.normal_(self.A, std=1.0 / max(1, self.r))
        nn.init.zeros_(self.B)

    def has_params(self) -> bool:
        return self.kv.numel() > 0 and self.r > 0


def build_imagelora_bank(
    model: Qwen2_5_VLForConditionalGeneration,
    *,
    r: int,
    alpha: float,
    kv_heads_per_layer: int = 0,
    kv_indices_by_layer: Optional[Dict[int, List[int]]] = None,
    lora_dtype: torch.dtype = torch.float32,
) -> Dict[int, ImageLoRALayer]:
    """
    Build ImageLoRA modules for each transformer layer.

    Selection priority (minimal & explicit):

      1) If kv_indices_by_layer is provided and contains an entry for this layer,
         we use those KV head indices (after sanitizing & clamping).
      2) Else fall back to kv_heads_per_layer:
           - 0  => select ALL KV heads in each layer
           - >0 => select first N KV heads in each layer (deterministic)
    """
    bank: Dict[int, ImageLoRALayer] = {}
    layers = model.model.language_model.layers
    hidden_size = int(model.config.hidden_size)

    for layer_idx, layer in enumerate(layers):
        sa = layer.self_attn
        head_dim = int(sa.head_dim)

        num_kv = getattr(sa, "num_key_value_heads", None)
        if num_kv is None:
            num_kv = int(sa.v_proj.out_features // head_dim)
        num_kv = int(num_kv)
        if num_kv <= 0:
            continue

        # --- 1) Optional per-layer KV list override (from head-selection JSON) ---
        kv_indices: List[int]
        if kv_indices_by_layer is not None and int(layer_idx) in kv_indices_by_layer:
            raw = kv_indices_by_layer[int(layer_idx)]
            # sanitize: ints, unique, in-range
            kv_indices = sorted({int(x) for x in raw if 0 <= int(x) < num_kv})
        else:
            # --- 2) Fallback selection by count ---
            if kv_heads_per_layer and kv_heads_per_layer > 0:
                kv_indices = list(range(min(num_kv, int(kv_heads_per_layer))))
            else:
                kv_indices = list(range(num_kv))

        # If you provide an empty list for a layer, skip it.
        if len(kv_indices) == 0:
            continue

        # Helpful warning to explain "kv_heads_per_layer doesn't change params".
        # Qwen2.5-VL-7B typically has num_key_value_heads=4.
        if (
            layer_idx == 0
            and kv_indices_by_layer is None
            and kv_heads_per_layer
            and int(kv_heads_per_layer) > num_kv
        ):
            print(
                f"[warn] kv_heads_per_layer={int(kv_heads_per_layer)} > num_kv={num_kv}; "
                f"clamping to {num_kv} (GQA) ⇒ param count won't increase."
            )

        dev = sa.v_proj.weight.device
        bank[layer_idx] = ImageLoRALayer(
            hidden_size=hidden_size,
            head_dim=head_dim,
            kv_indices=kv_indices,
            r=r,
            alpha=alpha,
            device=dev,
            dtype=lora_dtype,
        )

    return bank


# --------------------------
# Visual span utilities
# --------------------------

def find_visual_span(processor, inputs) -> Tuple[int, int]:
    """
    Return (v_start, v_end) indices in the input sequence such that the visual tokens are:
        input_ids[:, v_start+1 : v_end]
    """
    tok = processor.tokenizer
    ids = inputs["input_ids"][0].tolist()
    vstart_id = tok.convert_tokens_to_ids("<|vision_start|>")
    vend_id = tok.convert_tokens_to_ids("<|vision_end|>")
    try:
        v_start = ids.index(vstart_id)
        v_end = ids.index(vend_id, v_start + 1)
    except ValueError as e:
        raise RuntimeError("Could not locate <|vision_start|>/<|vision_end|> in input_ids") from e
    return int(v_start), int(v_end)


def _wrap_v_proj_forward(
    old_forward,
    lora: ImageLoRALayer,
    v_start: int,
    v_end: int,
    head_dim: int,
):
    """
    Patch v_proj.forward:
      out = v_proj(x) + ΔV  (ΔV applied only on visual tokens & selected KV heads)
    """
    def forward_with_imagelora(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        out = old_forward(x, *args, **kwargs)  # [B,T,Hkv*Dh]

        if (not torch.is_tensor(x)) or (not lora.has_params()):
            return out

        B, T, D = out.shape
        tv = slice(v_start + 1, v_end)  # visual-only
        if tv.stop is None or tv.stop > T or tv.start >= tv.stop:
            # During decode steps T can be 1; visual tokens are not present.
            return out

        Hkv = D // head_dim
        out4 = out.view(B, T, Hkv, head_dim)

        # Move x slice to the layer device/dtype where A/B live
        x_vis = x[:, tv, :].to(device=lora.A.device, dtype=lora.A.dtype)  # [B,Tv,H]

        # A shared per-layer => [B,Tv,r]
        pre = torch.einsum("bti,ir->btr", x_vis, lora.A)
        # B per selected KV head => [B,Tv,Hsel,Dh]
        delta = torch.einsum("btr,hrd->bthd", pre, lora.B)

        delta = delta * (lora.scaling * lora.norm * lora.gamma)
        delta = delta.to(device=out4.device, dtype=out4.dtype)

        idx = lora.kv.to(out4.device)  # [Hsel]
        out4[:, tv, idx, :] = out4[:, tv, idx, :] + delta
        return out4.view(B, T, D)

    return forward_with_imagelora


@contextmanager
def apply_imagelora(
    model: Qwen2_5_VLForConditionalGeneration,
    bank: Dict[int, ImageLoRALayer],
    v_start: int,
    v_end: int,
):
    """
    Temporarily patch each layer's v_proj.forward with ImageLoRA.
    """
    backups = []
    try:
        for layer_idx, lora in bank.items():
            if not lora.has_params():
                continue
            sa = model.model.language_model.layers[layer_idx].self_attn
            v_proj = sa.v_proj
            head_dim = int(sa.head_dim)

            old_forward = v_proj.forward
            v_proj.forward = _wrap_v_proj_forward(old_forward, lora, v_start, v_end, head_dim)  # type: ignore[assignment]
            backups.append((v_proj, old_forward))

        yield
    finally:
        for v_proj, old_forward in backups:
            v_proj.forward = old_forward  # type: ignore[assignment]


# --------------------------
# Instruction-tuning batch builder
# --------------------------

def _find_subsequence(haystack_1d: torch.Tensor, needle_1d: torch.Tensor) -> Optional[int]:
    """
    Find needle inside haystack. Returns starting index or None.
    (Simple O(n*m) is fine for minimal script.)
    """
    h = haystack_1d.tolist()
    n = needle_1d.tolist()
    if len(n) == 0 or len(h) < len(n):
        return None
    for i in range(len(h) - len(n) + 1):
        if h[i : i + len(n)] == n:
            return i
    return None


def maybe_load_or_make_image(image_path: str, *, save_if_missing: bool = True) -> Image.Image:
    """
    Load an image if it exists; otherwise create a simple white demo image.
    """
    if image_path and os.path.isfile(image_path):
        return Image.open(image_path).convert("RGB")

    img = Image.new("RGB", (512, 512), color="white")
    if save_if_missing and image_path:
        try:
            img.save(image_path)
        except Exception:
            pass
    return img


def build_instruction_tuning_batch(
    processor,
    image: Image.Image,
    question: str,
    answer: str,
):
    """
    Build:
      - inputs_full: tokenized prompt+answer
      - labels: -100 on prompt tokens, supervise assistant answer tokens only
    """
    # user-only messages
    user_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]

    # user + assistant messages
    full_messages = user_messages + [
        {"role": "assistant", "content": [{"type": "text", "text": answer}]}
    ]

    prompt_text = processor.apply_chat_template(
        user_messages,
        tokenize=False,
        add_generation_prompt=True,   # open assistant turn, no answer yet
    )
    full_text = processor.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,  # answer is included
    )

    images, videos = process_vision_info(user_messages)

    inputs_prompt = processor(
        text=[prompt_text],
        images=images,
        videos=videos,
        padding=True,
        return_tensors="pt",
    )
    inputs_full = processor(
        text=[full_text],
        images=images,
        videos=videos,
        padding=True,
        return_tensors="pt",
    )

    # Find prompt_len inside the full sequence
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


# --------------------------
# Saving / loading adapter
# --------------------------

def save_imagelora_adapter(
    path: str,
    *,
    model_name: str,
    r: int,
    alpha: float,
    bank: Dict[int, ImageLoRALayer],
) -> None:
    """
    Save a CPU-friendly checkpoint:
      - per-layer state_dict (A/B/gamma)
      - per-layer kv head indices
    """
    layers_payload = {}
    kv_payload = {}
    for layer_idx, layer in bank.items():
        layers_payload[str(layer_idx)] = {k: v.detach().cpu() for k, v in layer.state_dict().items()}
        kv_payload[str(layer_idx)] = [int(x) for x in layer.kv.detach().cpu().tolist()]

    obj = {
        "model_name": model_name,
        "r": int(r),
        "alpha": float(alpha),
        "kv_indices": kv_payload,
        "layers": layers_payload,
    }
    torch.save(obj, path)


# --------------------------
# Main
# --------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Minimal ImageLoRA training for Qwen2.5-VL")
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="HF model id or local path")
    ap.add_argument("--out", type=str, default="imagelora_qwen2_5_vl.pt", help="Where to save the ImageLoRA adapter")
    ap.add_argument("--steps", type=int, default=1000, help="Number of optimizer steps (demo repeats the same example)")
    ap.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    ap.add_argument("--r", type=int, default=8, help="ImageLoRA rank")
    ap.add_argument("--alpha", type=float, default=16.0, help="ImageLoRA alpha")
    ap.add_argument(
        "--kv-heads-per-layer",
        type=int,
        default=1,
        help="0=all KV heads in each layer; >0=first N KV heads per layer",
    )
    ap.add_argument(
        "--kv-indices-json",
        type=str,
        default="",
        help=(
            "Optional KV head selection JSON. If set, it overrides --kv-heads-per-layer. "
            "Use the output of imagelora_head_select_minimal.py (kv_indices mapping)."
        ),
    )
    ap.add_argument(
        "--lora-dtype",
        type=str,
        choices=["fp32", "bf16"],
        default="fp32",
        help="Datatype for ImageLoRA A/B/gamma parameters (fp32 recommended)",
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--image", type=str, default="imagelora_example.png", help="Path to an image (created if missing)")
    ap.add_argument("--question", type=str, default="What is the shape shown in this image?", help="User question")
    ap.add_argument("--answer", type=str, default="The image shows a circle inscribed inside a square, with the square divided by vertical and horizontal lines into four equal quadrants.", help="Assistant answer (supervised)")
    ap.add_argument("--log-every", type=int, default=50, help="Print loss every N steps")
    return ap.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 1) Load processor & model
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.train()
    device = next(model.parameters()).device

    # 2) (Optional) load KV head selections
    kv_indices_by_layer = None
    if args.kv_indices_json:
        if load_kv_indices_json is None:
            raise SystemExit(
                "--kv-indices-json was provided but imagelora_head_select_minimal.py is not importable. "
                "Place imagelora_head_select_minimal.py in the same directory."
            )
        kv_indices_by_layer = load_kv_indices_json(args.kv_indices_json)

    # 3) Build ImageLoRA bank
    lora_dtype = torch.float32 if args.lora_dtype == "fp32" else torch.bfloat16
    bank = build_imagelora_bank(
        model,
        r=int(args.r),
        alpha=float(args.alpha),
        kv_heads_per_layer=int(args.kv_heads_per_layer),
        kv_indices_by_layer=kv_indices_by_layer,
        lora_dtype=lora_dtype,
    )

    # 4) Freeze base model; train only ImageLoRA params
    for p in model.parameters():
        p.requires_grad_(False)

    lora_params: List[torch.nn.Parameter] = []
    for layer in bank.values():
        lora_params.extend(list(layer.parameters()))
    for p in lora_params:
        p.requires_grad_(True)

    nparams = sum(p.numel() for p in lora_params)
    per_layer_kv = {int(L): int(layer.kv.numel()) for L, layer in bank.items()}
    total_kv = int(sum(per_layer_kv.values()))
    eff_kv = int(max(per_layer_kv.values())) if per_layer_kv else 0

    print(f"[train] model={args.model}")
    print(
        f"[train] ImageLoRA params: {nparams:,} (r={args.r}, alpha={args.alpha}, "
        f"kv_heads_per_layer={args.kv_heads_per_layer}, effective_kv={eff_kv})"
    )
    if args.kv_indices_json:
        n_layers_used = len(per_layer_kv)
        print(
            f"[train] KV selection override: {args.kv_indices_json} | "
            f"layers_used={n_layers_used} | total_selected_kv={total_kv} | "
            f"per_layer_selected(min/mean/max)="
            f"{min(per_layer_kv.values()) if per_layer_kv else 0}/"
            f"{(total_kv / max(1, n_layers_used)):.2f}/"
            f"{max(per_layer_kv.values()) if per_layer_kv else 0}"
        )

    optimizer = torch.optim.AdamW(lora_params, lr=float(args.lr))

    # 4) Build one demo batch (replace with real data)
    image = maybe_load_or_make_image(args.image, save_if_missing=True)
    inputs_full, labels = build_instruction_tuning_batch(
        processor=processor,
        image=image,
        question=args.question,
        answer=args.answer,
    )

    # Visual span must come from the SAME inputs we feed the model
    v_start, v_end = find_visual_span(processor, inputs_full)

    # Move tensors
    inputs_full = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs_full.items()}
    labels = labels.to(device)

    # 5) Train (repeat the same batch for demo)
    for step in range(1, int(args.steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        with apply_imagelora(model, bank, v_start, v_end):
            out = model(**inputs_full, labels=labels, use_cache=False)
        loss = out.loss
        loss.backward()
        optimizer.step()

        if step % max(1, int(args.log_every)) == 0:
            print(f"[train] step={step:04d} loss={float(loss.detach().cpu()):.4f}")

    # 6) Save adapter
    save_imagelora_adapter(
        args.out,
        model_name=args.model,
        r=int(args.r),
        alpha=float(args.alpha),
        bank=bank,
    )
    print(f"[save] adapter -> {args.out}")
    print(f"[save] example image -> {args.image} (if it was missing, it was created)")


if __name__ == "__main__":
    main()


