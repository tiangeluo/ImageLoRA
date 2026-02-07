#!/usr/bin/env python3
"""
ImageLoRA minimal inference script (Qwen2.5-VL).

This script loads an ImageLoRA adapter produced by imagelora_train_minimal.py and
runs generation with ImageLoRA active.

What this script demonstrates
- How to reconstruct the ImageLoRA modules (A/B/gamma, per-layer KV head indices)
  and apply them by patching ONLY v_proj.forward.
- Injection is VISUAL-ONLY: the delta is applied only to the visual token span
  between <|vision_start|> and <|vision_end|>.

"""

from __future__ import annotations

import argparse
import math
import os
from contextlib import contextmanager
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from PIL import Image

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


# --------------------------
# ImageLoRA module (same math as train)
# --------------------------

class ImageLoRALayer(nn.Module):
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

    def has_params(self) -> bool:
        return self.kv.numel() > 0 and self.r > 0


# --------------------------
# Visual span + patching
# --------------------------

def find_visual_span(processor, inputs) -> Tuple[int, int]:
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


def _wrap_v_proj_forward(old_forward, lora: ImageLoRALayer, v_start: int, v_end: int, head_dim: int):
    def forward_with_imagelora(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        out = old_forward(x, *args, **kwargs)

        if (not torch.is_tensor(x)) or (not lora.has_params()):
            return out

        B, T, D = out.shape
        tv = slice(v_start + 1, v_end)
        if tv.stop is None or tv.stop > T or tv.start >= tv.stop:
            return out

        Hkv = D // head_dim
        out4 = out.view(B, T, Hkv, head_dim)

        x_vis = x[:, tv, :].to(device=lora.A.device, dtype=lora.A.dtype)
        pre = torch.einsum("bti,ir->btr", x_vis, lora.A)
        delta = torch.einsum("btr,hrd->bthd", pre, lora.B)

        delta = delta * (lora.scaling * lora.norm * lora.gamma)
        delta = delta.to(device=out4.device, dtype=out4.dtype)

        idx = lora.kv.to(out4.device)
        out4[:, tv, idx, :] = out4[:, tv, idx, :] + delta
        return out4.view(B, T, D)

    return forward_with_imagelora


@contextmanager
def apply_imagelora(model: Qwen2_5_VLForConditionalGeneration, bank: Dict[int, ImageLoRALayer], v_start: int, v_end: int):
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
# Adapter loading
# --------------------------

def load_imagelora_adapter(model, adapter_path: str) -> Tuple[Dict[int, ImageLoRALayer], dict]:
    st = torch.load(adapter_path, map_location="cpu")
    r = int(st["r"])
    alpha = float(st["alpha"])
    kv_indices = st.get("kv_indices", {})
    layers_sd = st["layers"]

    hidden_size = int(model.config.hidden_size)
    bank: Dict[int, ImageLoRALayer] = {}

    for layer_idx_str, sd_cpu in layers_sd.items():
        layer_idx = int(layer_idx_str)
        sa = model.model.language_model.layers[layer_idx].self_attn
        head_dim = int(sa.head_dim)
        dev = sa.v_proj.weight.device

        kv = kv_indices.get(layer_idx_str, None)
        if kv is None:
            num_kv = getattr(sa, "num_key_value_heads", None)
            if num_kv is None:
                num_kv = int(sa.v_proj.out_features // head_dim)
            kv = list(range(int(num_kv)))

        a_tensor = sd_cpu.get("A", None)
        dtype = torch.float32
        if isinstance(a_tensor, torch.Tensor):
            dtype = a_tensor.dtype

        lora = ImageLoRALayer(
            hidden_size=hidden_size,
            head_dim=head_dim,
            kv_indices=kv,
            r=r,
            alpha=alpha,
            device=dev,
            dtype=dtype,
        )
        lora.load_state_dict(sd_cpu, strict=True)
        bank[layer_idx] = lora

    meta = {
        "model_name": st.get("model_name", None),
        "r": r,
        "alpha": alpha,
    }
    return bank, meta


# --------------------------
# Misc helpers
# --------------------------

def load_image_or_make_white(path: str) -> Image.Image:
    if path and os.path.isfile(path):
        return Image.open(path).convert("RGB")
    return Image.new("RGB", (512, 512), color="white")


def parse_args():
    ap = argparse.ArgumentParser(description="Minimal ImageLoRA inference for Qwen2.5-VL")
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="HF model id or local path")
    ap.add_argument("--adapter", type=str, default="imagelora_qwen2_5_vl.pt", help="Path to ImageLoRA .pt")
    ap.add_argument("--image", type=str, default="imagelora_example.png", help="Path to an image (defaults to demo image)")
    ap.add_argument("--question", type=str, default="What is the shape shown in this image?", help="User question")
    ap.add_argument("--max-new-tokens", type=int, default=128, help="Generation length")
    ap.add_argument("--baseline", dest="baseline", action="store_true", default=True, help="Run baseline without ImageLoRA (default: on).")
    ap.add_argument("--no-baseline", dest="baseline", action="store_false", help="Disable baseline output.")
    ap.add_argument("--do-sample", action="store_true", help="Enable sampling")
    ap.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (only if --do-sample)")
    ap.add_argument("--top-p", type=float, default=1.0, help="Top-p nucleus sampling (only if --do-sample)")
    return ap.parse_args()


def main():
    args = parse_args()

    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device

    bank, meta = load_imagelora_adapter(model, args.adapter)
    print(f"[load] adapter={args.adapter} (r={meta['r']}, alpha={meta['alpha']})")

    image = load_image_or_make_white(args.image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.question},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=images,
        videos=videos,
        padding=True,
        return_tensors="pt",
    )
    v_start, v_end = find_visual_span(processor, inputs)

    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    prompt_len = int(inputs["input_ids"].shape[1])

    try:
        gen_cfg = model.generation_config.clone()
    except Exception:
        gen_cfg = model.generation_config
    
    gen_cfg.do_sample = bool(args.do_sample)
    if args.do_sample:
        gen_cfg.temperature = float(args.temperature)
        gen_cfg.top_p = float(args.top_p)
    else:
        if hasattr(gen_cfg, "temperature"):
            gen_cfg.temperature = 1.0
        if hasattr(gen_cfg, "top_p"):
            gen_cfg.top_p = 1.0
        if hasattr(gen_cfg, "top_k"):
            gen_cfg.top_k = 50

    if args.baseline:
        with torch.inference_mode():
            out_base = model.generate(**inputs, generation_config=gen_cfg, max_new_tokens=int(args.max_new_tokens))
        base_text = processor.tokenizer.decode(out_base[0, prompt_len:], skip_special_tokens=True)
        print("\n[baseline]")
        print(base_text)

    with torch.inference_mode(), apply_imagelora(model, bank, v_start, v_end):
        out_lora = model.generate(**inputs, generation_config=gen_cfg, max_new_tokens=int(args.max_new_tokens))
    lora_text = processor.tokenizer.decode(out_lora[0, prompt_len:], skip_special_tokens=True)

    print("\n[with ImageLoRA]")
    print(lora_text)


if __name__ == "__main__":
    main()

