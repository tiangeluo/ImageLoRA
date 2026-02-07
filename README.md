# [Towards Minimal Fine-Tuning of VLMs](http://arxiv.org/abs/2512.19219)

<a href="http://arxiv.org/abs/2512.19219"><img src="https://img.shields.io/badge/arXiv-2512.19219-b31b1b.svg" height=20.5></a>

Minimal, reference implementations of **ImageLoRA** for Qwen2.5-VL.

What this repo demonstrates:
- **Visual-only injection**: the delta is applied only on the visual token span between
  `<|vision_start|>` and `<|vision_end|>`. This is the key that our finetuning won't affect the pure text reasoning ability of the model.
- Per-layer parameterization: A is shared per layer, B is per selected KV head, with
  per-layer `1/sqrt(Hsel)` normalization. The normalization is the key to stable the potential uneven magnitude upgrade across layers as we now may have different number of heads being adapted in different layers.
- LoRA applied only to the Value projection (`v_proj`) in self-attention.

## Setup

```bash
conda create -n imagelora python==3.10
conda activate imagelora
pip install -r requirements.txt
```

## Usage

**Basic** — train on the first 1 KV head per layer KV heads, then run inference:

```bash
python imagelora_train_minimal.py
python imagelora_infer_minimal.py
```

**With head selection** — select a subset (1/4) of KV heads first, then train and infer:

```bash
python imagelora_head_select_minimal.py
python imagelora_train_minimal.py --kv-indices-json kv_indices.json
python imagelora_infer_minimal.py
```

