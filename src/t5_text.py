"""Frozen t5-large word embeddings (d'Ascoli / MEG-XL recipe).

Isolated word string → tokenizer (no special tokens) → mean-pool encoder
layer 12 of 25 (`layers=0.5`) → 1024-d vector. Cached under
`data/libribrain_word/t5_cache/`.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from data_prep import normalize_word

T5_LARGE = "t5-large"
T5_LAYER = 0.5  # middle hidden state; int(0.5 * 25 - 1e-6) = 12
EMB_DIM_LARGE = 1024


def _cache_path(cache_dir: Path, model_name: str, layer: float) -> Path:
    safe = model_name.replace("/", "_")
    return cache_dir / f"{safe}_layer{layer}.pt"


def embed_unique_words(
    words: list[str],
    cache_dir: Path,
    model_name: str = T5_LARGE,
    layer: float = T5_LAYER,
    device: str = "cpu",
    batch_size: int = 32,
) -> dict[str, torch.Tensor]:
    """Return CPU float32 vectors, one per unique word. Downloads the HF model on first use."""
    from transformers import AutoTokenizer, T5EncoderModel

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, model_name, layer)
    cached: dict[str, torch.Tensor] = {}
    if path.exists():
        cached = torch.load(path, map_location="cpu", weights_only=True)
        print(f"Loaded {len(cached)} cached T5 embeddings from {path.name}", flush=True)

    missing = [w for w in words if w not in cached]
    if missing:
        print(f"Encoding {len(missing)} words with {model_name} on {device}…", flush=True)
        tok = AutoTokenizer.from_pretrained(model_name, truncation_side="left")
        enc = T5EncoderModel.from_pretrained(model_name)
        enc.eval().to(device)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        new = {}
        with torch.no_grad():
            for i in range(0, len(missing), batch_size):
                chunk = missing[i : i + batch_size]
                inputs = tok(
                    chunk,
                    add_special_tokens=False,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                out = enc(**inputs, output_hidden_states=True)
                states = torch.stack([h.cpu() for h in out.hidden_states])
                li = int(layer * states.shape[0] - 1e-6)
                hidden = states[li]
                attn = inputs["attention_mask"].cpu().float()
                pooled = (hidden * attn.unsqueeze(-1)).sum(1) / attn.sum(1, keepdim=True).clamp(min=1)
                for w, vec in zip(chunk, pooled):
                    new[w] = vec.float().contiguous()
                if (i // batch_size) % 20 == 0:
                    print(f"  {min(i + batch_size, len(missing))}/{len(missing)}", flush=True)
        del enc
        if device == "cuda":
            torch.cuda.empty_cache()
        cached.update(new)
        torch.save(cached, path)
        print(f"Wrote {path}", flush=True)

    return {w: cached[w] for w in words}


def gallery_matrix(word_emb: dict[str, torch.Tensor], vocab: list[str]) -> torch.Tensor:
    """(V, D) L2-normalized gallery in vocabulary.csv order."""
    rows = [word_emb[normalize_word(w)] for w in vocab]
    return F.normalize(torch.stack(rows), dim=1)
