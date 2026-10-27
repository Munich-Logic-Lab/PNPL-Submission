"""BrainOmni (Xiao et al., NeurIPS 2025, arXiv 2505.18185) as the hybrid's word-window encoder.

BrainOmni = frozen BrainTokenizer (SEANet per-channel encoder + sensor-geometry embedding + cross-attention
onto 16 latent "neuro" channels + 4-level residual VQ; 2-s windows at 256 Hz -> 8 tokens) and a
spatial-temporal transformer pretrained by masked token prediction on public EEG + MEG. `encode`
(all blocks but the last) gives L2-normalised features (B, 16, W, D). Vendored model code:
third_party/BrainOmni (MIT). Weights: huggingface.co/OpenTSLab/BrainOmni ({base,tiny}/).

Input contract reproduced from BrainOmni's preprocessing (factory/utils.py):
  * 256 Hz (our 250 Hz windows are FFT-resampled),
  * per sensor type: common mode removed at every time point, then one std over the whole group
    (relative channel magnitudes kept) — our windows are per-channel z-scored by the frozen
    scaler, so the per-channel std is multiplied back first,
  * pos (C, 6) = MEG xyz centred and scaled to unit RMS radius + coil orientation (magnetometer
    normal, planar-gradiometer direction), sensor_type MAG = 1, GRAD = 2 (src/assets/omni_geometry.npz,
    built from MNE's Neuromag test_raw.fif loc; xyz identical to src/assets/sensor_geometry.npz).
A 1-s window gives 4 real tokens (the tokenizer zero-pads to 2 s); only those are pooled.

mode "vq" (default) = BrainOmni as published: frozen tokenizer, the transformer sees the residual-VQ codes.
mode "continuous" = bottleneck removed: the tokenizer encoder (SEANet + sensor embedding + neuro cross-attention)
is fine-tuned too and the transformer sees its L2-normalised continuous output (what the quantizer approximates,
so the pretrained transformer keeps its input scale); quantizer and decoder are unused.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.utils.checkpoint
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
OMNI_CODE = ROOT / "third_party" / "BrainOmni"
SFREQ_IN, SFREQ_OMNI, TOKEN_HOP = 250, 256, 64


def load_brainomni(ckpt_dir: str | Path):
    if str(OMNI_CODE) not in sys.path:
        sys.path.insert(0, str(OMNI_CODE))
    from brainomni.model import BrainOmni

    ckpt_dir = Path(ckpt_dir)
    cfg = json.loads((ckpt_dir / "model_cfg.json").read_text())
    model = BrainOmni(**cfg)
    w = ckpt_dir / "BrainOmni.pt"
    if w.exists():
        sd = torch.load(w, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"BrainOmni: loaded {w} (missing {len(missing)}, unexpected {len(unexpected)})", flush=True)
    else:
        print(f"BrainOmni: no weights at {w} — random init (architecture from model_cfg.json)", flush=True)
    return model, cfg


class OmniBackbone(nn.Module):
    """Drop-in replacement for CNN_TCN in the hybrid: features() -> (N, model_dim), classify()."""

    def __init__(self, n_classes: int, model_dim: int = 256, ckpt_dir: str = "", pool_queries: int = 4,
                 dropout: float = 0.3, mode: str = "vq"):
        super().__init__()
        if mode not in ("vq", "continuous"):
            raise ValueError(f"omni mode {mode}")
        self.mode = mode
        from cnn_tcn import AttentionPool

        self.omni, cfg = load_brainomni(ckpt_dir)
        # frozen: tokenizer (as in BrainOmni fine-tuning) and the parts encode() never uses
        for p in self.omni.tokenizer.parameters():
            p.requires_grad = False
        if mode == "continuous":
            for m in (self.omni.tokenizer.encoder, self.omni.tokenizer.sensor_embed):
                for p in m.parameters():
                    p.requires_grad = True
        for p in list(self.omni.predict_head.parameters()) + list(self.omni.blocks[-1].parameters()):
            p.requires_grad = False
        self.omni.mask_token.requires_grad = False

        g = np.load(ROOT / "src" / "assets" / "omni_geometry.npz")
        pos, st = g["pos"].astype(np.float32).copy(), g["sensor_type"].astype(np.int64)
        pos[:, :3] -= pos[:, :3].mean(0, keepdims=True)
        pos[:, :3] /= math.sqrt(3 * np.mean(np.sum(pos[:, :3] ** 2, axis=1)))
        std = np.load(ROOT / "artifacts" / "data_prep" / "train_channel_std.npy").astype(np.float32).reshape(-1)
        scale = np.empty_like(std)
        for t in (1, 2):                                   # O(1) per type, relative magnitudes kept
            scale[st == t] = std[st == t] / np.median(std[st == t])
        self.register_buffer("pos", torch.from_numpy(pos))
        self.register_buffer("sensor_type", torch.from_numpy(st))
        self.register_buffer("chan_scale", torch.from_numpy(scale))
        self.register_buffer("mag_idx", torch.from_numpy(np.flatnonzero(st == 1)))
        self.register_buffer("grad_idx", torch.from_numpy(np.flatnonzero(st == 2)))

        d, self.n_neuro = cfg["lm_dim"], cfg["n_neuro"]
        self.max_tok = cfg["window_length"] // TOKEN_HOP
        self.proj = nn.Linear(d, model_dim)
        self.tok_emb = nn.Parameter(torch.zeros(self.n_neuro * self.max_tok, model_dim))
        self.attn_pool = AttentionPool(model_dim, pool_queries, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(model_dim, n_classes)
        self.model_dim = model_dim

    def new_parameters(self):
        """Randomly initialised layers: trained at lr_new, the pretrained transformer at lr_backbone."""
        return [self.tok_emb, *self.proj.parameters(), *self.attn_pool.parameters(), *self.fc.parameters()]

    def to_omni(self, x: torch.Tensor) -> torch.Tensor:
        """(N, 306, T@250 Hz) frozen-scaler z-scores -> (N, 306, T'@256 Hz) BrainOmni input."""
        x = x.float() * self.chan_scale[None, :, None]
        t = x.shape[-1]
        t2 = int(round(t * SFREQ_OMNI / SFREQ_IN))
        y = torch.fft.irfft(torch.fft.rfft(x, dim=-1), n=t2, dim=-1) * (t2 / t)
        out = torch.empty_like(y)
        for idx in (self.mag_idx, self.grad_idx):
            g = y[:, idx]
            g = g - g.mean(dim=1, keepdim=True)
            out[:, idx] = g / (g.std(dim=(1, 2), keepdim=True) + 1e-6)
        return out

    def features(self, x, tap=None, tap_seq=False):
        if tap is not None:
            raise ValueError("BrainOmni backbone has no tap features")
        n = x.shape[0]
        with torch.autocast(device_type=x.device.type, enabled=False):
            y = self.to_omni(x)
        n_tok = min(self.max_tok, math.ceil(y.shape[-1] / TOKEN_HOP))
        pos = self.pos.unsqueeze(0).expand(n, -1, -1)
        st = self.sensor_type.unsqueeze(0).expand(n, -1)
        if self.mode == "vq":
            f = self.omni.encode(y, pos, st)
        elif self.training:
            # per-channel SEANet activations of a whole batch do not fit: recompute them chunk by chunk
            f = torch.utils.checkpoint.checkpoint(self.encode_continuous, y, pos, st, use_reentrant=False)
        else:
            f = self.encode_continuous(y, pos, st)
        f = f[:, :, :n_tok]                                                   # (N, 16, n_tok, D)
        h = self.proj(f.to(self.proj.weight.dtype))                          # (N, 16, n_tok, model_dim)
        emb = self.tok_emb.view(self.n_neuro, self.max_tok, -1)[:, :n_tok]
        h = (h + emb.unsqueeze(0)).flatten(1, 2)                              # (N, 16*n_tok, model_dim)
        return self.attn_pool(h.transpose(1, 2))

    def encode_continuous(self, y, pos, st):
        """BrainOmni.encode without the residual VQ (gradients reach the tokenizer encoder)."""
        tok = self.omni.tokenizer
        f = tok.encoder(tok.unfold(y, overlap_ratio=self.omni.overlap_ratio), tok.sensor_embed(pos, st))
        f = nn.functional.normalize(f, p=2.0, dim=-1)                         # B C N T D, as fed to the RVQ
        f = f.flatten(2, 3)                                                   # B C (N T) D
        f = f + tok.encoder.neuros.type_as(f).view(1, f.shape[1], 1, -1)
        f = self.omni.projection(f)
        for block in self.omni.blocks[:-1]:
            f = block(f)
        return nn.functional.normalize(f, p=2.0, dim=-1, eps=1e-6)

    def classify(self, features):
        return self.fc(self.drop(features))

    def forward(self, x):
        return self.classify(self.features(x))
