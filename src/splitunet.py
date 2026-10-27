"""MEG SplitUNet — B[FM]² (Hwang et al. 2026, arXiv 2606.20812) adapted to 306-channel MEG.

Input layout (B, 3, 102, T): 3 sensor types (magnetometer, two planar gradiometers) at the 102
helmet locations, 50 Hz. As in B[FM]², every block factorises a 2-D convolution into a 1-D
temporal conv and a 1-D "electrode" conv and downsamples only along time. MEG has no 10-20
ordering, so the electrode conv is a graph conv over each location's k nearest helmet neighbours
(fixed adjacency from the sensor geometry). Bottleneck: axial self-attention over time, then over
locations (the paper uses linear attention per stage + full attention at the bottleneck; we keep
only the bottleneck attention). Velocity field for rectified-flow pretraining; `encode` returns the
bottleneck features used for fine-tuning.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_LOC = 102


def knn_adjacency(k: int = 6) -> torch.Tensor:
    g = np.load(Path(__file__).resolve().parent / "assets" / "sensor_geometry.npz")
    xyz = g["xyz"][g["stype"] == 0]                        # 102 magnetometer positions, location order
    d = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    A = np.zeros((N_LOC, N_LOC), dtype=np.float32)
    nn_idx = np.argsort(d, axis=1)[:, :k]
    for i, js in enumerate(nn_idx):
        A[i, js] = 1.0
    A = np.maximum(A, A.T)                                  # symmetric kNN graph
    return torch.tensor(A / A.sum(1, keepdims=True))        # row-normalised neighbour mean


class TimeEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim * 4))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        f = torch.exp(torch.linspace(0, math.log(1000), self.dim // 2, device=t.device))
        a = t[:, None] * f[None] * 1000 / 1000
        return self.mlp(torch.cat([a.sin(), a.cos()], 1))


class ElectrodeConv(nn.Module):
    """Graph conv over helmet locations: y = W [x ; mean_{neighbours} x]."""

    def __init__(self, cin: int, cout: int, A: torch.Tensor):
        super().__init__()
        self.register_buffer("A", A.clone())
        self.lin = nn.Conv2d(2 * cin, cout, kernel_size=1)

    def forward(self, x):
        nb = torch.einsum("bcet,fe->bcft", x, self.A.to(x.dtype))
        return self.lin(torch.cat([x, nb], 1))


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, tdim: int, A: torch.Tensor, dropout: float = 0.1):
        super().__init__()
        self.n1 = nn.GroupNorm(min(8, cin), cin)
        self.tconv = nn.Conv2d(cin, cout, kernel_size=(1, 3), padding=(0, 1))
        self.film = nn.Linear(tdim, 2 * cout)
        self.n2 = nn.GroupNorm(min(8, cout), cout)
        self.econv = ElectrodeConv(cout, cout, A)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.tconv(F.silu(self.n1(x)))
        sc, sh = self.film(temb)[:, :, None, None].chunk(2, 1)
        h = h * (1 + sc) + sh
        h = self.econv(self.drop(F.silu(self.n2(h))))
        return h + self.skip(x)


class AxialAttention(nn.Module):
    def __init__(self, c: int, heads: int = 4):
        super().__init__()
        self.nt, self.ne = nn.GroupNorm(8, c), nn.GroupNorm(8, c)
        self.at = nn.MultiheadAttention(c, heads, batch_first=True)
        self.ae = nn.MultiheadAttention(c, heads, batch_first=True)

    def forward(self, x):
        b, c, e, t = x.shape
        h = self.nt(x).permute(0, 2, 3, 1).reshape(b * e, t, c)        # attention over time per location
        h, _ = self.at(h, h, h, need_weights=False)
        x = x + h.reshape(b, e, t, c).permute(0, 3, 1, 2)
        h = self.ne(x).permute(0, 3, 2, 1).reshape(b * t, e, c)        # attention over locations per time
        h, _ = self.ae(h, h, h, need_weights=False)
        return x + h.reshape(b, t, e, c).permute(0, 3, 2, 1)


class SplitUNet(nn.Module):
    def __init__(self, ch: int = 32, mults=(1, 2, 4, 8), nres: int = 1, k: int = 6, dropout: float = 0.1):
        super().__init__()
        A = knn_adjacency(k)
        tdim = ch * 4
        self.temb = TimeEmb(ch)
        self.stem = nn.Conv2d(3, ch, kernel_size=(1, 7), padding=(0, 3))
        chans = [ch * m for m in mults]
        self.down, self.downsample, skips = nn.ModuleList(), nn.ModuleList(), []
        c = ch
        for i, co in enumerate(chans):
            blocks = nn.ModuleList()
            for _ in range(nres):
                blocks.append(ResBlock(c, co, tdim, A, dropout)); c = co
            self.down.append(blocks); skips.append(c)
            self.downsample.append(nn.Conv2d(c, c, (1, 3), stride=(1, 2), padding=(0, 1)) if i < len(chans) - 1 else nn.Identity())
        self.mid1 = ResBlock(c, c, tdim, A, dropout)
        self.attn = AxialAttention(c)
        self.mid2 = ResBlock(c, c, tdim, A, dropout)
        self.up, self.upsample = nn.ModuleList(), nn.ModuleList()
        for i, co in reversed(list(enumerate(chans))):
            blocks = nn.ModuleList()
            for j in range(nres):
                blocks.append(ResBlock(c + (skips[i] if j == 0 else 0), co, tdim, A, dropout)); c = co
            self.up.append(blocks)
            self.upsample.append(nn.Conv2d(c, chans[i - 1], (1, 3), padding=(0, 1)) if i > 0 else nn.Identity())
            if i > 0:
                c = chans[i - 1]
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, 3, 1))
        nn.init.zeros_(self.out[2].weight); nn.init.zeros_(self.out[2].bias)
        self.levels = len(chans)
        self.feat_dim = chans[-1]

    def _encode(self, x, t):
        temb = self.temb(t)
        h = self.stem(x); hs = []
        for blocks, ds in zip(self.down, self.downsample):
            for b in blocks:
                h = b(h, temb)
            hs.append(h); h = ds(h)
        h = self.mid2(self.attn(self.mid1(h, temb)), temb)
        return h, hs, temb

    def encode(self, x, t):
        """Bottleneck features (B, C, 102, T / 2^(levels-1))."""
        return self._encode(x, t)[0]

    def forward(self, x, t):
        h, hs, temb = self._encode(x, t)
        for k, (blocks, us) in enumerate(zip(self.up, self.upsample)):
            i = self.levels - 1 - k
            for j, b in enumerate(blocks):
                h = b(torch.cat([h, hs[i]], 1) if j == 0 else h, temb)
            if i > 0:
                h = us(F.interpolate(h, scale_factor=(1, 2), mode="nearest"))
        return self.out(h)


def to_bfm_layout(x: torch.Tensor, scale: float, mult: int = 8) -> torch.Tensor:
    """(N, 306, T250) standardized windows -> (N, 3, 102, T50 padded to a multiple of `mult`),
    FFT-resampled to 50 Hz, scaled like bfm_prep.py, clipped ±5."""
    n, _, t = x.shape
    t50 = t // 5
    X = torch.fft.rfft(x.float(), dim=-1)[..., : t50 // 2 + 1]
    y = torch.fft.irfft(X, n=t50, dim=-1) * (t50 / t)
    y = torch.clamp(y * scale, -5, 5).reshape(n, N_LOC, 3, t50).transpose(1, 2)
    pad = (-t50) % mult
    return F.pad(y, (0, pad)) if pad else y


class BFMBackbone(nn.Module):
    """Drop-in replacement for CNN_TCN in the hybrid: features() -> (N, model_dim), classify()."""

    def __init__(self, n_classes: int, model_dim: int = 256, ckpt: str = "", scale: float = 1.0,
                 pool_queries: int = 4, dropout: float = 0.3, spatial_k: int = 8):
        super().__init__()
        from cnn_tcn import AttentionPool
        self.unet = SplitUNet()
        if ckpt and Path(ckpt).exists():
            sd = torch.load(ckpt, map_location="cpu", weights_only=False)
            sd = sd.get("ema", sd.get("model", sd))
            missing, unexpected = self.unet.load_state_dict(sd, strict=False)
            print(f"BFM backbone: loaded {ckpt} (missing {len(missing)}, unexpected {len(unexpected)})", flush=True)
        self.scale = float(scale)
        c = self.unet.feat_dim
        self.spatial = nn.Parameter(torch.zeros(spatial_k, N_LOC))          # learned location pooling
        self.proj = nn.Conv1d(c * spatial_k, model_dim, 1)
        self.attn_pool = AttentionPool(model_dim, pool_queries, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(model_dim, n_classes)
        self.model_dim = model_dim

    def features(self, x, tap=None, tap_seq=False):
        if tap is not None:
            raise ValueError("BFM backbone has no tap features")
        z = to_bfm_layout(x, self.scale)
        h = self.unet.encode(z, torch.ones(z.shape[0], device=z.device))   # t = 1: clean data
        w = torch.softmax(self.spatial, dim=-1).to(h.dtype)                  # (K, 102)
        h = torch.einsum("bcet,ke->bckt", h, w).flatten(1, 2)              # (N, C*K, T')
        return self.attn_pool(self.proj(h))

    def classify(self, features):
        return self.fc(self.drop(features))

    def forward(self, x):
        return self.classify(self.features(x))
