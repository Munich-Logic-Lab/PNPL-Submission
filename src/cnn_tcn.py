"""Parameter Team CNN+TCN and the 50-way Lightning wrapper."""

from __future__ import annotations

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F


class TCNBlock(nn.Module):
    """Residual dilated Conv1d block from Parameter Team CNN_TCN."""

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding=padding, dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(p=dropout)
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        out = self.bn2(self.conv2(out))
        out = self.dropout2(out)
        return self.relu(out + residual)


class AttentionPool(nn.Module):
    """Learned-query attention pooling over time: (B, D, T) → (B, D).

    n_queries learned vectors attend over the timestep sequence, so the model
    can focus on the informative 200–500 ms slice instead of averaging it
    away (occlusion study, HANDOVER §10)."""

    def __init__(self, dim: int, n_queries: int = 4, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"model_dim {dim} not divisible by pool heads {heads}")
        self.queries = nn.Parameter(torch.randn(n_queries, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.merge = nn.Linear(n_queries * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.transpose(1, 2)  # (B, T, D)
        q = self.queries.unsqueeze(0).expand(seq.size(0), -1, -1)
        out, _ = self.attn(q, seq, seq, need_weights=False)  # (B, nq, D)
        return self.merge(out.flatten(1))


def real_sph_harm(xyz, degree: int):
    """Real spherical harmonics (l <= degree) at unit directions from a fitted centre: (n, (degree+1)^2)."""
    import numpy as np
    from scipy import special
    p = xyz - xyz.mean(0)
    p = p / np.linalg.norm(p, axis=1, keepdims=True)
    theta, phi = np.arccos(np.clip(p[:, 2], -1, 1)), np.arctan2(p[:, 1], p[:, 0])
    cols = []
    for l in range(degree + 1):
        for m in range(-l, l + 1):
            if hasattr(special, "sph_harm_y"):
                y = special.sph_harm_y(l, abs(m), theta, phi)
            else:
                y = special.sph_harm(abs(m), l, phi, theta)
            cols.append(np.sqrt(2) * (y.imag if m < 0 else y.real) if m != 0 else y.real)
    Y = np.stack(cols, 1)
    return Y / (np.linalg.norm(Y, axis=0, keepdims=True) + 1e-8)


class SphericalSpatial(nn.Module):
    """306 sensors -> k virtual channels through smooth spatial filters (LISA 2026 low-rank /
    Défossez 2023 spatial attention, geometry-aware): virtual channel j = sum_i W[j, i] x_i with
    W[j, i] = sum_m z[j, type(i), m] Y_m(pos_i) — one spherical-harmonic field per sensor type
    (mag, grad-1, grad-2 share a location). Unconstrained (signed) weights; the first version
    used a softmax over sensors and collapsed to the global mean (HANDOVER §21.11)."""

    def __init__(self, k: int, degree: int = 8):
        super().__init__()
        import numpy as np
        from pathlib import Path
        g = np.load(Path(__file__).resolve().parent / "assets" / "sensor_geometry.npz")
        Y = torch.tensor(real_sph_harm(g["xyz"].astype("float64"), degree), dtype=torch.float32)  # (306, nb), unit columns
        self.register_buffer("Y", Y)
        self.register_buffer("onehot", F.one_hot(torch.tensor(g["stype"], dtype=torch.long), 3).float())  # (306, 3)
        # std so that each virtual channel starts with unit-ish gain over ~100 sensors of a type
        self.z = nn.Parameter(torch.randn(k, 3, Y.shape[1]) / np.sqrt(Y.shape[1]))

    def weights(self) -> torch.Tensor:
        field = torch.einsum("ktm,im->kti", self.z, self.Y)            # (k, 3, 306)
        return (field * self.onehot.T[None]).sum(1)                    # (k, 306), type-selected

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ki,bit->bkt", self.weights().to(x.dtype), x)


class CNN_TCN(nn.Module):
    """Parameter Team CNN+TCN, 50-way pooled head for word classification."""

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        model_dim: int = 100,
        tcn_layers: int = 20,
        dropout: float = 0.3,
        batch_norm: bool = False,
        pool: str = "avg",
        pool_queries: int = 4,
        time_emb_len: int = 0,
        spatial_k: int = 0,
    ):
        super().__init__()
        # spatial_k > 0: geometry-constrained spatial layer (306 -> k) before the 1x1 projection
        self.spatial = SphericalSpatial(spatial_k) if spatial_k else None
        self.proj = nn.Conv1d(spatial_k or n_channels, model_dim, kernel_size=1)
        # Learned per-timestep embedding added before pooling (long windows: the
        # attention pool must know where the onset sits; 0 = off, historic).
        self.time_emb = nn.Parameter(torch.zeros(1, model_dim, time_emb_len)) if time_emb_len else None
        self.proj_bn = nn.BatchNorm1d(model_dim) if batch_norm else nn.Identity()
        self.proj_drop = nn.Dropout(p=dropout)
        self.tcn = nn.Sequential(*[
            TCNBlock(model_dim, model_dim, dilation=2 ** (i % 8), dropout=dropout)
            for i in range(tcn_layers)
        ])
        # Keep this Sequential layout so QC checkpoints load with strict=True.
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(model_dim, n_classes),
        )
        if pool == "attn":
            self.attn_pool = AttentionPool(model_dim, pool_queries, dropout=dropout)
        elif pool != "avg":
            raise ValueError(f"pool must be avg|attn, got {pool!r}")
        else:
            self.attn_pool = None
        self.model_dim = model_dim

    def features(self, x, tap: int | None = None, tap_seq: bool = False):
        """Pooled TCN embedding (B, model_dim), before dropout + 50-way linear.

        With `tap` set, also returns the time-averaged activation of trunk
        block `tap` — intermediate-layer features for low-level auxiliary
        heads (Toshniwal et al. 2017: attach phoneme supervision below the
        final layer)."""
        if self.spatial is not None:
            x = self.spatial(x)
        x = self.proj_drop(self.proj_bn(self.proj(x)))
        if tap is not None:
            h_tap = self.tcn[:tap](x)
            x = self.tcn[tap:](h_tap)
        else:
            x = self.tcn(x)
        if self.time_emb is not None:
            te = self.time_emb[..., : x.shape[-1]]
            x = x + (F.pad(te, (0, x.shape[-1] - te.shape[-1])) if te.shape[-1] < x.shape[-1] else te)
        if self.attn_pool is not None:
            pooled = self.attn_pool(x)
        else:
            pooled = self.head[1](self.head[0](x))
        if tap is not None:
            # tap_seq: full (B, D, T) activation for frame-level heads
            return pooled, (h_tap if tap_seq else h_tap.mean(dim=2))
        return pooled

    def classify(self, features):
        """50-way logits from pooled features (B, model_dim)."""
        return self.head[3](self.head[2](features))

    def forward(self, x):
        return self.classify(self.features(x))


def balanced_topk_accuracy(logits, labels, k, n_classes):
    k = min(k, logits.shape[1])
    topk = logits.topk(k, dim=1).indices
    hit = (topk == labels.unsqueeze(1)).any(dim=1).float()
    recalls = []
    for c in range(n_classes):
        mask = labels == c
        if mask.any():
            recalls.append(hit[mask].mean())
    if not recalls:
        return 0.0
    return float(torch.stack(recalls).mean())


class LitWordClassifier(L.LightningModule):
    def __init__(
        self,
        n_channels,
        n_classes,
        lr=1e-3,
        weight_decay=0.01,
        class_weight=None,
        label_smoothing=0.1,
        model_dim=100,
        tcn_layers=20,
        dropout=0.3,
        sched_patience=2,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weight"])
        self.model = CNN_TCN(
            n_channels, n_classes,
            model_dim=model_dim, tcn_layers=tcn_layers, dropout=dropout,
        )
        self.loss_fn = nn.CrossEntropyLoss(
            weight=class_weight, label_smoothing=label_smoothing,
        )
        self._val_logits, self._val_labels = [], []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, batch_size=x.shape[0])
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        self.log("val_loss", self.loss_fn(logits, y), on_step=False,
                 on_epoch=True, prog_bar=True, batch_size=x.shape[0])
        self._val_logits.append(logits.detach().cpu())
        self._val_labels.append(y.detach().cpu())

    def on_validation_epoch_end(self):
        if not self._val_logits:
            return
        logits = torch.cat(self._val_logits)
        labels = torch.cat(self._val_labels)
        n = self.hparams.n_classes
        self.log("val_bacc10", balanced_topk_accuracy(logits, labels, 10, n), prog_bar=True)
        self.log("val_bacc1", balanced_topk_accuracy(logits, labels, 1, n), prog_bar=True)
        self._val_logits.clear()
        self._val_labels.clear()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", patience=self.hparams.sched_patience, factor=0.5,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "monitor": "val_bacc10",
                "interval": "epoch",
            },
        }
