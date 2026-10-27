"""AdaBN (Li et al. 2016): re-estimate BatchNorm statistics on unlabelled target-domain
windows (e.g. the competition holdout's new recording sessions). Only BN running
mean/var change; weights, dropout (kept in eval) and the mixer are untouched."""

from __future__ import annotations

import torch
import torch.nn as nn


@torch.no_grad()
def adapt_bn(model, windows: torch.Tensor, device: str, batch: int = 256) -> int:
    """windows (N, C, T) standardized like training input. Returns #BN layers adapted."""
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None          # cumulative average over all target batches
    model.eval()
    for m in bns:
        m.train()
    for i in range(0, windows.shape[0], batch):
        model.encode_windows(windows[i : i + batch].float().to(device))
    model.eval()
    return len(bns)
