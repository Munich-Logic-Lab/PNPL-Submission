"""Build a Deep-track Kaggle CSV from a hybrid TCN checkpoint.

Notebook 01's holdout loop is isolated `model(x)` on 1 s windows. That drops
the sentence mixer. This script groups holdout sentence-source rows (868) by
epoch, runs mixer + CE/retrieval blend, and scores the 92 isolated words as
single-token sentences. Canonical `holdout.indices` order is preserved.

Holdout MEG is raw: z-score with the frozen QC train scaler, clip ±10.

  python scripts/submit_tcn_hybrid_holdout.py \\
      --ckpt lightning_logs/tcn_hybrid/version_2/checkpoints/best-epoch=21-val_bacc10=0.704.ckpt

  # upload (2/day). Needs KAGGLE_API_TOKEN or ~/.kaggle/kaggle.json
  python scripts/submit_tcn_hybrid_holdout.py --ckpt ... --submit \\
      --message "hybrid TCN blend ep21"
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F

from data_prep import normalize_word
from hybrid_tcn import LitHybridTCN
from pnpl.competition import (
    LibriBrainCompetitionHoldout,
    load_vocabulary,
    submit_to_kaggle,
    write_submission,
)
from t5_text import T5_LARGE, T5_LAYER, embed_unique_words, gallery_matrix

DATA_PATH = ROOT / "data" / "libribrain_word"
PREP = ROOT / "artifacts" / "data_prep"
EMB_DIR = DATA_PATH / "t5_cache"
MEAN_NPY = PREP / "train_channel_mean.npy"
STD_NPY = PREP / "train_channel_std.npy"
CLIP = 10.0
MAX_WORDS = 32
KAGGLE_DEEP = "pnpl-competition-2026-deep"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def long_windows(holdout, rows, mean, std, tmin: float, tmax: float, holdout_dir: Path, sf: int = 250) -> torch.Tensor:
    """Windows [onset + tmin, onset + tmax) cut from the standardized continuous
    sentence epochs (subj00_holdout01_sentence.npz); samples outside the epoch
    stay 0 (= train mean), exactly as --epoch-mask does in training. Isolated rows
    only have their 1-s window."""
    n = int(round((tmax - tmin) * sf))
    out = torch.zeros(len(rows), mean.shape[1], n)
    if not hasattr(long_windows, "cache"):
        z = np.load(Path(holdout_dir) / "COMPETITION_HOLDOUT" / "subj00_holdout01_sentence.npz")
        long_windows.cache = {"meg": z["meg"], "sentence_n_times": z["sentence_n_times"]}
    npz = long_windows.cache
    for r, i in enumerate(rows):
        m = holdout.metadata[i]
        if m["source"] == "sentence":
            n_t = int(npz["sentence_n_times"][m["epoch"]])
            ep = standardize(npz["meg"][m["epoch"]][None, :, :n_t], mean, std)[0]
            s0 = int(m["start_sample"]) + int(round(tmin * sf))
        else:
            ep = standardize(holdout[i][0][None], mean, std)[0]
            s0 = int(round(tmin * sf))
        lo, hi = max(0, s0), min(ep.shape[-1], s0 + n)
        if hi > lo:
            out[r, :, lo - s0 : hi - s0] = ep[:, lo:hi]
    return out


def synthetic_windows(holdout, rows, holdout_dir: Path, win_len: int = 250, seed: int = 0) -> torch.Tensor:
    """Timing-audit input (HANDOVER 21.18): no brain signal, only word onsets — the same construction as
    SentenceWordDataset(synthetic=True): fixed spatial pattern (torch seed 0) x 300-ms half-sine bumps at all
    word onsets of the row's sentence epoch (relative to the row's onset) + 0.1 white noise (seeded)."""
    from hybrid_tcn import SYN_KERNEL_S
    if not hasattr(synthetic_windows, "cache"):
        z = np.load(Path(holdout_dir) / "COMPETITION_HOLDOUT" / "subj00_holdout01_sentence.npz")
        synthetic_windows.cache = {"on": z["word_onsets_s"], "wm": z["word_mask"]}
    c = synthetic_windows.cache
    pattern = torch.randn(306, generator=torch.Generator().manual_seed(0))
    gen = torch.Generator().manual_seed(seed + int(rows[0]))
    t = torch.arange(win_len) / 250.0
    out = torch.empty(len(rows), 306, win_len)
    for r, i in enumerate(rows):
        m = holdout.metadata[i]
        if m["source"] == "sentence":
            ons = torch.tensor(np.asarray(c["on"][m["epoch"]])[np.asarray(c["wm"][m["epoch"]])], dtype=torch.float32)
            rel = ons - float(m["start_sample"]) / 250.0
        else:
            rel = torch.zeros(1)
        u = t[None, :] - rel[:, None]
        bump = torch.where((u >= 0) & (u < SYN_KERNEL_S), torch.sin(math.pi * u / SYN_KERNEL_S), torch.zeros(()))
        out[r] = pattern[:, None] * bump.sum(0)[None, :] + 0.1 * torch.randn(306, win_len, generator=gen)
    return out


def standardize(meg: np.ndarray, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    x = (torch.from_numpy(np.asarray(meg, dtype=np.float32)) - mean) / std
    return x.clamp(-CLIP, CLIP)


def sentence_groups(holdout: LibriBrainCompetitionHoldout) -> list[list[int]]:
    """Canonical-order groups: one isolated word, or all words in one sentence."""
    groups = []
    i, n = 0, len(holdout)
    while i < n:
        m = holdout.metadata[i]
        if m["source"] == "word":
            groups.append([i])
            i += 1
            continue
        ep = m["epoch"]
        j = i + 1
        while (
            j < n
            and holdout.metadata[j]["source"] == "sentence"
            and holdout.metadata[j]["epoch"] == ep
        ):
            j += 1
        groups.append(list(range(i, j)))
        i = j
    return groups


@torch.no_grad()
def predict_group(model, windows: torch.Tensor, device: str, mode: str = "full") -> torch.Tensor:
    """windows (W, 306, 250) → logits (W, 50) for CE / ret / blend (log-space blend).

    mode: full = mixer over the sentence (as trained); nomix = encoder only
    (each word from its own window); shuffle = mixer over a random word order,
    outputs mapped back to the original rows.
    """
    n = windows.shape[0]
    perm = None
    if mode == "shuffle" and n > 1:
        perm = torch.randperm(n)
        windows = windows[perm]
    out = {"ce": [], "ret": [], "blend": []}
    for start in range(0, n, MAX_WORDS):
        chunk = windows[start : start + MAX_WORDS].to(device)
        w = chunk.shape[0]
        meg = chunk.unsqueeze(0)
        mask = torch.ones(1, w, dtype=torch.bool, device=device)
        if mode == "nomix":
            h = model.encode_windows(chunk).unsqueeze(0)
        else:
            h = model.contextualize(meg, mask)
        ce, pred = model.heads(h)
        ce, pred = ce[0], pred[0]
        out["ce"].append(ce.cpu())
        out["ret"].append(model.retrieval_logits(pred).cpu())
        out["blend"].append(model.blended_logits(ce, pred).cpu())
    res = {k: torch.cat(v, dim=0) for k, v in out.items()}
    if perm is not None:
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(n)
        res = {k: v[inv] for k, v in res.items()}
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--head", choices=["ce", "ret", "blend"], default="blend")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--holdout-dir", type=Path, default=ROOT / "data" / "holdout")
    p.add_argument(
        "--holdout-scaler", choices=["train", "self"], default="train",
        help="train = frozen train npy scaler (historic; but holdout amplitudes "
             "are ~9-10x smaller than train, so standardized inputs land at "
             "std~0.16 instead of ~1). self = fit per-channel mean/std on the "
             "960 unlabeled holdout windows (transductive, label-free).",
    )
    p.add_argument("--batch-groups", type=int, default=1)
    p.add_argument("--mode", choices=["full", "nomix", "shuffle"], default="full")
    p.add_argument("--adabn", action="store_true",
                   help="re-estimate BatchNorm stats on the 960 unlabelled holdout windows (AdaBN)")
    p.add_argument("--synthetic-onsets", action="store_true",
                   help="timing audit: score a --synthetic-onsets model on onset-only holdout windows (diagnostic)")
    p.add_argument("--shift-ms", type=float, default=0.0,
                   help="test-time onset shift: windows [onset + shift, onset + shift + 1 s) cut from the "
                        "continuous sentence epochs (epoch-masked, as dump_ensemble_preds --conds sh<ms>)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--submit", action="store_true")
    p.add_argument("--message", type=str, default="hybrid TCN")
    p.add_argument(
        "--kaggle-competition",
        type=str,
        default=KAGGLE_DEEP,
        help="Kaggle slug. Intro notebook uses 'deep'; the real slug is this default.",
    )
    args = p.parse_args()
    if not args.ckpt.exists():
        raise FileNotFoundError(args.ckpt)
    for path in (MEAN_NPY, STD_NPY):
        if not path.exists():
            raise FileNotFoundError(path)

    device = pick_device()
    print(f"ckpt: {args.ckpt}", flush=True)
    print(f"device: {device}  head: {args.head}", flush=True)

    mean = torch.tensor(np.load(MEAN_NPY), dtype=torch.float32)[None, :, None]
    std = torch.tensor(np.load(STD_NPY), dtype=torch.float32)[None, :, None]

    vocab = load_vocabulary("primary")
    word_emb = embed_unique_words(
        [normalize_word(w) for w in vocab],
        EMB_DIR,
        model_name=T5_LARGE,
        layer=T5_LAYER,
        device="cpu",
    )
    gallery = gallery_matrix(word_emb, vocab)
    model = LitHybridTCN.load_from_checkpoint(
        str(args.ckpt),
        gallery=gallery,
        class_weight=torch.ones(len(vocab)),
        map_location=device,
        strict=False,
    )
    model.eval().to(device)
    print(f"mixer_gate={float(model.mixer.gate.detach().cpu()):.4f}", flush=True)

    args.holdout_dir.mkdir(parents=True, exist_ok=True)
    print("Loading Deep holdout (subj00 sentence+word)…", flush=True)
    t0 = time.perf_counter()
    holdout = LibriBrainCompetitionHoldout(
        track="deep",
        data_path=str(args.holdout_dir),
        download=True,
    )
    print(
        f"  {holdout.counts()}  ({time.perf_counter() - t0:.1f}s)",
        flush=True,
    )

    if args.holdout_scaler == "self":
        allw = np.stack([holdout[i][0] for i in range(len(holdout))])
        m_self = allw.mean(axis=(0, 2))
        s_self = allw.std(axis=(0, 2))
        ratio = float(np.median(s_self / np.load(STD_NPY)))
        mean = torch.tensor(m_self, dtype=torch.float32)[None, :, None]
        std = torch.tensor(s_self, dtype=torch.float32)[None, :, None]
        print(f"holdout-fit scaler (median std ratio vs train: {ratio:.3f})", flush=True)

    torch.manual_seed(args.seed)
    if args.adabn:
        from adabn import adapt_bn
        allw = standardize(np.stack([holdout[i][0] for i in range(len(holdout))]), mean, std)
        if allw.shape[-1] < 250:
            allw = F.pad(allw, (0, 250 - allw.shape[-1]))
        n_bn = adapt_bn(model, allw[..., :250], device)
        print(f"AdaBN: re-estimated {n_bn} BatchNorm layers on {allw.shape[0]} holdout windows", flush=True)
    groups = sentence_groups(holdout)
    n = len(holdout)
    logits = torch.empty(n, len(vocab))
    t1 = time.perf_counter()
    n_sent = n_word = 0
    tmin, tmax = float(model.hparams.get("win_tmin", 0.0)), float(model.hparams.get("win_tmax", 1.0))
    if args.shift_ms:
        tmin, tmax = tmin + args.shift_ms / 1000.0, tmax + args.shift_ms / 1000.0
        print(f"onset shift {args.shift_ms:+.0f} ms", flush=True)
    long = (tmin, tmax) != (0.0, 1.0)
    if long:
        print(f"long windows [{tmin}, {tmax}) s cut from the continuous sentence epochs", flush=True)
    for g in groups:
        if long:
            pred = predict_group(model, long_windows(holdout, g, mean, std, tmin, tmax, args.holdout_dir), device, args.mode)[args.head]
            for row, idx in enumerate(g):
                logits[idx] = pred[row]
            n_sent += len(g) if holdout.metadata[g[0]]["source"] == "sentence" else 0
            n_word += 0 if holdout.metadata[g[0]]["source"] == "sentence" else len(g)
            continue
        if args.synthetic_onsets:
            windows = synthetic_windows(holdout, g, args.holdout_dir)
        else:
            raw = np.stack([holdout[i][0] for i in g])
            windows = standardize(raw, mean, std)
        if windows.shape[-1] < 250:
            windows = F.pad(windows, (0, 250 - windows.shape[-1]))
        elif windows.shape[-1] > 250:
            windows = windows[..., :250]
        pred = predict_group(model, windows, device, args.mode)[args.head]
        for row, idx in enumerate(g):
            logits[idx] = pred[row]
        if holdout.metadata[g[0]]["source"] == "sentence":
            n_sent += len(g)
        else:
            n_word += len(g)
    print(
        f"scored sentence={n_sent} word={n_word} in {time.perf_counter() - t1:.1f}s",
        flush=True,
    )

    probs = torch.softmax(logits, dim=1).numpy()
    # Primary is scored; moses columns are required by the sample CSV but
    # unused. Fill a uniform 50-way so the header matches Kaggle's template.
    moses = load_vocabulary("moses")
    secondary = np.full((len(probs), len(moses)), 1.0 / len(moses), dtype=np.float64)
    out = args.out or (
        ROOT / "submissions" / f"submission_deep_hybrid_{args.head}_{args.ckpt.stem}.csv"
    )
    path = write_submission(
        path=str(out),
        indices=holdout.indices,
        primary_probs=probs,
        secondary_probs=secondary,
        normalize=True,
    )
    print(f"wrote {path}  rows={len(probs)} cols=1+{probs.shape[1]}", flush=True)

    if args.submit:
        result = submit_to_kaggle(
            path,
            competition=args.kaggle_competition,
            message=args.message,
        )
        print(result, flush=True)


if __name__ == "__main__":
    main()
