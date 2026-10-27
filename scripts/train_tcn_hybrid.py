"""Hybrid Deep TCN: sentence mixer + open-vocab T5 retrieval + 50-way CE.

Starts from the QC Sherlock ckpt (val 0.443 / test 0.460). Encoder weights
are loaded; mixer gate is 0 at H0 so isolated CE must still sit near 0.443.

Four pieces (all on CNN+TCN, not d'Ascoli's SimpleConv):
  1. Frozen t5-large targets (layer 12, mean-pool) instead of class-ids-only
  2. Open-vocab SigLIP on every word; 50-word gallery only at eval
  3. Hybrid CE + SigLIP heads on the same pooled TCN features
  4. Residual sentence mixer at train and val (holdout-style context)

Intended machine: gpu-sandbox RTX 5090 32 GB (`~/venv`, torch cu128).
QC TCN sat at ~1.4 GB / batch 32 windows; this run is sentence-batched but
windows are chunked (`encode_chunk=128` on CUDA), so 32 GB is ample.

Default is a *screen*: 800 steps × 3 epochs on Sherlock QC data.
Full epoch only after H0 + screen keep.

On the 5090 (do NOT uv sync — that replaces CUDA torch):

  nvidia-smi   # Ismail may already be on the GPU
  pip install transformers sentencepiece
  python scripts/train_tcn_hybrid.py --host sandbox

CUDA auto-sets --num-workers 4, --batch-size 16, TensorBoard :6008.
Mac MPS: pass --batch-size 2 --num-workers 0 --host local.

First run downloads t5-large (~3 GB) into the HF cache and writes
data/libribrain_word/t5_cache/.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Timer
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from cnn_tcn import balanced_topk_accuracy
from data_prep import corpus_family, load_drop_keys, normalize_word
from hybrid_tcn import (
    LitHybridTCN,
    SentenceWordDataset,
    collate_sentences,
)
from pnpl.competition import load_vocabulary
from pnpl.datasets import LibriBrain100Word
from t5_text import T5_LARGE, T5_LAYER, embed_unique_words, gallery_matrix

DATA_PATH = ROOT / "data" / "libribrain_word"
PREP = ROOT / "artifacts" / "data_prep"
LOG_DIR = ROOT / "lightning_logs"
EMB_DIR = DATA_PATH / "t5_cache"
CKPT = LOG_DIR / "tcn_deep_qc/version_2/checkpoints/best-epoch=05-val_bacc10=0.443.ckpt"
DROP_LIST = PREP / "drop_list_train_rms_plus_jump.csv"
DROP_LIST_ALL = PREP / "drop_list_train_rms_plus_jump_all.csv"
MEAN_NPY = PREP / "train_channel_mean.npy"
STD_NPY = PREP / "train_channel_std.npy"
LOG_NAME = "tcn_hybrid"

TMIN, TMAX = 0.0, 1.0
BATCH_SIZE = 8
EPOCHS = 15
LR_BACKBONE = 1e-4
LR_NEW = 3e-4
WEIGHT_DECAY = 0.01
DROPOUT = 0.3
ES_PATIENCE = 5
SCHED_PATIENCE = 2
LABEL_SMOOTHING = 0.1
TB_PORT = 6008
H0_MIN = 0.38
PROMOTE_VAL = 0.424
PROMOTE_TEST = 0.42
SEED = 0
MAX_WORDS = 32
ENCODE_CHUNK = 64
SFREQ = 250
# Competition Deep holdout: 156 sentences (+92 isolated words) — words per
# sentence -> count, measured from artifacts/errors/holdout_meta.parquet
# (HANDOVER §17). Median 5, mean 5.56, max 9. Isolated items are 92/248 = 37%
# of holdout *items*, which is also what makes ~10% of holdout *rows* isolated.
HOLDOUT_SENT_LEN_HIST = {1: 2, 3: 2, 4: 24, 5: 55, 6: 37, 7: 23, 8: 11, 9: 2}
HOLDOUT_ISO_FRAC = 92 / (92 + 156)

VOCAB = load_vocabulary("primary")
VOCAB_TO_ID = {normalize_word(w): i for i, w in enumerate(VOCAB)}


class CompetitionVocabDataset(Dataset):
    def __init__(self, base):
        self.base = base
        self.keep = []
        for i, sample in enumerate(base.samples):
            vid = VOCAB_TO_ID.get(normalize_word(sample[5]))
            if vid is not None:
                self.keep.append((i, vid))

    def __len__(self):
        return len(self.keep)

    def __getitem__(self, idx):
        base_idx, vocab_id = self.keep[idx]
        meg = self.base[base_idx][0].float()
        return meg, torch.tensor(vocab_id, dtype=torch.long)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SampleView:
    """Index view so TIMIT val can be one held-out session of the train partition."""

    def __init__(self, base, keep_fn):
        self._base = base
        src = list(base.samples)
        self._idx = [i for i, s in enumerate(src) if keep_fn(s)]
        if not self._idx:
            raise RuntimeError("SampleView is empty — check session/corpus filters.")
        self._samples = [src[i] for i in self._idx]

    @property
    def samples(self):
        return self._samples

    def __len__(self):
        return len(self._idx)

    def __getitem__(self, i):
        return self._base[self._idx[i]]


def rate_match_weights(base, vocab_to_id, cap: float = 8.0):
    """p_holdout(IOI) / p_train(IOI after a vocab word), per RATE_BIN bin, capped and
    normalised to max 1. Holdout IOIs come from its word onsets (no labels)."""
    from hybrid_tcn import RATE_BIN
    z = np.load(ROOT / "data" / "holdout" / "COMPETITION_HOLDOUT" / "subj00_holdout01_sentence.npz")
    on, wm = z["word_onsets_s"], z["word_mask"]
    hold = np.concatenate([np.diff(np.sort(on[i][wm[i]])) for i in range(len(on))])
    by_run: dict = defaultdict(list)
    for s in base.samples:
        by_run[(s[1], s[2], s[3])].append((float(s[4]), normalize_word(s[5])))
    tr = []
    for v in by_run.values():
        v.sort()
        tr += [b[0] - a[0] for a, b in zip(v, v[1:]) if a[1] in vocab_to_id and 0 < b[0] - a[0] < 2.0]
    bins = np.arange(0, 1.0 + RATE_BIN, RATE_BIN)
    ph = np.histogram(np.clip(hold, 0, 0.999), bins)[0] + 1.0
    pt = np.histogram(np.clip(tr, 0, 0.999), bins)[0] + 1.0
    w = np.minimum((ph / ph.sum()) / (pt / pt.sum()), cap)
    w = w / w.max()
    print(f"  rate match: holdout IOI median {np.median(hold):.3f}s, train-after-vocab median {np.median(tr):.3f}s; "
          f"weights {np.round(w, 2).tolist()}", flush=True)
    return w


def holdout_onset_gaps() -> np.ndarray:
    """Onset gaps (s) between consecutive words of the holdout sentence epochs (no labels)."""
    z = np.load(ROOT / "data" / "holdout" / "COMPETITION_HOLDOUT" / "subj00_holdout01_sentence.npz")
    on, wm = z["word_onsets_s"], z["word_mask"]
    return np.concatenate([np.diff(np.sort(on[i][wm[i]])) for i in range(len(on))])


def load_poe_table(path: str, beta_t: float, beta_l: float) -> dict:
    z = np.load(path, allow_pickle=False)
    lp = beta_t * z["lp_timing"].astype(np.float32) + beta_l * z["lp_lm"].astype(np.float32)
    return dict(zip(z["keys"].tolist(), lp))


def make_split(partition, channel_means, channel_stds, corpus: str, subjects="deep",
               pad_s: float = 0.0):
    # pad_s > 0 widens the window on both sides so the dataset can take a
    # random 1 s crop (temporal jitter). Val/test/holdout always use pad 0.
    kw = dict(
        data_path=str(DATA_PATH),
        partition=partition,
        subjects=subjects,
        corpus=corpus,
        tmin=TMIN - pad_s,
        tmax=TMAX + pad_s,
        standardize=True,
        include_info=False,
        preload_files=False,
        download=True,
    )
    if channel_means is not None:
        kw["channel_means"] = channel_means
        kw["channel_stds"] = channel_stds
    return LibriBrain100Word(**kw)


class ScaledBase:
    """Multiply a base's already-standardized windows by a constant.

    Broad subjects must be standardized with *their own* channel stats — 12
    subjects means 12 head positions, so the same sensor does not see the
    same cortex (LibriBrain paper; docs/RESEARCH.md). Self-fit stats put
    their windows at std~1, while deep windows under the frozen sub-0 npy
    land at ~0.165 (the npy is fit on higher-variance statistics, HANDOVER
    §6c). Un-corrected that is a ~6x amplitude split between the two
    populations in one batch. The factor is measured at run time, not
    assumed, and printed — §6c's rule after the self-scaler false alarm.
    """

    def __init__(self, base, scale):
        self._base = base
        # float, or a per-channel (C,) array (--broad-scale profile)
        self.scale = (torch.tensor(np.asarray(scale), dtype=torch.float32)[:, None]
                      if np.ndim(scale) else float(scale))

    @property
    def samples(self):
        return self._base.samples

    def __len__(self):
        return len(self._base)

    def __getitem__(self, i):
        out = self._base[i]
        return (out[0] * self.scale, *out[1:])


class MultiBase:
    """Concatenate bases that carry different scalers, keeping the SampleView API.

    ``SentenceWordDataset`` only needs ``.samples`` (for grouping/labels) and
    integer indexing. Sentence uids include ``sample[0]`` (subject), so
    per-subject sentences never merge across subjects even though every
    broad subject heard the identical audio.
    """

    def __init__(self, bases):
        self._bases = list(bases)
        self._samples = []
        self._map: list[tuple[int, int]] = []
        for bi, b in enumerate(self._bases):
            for li in range(len(b.samples)):
                self._samples.append(b.samples[li])
                self._map.append((bi, li))

    @property
    def samples(self):
        return self._samples

    def __len__(self):
        return len(self._map)

    def __getitem__(self, i):
        bi, li = self._map[i]
        return self._bases[bi][li]


def window_std(base, n: int = 256, seed: int = 0) -> float:
    """Empirical std of n randomly drawn standardized windows."""
    rng = np.random.default_rng(seed)
    take = min(n, len(base))
    idx = rng.choice(len(base), size=take, replace=False)
    stack = np.stack([np.asarray(base[int(i)][0], dtype=np.float64) for i in idx])
    return float(stack.std())


def window_std_ch(base, n: int = 256, seed: int = 0) -> np.ndarray:
    """Per-channel empirical std of n randomly drawn standardized windows, (C,)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(base), size=min(n, len(base)), replace=False)
    stack = np.stack([np.asarray(base[int(i)][0], dtype=np.float64) for i in idx])
    return stack.std(axis=(0, 2))


def clip_to_recorded(base, tail_s: float):
    """Drop samples whose window runs past the recorded data. The broad 50%/25%
    groups' competition uploads (desc-firsthalf / desc-firstquarter H5, symlinked
    under the standard name) keep the full events TSV but only part of the MEG."""
    ends = {rk: base.get_h5_dataset(rk).shape[1] / SFREQ for rk in base.run_keys}
    keep = lambda s: float(s[4]) + tail_s <= ends[tuple(s[:4])] - 0.01  # noqa: E731
    if all(keep(s) for s in base.samples):
        return base
    view = SampleView(base, keep)
    print(f"    clipped to recorded data: {len(base.samples)} -> {len(view.samples)} windows", flush=True)
    return view


def parse_subject_spec(spec: str) -> list[int]:
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


class KeepEpochCkpt(L.Callback):
    """Write a full checkpoint every val epoch.

    On the first corpus=all job, Lightning ``save_last`` never moved past
    epoch 0 while ``save_top_k=1`` only kept the Sherlock peak. Intermediate
    weights were unrecoverable after SIGTERM.
    """

    def __init__(self, dirpath: Path, keep: int = 8):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.keep = keep

    def on_validation_end(self, trainer, _pl_module):
        if trainer.sanity_checking:
            return
        self.dirpath.mkdir(parents=True, exist_ok=True)
        path = self.dirpath / f"epoch={int(trainer.current_epoch):02d}.ckpt"
        trainer.save_checkpoint(str(path))
        snaps = sorted(self.dirpath.glob("epoch=*.ckpt"))
        for old in snaps[: max(0, len(snaps) - self.keep)]:
            old.unlink(missing_ok=True)


class ValDropStop(L.Callback):
    def __init__(self, monitor="val_bacc10", patience: int = 2):
        self.monitor = monitor
        self.patience = patience
        self.best = None
        self.drops = 0

    def on_validation_epoch_end(self, trainer, _pl_module):
        raw = trainer.callback_metrics.get(self.monitor)
        if raw is None:
            return
        val = float(raw)
        if self.best is None or val > self.best + 1e-6:
            print(f"  new peak {self.monitor}={val:.3f}", flush=True)
            self.best = val
            self.drops = 0
            return
        self.drops += 1
        print(
            f"  {self.monitor}={val:.3f}  peak={self.best:.3f}  "
            f"drops_after_peak={self.drops}/{self.patience}",
            flush=True,
        )
        if self.drops >= self.patience:
            print(
                f"Operational stop: {self.monitor} fell {self.drops} epochs "
                f"after peak {self.best:.3f}",
                flush=True,
            )
            trainer.should_stop = True


@torch.no_grad()
def eval_isolated_ce(backbone, ds, device, name):
    was_training = backbone.training
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    backbone.eval().to(device)
    logits, labels = [], []
    t0 = time.perf_counter()
    for x, y in loader:
        logits.append(backbone(x.to(device)).cpu())
        labels.append(y)
    logits = torch.cat(logits)
    labels = torch.cat(labels)
    n = len(VOCAB)
    b10 = balanced_topk_accuracy(logits, labels, 10, n)
    b1 = balanced_topk_accuracy(logits, labels, 1, n)
    print(
        f"{name}: n={len(ds)}  BAcc@10={b10:.3f}  BAcc@1={b1:.3f}  "
        f"({time.perf_counter() - t0:.1f}s)",
        flush=True,
    )
    backbone.train(was_training)
    return b10, b1


@torch.no_grad()
def eval_sentence(model, loader, device, name):
    model.eval().to(device)
    buckets = {"ce": [], "ret": [], "blend": [], "y": []}
    t0 = time.perf_counter()
    for meg, emb, vid, mask, *_rest in loader:
        meg, emb, vid, mask = meg.to(device), emb.to(device), vid.to(device), mask.to(device)
        h = model.contextualize(meg, mask)
        ce_logits, pred = model.heads(h)
        y = vid[mask]
        keep = y >= 0
        if not keep.any():
            continue
        ce = ce_logits[mask][keep]
        pr = pred[mask][keep]
        buckets["ce"].append(ce.cpu())
        buckets["ret"].append(model.retrieval_logits(pr).cpu())
        buckets["blend"].append(model.blended_logits(ce, pr).cpu())
        buckets["y"].append(y[keep].cpu())
    y = torch.cat(buckets["y"])
    n = len(VOCAB)
    out = {}
    for key in ("ce", "ret", "blend"):
        logits = torch.cat(buckets[key])
        out[f"{key}_bacc10"] = balanced_topk_accuracy(logits, y, 10, n)
        out[f"{key}_bacc1"] = balanced_topk_accuracy(logits, y, 1, n)
    print(
        f"{name}: n={len(y)}  "
        f"CE@10={out['ce_bacc10']:.3f}  ret@10={out['ret_bacc10']:.3f}  "
        f"blend@10={out['blend_bacc10']:.3f}  ({time.perf_counter() - t0:.1f}s)",
        flush=True,
    )
    return out


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def start_tensorboard(logdir: Path, port: int, bind_all: bool = False) -> None:
    if _port_open(port):
        print(f"TensorBoard already running → http://localhost:{port}", flush=True)
        return
    log = LOG_DIR / "tensorboard.server.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    tb = shutil.which("tensorboard")
    if tb:
        cmd = [tb, "--logdir", str(logdir), "--port", str(port), "--reload_interval", "5"]
    else:
        cmd = [
            sys.executable, "-m", "tensorboard.main", "--logdir", str(logdir),
            "--port", str(port), "--reload_interval", "5",
        ]
    if bind_all:
        cmd.append("--bind_all")
    with open(log, "ab") as err:
        subprocess.Popen(cmd, stdout=err, stderr=err, start_new_session=True)
    for _ in range(40):
        if _port_open(port):
            extra = " (bind_all)" if bind_all else ""
            print(f"TensorBoard live → http://localhost:{port}{extra}", flush=True)
            return
        time.sleep(0.25)
    print(f"TensorBoard did not bind :{port}; check {log}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit-train-batches", type=int, default=800)
    p.add_argument("--max-epochs", type=int, default=3)
    p.add_argument("--log-name", type=str, default=LOG_NAME)
    p.add_argument("--ckpt", type=Path, default=CKPT)
    p.add_argument("--drop-patience", type=int, default=3)
    p.add_argument(
        "--es-patience",
        type=int,
        default=ES_PATIENCE,
        help="Lightning EarlyStopping patience on val_bacc10. Raise this with "
             "--drop-patience or ES still stops at 5.",
    )
    p.add_argument(
        "--num-workers", type=int, default=None,
        help="DataLoader workers. Default: 4 on CUDA (5090), 0 on Mac.",
    )
    p.add_argument("--tb-port", type=int, default=TB_PORT)
    p.add_argument("--tb-bind-all", action="store_true")
    p.add_argument("--corpus", type=str, default="sherlock")
    p.add_argument(
        "--val-corpus",
        type=str,
        default="auto",
        help="sherlock | mixed | auto. auto=mixed when train corpus is not sherlock. "
             "mixed = equal-vote family val (Sherlock ses-11 + MOCHA val + Moth val + "
             "TIMIT held-out session). Stops on the macro mean, not Sherlock alone.",
    )
    p.add_argument(
        "--timit-val-session",
        type=str,
        default="14",
        help="TIMIT has no pnpl val split; this train session is held out of train.",
    )
    p.add_argument(
        "--sampler",
        type=str,
        default="auto",
        choices=("auto", "book", "family"),
        help="auto: book on sherlock, family on mixed corpora so TIMIT/Moth are not drowned.",
    )
    p.add_argument(
        "--sampler-alpha",
        type=float,
        default=0.0,
        help="Group weight = n_group**(alpha-1). 0.0 = equal mass per group "
             "(historic behavior), 0.5 = sqrt smoothing (big groups keep majority "
             "mass, tiny groups like MOCHA stop being 25%% of every epoch), "
             "1.0 = natural frequency.",
    )
    p.add_argument(
        "--lr-backbone",
        type=float,
        default=None,
        help=f"Backbone AdamW LR. Default {LR_BACKBONE} ({'{:g}'.format(LR_BACKBONE)}), "
             "or lr-new when --from-scratch.",
    )
    p.add_argument(
        "--lr-new",
        type=float,
        default=None,
        help=f"Mixer/retrieval/SigLIP LR. Default {LR_NEW}.",
    )
    p.add_argument(
        "--channel-dropout",
        type=float,
        default=0.0,
        help="Train-time sensor dropout probability per channel per window "
             "(0 = off; kept channels rescaled by 1/(1-p)).",
    )
    p.add_argument(
        "--broad-subjects",
        type=str,
        default="",
        help="Broad multi-subject ids to add to TRAIN, e.g. '1-12'. Empty = off "
             "(historic). These subjects only ever heard Sherlock1 ses-11/ses-12; "
             "pnpl has no broad train partition, so they enter train via "
             "--broad-partition. Each subject is standardized with its own "
             "channel stats and rescaled to the deep window amplitude.",
    )
    p.add_argument(
        "--broad-partition",
        type=str,
        default="test",
        choices=("test", "validation", "both"),
        help="test = broad ses-12 (Study in Scarlet Book 2 ch 5). ses-12 text is "
             "in neither our deep train nor our val rail, so the Sherlock val "
             "rail stays clean. validation = ses-11 = the exact stimulus of our "
             "val rail; that leaks and makes val_bacc10_sherlock meaningless.",
    )
    p.add_argument(
        "--broad-share",
        type=float,
        default=-1.0,
        help="Fraction of each epoch drawn from broad subjects. -1 = natural "
             "frequency (broad sentences / all sentences, ~4%%). The family "
             "sampler's equal-mass rule would otherwise hand broad 1/5 of every "
             "epoch, and broad is only ~220 unique sentences seen 12x — the same "
             "repetition shape HANDOVER §15/6a blames for the 0.48178 run.",
    )
    p.add_argument(
        "--broad-scale",
        choices=("scalar", "profile"),
        default="scalar",
        help="scalar = one factor per subject (historic). profile = per-channel factor "
             "that maps each subject's self-standardized channel stds onto the deep "
             "train windows' per-channel std profile (matches broad_holdout_preds.py).",
    )
    p.add_argument(
        "--macro-families",
        type=str,
        default="",
        help="Comma list of families whose per-family BAcc@10 votes in "
             "val_bacc10 (stop/select metric). Empty = all present (historic). "
             "Use timit,mocha,podcasts to keep Sherlock out of selection; "
             "Sherlock is still logged as val_bacc10_sherlock.",
    )
    p.add_argument(
        "--drop-list",
        type=Path,
        default=None,
        help="Train drop CSV. Default: Sherlock QC list, or *_all.csv when --corpus all.",
    )
    p.add_argument(
        "--batch-size", type=int, default=None,
        help="Sentences per batch. Default: 16 on CUDA (5090), 8 otherwise.",
    )
    p.add_argument("--max-words", type=int, default=MAX_WORDS)
    p.add_argument(
        "--span-dist", type=str, default="off", choices=("off", "holdout"),
        help="Train-time span sampling. off = historic (whole sentences, chunked "
             "at --max-words). holdout = every fetched item is a random span of "
             "its sentence whose length is drawn from the competition holdout's "
             "sentence-length histogram (median 5), or a single word with "
             "probability --span-iso-frac. HANDOVER §17: the holdout is short "
             "contexts + 92 isolated words, on which the 0.646 model is at chance.",
    )
    p.add_argument(
        "--span-iso-frac", type=float, default=HOLDOUT_ISO_FRAC,
        help="With --span-dist holdout: probability an item is a single word "
             f"(default {HOLDOUT_ISO_FRAC:.2f} = the holdout's isolated share of items).",
    )
    p.add_argument(
        "--jitter-ms", type=float, default=0.0,
        help="Train-time temporal jitter: load windows widened by this much on "
             "each side and take a random 1 s crop per word (0 = off). Val/test/"
             "holdout windows are untouched.",
    )
    p.add_argument(
        "--amp-scale", type=float, default=0.0,
        help="Train-time per-window gain U(1-a, 1+a) (0 = off).",
    )
    p.add_argument(
        "--time-mask-ms", type=float, default=0.0,
        help="Train-time time masking: zero one random span of up to this many ms "
             "in half of the windows (MEGAugment-style). 0 = off.",
    )
    p.add_argument(
        "--band-perturb", type=float, default=0.0,
        help="Train-time frequency-band perturbation: scale one random rFFT band "
             "by U(1-a, 1+a) per window. 0 = off.",
    )
    p.add_argument(
        "--frame-phoneme-lambda", type=float, default=0.0,
        help="Weight of the frame-level phoneme auxiliary loss (40-way per 4 ms "
             "frame from the events TSVs, all words in the window, head on the "
             "--phoneme-tap block). 0 = off. Mutually exclusive with --phoneme-lambda.",
    )
    p.add_argument(
        "--frame-phoneme-lag-ms", type=float, default=0.0,
        help="Shift frame phoneme labels: frame k is labelled with the phoneme heard "
             "lag ms earlier (MEG response latency; 0 = acoustic alignment).",
    )
    p.add_argument(
        "--speech-lambda", type=float, default=0.0,
        help="Weight of the MEG<->speech contrastive loss (HuBERT targets from "
             "scripts/speech_targets.py in --speech-dir). 0 = off.",
    )
    p.add_argument(
        "--instance-norm", action="store_true",
        help="Per-window per-channel z-score over time before the encoder (session-shift robustness).",
    )
    p.add_argument("--win-tmin", type=float, default=0.0, help="Window start rel. to onset (s). Historic 0.")
    p.add_argument("--win-tmax", type=float, default=1.0, help="Window end rel. to onset (s). Historic 1.")
    p.add_argument("--time-emb", action="store_true", help="Learned time embedding before pooling (long windows).")
    p.add_argument("--backbone", choices=["tcn", "bfm", "omni"], default="tcn",
                   help="bfm = B[FM]²-style MEG SplitUNet encoder (src/splitunet.py)")
    p.add_argument("--bfm-ckpt", type=str, default="", help="flow-matching pretrained SplitUNet (bfm_pretrain.py); empty = random init")
    p.add_argument("--bfm-scale", type=float, default=1.0, help="BFM_SCALE from the bfm_prep.py cache meta.json")
    p.add_argument("--omni-ckpt", type=str, default="",
                   help="BrainOmni weights dir (model_cfg.json + BrainOmni.pt, HF OpenTSLab/BrainOmni base/ or tiny/)")
    # research ideas (HANDOVER 21.18)
    p.add_argument("--subseq-frac", type=float, default=0.0,
                   help="idea 1: share of train items replaced by holdout-shaped vocab-only subsequences")
    p.add_argument("--val-vsub", action="store_true",
                   help="log-only val view 'vsub': every val sentence as a seeded holdout-shaped vocab subsequence")
    p.add_argument("--synthetic-onsets", action="store_true",
                   help="timing audit: replace all MEG by a no-brain signal encoding only word onsets")
    p.add_argument("--poe-priors", type=str, default="",
                   help="idea 2: npz from scripts/poe_priors.py (per-token log-priors added to CE logits in training)")
    p.add_argument("--poe-beta-timing", type=float, default=1.0)
    p.add_argument("--poe-beta-lm", type=float, default=1.0)
    p.add_argument("--neighbor-lambda", type=float, default=0.0, help="idea 3: neighbour-word heads on pre-mixer windows")
    p.add_argument("--neighbor-offsets", type=str, default="-1,1,2,3")
    p.add_argument("--bank-lambda", type=float, default=0.0, help="idea 4: softmax over a fixed bank of frequent words")
    p.add_argument("--bank-size", type=int, default=3000)
    p.add_argument("--accumulate-grad-batches", type=int, default=1,
                   help="16-GB cards (RTX 5080): --batch-size 8 --accumulate-grad-batches 3 = effective batch 24 "
                        "(BatchNorm stats come from encode chunks, not the sentence batch)")
    p.add_argument("--omni-mode", choices=["vq", "continuous"], default="vq",
                   help="vq = frozen tokenizer + RVQ codes (published); continuous = fine-tune the tokenizer encoder, no VQ")
    p.add_argument("--spatial-k", type=int, default=0,
                   help="Geometry-constrained spatial layer 306 -> k virtual channels (spherical harmonics, LISA/Défossez); 0 = off.")
    p.add_argument("--rate-match", action="store_true",
                   help="With --span-dist holdout: rejection-sample spans so the inter-onset intervals after vocab "
                        "words follow the holdout's (label-free, from its onsets); timing never enters the model.")
    p.add_argument("--epoch-mask", action="store_true",
                   help="Zero window samples outside the chunk's continuous epoch [first onset, last onset + 1.35 s] (holdout shape).")
    p.add_argument(
        "--lowpass-hz", type=float, default=0.0,
        help="Brick-wall low-pass of every window before the encoder (0 = off; train and inference).",
    )
    p.add_argument("--speech-dir", type=Path, default=ROOT / "artifacts" / "speech_emb")
    p.add_argument(
        "--no-val-short", action="store_true",
        help="Skip the two log-only val views (5-word chunks, single words) "
             "that mirror the holdout's context shape.",
    )
    p.add_argument("--text-model", type=str, default=T5_LARGE)
    p.add_argument(
        "--model-dim", type=int, default=100,
        help="Backbone width. Historic runs are 100; changing it requires --from-scratch.",
    )
    p.add_argument(
        "--tcn-layers", type=int, default=20,
        help="TCN depth (dilations still cycle 2**(i%%8)).",
    )
    p.add_argument(
        "--pool", type=str, default="avg", choices=("avg", "attn"),
        help="Temporal pooling: avg (historic) or attn (learned-query attention pool).",
    )
    p.add_argument("--pool-queries", type=int, default=4)
    p.add_argument("--mixer-heads", type=int, default=4)
    p.add_argument(
        "--precision", type=str, default="32-true",
        help="Lightning precision. Historic runs are 32-true; bf16-mixed halves "
             "activation VRAM (needed for model_dim>=256 on the 32 GB 5090).",
    )
    p.add_argument(
        "--phoneme-lambda", type=float, default=0.0,
        help="Weight of the phoneme auxiliary losses (bag-of-phonemes + first "
             "phoneme, CMU dict targets, open-vocab). 0 = off (historic).",
    )
    p.add_argument(
        "--phoneme-tap", type=int, default=16,
        help="Trunk block whose time-averaged output feeds the phoneme heads "
             "(intermediate layer per Toshniwal et al. 2017).",
    )
    p.add_argument("--ce-lambda", type=float, default=1.0)
    p.add_argument("--siglip-lambda", type=float, default=1.0)
    p.add_argument("--blend", type=float, default=0.5)
    p.add_argument("--mixer-depth", type=int, default=2)
    p.add_argument("--promote-val", type=float, default=PROMOTE_VAL)
    p.add_argument("--promote-test", type=float, default=PROMOTE_TEST)
    p.add_argument("--h0-min", type=float, default=H0_MIN)
    p.add_argument("--host", type=str, default="")
    p.add_argument(
        "--grad-clip", type=float, default=0.0,
        help="Gradient-norm clipping (Lightning gradient_clip_val). 0 = off (historic). "
             "The 70M from-scratch run diverged at lr 3e-4 without it (HANDOVER §18).",
    )
    p.add_argument(
        "--seed", type=int, default=SEED,
        help="Global seed (Lightning + sampler). Historic runs are 0; use other values "
             "to grow decorrelated ensemble members (HANDOVER §18).",
    )
    p.add_argument(
        "--from-scratch",
        action="store_true",
        help="Do not load the QC TCN. Random init (better for T5+sentence; H0 0.38 does not apply).",
    )
    p.add_argument(
        "--fit-ckpt",
        type=Path,
        default=None,
        help="Resume Trainer.fit from a hybrid last.ckpt (optimizer + epoch).",
    )
    p.add_argument(
        "--weights-ckpt",
        type=Path,
        default=None,
        help="Load hybrid weights only (fresh AdamW, epoch 0). Use when changing batch size.",
    )
    p.add_argument(
        "--encode-chunk",
        type=int,
        default=None,
        help="Windows per TCN forward. Default: 128 on CUDA, 64 otherwise.",
    )
    p.add_argument(
        "--full-epoch",
        action="store_true",
        help="Ignore --limit-train-batches (use after a screen keep).",
    )
    p.add_argument(
        "--save-top-k",
        type=int,
        default=5,
        help="Keep this many best-val checkpoints (not just the peak).",
    )
    p.add_argument(
        "--epoch-ckpt-keep",
        type=int,
        default=8,
        help="Keep this many most-recent epoch=NN.ckpt snapshots (independent of top-k).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    try:
        import transformers  # noqa: F401
    except ImportError:
        raise SystemExit(
            "Need transformers + sentencepiece. Farm: pip install transformers sentencepiece "
            "(do not uv sync). Local: uv add transformers sentencepiece"
        )

    if args.fit_ckpt is not None and args.weights_ckpt is not None:
        raise SystemExit("Pass only one of --fit-ckpt (optimizer resume) or --weights-ckpt (fresh AdamW).")

    sampler_mode = args.sampler
    if sampler_mode == "auto":
        sampler_mode = "book" if args.corpus == "sherlock" else "family"
    val_mode = args.val_corpus
    if val_mode == "auto":
        val_mode = "mixed" if args.corpus != "sherlock" else "sherlock"
    if val_mode == "all":
        val_mode = "mixed"
    timit_val_ses = str(args.timit_val_session)
    drop_path = args.drop_list
    if drop_path is None:
        drop_path = DROP_LIST_ALL if args.corpus != "sherlock" else DROP_LIST

    to_check = [drop_path, MEAN_NPY, STD_NPY]
    if args.fit_ckpt is not None:
        to_check.append(args.fit_ckpt)
    elif args.weights_ckpt is not None:
        to_check.append(args.weights_ckpt)
    elif not args.from_scratch:
        to_check.append(args.ckpt)
    for p in to_check:
        if not p.exists():
            raise FileNotFoundError(p)

    L.seed_everything(args.seed)
    device = pick_device()
    host = args.host or ("sandbox" if device == "cuda" else "local")
    limit = None if args.full_epoch else args.limit_train_batches
    batch_size = args.batch_size if args.batch_size is not None else (16 if device == "cuda" else BATCH_SIZE)
    nw = args.num_workers if args.num_workers is not None else (4 if device == "cuda" else 0)
    if args.encode_chunk is not None:
        encode_chunk = args.encode_chunk
    else:
        encode_chunk = 128 if device == "cuda" else ENCODE_CHUNK
    print("device:", device, flush=True)
    if device == "cuda":
        print("gpu:", torch.cuda.get_device_name(0), flush=True)
        free, total = torch.cuda.mem_get_info()
        print(f"vram_free_gb: {free / 1e9:.1f} / {total / 1e9:.1f}", flush=True)
    print("host:", host, flush=True)
    if args.fit_ckpt is not None:
        init_msg = f"resume {args.fit_ckpt}"
    elif args.weights_ckpt is not None:
        init_msg = f"weights-only {args.weights_ckpt} (fresh AdamW)"
    elif args.from_scratch:
        init_msg = "from-scratch"
    else:
        init_msg = f"QC ckpt {args.ckpt}"
    print("init:", init_msg, flush=True)
    print(
        f"corpus={args.corpus}  val_corpus={val_mode}  sampler={sampler_mode}  "
        f"batch={batch_size} sentences  max_words={args.max_words}  "
        f"encode_chunk={encode_chunk}  num_workers={nw}",
        flush=True,
    )
    print(
        f"drop_list={drop_path.name}  scaler=frozen Sherlock npy "
        f"(not rebuilt — 0.716 weights expect this)",
        flush=True,
    )
    print(
        f"text={args.text_model}  ce_λ={args.ce_lambda}  siglip_λ={args.siglip_lambda}  "
        f"blend={args.blend}  mixer_depth={args.mixer_depth}",
        flush=True,
    )
    print(
        f"model_dim={args.model_dim}  tcn_layers={args.tcn_layers}  pool={args.pool}"
        f"({args.pool_queries}q)  mixer_heads={args.mixer_heads}",
        flush=True,
    )
    lr_new = args.lr_new if args.lr_new is not None else LR_NEW
    if args.lr_backbone is not None:
        lr_backbone = args.lr_backbone
    else:
        lr_backbone = lr_new if args.from_scratch else LR_BACKBONE
    print(
        f"lr_backbone={lr_backbone}  lr_new={lr_new}  "
        f"sampler_alpha={args.sampler_alpha}  channel_dropout={args.channel_dropout}  "
        f"macro_families={args.macro_families or 'all'}  "
        f"span_dist={args.span_dist}"
        + (f"(iso_frac={args.span_iso_frac:.2f})  " if args.span_dist != "off" else "  ")
        + f"jitter_ms={args.jitter_ms}  amp_scale={args.amp_scale}  time_mask_ms={args.time_mask_ms}  band_perturb={args.band_perturb}  frame_ph_λ={args.frame_phoneme_lambda}  speech_λ={args.speech_lambda}  "
        f"broad_subjects={args.broad_subjects or 'off'}"
        + (f"/{args.broad_partition} share={args.broad_share}  " if args.broad_subjects else "  ")
        +
        f"seed={args.seed}  grad_clip={args.grad_clip}  drop_patience={args.drop_patience}  es_patience={args.es_patience}  "
        f"save_top_k={args.save_top_k}  epoch_ckpt_keep={args.epoch_ckpt_keep}  "
        f"limit_train_batches={limit or 'full epoch'}",
        flush=True,
    )

    t0 = time.perf_counter()
    mean = np.load(MEAN_NPY)
    std = np.load(STD_NPY)
    print(f"frozen scaler {MEAN_NPY.name}  shape={mean.shape}", flush=True)
    global TMIN, TMAX
    TMIN, TMAX = args.win_tmin, args.win_tmax
    win_len = int(round((TMAX - TMIN) * SFREQ))
    WIN_KW = dict(win_len=win_len, win_t0=TMIN, epoch_mask=args.epoch_mask)
    jitter = int(round(args.jitter_ms / 1000.0 * SFREQ))
    pad_s = jitter / SFREQ
    print(f"Train split… (jitter {jitter} samples -> windows of {250 + 2 * jitter})", flush=True)
    train_base = make_split("train", mean, std, args.corpus, pad_s=pad_s)
    if args.corpus != "sherlock":
        n_before = len(train_base.samples)
        train_base = SampleView(
            train_base,
            lambda s, ses=timit_val_ses: not (
                corpus_family(str(s[2])) == "timit" and str(s[1]) == ses
            ),
        )
        print(
            f"  held out TIMIT ses-{timit_val_ses} from train  "
            f"({n_before} -> {len(train_base.samples)} windows)",
            flush=True,
        )

    broad_ids = parse_subject_spec(args.broad_subjects)
    if broad_ids:
        if 0 in broad_ids:
            raise SystemExit("--broad-subjects must not contain 0 (that is the deep subject).")
        parts = (
            ("validation", "test") if args.broad_partition == "both"
            else (args.broad_partition,)
        )
        sigma_deep = window_std(train_base)
        prof_deep = window_std_ch(train_base) if args.broad_scale == "profile" else None
        print(
            f"broad: subjects={broad_ids}  partitions={list(parts)}  "
            f"deep standardized window std={sigma_deep:.4f}",
            flush=True,
        )
        broad_bases = []
        for sid in broad_ids:
            for part in parts:
                # channel_means=None -> pnpl self-fits this subject's own stats.
                try:
                    raw = make_split(part, None, None, "sherlock", subjects=[sid], pad_s=pad_s)
                except Exception as e:  # reduced competition upload missing for this subject/session
                    print(f"  sub-{sid} {part}: skipped ({type(e).__name__}: {e})", flush=True)
                    continue
                if not len(raw.samples):
                    print(f"  sub-{sid} {part}: no samples, skipped", flush=True)
                    continue
                raw = clip_to_recorded(raw, TMAX + pad_s)
                if args.broad_scale == "profile":
                    scale = prof_deep / np.maximum(window_std_ch(raw, n=128, seed=sid), 1e-6)
                    sigma_self = float(np.median(1.0 / scale * prof_deep))
                    scale_txt = f"profile median {np.median(scale):.4f}"
                else:
                    sigma_self = window_std(raw, n=128, seed=sid)
                    scale = sigma_deep / sigma_self if sigma_self > 0 else 1.0
                    scale_txt = f"{scale:.4f}"
                broad_bases.append(ScaledBase(raw, scale))
                print(
                    f"  sub-{sid} {part}: {len(raw.samples)} windows  "
                    f"self std={sigma_self:.4f}  scale={scale_txt}",
                    flush=True,
                )
        n_deep = len(train_base.samples)
        train_base = MultiBase([train_base, *broad_bases])
        n_broad = len(train_base.samples) - n_deep
        print(
            f"  train windows deep={n_deep} + broad={n_broad} "
            f"= {len(train_base.samples)}",
            flush=True,
        )
        check = window_std(broad_bases[0], n=128, seed=99)
        print(
            f"  amplitude guard: rescaled broad window std={check:.4f} "
            f"vs deep {sigma_deep:.4f} (must match within ~10%)",
            flush=True,
        )

    print("Val / test…", flush=True)
    sherlock_val = make_split("validation", mean, std, "sherlock")
    sherlock_test = make_split("test", mean, std, "sherlock")
    val_bases = {"sherlock": sherlock_val}
    if val_mode == "mixed":
        for extra in ("mocha", "podcasts"):
            print(f"  loading {extra} validation…", flush=True)
            val_bases[extra] = make_split("validation", mean, std, extra)
        print(
            f"  loading TIMIT ses-{timit_val_ses} as val (pnpl has no TIMIT val split)…",
            flush=True,
        )
        val_bases["timit"] = SampleView(
            make_split("train", mean, std, "timit"),
            lambda s, ses=timit_val_ses: str(s[1]) == ses,
        )
    val_base = sherlock_val
    test_base = sherlock_test

    unique_bases = [train_base, *val_bases.values(), test_base]
    unique = sorted({
        normalize_word(s[5])
        for base in unique_bases
        for s in base.samples
        if str(s[5]).strip()
    })
    unique = sorted(set(unique) | {normalize_word(w) for w in VOCAB})
    print(
        f"  unique word types={len(unique)}  "
        f"({time.perf_counter() - t0:.1f}s)",
        flush=True,
    )
    t5_device = device if device != "mps" else "cpu"
    word_emb = embed_unique_words(
        unique, EMB_DIR, model_name=args.text_model, layer=T5_LAYER, device=t5_device,
    )
    gallery = gallery_matrix(word_emb, VOCAB)
    print("gallery:", tuple(gallery.shape), flush=True)

    drop_keys = load_drop_keys(drop_path)
    print(f"  drop keys={len(drop_keys)} from {drop_path.name}", flush=True)
    frame_labeler = None
    if args.frame_phoneme_lambda > 0:
        from phonemes import FrameLabeler
        frame_labeler = FrameLabeler(DATA_PATH)
        probe_runs = {(str(x[0]), str(x[1]), str(x[2]), str(x[3])) for x in train_base.samples[::5000]}
        n_ok = sum(frame_labeler._load(*k) is not None for k in probe_runs)
        print(f"frame phoneme aux: λ={args.frame_phoneme_lambda} tap=block{args.phoneme_tap} lag={args.frame_phoneme_lag_ms:.0f}ms  "
              f"events TSV with phonemes found for {n_ok}/{len(probe_runs)} sampled runs", flush=True)
    speech_store = None
    if args.speech_lambda > 0:
        from phonemes import SpeechStore
        speech_store = SpeechStore(args.speech_dir)
        print(f"speech contrastive: λ={args.speech_lambda}  targets for {speech_store.n_runs()} runs in {args.speech_dir}", flush=True)
        if speech_store.n_runs() == 0:
            raise SystemExit("--speech-lambda > 0 but no speech targets found; run scripts/speech_targets.py first")
    span_lengths = None
    if args.span_dist == "holdout":
        span_lengths = [n for n, c in HOLDOUT_SENT_LEN_HIST.items() for _ in range(c)]
    rate_match = None
    if args.rate_match:
        if not span_lengths:
            raise SystemExit("--rate-match needs --span-dist holdout")
        rate_match = rate_match_weights(train_base, VOCAB_TO_ID)
    subseq_kw = {}
    if args.subseq_frac > 0 or args.val_vsub:
        gaps = holdout_onset_gaps()
        subseq_kw = dict(subseq_gaps=gaps, subseq_lengths=[n for n, c in HOLDOUT_SENT_LEN_HIST.items() for _ in range(c)])
        print(f"  holdout-shaped subsequences: gap median {np.median(gaps):.3f}s  train share {args.subseq_frac:.2f}  "
              f"val view vsub={args.val_vsub}", flush=True)
    prior_table = None
    if args.poe_priors:
        prior_table = load_poe_table(args.poe_priors, args.poe_beta_timing, args.poe_beta_lm)
        print(f"  PoE priors: {len(prior_table)} tokens from {args.poe_priors}  beta_timing={args.poe_beta_timing} "
              f"beta_lm={args.poe_beta_lm}", flush=True)
    if args.synthetic_onsets:
        print("  TIMING AUDIT: all MEG replaced by synthetic onset-only signals", flush=True)
    train_ds = SentenceWordDataset(
        train_base, word_emb, VOCAB_TO_ID, drop_keys=drop_keys, max_words=args.max_words,
        subseq_frac=args.subseq_frac, synthetic=args.synthetic_onsets, prior_table=prior_table,
        **(subseq_kw if args.subseq_frac > 0 else {}),
        rate_match=rate_match,
        span_lengths=span_lengths,
        span_iso_frac=args.span_iso_frac if span_lengths else 0.0,
        jitter=jitter,
        frame_labeler=frame_labeler,
        win_len=win_len,
        win_t0=TMIN - pad_s,
        epoch_mask=args.epoch_mask,
        frame_lag_s=args.frame_phoneme_lag_ms / 1000.0,
        speech_store=speech_store,
    )
    if span_lengths:
        exp_len = (1 - args.span_iso_frac) * (
            sum(n * c for n, c in HOLDOUT_SENT_LEN_HIST.items()) / sum(HOLDOUT_SENT_LEN_HIST.values())
        ) + args.span_iso_frac
        print(
            f"  span sampling: holdout hist {HOLDOUT_SENT_LEN_HIST}  iso_frac={args.span_iso_frac:.2f}  "
            f"expected words/item={exp_len:.2f} (historic ~12 -> ~{exp_len / 12:.0%} word exposures per epoch)",
            flush=True,
        )
    val_parts = []
    for fam, base in val_bases.items():
        part = SentenceWordDataset(
            base, word_emb, VOCAB_TO_ID, drop_keys=None, max_words=args.max_words,
            synthetic=args.synthetic_onsets, **WIN_KW,
        )
        print(f"  val {fam}: {len(part)} sentences", flush=True)
        val_parts.append(part)
    val_ds = val_parts[0] if len(val_parts) == 1 else ConcatDataset(val_parts)
    # Log-only val views with the holdout's context shape (HANDOVER §17):
    # the same val words re-chunked into <=5-word groups and into single words.
    val_views = {}
    if not args.no_val_short:
        for view, mw in (("short5", 5), ("iso1", 1)):
            parts = [
                SentenceWordDataset(base, word_emb, VOCAB_TO_ID, drop_keys=None, max_words=mw,
                                    synthetic=args.synthetic_onsets, **WIN_KW)
                for base in val_bases.values()
            ]
            val_views[view] = parts[0] if len(parts) == 1 else ConcatDataset(parts)
            print(f"  val view {view}: {len(val_views[view])} items", flush=True)
    if args.val_vsub:
        parts = [SentenceWordDataset(base, word_emb, VOCAB_TO_ID, drop_keys=None, max_words=args.max_words,
                                     subseq_frac=1.0, subseq_det=True, synthetic=args.synthetic_onsets,
                                     **subseq_kw, **WIN_KW) for base in val_bases.values()]
        val_views["vsub"] = parts[0] if len(parts) == 1 else ConcatDataset(parts)
        print(f"  val view vsub: {len(val_views['vsub'])} items", flush=True)
    test_ds = SentenceWordDataset(
        test_base, word_emb, VOCAB_TO_ID, drop_keys=None, max_words=args.max_words, **WIN_KW,
    )
    n_train_words = sum(len(s) for s in train_ds.sentences)
    n_invocab = sum(
        1
        for s in train_ds.sentences
        for _i, w, _k in s
        if w in VOCAB_TO_ID
    )
    print(
        f"  sentences train/val/test={len(train_ds)}/{len(val_ds)}/{len(test_ds)}  "
        f"train words={n_train_words} (in-vocab {n_invocab})",
        flush=True,
    )
    if val_mode == "mixed":
        print(
            "  val_bacc10 = mean of per-family BAcc@10 (Sherlock/TIMIT/MOCHA/Moth), "
            "not Sherlock n-weighted",
            flush=True,
        )

    val_iso = CompetitionVocabDataset(sherlock_val)
    test_iso = CompetitionVocabDataset(sherlock_test)

    counts = torch.zeros(len(VOCAB))
    for sent in train_ds.sentences:
        for _i, w, _k in sent:
            vid = VOCAB_TO_ID.get(w)
            if vid is not None:
                counts[vid] += 1
    class_weight = counts.clamp(min=1).pow(-0.5)
    class_weight = class_weight / class_weight.mean()
    print(
        f"  in-vocab counts min/max {int(counts.min())}/{int(counts.max())}",
        flush=True,
    )

    train_samples = train_ds.base.samples
    books = [str(train_samples[sent[0][0]][2]) for sent in train_ds.sentences]
    subs = [str(train_samples[sent[0][0]][0]) for sent in train_ds.sentences]
    if sampler_mode == "family":
        # Broad subjects are task=Sherlock1, so corpus_family() would fold them
        # into the sherlock family and hide them from the share accounting.
        group_keys = [
            "broad" if s != "0" else corpus_family(b) for s, b in zip(subs, books)
        ]
    else:
        # Book sampler: broad subjects get their own group too, so --broad-share
        # applies in phase-1 (Sherlock-only) runs as well (HANDOVER §16 next step).
        group_keys = ["broad" if s != "0" else b for s, b in zip(subs, books)]
    n_group = Counter(group_keys)
    # weight = n**(alpha-1): alpha=0 → equal mass per group, alpha=1 → natural.
    alpha = args.sampler_alpha
    weight = {g: float(n) ** (alpha - 1.0) for g, n in n_group.items()}
    if "broad" in n_group:
        share = args.broad_share
        if share < 0:
            share = n_group["broad"] / len(group_keys)  # natural frequency
        if not 0.0 <= share < 1.0:
            raise SystemExit(f"--broad-share must be in [0, 1), got {share}")
        # Give broad exactly `share` of the epoch mass; the real families split
        # the remainder by the alpha rule they would have used on their own.
        rest = sum(weight[g] * n_group[g] for g in n_group if g != "broad")
        for g in n_group:
            if g != "broad":
                weight[g] *= (1.0 - share) / rest if rest else 0.0
        weight["broad"] = share / n_group["broad"]
    sample_w = torch.tensor([weight[k] for k in group_keys], dtype=torch.double)
    gen = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        sample_w, num_samples=len(sample_w), replacement=True, generator=gen,
    )
    print(f"{sampler_mode} sentences:", dict(sorted(n_group.items())), flush=True)
    group_mass = {g: weight[g] * n_group[g] for g in n_group}
    total_mass = sum(group_mass.values())
    print(
        f"epoch share (alpha={alpha}"
        + (f", broad_share={args.broad_share}" if "broad" in n_group else "")
        + "):",
        {g: f"{m / total_mass:.1%}" for g, m in sorted(group_mass.items())},
        flush=True,
    )
    if sampler_mode == "family":
        print("book sentences:", dict(sorted(Counter(books).items())), flush=True)

    pin = device == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler, shuffle=False,
        drop_last=True, num_workers=nw, pin_memory=pin,
        persistent_workers=nw > 0, collate_fn=collate_sentences,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=pin, persistent_workers=nw > 0,
        collate_fn=collate_sentences,
    )
    val_loaders = [val_loader] + [
        DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=nw, pin_memory=pin, persistent_workers=nw > 0,
            collate_fn=collate_sentences,
        )
        for ds in val_views.values()
    ]
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=pin, persistent_workers=nw > 0,
        collate_fn=collate_sentences,
    )

    phoneme_map = None
    if args.phoneme_lambda > 0:
        from phonemes import phoneme_table
        phoneme_map = phoneme_table(word_emb.keys())
        print(
            f"phoneme aux: λ={args.phoneme_lambda} tap=block{args.phoneme_tap}  "
            f"cmudict coverage {len(phoneme_map)}/{len(word_emb)} words "
            f"({len(phoneme_map)/max(len(word_emb),1):.0%})",
            flush=True,
        )

    bank, bank_words = None, None
    if args.bank_lambda > 0:
        cnt = Counter(w for sent in train_ds.sentences for _i, w, _k in sent)
        bank_words = [w for w, _c in cnt.most_common(args.bank_size)]
        bank = torch.stack([word_emb[w] for w in bank_words])
        cover = sum(cnt[w] for w in bank_words) / max(sum(cnt.values()), 1)
        print(f"  word bank: {len(bank_words)} words cover {cover:.1%} of training tokens", flush=True)
    print("Building hybrid…", flush=True)
    if args.from_scratch:
        print(f"from-scratch: one AdamW lr={lr_new} on all params", flush=True)
    model = LitHybridTCN(
        n_channels=306,
        n_classes=len(VOCAB),
        gallery=gallery,
        class_weight=class_weight,
        lr_backbone=lr_backbone,
        lr_new=lr_new,
        weight_decay=WEIGHT_DECAY,
        dropout=DROPOUT,
        label_smoothing=LABEL_SMOOTHING,
        model_dim=args.model_dim,
        tcn_layers=args.tcn_layers,
        pool=args.pool,
        pool_queries=args.pool_queries,
        mixer_heads=args.mixer_heads,
        mixer_depth=args.mixer_depth,
        ce_lambda=args.ce_lambda,
        siglip_lambda=args.siglip_lambda,
        blend=args.blend,
        encode_chunk=encode_chunk,
        sched_patience=SCHED_PATIENCE,
        channel_dropout=args.channel_dropout,
        macro_families=args.macro_families,
        phoneme_lambda=args.phoneme_lambda,
        phoneme_tap=args.phoneme_tap,
        phoneme_map=phoneme_map,
        emb_dim=gallery.shape[1],
        amp_scale=args.amp_scale,
        time_mask_ms=args.time_mask_ms,
        band_perturb=args.band_perturb,
        frame_phoneme_lambda=args.frame_phoneme_lambda,
        speech_lambda=args.speech_lambda,
        val_views=tuple(val_views.keys()),
        instance_norm=args.instance_norm,
        lowpass_hz=args.lowpass_hz,
        win_tmin=args.win_tmin,
        win_tmax=args.win_tmax,
        time_emb=args.time_emb,
        epoch_mask=args.epoch_mask,
        spatial_k=args.spatial_k,
        backbone=args.backbone,
        bfm_ckpt=args.bfm_ckpt,
        bfm_scale=args.bfm_scale,
        omni_ckpt=args.omni_ckpt,
        omni_mode=args.omni_mode,
        neighbor_lambda=args.neighbor_lambda,
        neighbor_offsets=args.neighbor_offsets,
        bank_lambda=args.bank_lambda,
        bank=bank,
        bank_words=bank_words,
    )
    hybrid_src = args.fit_ckpt or args.weights_ckpt
    if hybrid_src is not None:
        ck = torch.load(hybrid_src, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
        kind = "resume" if args.fit_ckpt is not None else "weights-only"
        print(
            f"loaded hybrid {kind} {hybrid_src.name}: "
            f"{len(missing)} missing, {len(unexpected)} unexpected",
            flush=True,
        )
    elif args.from_scratch:
        print("random init (no QC backbone)", flush=True)
    else:
        model.load_qc_backbone(str(args.ckpt))
    model.ce.weight = class_weight.detach().clone()
    print(f"mixer_gate={model.mixer.gate.detach().item():.4f} (expect 0)", flush=True)

    print("H0-init: isolated CE (mixer bypassed)…", flush=True)
    init_b10, init_b1 = eval_isolated_ce(model.backbone, val_iso, device, "H0-init val CE")
    skip_h0 = args.fit_ckpt is not None or args.weights_ckpt is not None or args.from_scratch
    if not skip_h0 and init_b10 < args.h0_min:
        print("---", flush=True)
        print(f"h0_val_bacc10:    {init_b10:.6f}", flush=True)
        print(f"val_bacc10:       {init_b10:.6f}", flush=True)
        print("test_bacc10:      0.000000", flush=True)
        print("status:           abort", flush=True)
        print(f"host:             {host}", flush=True)
        raise SystemExit(
            f"H0-init fail: isolated CE val={init_b10:.3f} < {args.h0_min}. "
            "Backbone did not load — aborting."
        )
    if args.from_scratch:
        print(
            f"from-scratch H0 isolated CE={init_b10:.3f} (expect ~0.20; not an abort gate)",
            flush=True,
        )

    start_tensorboard(LOG_DIR, args.tb_port, bind_all=args.tb_bind_all)
    tb_logger = TensorBoardLogger(save_dir=str(LOG_DIR), name=args.log_name)
    csv_logger = CSVLogger(save_dir=str(LOG_DIR), name=args.log_name, version=tb_logger.version)
    ckpt_dir = Path(tb_logger.log_dir) / "checkpoints"
    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="best-{epoch:02d}-{val_bacc10:.3f}",
        monitor="val_bacc10",
        mode="max",
        save_top_k=max(1, args.save_top_k),
        save_last=True,
        every_n_epochs=1,
    )
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        precision=args.precision,
        logger=[tb_logger, csv_logger],
        callbacks=[
            Timer(),
            ckpt_cb,
            KeepEpochCkpt(ckpt_dir, keep=args.epoch_ckpt_keep),
            ValDropStop(patience=args.drop_patience),
            EarlyStopping(monitor="val_bacc10", mode="max", patience=args.es_patience),
        ],
        enable_checkpointing=True,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=5,
        num_sanity_val_steps=0,
        limit_train_batches=limit or 1.0,
        gradient_clip_val=args.grad_clip if args.grad_clip > 0 else None,
    )

    t1 = time.perf_counter()
    model.train()
    fit_kw = {}
    if args.fit_ckpt is not None:
        fit_kw["ckpt_path"] = str(args.fit_ckpt)
        print(f"Trainer.fit resume {args.fit_ckpt}", flush=True)
    trainer.fit(model, train_loader, val_loaders, **fit_kw)
    print(f"\nfit wall: {time.perf_counter() - t1:.1f}s", flush=True)
    print("best ckpt:", ckpt_cb.best_model_path, flush=True)
    print(
        "best val :",
        None if ckpt_cb.best_model_score is None else float(ckpt_cb.best_model_score),
        flush=True,
    )

    best = LitHybridTCN.load_from_checkpoint(
        ckpt_cb.best_model_path,
        gallery=gallery,
        class_weight=class_weight,
        map_location=device,
    )
    best.ce.weight = class_weight.detach().clone()
    iso_val = eval_isolated_ce(best.backbone, val_iso, device, "isolated CE val")
    iso_test = eval_isolated_ce(best.backbone, test_iso, device, "isolated CE test")
    # Mixed val_loader is n-weighted (~80% Sherlock). The rail that matches
    # historical 0.716 numbers is Sherlock ses-11 only.
    sherlock_val_loader = DataLoader(
        val_parts[0], batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=pin, persistent_workers=False,
        collate_fn=collate_sentences,
    )
    val_label = (
        "sentence val  (ses-11 rail)" if val_mode == "mixed"
        else "sentence val  (ses-11)"
    )
    val_m = eval_sentence(best, sherlock_val_loader, device, val_label)
    test_m = eval_sentence(best, test_loader, device, "sentence test (ses-12)")

    chosen_val = max(val_m["ce_bacc10"], val_m["blend_bacc10"])
    chosen_test = (
        test_m["blend_bacc10"] if val_m["blend_bacc10"] >= val_m["ce_bacc10"]
        else test_m["ce_bacc10"]
    )
    scoring = "blend" if val_m["blend_bacc10"] >= val_m["ce_bacc10"] else "ce"
    promote = chosen_val > args.promote_val and chosen_test >= args.promote_test
    skip_promote = (
        bool(args.full_epoch)
        or val_mode == "mixed"
        or args.weights_ckpt is not None
        or args.fit_ckpt is not None
    )
    ret_alive = val_m["ret_bacc10"] >= 0.25
    status = "keep" if (promote or skip_promote) else "abort"
    print("QC baseline (version_2 ep5): val 0.443 / test 0.460 isolated CE", flush=True)
    print(f"H0-init isolated CE: {init_b10:.3f}  (need >={args.h0_min})", flush=True)
    print(f"scoring rule: {scoring}  ret_alive={ret_alive} (ret val>=0.25)", flush=True)
    print(f"mixer_gate={best.mixer.gate.detach().item():.4f}", flush=True)
    print("---", flush=True)
    print(f"val_bacc10:       {chosen_val:.6f}", flush=True)
    print(f"test_bacc10:      {chosen_test:.6f}", flush=True)
    print(f"h0_val_bacc10:    {init_b10:.6f}", flush=True)
    print(f"val_bacc10_ce:    {val_m['ce_bacc10']:.6f}", flush=True)
    print(f"val_bacc10_ret:   {val_m['ret_bacc10']:.6f}", flush=True)
    print(f"val_bacc10_blend: {val_m['blend_bacc10']:.6f}", flush=True)
    print(f"test_bacc10_ce:   {test_m['ce_bacc10']:.6f}", flush=True)
    print(f"test_bacc10_ret:  {test_m['ret_bacc10']:.6f}", flush=True)
    print(f"test_bacc10_blend:{test_m['blend_bacc10']:.6f}", flush=True)
    print(f"iso_val_bacc10:   {iso_val[0]:.6f}", flush=True)
    print(f"iso_test_bacc10:  {iso_test[0]:.6f}", flush=True)
    print(f"training_seconds: {time.perf_counter() - t1:.1f}", flush=True)
    print(f"num_params_M:     {sum(p.numel() for p in best.parameters()) / 1e6:.2f}", flush=True)
    print(f"status:           {status}", flush=True)
    print(f"host:             {host}", flush=True)
    print(f"scoring:          {scoring}", flush=True)
    if not promote and not skip_promote:
        raise SystemExit(
            f"screen abort: chosen val={chosen_val:.3f} test={chosen_test:.3f} "
            f"(need val>{args.promote_val:.3f} and test>={args.promote_test:.3f})"
        )


if __name__ == "__main__":
    main()
