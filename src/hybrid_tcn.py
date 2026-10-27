"""Hybrid TCN: 50-way CE + open-vocab T5 retrieval + residual sentence mixer.

The QC CNN+TCN encoder is unchanged. New pieces (zero-init residual mixer,
retrieval MLP, SigLIP) sit on pooled features so H0-init of the isolated CE
head still matches the 0.443 Sherlock val of the source checkpoint.
"""

from __future__ import annotations

import math
from collections import defaultdict

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from cnn_tcn import CNN_TCN, balanced_topk_accuracy
from data_prep import corpus_family, sample_trial_key
from t5_text import normalize_word


class ResidualSentenceMixer(nn.Module):
    """Word-level transformer with a zero-init residual gate (identity at H0)."""

    def __init__(self, dim: int, depth: int = 2, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"model_dim {dim} is not divisible by heads {heads}")
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=4 * dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        pe = torch.zeros(256, dim)
        pos = torch.arange(256).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=depth, enable_nested_tensor=False,
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
        # keep_mask True = real word; Transformer wants True = pad
        y = x + self.pe[: x.size(1)]
        y = self.encoder(y, src_key_padding_mask=~keep_mask)
        return x + self.gate * y


class SigLipLoss(nn.Module):
    """SigLIP (Zhai et al. 2023), matching neuraltrain / notebook 05 defaults."""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(10.0).log())
        self.bias = nn.Parameter(torch.tensor(-10.0))

    def forward(self, estimate: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        est = F.normalize(estimate, dim=1)
        cand = F.normalize(candidate, dim=1)
        scores = self.temperature.exp() * (est @ cand.T) + self.bias
        sim = cand @ cand.T
        n = scores.shape[0]
        targets = (sim >= 0.999).float()[:n]
        weights = 1.0 - (sim >= 0.999).float()
        weights = weights[:n] + torch.eye(n, device=scores.device)
        loss = F.binary_cross_entropy_with_logits(scores, targets, weights, reduction="sum")
        return loss / n


class RetrievalHead(nn.Module):
    def __init__(self, model_dim: int, emb_dim: int = 1024, dropout: float = 0.3):
        super().__init__()
        hidden = max(model_dim * 4, 256)
        self.net = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, hidden),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(h), dim=-1)


RATE_BIN = 0.05     # s, inter-onset-interval bins for --rate-match
SUBSEQ_MIN = 3       # --subseq: shortest accepted holdout-shaped item
SUBSEQ_MAX_GAP = 0.8  # s, longest onset gap allowed between consecutive picks
SYN_KERNEL_S = 0.3    # --synthetic-onsets: half-sine "response" per word onset
EPOCH_TAIL_S = 1.35  # holdout epoch end - last onset (last word offset + 1.0 s): median 1.36 s, q10 1.18, q90 1.58


class SentenceWordDataset(Dataset):
    """One item = one sentence (open-vocab words ordered by wordidx).

    Two train-time augmentations, both off by default (historic behaviour):

    * **Span sampling** (`span_lengths`, `span_iso_frac`). Each item is an
      anchor chunk of a training sentence; at fetch time a *random span* of the
      parent sentence overlapping that anchor is returned instead, with its
      length drawn from `span_lengths` (e.g. the competition holdout's 156
      sentence lengths) or, with probability `span_iso_frac`, a single word.
      HANDOVER §17: the 0.646 model is a sentence decoder trained on ~12-word
      sentences; the holdout is median-5-word sentences plus 92 isolated
      words, on which it scores chance. This makes the mixer see the
      holdout's context shape every step.
    * **Temporal jitter** (`jitter` samples). The base must then serve windows
      of `win_len + 2*jitter` samples (tmin/tmax widened by the caller); a
      random `win_len` crop is taken per word per fetch. `jitter=0` keeps the
      historic pad/truncate-to-250 path.
    """

    def __init__(
        self,
        base,
        word_emb: dict[str, torch.Tensor],
        vocab_to_id: dict[str, int],
        drop_keys: set[tuple] | None = None,
        max_words: int = 32,
        span_lengths=None,
        span_iso_frac: float = 0.0,
        jitter: int = 0,
        win_len: int = 250,
        frame_labeler=None,
        win_t0: float = 0.0,
        frame_lag_s: float = 0.0,
        speech_store=None,
        epoch_mask: bool = False,
        rate_match=None,
        subseq_frac: float = 0.0,
        subseq_gaps=None,
        subseq_lengths=None,
        subseq_det: bool = False,
        synthetic: bool = False,
        prior_table: dict | None = None,
    ):
        self.base = base
        # Holdout-shaped items (research idea 1): with probability subseq_frac an item is a
        # vocab-only subsequence of its sentence whose consecutive onsets follow the holdout's
        # gap distribution (subseq_gaps, s) and whose length follows subseq_lengths. Timing only
        # selects training items, it is never a model input. subseq_det = seeded per item (val view).
        self.subseq_frac = float(subseq_frac)
        self.subseq_gaps = None if subseq_gaps is None else [float(g) for g in subseq_gaps]
        self.subseq_lengths = None if subseq_lengths is None else [int(n) for n in subseq_lengths if int(n) >= SUBSEQ_MIN]
        self.subseq_det = bool(subseq_det)
        if self.subseq_frac > 0 and (not self.subseq_gaps or not self.subseq_lengths):
            raise ValueError("subseq_frac needs subseq_gaps and subseq_lengths")
        # Timing audit (SimpleB2T control): replace every window by a no-brain signal that
        # only encodes the word onsets of its sentence (fixed spatial pattern x half-sine bumps).
        self.synthetic = bool(synthetic)
        if self.synthetic:
            g = torch.Generator().manual_seed(0)
            self._syn_pattern = torch.randn(306, generator=g)
        # Product-of-experts priors (research idea 2): sample key -> (50,) log-prior added to
        # the CE logits in training only.
        self.prior_table = prior_table
        # Speech-rate matching (train only): array of per-bin acceptance weights over
        # the inter-onset interval after each vocab word (bins of RATE_BIN s), so that
        # sampled spans follow the holdout's slower careful-phrase timing. Timing is
        # used only to choose training spans, never as a model input.
        self.rate_match = None if rate_match is None else torch.as_tensor(rate_match, dtype=torch.float32)
        # Holdout sentence epochs run from the first onset to ~last onset + 1.35 s
        # (last word offset + 1.0 s): zero every window sample outside that span.
        self.epoch_mask = bool(epoch_mask)
        # Frame-level phoneme labels (train only). win_t0 = start of the loaded
        # window relative to word onset (−pad when jitter widens the window).
        self.frame_labeler = frame_labeler
        self.win_t0 = float(win_t0)
        # Neural response lags the acoustics by ~100-300 ms: label frame k with
        # the phoneme heard at t_k - lag so the head predicts what the brain is
        # processing *now*, not what the ear hears now.
        self.frame_lag_s = float(frame_lag_s)
        # Speech-embedding targets (HuBERT of the audio heard in the window).
        self.speech_store = speech_store
        self.word_emb = word_emb
        self.max_words = max_words
        self.span_lengths = (
            torch.tensor(list(span_lengths), dtype=torch.long)
            if span_lengths is not None and len(span_lengths) > 0 else None
        )
        self.span_iso_frac = float(span_iso_frac)
        self.jitter = int(jitter)
        self.win_len = int(win_len)
        if self.span_lengths is None and self.span_iso_frac > 0:
            raise ValueError("span_iso_frac needs span_lengths")
        grouped: dict[tuple, list] = defaultdict(list)
        for i, sample in enumerate(base.samples):
            word = normalize_word(sample[5])
            if word not in word_emb:
                continue
            if drop_keys is not None and sample_trial_key(sample) in drop_keys:
                continue
            if len(sample) > 6:
                sent_uid = (sample[0], sample[1], sample[2], sample[3], int(sample[6]))
                word_idx = int(sample[7]) if len(sample) > 7 else i
            else:
                sent_uid = ("iso", i)
                word_idx = 0
            grouped[sent_uid].append((i, word, word_idx))

        sentences, parents, anchors = [], [], []
        for items in grouped.values():
            items.sort(key=lambda t: t[2])
            for start in range(0, len(items), max_words):
                sentences.append(items[start : start + max_words])
                parents.append(items)
                anchors.append((start, min(start + max_words, len(items))))
        self.sentences = sentences
        # Full parent sentence + anchor bounds per item, for span sampling.
        self._parents = parents
        self._anchors = anchors
        self.vocab_to_id = vocab_to_id

    def __len__(self):
        return len(self.sentences)

    def _span_weight(self, parent, start: int, length: int) -> float:
        """Mean acceptance weight of the vocab words in parent[start:start+length]
        (IOI to the next word of the parent sentence; last word of the parent: 1)."""
        ws = []
        for j in range(start, start + length):
            if parent[j][1] not in self.vocab_to_id or j + 1 >= len(parent):
                continue
            ioi = float(self.base.samples[parent[j + 1][0]][4]) - float(self.base.samples[parent[j][0]][4])
            b = min(max(int(ioi / RATE_BIN), 0), len(self.rate_match) - 1)
            ws.append(float(self.rate_match[b]))
        return sum(ws) / len(ws) if ws else 1.0

    def _sample_span(self, idx: int) -> list:
        """Random span of the parent sentence that overlaps this item's anchor
        (with rate matching: rejection-sampled by holdout-likeness of its timing)."""
        if self.rate_match is not None:
            wmax = float(self.rate_match.max())
            for _ in range(8):
                span = self._sample_span_once(idx)
                parent = self._parents[idx]
                start = next(i for i, it in enumerate(parent) if it is span[0])
                if float(torch.rand(())) * wmax < self._span_weight(parent, start, len(span)):
                    return span
            return span
        return self._sample_span_once(idx)

    def _sample_span_once(self, idx: int) -> list:
        parent, (a0, a1) = self._parents[idx], self._anchors[idx]
        n = len(parent)
        if self.span_iso_frac > 0 and float(torch.rand(())) < self.span_iso_frac:
            length = 1
        else:
            length = int(self.span_lengths[int(torch.randint(len(self.span_lengths), ()))])
        length = max(1, min(length, n))
        lo = max(0, a0 - length + 1)
        hi = min(a1 - 1, n - length)
        start = lo if hi <= lo else lo + int(torch.randint(hi - lo + 1, ()))
        return parent[start : start + length]

    def _onset(self, item) -> float:
        return float(self.base.samples[item[0]][4])

    def _vocab_subseq(self, idx: int, rng=None):
        """Vocab-only subsequence of the parent sentence, consecutive picks spaced like the
        holdout (closest onset gap to a gap drawn from subseq_gaps); None if too short."""
        draw = rng.random if rng is not None else (lambda: float(torch.rand(())))
        cand = [it for it in self._parents[idx] if it[1] in self.vocab_to_id]
        if len(cand) < SUBSEQ_MIN:
            return None
        ons = [self._onset(it) for it in cand]
        length = self.subseq_lengths[int(draw() * len(self.subseq_lengths))]
        cur = int(draw() * (len(cand) - SUBSEQ_MIN + 1))
        seq = [cur]
        while len(seq) < length:
            nxt = [j for j in range(cur + 1, len(cand)) if ons[j] - ons[cur] <= SUBSEQ_MAX_GAP]
            if not nxt:
                break
            g = self.subseq_gaps[int(draw() * len(self.subseq_gaps))]
            cur = min(nxt, key=lambda j: abs(ons[j] - ons[cur] - g))
            seq.append(cur)
        return [cand[j] for j in seq] if len(seq) >= SUBSEQ_MIN else None

    def _synthetic_window(self, onsets_rel: torch.Tensor) -> torch.Tensor:
        """(306, win_len) no-brain window: sum of half-sine bumps at the sentence's word onsets
        (relative to this word's onset) on a fixed spatial pattern, plus white noise."""
        t = self.win_t0 + torch.arange(self.win_len) / 250.0
        u = t[None, :] - onsets_rel[:, None]
        bump = torch.where((u >= 0) & (u < SYN_KERNEL_S), torch.sin(math.pi * u / SYN_KERNEL_S), torch.zeros(()))
        return self._syn_pattern[:, None] * bump.sum(0)[None, :] + 0.1 * torch.randn(306, self.win_len)

    @staticmethod
    def prior_key(sample) -> str:
        return f"{sample[1]}|{sample[2]}|{sample[3]}|{round(float(sample[4]) * 1000)}"

    def _window(self, base_idx: int):
        meg = self.base[base_idx][0].float()
        want = self.win_len + 2 * self.jitter
        if meg.shape[-1] < want:
            meg = F.pad(meg, (0, want - meg.shape[-1]))
        elif meg.shape[-1] > want:
            meg = meg[:, :want]
        off = 0
        if self.jitter > 0:
            off = int(torch.randint(2 * self.jitter + 1, ()))
            meg = meg[:, off : off + self.win_len]
        return meg, off

    def __getitem__(self, idx):
        items = None
        if self.subseq_frac > 0:
            rng = np.random.default_rng(idx) if self.subseq_det else None
            if (rng.random() if rng is not None else float(torch.rand(()))) < self.subseq_frac:
                items = self._vocab_subseq(idx, rng)
        if items is None:
            items = self._sample_span(idx) if self.span_lengths is not None else self.sentences[idx]
        megs, embs, vids, frames, sembs, smask, priors = [], [], [], [], [], [], []
        if self.synthetic:
            par_on = torch.tensor([self._onset(it) for it in self._parents[idx]])
        if self.epoch_mask:
            ons = [float(self.base.samples[b][4]) for b, _w, _k in items]
            t_lo, t_hi = min(ons), max(ons) + EPOCH_TAIL_S
        for base_idx, word, _ in items:
            if self.synthetic:
                meg, off = self._synthetic_window(par_on - float(self.base.samples[base_idx][4])), 0
            else:
                meg, off = self._window(base_idx)
            if self.epoch_mask:
                t = float(self.base.samples[base_idx][4]) + self.win_t0 + (off + torch.arange(meg.shape[-1])) / 250.0
                meg = meg.masked_fill(((t < t_lo) | (t >= t_hi))[None, :], 0.0)
            megs.append(meg)
            embs.append(self.word_emb[word])
            vids.append(self.vocab_to_id.get(word, -1))
            smp = self.base.samples[base_idx]
            if self.frame_labeler is not None:
                t_start = float(smp[4]) + self.win_t0 + off / 250.0 - self.frame_lag_s
                frames.append(self.frame_labeler.labels(smp, t_start, self.win_len))
            if self.speech_store is not None:
                e, ok = self.speech_store.get(smp)
                sembs.append(e); smask.append(ok)
            if self.prior_table is not None:
                lp = self.prior_table.get(self.prior_key(smp))
                priors.append(torch.zeros(len(self.vocab_to_id)) if lp is None else torch.as_tensor(lp, dtype=torch.float32))
        return (
            torch.stack(megs),
            torch.stack(embs),
            torch.tensor(vids, dtype=torch.long),
            torch.tensor(len(megs), dtype=torch.long),
            str(self.base.samples[items[0][0]][2]),
            [word for _i, word, _w in items],
            torch.stack(frames) if frames else None,
            (torch.stack(sembs), torch.tensor(smask, dtype=torch.bool)) if sembs else None,
            torch.stack(priors) if priors else None,
        )


def collate_sentences(batch):
    n_ch = batch[0][0].shape[1]
    t = batch[0][0].shape[2]
    d = batch[0][1].shape[1]
    max_len = max(int(x[3]) for x in batch)
    bsz = len(batch)
    meg = torch.zeros(bsz, max_len, n_ch, t)
    emb = torch.zeros(bsz, max_len, d)
    vid = torch.full((bsz, max_len), -1, dtype=torch.long)
    mask = torch.zeros(bsz, max_len, dtype=torch.bool)
    books, words = [], []
    has_frames = len(batch[0]) > 6 and batch[0][6] is not None
    has_speech = len(batch[0]) > 7 and batch[0][7] is not None
    has_prior = len(batch[0]) > 8 and batch[0][8] is not None
    frames = torch.full((bsz, max_len, t), -1, dtype=torch.long) if has_frames else None
    semb = torch.zeros(bsz, max_len, batch[0][7][0].shape[1]) if has_speech else None
    smask = torch.zeros(bsz, max_len, dtype=torch.bool) if has_speech else None
    prior = torch.zeros(bsz, max_len, batch[0][8].shape[1]) if has_prior else None
    for i, item in enumerate(batch):
        m, e, v, n, book, ws = item[:6]
        n = int(n)
        meg[i, :n] = m
        emb[i, :n] = e
        vid[i, :n] = v
        mask[i, :n] = True
        books.append(book)
        words.append(ws)
        if has_frames:
            frames[i, :n] = item[6]
        if has_speech:
            semb[i, :n] = item[7][0]; smask[i, :n] = item[7][1]
        if has_prior:
            prior[i, :n] = item[8]
    if has_prior:
        return meg, emb, vid, mask, books, words, frames, (semb, smask) if has_speech else None, prior
    if has_frames or has_speech:
        return meg, emb, vid, mask, books, words, frames, (semb, smask) if has_speech else None
    return meg, emb, vid, mask, books, words


def load_backbone_state(ckpt_path: str) -> dict:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    return {k[len("model.") :]: v for k, v in sd.items() if k.startswith("model.")}


class LitHybridTCN(L.LightningModule):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        gallery: torch.Tensor,
        class_weight=None,
        lr_backbone: float = 1e-4,
        lr_new: float = 3e-4,
        weight_decay: float = 0.01,
        dropout: float = 0.3,
        label_smoothing: float = 0.1,
        model_dim: int = 100,
        tcn_layers: int = 20,
        pool: str = "avg",
        pool_queries: int = 4,
        mixer_depth: int = 2,
        mixer_heads: int = 4,
        ce_lambda: float = 1.0,
        siglip_lambda: float = 1.0,
        blend: float = 0.5,
        encode_chunk: int = 64,
        sched_patience: int = 2,
        channel_dropout: float = 0.0,
        macro_families: str = "",
        phoneme_lambda: float = 0.0,
        phoneme_tap: int = 16,
        phoneme_map: dict | None = None,
        emb_dim: int = 1024,
        amp_scale: float = 0.0,
        time_mask_ms: float = 0.0,
        band_perturb: float = 0.0,
        frame_phoneme_lambda: float = 0.0,
        speech_lambda: float = 0.0,
        speech_dim: int = 768,
        val_views: tuple[str, ...] = (),
        instance_norm: bool = False,
        lowpass_hz: float = 0.0,
        win_tmin: float = 0.0,
        win_tmax: float = 1.0,
        time_emb: bool = False,
        epoch_mask: bool = False,
        spatial_k: int = 0,
        backbone: str = "tcn",
        bfm_ckpt: str = "",
        bfm_scale: float = 1.0,
        omni_ckpt: str = "",
        omni_mode: str = "vq",
        neighbor_lambda: float = 0.0,
        neighbor_offsets: str = "-1,1,2,3",
        bank_lambda: float = 0.0,
        bank: torch.Tensor | None = None,
        bank_words: list | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["gallery", "class_weight", "phoneme_map", "bank", "bank_words"])
        if backbone == "omni":
            # BrainOmni foundation model (frozen tokenizer + fine-tuned transformer), src/omni_backbone.py
            from omni_backbone import OmniBackbone
            self.backbone = OmniBackbone(n_classes, model_dim, ckpt_dir=omni_ckpt,
                                         pool_queries=pool_queries, dropout=dropout, mode=omni_mode)
        elif backbone == "bfm":
            # B[FM]²-style SplitUNet encoder (flow-matching pretrained or random init), src/splitunet.py
            from splitunet import BFMBackbone
            self.backbone = BFMBackbone(n_classes, model_dim, ckpt=bfm_ckpt, scale=bfm_scale,
                                        pool_queries=pool_queries, dropout=dropout)
        else:
            self.backbone = CNN_TCN(
                n_channels, n_classes,
                model_dim=model_dim, tcn_layers=tcn_layers, dropout=dropout,
                pool=pool, pool_queries=pool_queries,
                time_emb_len=int(round((win_tmax - win_tmin) * 250)) if time_emb else 0,
                spatial_k=spatial_k,
            )
        if neighbor_lambda > 0:
            # Research idea 3: each pre-mixer window embedding also predicts the vocab label of the
            # words at these sentence offsets, so the window vector carries the neighbour evidence
            # the mixer relies on.
            self.nb_heads = nn.ModuleDict({
                str(k): nn.Linear(model_dim, n_classes) for k in map(int, neighbor_offsets.split(","))
            })
        if bank_lambda > 0:
            # Research idea 4: softmax of the retrieval prediction over a fixed bank of the most
            # frequent training words (T5 embeddings), so every step contrasts against all phonetic
            # neighbours, not only the words of the batch. The bank is not persisted (training only).
            self.bank_scale = nn.Parameter(torch.tensor(10.0).log())
        self._bank = None if bank is None else F.normalize(bank.float(), dim=1)
        self._bank_index = {w: i for i, w in enumerate(bank_words or [])}
        self.mixer = ResidualSentenceMixer(
            model_dim, depth=mixer_depth, heads=mixer_heads, dropout=dropout,
        )
        self.ret_head = RetrievalHead(model_dim, emb_dim=emb_dim, dropout=dropout)
        self.siglip = SigLipLoss()
        self.ce = nn.CrossEntropyLoss(
            weight=class_weight, label_smoothing=label_smoothing, ignore_index=-1,
        )
        self.register_buffer("gallery", F.normalize(gallery.float(), dim=1))
        self.phoneme_map = phoneme_map or {}
        if phoneme_lambda > 0:
            from phonemes import N_PHONES
            self.ph_bag = nn.Linear(model_dim, N_PHONES)
            self.ph_first = nn.Linear(model_dim, N_PHONES)
        if frame_phoneme_lambda > 0:
            if phoneme_lambda > 0:
                raise ValueError("use either phoneme_lambda (bag) or frame_phoneme_lambda, not both")
            from phonemes import N_PHONES
            # Frame-level phoneme head on the time-resolved tap activation:
            # 40-way (39 phones + none) per 4 ms frame, for every word in the window.
            self.ph_frame = nn.Conv1d(model_dim, N_PHONES + 1, kernel_size=1)
        if speech_lambda > 0:
            # MEG↔speech contrastive head (Défossez et al. 2023 style): encoder
            # features (pre-mixer) are aligned with the HuBERT embedding of the
            # audio heard in the window, SigLIP loss over the batch's windows.
            self.speech_head = RetrievalHead(model_dim, emb_dim=speech_dim, dropout=dropout)
            self.siglip_speech = SigLipLoss()
        # Buffer 0 is the stop/select view (full sentences). Extra views
        # (e.g. "short5", "iso1") are logged only: HANDOVER §17 says the
        # holdout is short contexts, so watch those beside the main rail.
        self._views = ("full",) + tuple(val_views)
        self._val = [
            {"ce": [], "ret": [], "blend": [], "y": [], "fam": []} for _ in self._views
        ]

    def load_qc_backbone(self, ckpt_path: str) -> None:
        mapped = load_backbone_state(ckpt_path)
        self.backbone.load_state_dict(mapped, strict=True)
        print(f"QC backbone loaded ({len(mapped)} tensors)", flush=True)

    def encode_windows(self, meg: torch.Tensor, want_tap: bool = False, tap_seq: bool = False):
        """(N, 306, 250) → (N, D) [, tap: (N, D) mean or (N, D, T) sequence], chunked to fit VRAM."""
        lp = self.hparams.get("lowpass_hz", 0.0)
        if lp:
            # Brick-wall low-pass per window (rFFT): word identity lives < ~4 Hz
            # (HANDOVER §21.8), higher bands are mostly session/noise variance.
            f = torch.fft.rfftfreq(meg.shape[-1], d=1.0 / 250.0).to(meg.device)
            X = torch.fft.rfft(meg.float(), dim=-1) * (f <= lp).to(torch.float32)
            meg = torch.fft.irfft(X, n=meg.shape[-1], dim=-1).to(meg.dtype)
        if self.hparams.get("instance_norm", False):
            # Per-window, per-channel z-score over time (MEGConformer, LibriBrain
            # 2025 phoneme winner): removes session-level gain/offset shift.
            meg = (meg - meg.mean(-1, keepdim=True)) / (meg.std(-1, keepdim=True) + 1e-5)
        chunk = self.hparams.encode_chunk
        tap = self.hparams.phoneme_tap if want_tap else None
        parts, taps = [], []
        for i in range(0, meg.shape[0], chunk):
            out = self.backbone.features(meg[i : i + chunk], tap=tap, tap_seq=tap_seq)
            if want_tap:
                parts.append(out[0]); taps.append(out[1])
            else:
                parts.append(out)
        h = torch.cat(parts, dim=0)
        if want_tap:
            return h, torch.cat(taps, dim=0)
        return h

    def contextualize(self, meg, mask):
        bsz, n_w, n_ch, t = meg.shape
        h = self.encode_windows(meg.reshape(bsz * n_w, n_ch, t))
        h = h.reshape(bsz, n_w, -1)
        h = h * mask.unsqueeze(-1).float()
        return self.mixer(h, mask)

    def heads(self, h_ctx):
        bsz, n_w, dim = h_ctx.shape
        flat = h_ctx.reshape(bsz * n_w, dim)
        logits = self.backbone.classify(flat).reshape(bsz, n_w, -1)
        pred = self.ret_head(flat).reshape(bsz, n_w, -1)
        return logits, pred

    def retrieval_logits(self, pred_norm: torch.Tensor) -> torch.Tensor:
        return pred_norm @ self.gallery.T

    def blended_logits(self, ce_logits: torch.Tensor, pred_norm: torch.Tensor) -> torch.Tensor:
        ret = self.retrieval_logits(pred_norm)
        temp = self.siglip.temperature.exp().detach().clamp(min=1.0)
        p_ce = F.softmax(ce_logits, dim=-1)
        p_ret = F.softmax(ret * temp, dim=-1)
        mix = self.hparams.blend * p_ce + (1.0 - self.hparams.blend) * p_ret
        return mix.clamp(min=1e-8).log()

    def _flat_valid(self, tensor, mask):
        return tensor[mask]

    def training_step(self, batch, _):
        meg, emb, vid, mask, _books, words, *extra = batch
        frames = extra[0] if extra else None
        speech = extra[1] if len(extra) > 1 else None
        prior = extra[2] if len(extra) > 2 else None
        h_enc = None
        p = float(self.hparams.channel_dropout)
        if self.training and p > 0:
            # Sensor dropout: zero whole channels per window, rescale the rest
            # so activation scale matches eval (no dropout there).
            keep = (
                torch.rand(meg.shape[0], meg.shape[1], meg.shape[2], 1, device=meg.device)
                >= p
            )
            meg = meg * keep.float() / (1.0 - p)
        a = float(self.hparams.amp_scale)
        if self.training and a > 0:
            # Per-window gain U(1-a, 1+a): robustness to session/holdout
            # amplitude drift (the ±10% band of the 2026 MEG robustness stack).
            gain = 1.0 + a * (2.0 * torch.rand(meg.shape[0], meg.shape[1], 1, 1, device=meg.device) - 1.0)
            meg = meg * gain
        tm = float(self.hparams.time_mask_ms)
        if self.training and tm > 0:
            # MEGAugment-style time masking (MEGConformer, LibriBrain 2025): zero one
            # random contiguous span of up to tm ms in half of the windows.
            bsz, n_w, n_ch, t = meg.shape
            max_len = max(1, int(round(tm / 1000.0 * 250)))
            lens = torch.randint(1, max_len + 1, (bsz, n_w, 1, 1), device=meg.device)
            starts = (torch.rand(bsz, n_w, 1, 1, device=meg.device) * (t - lens)).long()
            idx = torch.arange(t, device=meg.device).view(1, 1, 1, t)
            masked = (idx >= starts) & (idx < starts + lens)
            apply = torch.rand(bsz, n_w, 1, 1, device=meg.device) < 0.5
            meg = meg.masked_fill(masked & apply, 0.0)
        bp = float(self.hparams.band_perturb)
        if self.training and bp > 0:
            # Frequency-band perturbation (2026 MEG robustness stack): rescale the
            # magnitude of one random band of the rFFT by U(1-bp, 1+bp) per window.
            bsz, n_w, n_ch, t = meg.shape
            spec = torch.fft.rfft(meg.float(), dim=-1)
            n_f = spec.shape[-1]
            lo = torch.randint(0, n_f - 2, (bsz, n_w, 1, 1), device=meg.device)
            width = torch.randint(2, max(3, n_f // 4), (bsz, n_w, 1, 1), device=meg.device)
            fidx = torch.arange(n_f, device=meg.device).view(1, 1, 1, n_f)
            band = (fidx >= lo) & (fidx < lo + width)
            scale = 1.0 + bp * (2.0 * torch.rand(bsz, n_w, 1, 1, device=meg.device) - 1.0)
            spec = torch.where(band, spec * scale, spec)
            meg = torch.fft.irfft(spec, n=t, dim=-1).to(meg.dtype)
        lam_ph = float(self.hparams.phoneme_lambda)
        loss_ph = None
        if lam_ph > 0:
            bsz, n_w, n_ch, t = meg.shape
            h_flat, tap_flat = self.encode_windows(
                meg.reshape(bsz * n_w, n_ch, t), want_tap=True,
            )
            h = h_flat.reshape(bsz, n_w, -1) * mask.unsqueeze(-1).float()
            h_enc = h
            h_ctx = self.mixer(h, mask)
            tap_v = tap_flat.reshape(bsz, n_w, -1)[mask]
            flat_words = [w for ws in words for w in ws]
            from phonemes import N_PHONES
            bag_t = torch.zeros(len(flat_words), N_PHONES)
            first_t = torch.full((len(flat_words),), -1, dtype=torch.long)
            has = torch.zeros(len(flat_words), dtype=torch.bool)
            for k, w in enumerate(flat_words):
                entry = self.phoneme_map.get(w)
                if entry is not None:
                    bag_t[k] = entry[0]
                    first_t[k] = entry[1]
                    has[k] = True
            if has.any():
                tap_h = tap_v[has.to(tap_v.device)]
                loss_bag = F.binary_cross_entropy_with_logits(
                    self.ph_bag(tap_h), bag_t[has].to(tap_v.device),
                )
                loss_first = F.cross_entropy(
                    self.ph_first(tap_h), first_t[has].to(tap_v.device),
                )
                loss_ph = 0.5 * loss_bag + 0.5 * loss_first
                self.log("train_ph_bag", loss_bag, on_step=False, on_epoch=True)
                self.log("train_ph_first", loss_first, on_step=False, on_epoch=True)
        elif float(self.hparams.frame_phoneme_lambda) > 0 and frames is not None:
            bsz, n_w, n_ch, t = meg.shape
            h_flat, tap_seq = self.encode_windows(
                meg.reshape(bsz * n_w, n_ch, t), want_tap=True, tap_seq=True,
            )
            h = h_flat.reshape(bsz, n_w, -1) * mask.unsqueeze(-1).float()
            h_enc = h
            h_ctx = self.mixer(h, mask)
            flat_mask = mask.reshape(-1)
            fr = frames.reshape(bsz * n_w, -1)[flat_mask]
            if (fr >= 0).any():
                logits_fr = self.ph_frame(tap_seq[flat_mask].float())      # (N, C, T)
                loss_ph = F.cross_entropy(logits_fr, fr, ignore_index=-1)
                with torch.no_grad():
                    valid = fr >= 0
                    acc = (logits_fr.argmax(1)[valid] == fr[valid]).float().mean()
                self.log("train_ph_frame", loss_ph, on_step=False, on_epoch=True)
                self.log("train_ph_frame_acc", acc, on_step=False, on_epoch=True)
                lam_ph = float(self.hparams.frame_phoneme_lambda)
        else:
            bsz, n_w, n_ch, t = meg.shape
            h_enc = self.encode_windows(meg.reshape(bsz * n_w, n_ch, t)).reshape(bsz, n_w, -1) * mask.unsqueeze(-1).float()
            h_ctx = self.mixer(h_enc, mask)
        loss_sp = None
        lam_sp = float(self.hparams.speech_lambda)
        if lam_sp > 0 and speech is not None:
            semb, smask = speech
            sel = mask & smask
            if int(sel.sum()) >= 8:
                loss_sp = self.siglip_speech(self.speech_head(h_enc[sel]), semb[sel])
                self.log("train_speech", loss_sp, on_step=False, on_epoch=True)
                self.log("train_speech_frac", sel.float().sum() / mask.float().sum(), on_step=False, on_epoch=True)
        ce_logits, pred = self.heads(h_ctx)
        y = self._flat_valid(vid, mask)
        logits = self._flat_valid(ce_logits, mask)
        pred_v = self._flat_valid(pred, mask)
        t5 = self._flat_valid(emb, mask)
        if prior is not None:
            # Research idea 2 (product of experts): fixed log-priors (timing-only and masked-LM
            # experts) join the logits in training only, so the network learns the residual and the
            # holdout-domain prior comes from the decoder (few-shot LLM + prior correction) instead.
            logits = logits + self._flat_valid(prior.to(logits.dtype), mask)

        in_vocab = y >= 0
        loss_ce = logits.new_zeros(())
        if in_vocab.any():
            loss_ce = self.ce(logits[in_vocab], y[in_vocab])
        loss_sig = self.siglip(pred_v, t5)
        loss = self.hparams.ce_lambda * loss_ce + self.hparams.siglip_lambda * loss_sig
        lam_nb = float(self.hparams.get("neighbor_lambda", 0.0))
        if lam_nb > 0:
            nb_losses = []
            for k_str, head in self.nb_heads.items():
                k = int(k_str)
                if k > 0:
                    src, tgt, m = h_enc[:, :-k], vid[:, k:], mask[:, :-k] & mask[:, k:]
                else:
                    src, tgt, m = h_enc[:, -k:], vid[:, :k], mask[:, -k:] & mask[:, :k]
                m = m & (tgt >= 0)
                if m.any():
                    nb_losses.append(self.ce(head(src[m]), tgt[m]))
            if nb_losses:
                loss_nb = torch.stack(nb_losses).mean()
                loss = loss + lam_nb * loss_nb
                self.log("train_neighbor", loss_nb, on_step=False, on_epoch=True, batch_size=int(mask.sum()))
        lam_bank = float(self.hparams.get("bank_lambda", 0.0))
        if lam_bank > 0 and self._bank is not None:
            flat_words = [w for ws in words for w in ws]
            tgt = torch.tensor([self._bank_index.get(w, -1) for w in flat_words], device=pred_v.device)
            ok = tgt >= 0
            if ok.any():
                bank = self._bank.to(pred_v.device, pred_v.dtype)
                bl = self.bank_scale.exp() * (pred_v[ok] @ bank.T)
                loss_bank = F.cross_entropy(bl.float(), tgt[ok])
                loss = loss + lam_bank * loss_bank
                self.log("train_bank", loss_bank, on_step=False, on_epoch=True, batch_size=int(ok.sum()))
        if loss_ph is not None:
            loss = loss + lam_ph * loss_ph
        if loss_sp is not None:
            loss = loss + lam_sp * loss_sp
        n = int(mask.sum())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=n)
        self.log("train_ce", loss_ce, on_step=False, on_epoch=True, batch_size=n)
        self.log("train_siglip", loss_sig, on_step=False, on_epoch=True, batch_size=n)
        self.log("mixer_gate", self.mixer.gate.detach(), on_step=False, on_epoch=True, batch_size=n)
        return loss

    def validation_step(self, batch, _, dataloader_idx: int = 0):
        meg, emb, vid, mask, books, _words, *_extra = batch
        buf = self._val[dataloader_idx]
        h_ctx = self.contextualize(meg, mask)
        ce_logits, pred = self.heads(h_ctx)
        y = self._flat_valid(vid, mask)
        keep = y >= 0
        if not keep.any():
            return
        ce = self._flat_valid(ce_logits, mask)[keep]
        pr = self._flat_valid(pred, mask)[keep]
        yy = y[keep]
        fams = []
        for i, book in enumerate(books):
            n = int((mask[i] & (vid[i] >= 0)).sum().item())
            if n:
                fams.extend([corpus_family(str(book))] * n)
        if len(fams) != int(keep.sum()):
            raise RuntimeError(
                f"val family labels {len(fams)} != kept words {int(keep.sum())}"
            )
        buf["ce"].append(ce.detach().cpu())
        buf["ret"].append(self.retrieval_logits(pr).detach().cpu())
        buf["blend"].append(self.blended_logits(ce, pr).detach().cpu())
        buf["y"].append(yy.detach().cpu())
        buf["fam"].extend(fams)
        if dataloader_idx == 0:
            loss = self.ce(ce, yy)
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True,
                     batch_size=int(keep.sum()), add_dataloader_idx=False)

    def _wanted_families(self) -> set[str]:
        return {
            f.strip().lower()
            for f in str(self.hparams.macro_families or "").split(",")
            if f.strip()
        }

    def on_validation_epoch_end(self):
        self._end_main_view()
        for k, view in enumerate(self._views):
            if k == 0:
                continue
            self._end_extra_view(k, view)
        for buf in self._val:
            for key in buf:
                buf[key].clear()

    def _end_extra_view(self, k: int, view: str):
        """Log-only BAcc@10 of an alternate val chunking (no stop/select role)."""
        buf = self._val[k]
        if not buf["y"]:
            return
        y = torch.cat(buf["y"])
        n = self.hparams.n_classes
        ce = torch.cat(buf["ce"])
        blend = torch.cat(buf["blend"])
        wanted = self._wanted_families()
        fam_arr = np.array(buf["fam"])
        parts = []
        for fam in sorted(set(buf["fam"])):
            idx = torch.from_numpy(np.flatnonzero(fam_arr == fam))
            ch = max(
                balanced_topk_accuracy(ce[idx], y[idx], 10, n),
                balanced_topk_accuracy(blend[idx], y[idx], 10, n),
            )
            self.log(f"val_{view}_bacc10_{fam}", ch, add_dataloader_idx=False)
            if not wanted or fam.lower() in wanted:
                parts.append(ch)
            print(f"    val[{view}] {fam}: n={int(idx.numel())}  BAcc@10={ch:.3f}", flush=True)
        if parts:
            self.log(f"val_{view}_bacc10", float(sum(parts) / len(parts)), add_dataloader_idx=False)

    def _end_main_view(self):
        buf = self._val[0]
        if not buf["y"]:
            return
        y = torch.cat(buf["y"])
        n = self.hparams.n_classes
        ce = torch.cat(buf["ce"])
        ret = torch.cat(buf["ret"])
        blend = torch.cat(buf["blend"])
        b_ce = balanced_topk_accuracy(ce, y, 10, n)
        b_ret = balanced_topk_accuracy(ret, y, 10, n)
        b_blend = balanced_topk_accuracy(blend, y, 10, n)
        pooled = max(b_ce, b_blend)
        fams = buf["fam"]
        macro_parts = []
        # Empty = every family votes (historic). Set e.g. "timit,mocha,podcasts"
        # to keep Sherlock out of the stop/select metric: the 2026-08-30
        # calibration showed Sherlock val gains anti-correlate with the
        # public holdout (0.703 sherlock → 0.558 public vs 0.677 → 0.626).
        wanted = self._wanted_families()
        if fams:
            fam_arr = np.array(fams)
            for fam in sorted(set(fams)):
                idx = torch.from_numpy(np.flatnonzero(fam_arr == fam))
                if idx.numel() == 0:
                    continue
                ch = max(
                    balanced_topk_accuracy(ce[idx], y[idx], 10, n),
                    balanced_topk_accuracy(blend[idx], y[idx], 10, n),
                )
                self.log(f"val_bacc10_{fam}", ch, add_dataloader_idx=False)
                if not wanted or fam.lower() in wanted:
                    macro_parts.append(ch)
                print(
                    f"    val {fam}: n={int(idx.numel())}  BAcc@10={ch:.3f}",
                    flush=True,
                )
        chosen = float(sum(macro_parts) / len(macro_parts)) if macro_parts else pooled
        self.log("val_bacc10_ce", b_ce, add_dataloader_idx=False)
        self.log("val_bacc10_ret", b_ret, add_dataloader_idx=False)
        self.log("val_bacc10_blend", b_blend, add_dataloader_idx=False)
        self.log("val_bacc10_pooled", pooled, add_dataloader_idx=False)
        # Family-macro so Sherlock-only val cannot stop a run that is generalising.
        self.log("val_bacc10", chosen, prog_bar=True, add_dataloader_idx=False)
        self.log(
            "val_bacc1",
            balanced_topk_accuracy(blend if b_blend >= b_ce else ce, y, 1, n),
            prog_bar=True, add_dataloader_idx=False,
        )

    def configure_optimizers(self):
        # backbones with pretrained weights may expose their freshly initialised layers
        # (new_parameters) so those train at lr_new; frozen parameters stay out of the optimizer
        bb_new = list(self.backbone.new_parameters()) if hasattr(self.backbone, "new_parameters") else []
        bb_new_ids = {id(p) for p in bb_new}
        backbone = [p for p in self.backbone.parameters() if p.requires_grad and id(p) not in bb_new_ids]
        new = bb_new + (
            list(self.mixer.parameters())
            + list(self.ret_head.parameters())
            + list(self.siglip.parameters())
        )
        # Any other trainable module (phoneme heads ph_bag / ph_first / ph_frame)
        # joins the "new" group. Before 2026-09-02 these were in NO optimizer
        # group at all, so the v3 bag-of-phonemes heads and the first
        # frame-phoneme run trained the encoder against a frozen random head.
        seen = {id(p) for p in backbone + new} | {id(p) for p in self.backbone.parameters()}
        extra = [p for p in self.parameters() if p.requires_grad and id(p) not in seen]
        new = new + extra
        opt = torch.optim.AdamW(
            [
                {"params": backbone, "lr": self.hparams.lr_backbone},
                {"params": new, "lr": self.hparams.lr_new},
            ],
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
