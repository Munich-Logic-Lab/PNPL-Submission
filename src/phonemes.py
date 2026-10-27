"""ARPAbet phoneme targets from the CMU pronouncing dictionary.

Used as auxiliary supervision (Toshniwal et al. 2017 style): per word we
derive a 39-dim bag-of-phonemes vector and the first-phoneme id. Stress
digits are stripped; words missing from cmudict get no targets (masked).

Needs `pip install cmudict` (pure data package).
"""

from __future__ import annotations

import torch

# canonical 39 ARPAbet phones (cmudict.phones() order, stress-stripped)
PHONES = [
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER",
    "EY", "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW",
    "OY", "P", "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z",
    "ZH",
]
PHONE_TO_ID = {p: i for i, p in enumerate(PHONES)}
N_PHONES = len(PHONES)


def _strip(ph: str) -> str:
    return "".join(c for c in ph if not c.isdigit())


def phoneme_table(words) -> dict[str, tuple[torch.Tensor, int]]:
    """word -> (bag float tensor (39,), first-phoneme id). Missing words omitted."""
    import cmudict

    d = cmudict.dict()
    table: dict[str, tuple[torch.Tensor, int]] = {}
    for w in words:
        prons = d.get(w)
        if not prons:
            continue
        phones = [_strip(p) for p in prons[0]]
        ids = [PHONE_TO_ID[p] for p in phones if p in PHONE_TO_ID]
        if not ids:
            continue
        bag = torch.zeros(N_PHONES)
        bag[ids] = 1.0
        table[w] = (bag, ids[0])
    return table


class FrameLabeler:
    """Frame-level (250 Hz) phoneme labels from the BIDS events TSVs.

    Every `kind == "phoneme"` row carries its MEG onset and duration, so a
    word window can be labelled at every 4 ms frame with the phoneme being
    heard at that moment — for *all* words inside the window, not just the
    target (HANDOVER §18: the v3 bag-of-phonemes head averaged the whole
    second and asked for the target word's phonemes; that discarded the timing
    the evoked response is locked to).

    Label 0 = no phoneme (silence / gap), 1..39 = PHONES, -1 = unknown run
    (no TSV) -> ignored by the loss.
    """

    N_CLASSES = N_PHONES + 1

    def __init__(self, data_path, sfreq: int = 250):
        from pathlib import Path as _P

        self.data_path = _P(data_path)
        self.sfreq = sfreq
        self._timelines: dict[tuple, object] = {}

    def _load(self, subject, session, task, run):
        import numpy as np
        import pandas as pd

        key = (str(subject), str(session), str(task), str(run))
        if key in self._timelines:
            return self._timelines[key]
        path = (
            self.data_path / str(task) / "derivatives" / "events"
            / f"sub-{subject}_ses-{session}_task-{task}_run-{run}_events.tsv"
        )
        tl = None
        if path.exists():
            ev = pd.read_csv(path, sep="\t", usecols=["kind", "segment", "timemeg", "duration"])
            ph = ev[ev.kind == "phoneme"].dropna(subset=["timemeg", "duration"])
            if len(ph):
                end = int(np.ceil((ph.timemeg + ph.duration).max() * self.sfreq)) + self.sfreq
                tl = np.zeros(end, dtype=np.int8)
                for seg, t0, dur in zip(ph.segment, ph.timemeg, ph.duration):
                    pid = PHONE_TO_ID.get(_strip(str(seg).split("_")[0].upper()))
                    if pid is None:
                        continue
                    a = int(round(float(t0) * self.sfreq)); b = int(round((float(t0) + float(dur)) * self.sfreq))
                    tl[a:max(b, a + 1)] = pid + 1
        self._timelines[key] = tl
        return tl

    def labels(self, sample, t_start: float, n: int = 250):
        """(n,) LongTensor for the window starting at MEG time `t_start` (s)."""
        import numpy as np

        tl = self._load(sample[0], sample[1], sample[2], sample[3])
        out = torch.full((n,), -1, dtype=torch.long)
        if tl is None:
            return out
        a = max(0, int(round(t_start * self.sfreq)))
        seg = tl[a:a + n]
        out[: len(seg)] = torch.from_numpy(seg.astype(np.int64))
        if len(seg) < n:
            out[len(seg):] = 0
        return out


class SpeechStore:
    """Per-word speech-embedding targets written by scripts/speech_targets.py.

    Keyed by (task, session, run) -> {rounded MEG onset: emb}. Loaded lazily per
    run (one npz each, ~N x 768 fp16). Windows without a target (other corpora,
    Sherlock8/9, broad subjects) get a zero vector and mask False.
    """

    def __init__(self, emb_dir, dim: int = 768):
        from pathlib import Path as _P

        self.dir = _P(emb_dir); self.dim = dim
        self._runs: dict[tuple, dict | None] = {}

    def _load(self, task, session, run):
        import numpy as np

        key = (str(task), str(session), str(run))
        if key not in self._runs:
            path = self.dir / f"{task}_ses-{session}_run-{run}.npz"
            if path.exists():
                d = np.load(path)
                self._runs[key] = {round(float(o), 4): e for o, e in zip(d["onset"], d["emb"])}
            else:
                self._runs[key] = None
        return self._runs[key]

    def get(self, sample):
        """(emb (dim,) float32, found bool) for a pnpl sample tuple (subject, session, task, run, onset, ...)."""
        if str(sample[0]) != "0":            # broad subjects heard the same audio but we only index sub-0 runs
            return torch.zeros(self.dim), False
        tbl = self._load(sample[2], sample[1], sample[3])
        if tbl is None:
            return torch.zeros(self.dim), False
        e = tbl.get(round(float(sample[4]), 4))
        if e is None:
            return torch.zeros(self.dim), False
        return torch.from_numpy(e.astype("float32")), True

    def n_runs(self):
        return len(list(self.dir.glob("*.npz")))
