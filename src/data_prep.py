"""QC for already-SSS LibriBrain H5 word windows.

Follows the MEG QC dossier (Section 10): physical-unit tests on raw H5,
magnetometers vs gradiometers kept separate, MAD-z fitted on train only,
val/test dropped only for integrity failures. Do not apply Tesla cuts
after z-scoring. Short words and Sherlock 9 are not artifacts.

Neuromag VectorView / MEGIN TRIUX 306-channel order is assumed:
triplets (grad, grad, mag). H5 files here have no channel-name attrs.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SFREQ = 250.0
WIN_S = 1.0
N_WIN = int(WIN_S * SFREQ)
CLIP_Z = 10.0
# Iglewicz, Hoaglin & Tukey: z = 0.6745 (x - median) / MAD_raw
IGLEWICZ = 0.67448975
# MNE BaseEpochs *example* PTP (not a decoder optimum; plot as reference only)
MNE_MAG_PTP = 4e-12  # T
MNE_GRAD_PTP = 4e-10  # T/m
FLAT_STD = 1e-18
HF_BAND = (70.0, 120.0)
TOT_BAND = (1.0, 120.0)

READER = {f"Sherlock{i}": "D. Clarke" for i in range(1, 9)}
READER["Sherlock9"] = "T. Copeland"


def corpus_family(task: str) -> str:
    """Map a pnpl task name to a stimulus family for mixed-corpus sampling."""
    t = str(task).lower()
    if t.startswith("sherlock"):
        return "sherlock"
    if "mocha" in t:
        return "mocha"
    if "timit" in t:
        return "timit"
    if "moth" in t or "podcast" in t:
        return "podcasts"
    return str(task)


def vectorview_indices(n_ch: int = 306) -> tuple[np.ndarray, np.ndarray]:
    if n_ch % 3 != 0:
        mag = np.arange(n_ch)
        return mag, np.array([], dtype=int)
    mag = np.arange(2, n_ch, 3, dtype=int)
    grad = np.setdiff1d(np.arange(n_ch), mag)
    return mag, grad


MAG_IDX, GRAD_IDX = vectorview_indices(306)


def normalize_word(w) -> str:
    return str(w).strip().lower().replace("’", "'")


def parse_bids_name(path: Path) -> tuple[str, str, str, str] | None:
    m = re.search(r"sub-(\d+)_ses-(\d+)_task-([^_]+)_run-(\d+)", path.name)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3), m.group(4)


def parse_h5_name(path: Path) -> tuple[str, str, str, str] | None:
    return parse_bids_name(path)


def reader_for_task(task: str) -> str:
    return READER.get(str(task), "unknown")


def trial_key(task, session, run, onset_s, word) -> tuple:
    return (
        str(task),
        str(session),
        str(run),
        round(float(onset_s), 4),
        normalize_word(word),
    )


def sample_trial_key(sample) -> tuple:
    return trial_key(sample[2], sample[1], sample[3], sample[4], sample[5])


def mad_raw(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    med = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - med)))
    return med, mad


def mad_z(x: np.ndarray, med: float | None = None, mad: float | None = None) -> np.ndarray:
    """Modified z (Iglewicz). If MAD is 0, return zeros (degenerate; inspect flatness)."""
    x = np.asarray(x, dtype=np.float64)
    if med is None or mad is None:
        med, mad = mad_raw(x)
    if not np.isfinite(mad) or mad < 1e-30:
        return np.zeros(len(x), dtype=np.float64)
    return IGLEWICZ * (x - med) / mad


def mad_flags(x: np.ndarray, k: float = 5.0, med=None, mad=None) -> np.ndarray:
    return np.abs(mad_z(x, med, mad)) > k


def group_samples_by_run(samples: list) -> dict[tuple, list]:
    g = defaultdict(list)
    for i, s in enumerate(samples):
        g[tuple(s[:4])].append((i, s))
    return g


def pool_channel_stats(
    means: np.ndarray, stds: np.ndarray, n_times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Sample-count-weighted pool of per-run channel mean/std (PNPL parallel algorithm)."""
    means = np.asarray(means, dtype=np.float64)
    vars_ = np.asarray(stds, dtype=np.float64) ** 2
    n = np.asarray(n_times, dtype=np.float64)
    wmean = np.average(means, axis=0, weights=n)
    tile_n = np.tile(n, (vars_.shape[1], 1)).T
    ss_within = np.sum(vars_ * tile_n, axis=0)
    ss_between = np.sum((means - np.tile(wmean, (means.shape[0], 1))) ** 2 * tile_n, axis=0)
    std = np.sqrt((ss_within + ss_between) / np.sum(n))
    return wmean, std


def _rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a * a))) if a.size else np.nan


def _ptp_max(a: np.ndarray) -> float:
    if a.size == 0:
        return np.nan
    return float((a.max(axis=1) - a.min(axis=1)).max())


def _jump_max(a: np.ndarray) -> float:
    if a.shape[-1] < 2:
        return np.nan
    return float(np.max(np.abs(np.diff(a, axis=1))))


def _gfp_mean(a: np.ndarray) -> float:
    if a.size == 0:
        return np.nan
    gfp = np.sqrt(np.mean(a * a, axis=0))
    return float(gfp.mean())


def _hf_ratio(gfp: np.ndarray, sfreq: float = SFREQ) -> float:
    spec = np.abs(np.fft.rfft(gfp)) ** 2
    freqs = np.fft.rfftfreq(gfp.size, 1.0 / sfreq)
    hf = spec[(freqs >= HF_BAND[0]) & (freqs <= HF_BAND[1])].sum()
    tot = spec[(freqs >= TOT_BAND[0]) & (freqs <= TOT_BAND[1])].sum()
    if tot <= 0:
        return np.nan
    return float(hf / tot)


def window_metrics(
    data: np.ndarray,
    onset_s: float,
    ch_mean: np.ndarray | None = None,
    ch_std: np.ndarray | None = None,
    n_times: int | None = None,
    mag_idx: np.ndarray = MAG_IDX,
    grad_idx: np.ndarray = GRAD_IDX,
    n: int = N_WIN,
    sfreq: float = SFREQ,
) -> dict:
    """Raw-unit 1 s window stats. Mag and grad never share one Tesla cutoff."""
    start = int(round(float(onset_s) * sfreq))
    n_ch, n_t = data.shape
    n_times = n_t if n_times is None else n_times
    out = {
        "oob": True,
        "nan_frac": np.nan,
        "n_flat_ch": 0,
        "globally_flat": False,
        "mag_rms": np.nan,
        "grad_rms": np.nan,
        "mag_gfp": np.nan,
        "grad_gfp": np.nan,
        "mag_ptp_max": np.nan,
        "grad_ptp_max": np.nan,
        "mag_jump": np.nan,
        "grad_jump": np.nan,
        "mag_hf_ratio": np.nan,
        "grad_hf_ratio": np.nan,
        "dist_start_s": float(onset_s),
        "dist_end_s": np.nan,
        "edge_1s": False,
        "clip_frac_z10": np.nan,
        "z_rms": np.nan,
        "z_max_abs": np.nan,
        "jump_ch": -1,
    }
    out["dist_end_s"] = n_times / sfreq - float(onset_s) - n / sfreq
    out["edge_1s"] = (float(onset_s) < 1.0) or (out["dist_end_s"] < 1.0)
    if start < 0 or start + n > n_t:
        return out
    w = data[:, start : start + n]
    finite = np.isfinite(w)
    nan_frac = float(1.0 - finite.mean())
    if nan_frac > 0:
        w = np.where(finite, w, 0.0)
    ch_ptp = w.max(axis=1) - w.min(axis=1)
    n_flat = int((ch_ptp < FLAT_STD).sum())
    w_m = w[mag_idx] if mag_idx.size else w
    w_g = w[grad_idx] if grad_idx.size else w[:0]
    mag_rms = _rms(w_m)
    grad_rms = _rms(w_g)
    gfp_m = np.sqrt(np.mean(w_m * w_m, axis=0)) if w_m.size else np.zeros(n)
    gfp_g = np.sqrt(np.mean(w_g * w_g, axis=0)) if w_g.size else np.zeros(n)
    dabs = np.max(np.abs(np.diff(w, axis=1)), axis=1) if n > 1 else np.zeros(n_ch)
    out.update(
        {
            "oob": False,
            "nan_frac": nan_frac,
            "n_flat_ch": n_flat,
            "globally_flat": bool(n_flat >= 0.5 * n_ch or mag_rms == 0.0),
            "mag_rms": mag_rms,
            "grad_rms": grad_rms,
            "mag_gfp": float(gfp_m.mean()),
            "grad_gfp": float(gfp_g.mean()) if w_g.size else np.nan,
            "mag_ptp_max": _ptp_max(w_m),
            "grad_ptp_max": _ptp_max(w_g),
            "mag_jump": _jump_max(w_m),
            "grad_jump": _jump_max(w_g),
            "mag_hf_ratio": _hf_ratio(gfp_m, sfreq),
            "grad_hf_ratio": _hf_ratio(gfp_g, sfreq) if w_g.size else np.nan,
            "jump_ch": int(np.argmax(dabs)),
        }
    )
    if ch_mean is not None and ch_std is not None:
        z = (w - ch_mean[:, None]) / np.maximum(ch_std[:, None], 1e-20)
        out["clip_frac_z10"] = float(np.mean(np.abs(z) > CLIP_Z))
        out["z_rms"] = _rms(z)
        out["z_max_abs"] = float(np.max(np.abs(z)))
    return out


def run_array_qa(
    data: np.ndarray,
    mag_idx: np.ndarray = MAG_IDX,
    grad_idx: np.ndarray = GRAD_IDX,
    sfreq: float = SFREQ,
) -> dict:
    n_ch, n_t = data.shape
    finite = np.isfinite(data)
    nan_frac = float(1.0 - finite.mean())
    x = data if nan_frac == 0.0 else np.where(finite, data, 0.0)
    ch_mean = x.mean(axis=1)
    ch_std = x.std(axis=1)
    ch_med = np.median(x, axis=1)
    mag = x[mag_idx] if mag_idx.size else x
    grad = x[grad_idx] if grad_idx.size else x[:0]
    head = x[:, : min(int(sfreq), n_t)]
    return {
        "n_channels": int(n_ch),
        "n_times": int(n_t),
        "duration_s": n_t / sfreq,
        "nan_frac": nan_frac,
        "n_nonfinite": int((~finite).sum()),
        "rms": _rms(x),
        "mag_rms": _rms(mag),
        "grad_rms": _rms(grad),
        "max_abs": float(np.max(np.abs(x))),
        "median_ch_std": float(np.median(ch_std)),
        "min_ch_std": float(np.min(ch_std)),
        "n_flat_channels": int((ch_std < FLAT_STD).sum()),
        "n_flat_mag": int((ch_std[mag_idx] < FLAT_STD).sum()) if mag_idx.size else 0,
        "n_flat_grad": int((ch_std[grad_idx] < FLAT_STD).sum()) if grad_idx.size else 0,
        "n_const_first_sec": int((head.std(axis=1) < FLAT_STD).sum()),
        "mag_median_std": float(np.median(ch_std[mag_idx])) if mag_idx.size else np.nan,
        "grad_median_std": float(np.median(ch_std[grad_idx])) if grad_idx.size else np.nan,
        "ch_mean": ch_mean,
        "ch_std": ch_std,
        "ch_med": ch_med,
    }


def session_bin_gfp(
    data: np.ndarray,
    mag_idx: np.ndarray = MAG_IDX,
    grad_idx: np.ndarray = GRAD_IDX,
    bin_s: float = 1.0,
    sfreq: float = SFREQ,
    max_bins: int = 1200,
) -> tuple[np.ndarray, np.ndarray]:
    """1 s GFP bins for mag and grad (first max_bins seconds)."""
    hop = int(bin_s * sfreq)
    n_bins = min(data.shape[1] // hop, max_bins)
    if n_bins == 0:
        return np.array([]), np.array([])
    stop = n_bins * hop
    mag = np.asarray(data[mag_idx, :stop]).reshape(mag_idx.size, n_bins, hop)
    mag_g = np.sqrt((mag.astype(np.float64) ** 2).mean(axis=(0, 2)))
    if grad_idx.size:
        grad = np.asarray(data[grad_idx, :stop]).reshape(grad_idx.size, n_bins, hop)
        grad_g = np.sqrt((grad.astype(np.float64) ** 2).mean(axis=(0, 2)))
    else:
        grad_g = np.full(n_bins, np.nan)
    return mag_g, grad_g


def mean_welch(
    x: np.ndarray, sfreq: float = SFREQ, nperseg: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.signal import welch

    nperseg = int(min(nperseg, max(64, x.size // 8)))
    freqs, pxx = welch(x, fs=sfreq, nperseg=nperseg)
    return freqs.astype(np.float64), pxx.astype(np.float64)


def band_power(freqs: np.ndarray, pxx: np.ndarray, lo: float, hi: float) -> float:
    m = (freqs >= lo) & (freqs < hi)
    if not m.any():
        return np.nan
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trap(pxx[m], freqs[m]))


def annotate_speech_layout(words: pd.DataFrame, win_s: float = WIN_S) -> pd.DataFrame:
    """Layout of later speech inside a fixed onset window. Not a drop rule."""
    n_all = len(words)
    isi_next = np.full(n_all, np.nan)
    n_follow = np.zeros(n_all, dtype=np.int32)
    n_follow_iv = np.zeros(n_all, dtype=np.int32)
    speech_s = np.full(n_all, np.nan)
    own_s = np.full(n_all, np.nan)
    next_in_vocab = np.zeros(n_all, dtype=bool)
    pos = {i: p for p, i in enumerate(words.index.to_numpy())}

    for _, g in words.groupby(["task", "session", "run"], sort=False):
        g = g.sort_values("onset_s")
        onsets = g["onset_s"].to_numpy()
        durs = g["duration_s"].to_numpy()
        in_v = g["in_vocab"].to_numpy() if "in_vocab" in g.columns else np.ones(len(g), dtype=bool)
        ends = onsets + np.where(np.isfinite(durs), durs, 0.0)
        g_index = g.index.to_numpy()
        n = len(g)
        for j in range(n):
            p = pos[g_index[j]]
            t0 = onsets[j]
            t1 = t0 + win_s
            if j + 1 < n:
                isi_next[p] = onsets[j + 1] - t0
                next_in_vocab[p] = bool(in_v[j + 1])
            later = (onsets > t0) & (onsets < t1)
            n_follow[p] = int(later.sum())
            n_follow_iv[p] = int((later & in_v).sum())
            dur = durs[j]
            own_s[p] = float(np.clip(dur, 0, win_s)) if np.isfinite(dur) else np.nan
            hit = (ends > t0) & (onsets < t1)
            if not hit.any():
                speech_s[p] = 0.0
                continue
            lo = np.maximum(onsets[hit], t0)
            hi = np.minimum(ends[hit], t1)
            order = np.argsort(lo)
            lo, hi = lo[order], hi[order]
            total = 0.0
            cs, ce = lo[0], hi[0]
            for a, b in zip(lo[1:], hi[1:]):
                if a <= ce:
                    ce = max(ce, b)
                else:
                    total += ce - cs
                    cs, ce = a, b
            total += ce - cs
            speech_s[p] = total

    out = words.copy()
    out["isi_next_s"] = isi_next
    out["n_following_in_win"] = n_follow
    out["n_following_in_vocab"] = n_follow_iv
    out["next_in_vocab"] = next_in_vocab
    out["own_speech_s"] = own_s
    out["union_speech_s"] = speech_s
    out["own_frac"] = out["own_speech_s"] / win_s
    out["foreign_speech_s"] = np.clip(out["union_speech_s"] - out["own_speech_s"].fillna(0), 0, None)
    out["isolated_1s"] = out["isi_next_s"] >= win_s
    return out


def combined_rms_z(mag_rms: np.ndarray, grad_rms: np.ndarray, fit: dict | None = None) -> tuple[np.ndarray, dict]:
    """Train-fitted type-specific MAD-z; combine with max(|z_mag|, |z_grad|)."""
    mag_rms = np.asarray(mag_rms, dtype=np.float64)
    grad_rms = np.asarray(grad_rms, dtype=np.float64)
    if fit is None:
        fit = {
            "mag_med": mad_raw(mag_rms)[0],
            "mag_mad": mad_raw(mag_rms)[1],
            "grad_med": mad_raw(grad_rms)[0],
            "grad_mad": mad_raw(grad_rms)[1],
        }
    z_m = mad_z(mag_rms, fit["mag_med"], fit["mag_mad"])
    z_g = mad_z(grad_rms, fit["grad_med"], fit["grad_mad"])
    return np.maximum(np.abs(z_m), np.abs(z_g)), fit


def load_drop_keys(path: Path) -> set[tuple]:
    keys: set[tuple] = set()
    with path.open() as f:
        for row in csv.DictReader(f):
            keys.add(
                trial_key(row["task"], row["session"], row["run"], row["onset_s"], row["word"])
            )
    return keys


def filter_vocab_keep(base, keep: list[tuple[int, int]], drop_keys: set[tuple]) -> list[tuple[int, int]]:
    out = []
    for base_idx, vid in keep:
        if sample_trial_key(base.samples[base_idx]) not in drop_keys:
            out.append((base_idx, vid))
    return out


def write_drop_list(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["split", "task", "session", "run", "onset_s", "word", "reasons"]
    if not rows:
        pd.DataFrame(columns=cols).to_csv(path, index=False)
        return
    pd.DataFrame(rows)[cols].to_csv(path, index=False)


class Reservoir:
    def __init__(self, k: int, rng: np.random.Generator):
        self.k = k
        self.rng = rng
        self.items: list = []
        self.seen = 0

    def add(self, item) -> None:
        self.seen += 1
        if len(self.items) < self.k:
            self.items.append(item)
            return
        j = int(self.rng.integers(0, self.seen))
        if j < self.k:
            self.items[j] = item
