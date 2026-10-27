"""Transductive prior correction for the class-balanced BAcc@10 metric.

Balanced recall is maximised by ranking classes on p(x | c) ∝ p(c | x) / pi_c,
with pi_c the class prior of the scored rows. pi is estimated label-free as
the mean predicted distribution over the holdout's sentence rows (iterated a
few times, EM-style, Saerens et al. 2002), and the rows are rescored with

    score = log p(c | x) - alpha * log pi_c

alpha = 0 leaves the CSV unchanged; alpha = 1 is the full correction.
Isolated rows (index >= 868) get the same correction.

    python scripts/prior_correct.py in.csv --alpha 0.5 --out out.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def balance_occupancy(logp: np.ndarray, pi: np.ndarray, k: int = 10, iters: int = 60,
                      floor: float = 4.0, step: float = 0.5) -> np.ndarray:
    """Per-class biases so class c appears in ~k * max(N*pi_c, floor) top-k lists
    (balanced recall@k needs every class in at least ~its count of lists). Label-free."""
    n, V = logp.shape
    target = np.maximum(n * pi, floor)
    target = target / target.sum() * n * k
    b = np.zeros(V)
    for _ in range(iters):
        top = np.argpartition(-(logp + b), k, axis=1)[:, :k]
        cnt = np.bincount(top.ravel(), minlength=V) + 0.5
        b += step * (np.log(target) - np.log(cnt))
    out = logp + b
    return out - np.logaddexp.reduce(out, axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--alpha", type=float, nargs="+", default=[0.5])
    ap.add_argument("--n-sentence-rows", type=int, default=868)
    ap.add_argument("--em-iters", type=int, default=5)
    ap.add_argument("--iso", choices=["shared", "own", "none"], default="own",
                    help="isolated rows (index >= n-sentence-rows): shared = sentence pi (historic), "
                         "own = pi estimated on the isolated rows themselves, none = leave uncorrected")
    ap.add_argument("--iso-shrink", type=float, default=0.0,
                    help="with --iso own: pi_iso = s*uniform + (1-s)*mean (92 rows give a noisy prior)")
    ap.add_argument("--occupancy", action="store_true",
                    help="after the prior correction, balance top-10 occupancy to k*max(N*pi, 4) per class")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    vocab = list(df.columns[1:51])
    p = np.clip(df[vocab].values.astype(np.float64), 1e-12, None)
    p /= p.sum(1, keepdims=True)
    logp = np.log(p)
    for alpha in args.alpha:
        # EM estimate of the holdout prior under a uniform-trained view:
        # pi <- mean_x softmax(log p - log pi_train + log pi); start from the plain mean.
        pi = p[: args.n_sentence_rows].mean(0)
        for _ in range(args.em_iters):
            q = logp[: args.n_sentence_rows] - alpha * np.log(pi)
            q = np.exp(q - q.max(1, keepdims=True))
            q /= q.sum(1, keepdims=True)
            pi = 0.5 * pi + 0.5 * q.mean(0)
        s = logp - alpha * np.log(pi)
        n = args.n_sentence_rows
        if len(logp) > n and args.iso != "shared":
            # The 92 isolated rows are private-only and have their own class mix
            # (~uniform: 92 rows / 50 words); the sentence prior does not apply.
            if args.iso == "none":
                s[n:] = logp[n:]
            else:
                pi_iso = p[n:].mean(0)
                pi_iso = args.iso_shrink / pi_iso.size + (1 - args.iso_shrink) * pi_iso
                s[n:] = logp[n:] - alpha * np.log(pi_iso)
        if args.occupancy:
            s[:n] = balance_occupancy(s[:n], pi)
        s = np.exp(s - s.max(1, keepdims=True))
        s /= s.sum(1, keepdims=True)
        out = df.copy()
        out[vocab] = s
        od = args.out_dir or args.csv.parent
        path = od / f"{args.csv.stem}_pc{alpha}{'' if args.iso == 'own' else '_iso' + args.iso}{'_occ' if args.occupancy else ''}.csv"
        out.to_csv(path, index=False)
        top1 = pd.Series(np.array(vocab)[s[: args.n_sentence_rows].argmax(1)]).value_counts()
        print(f"alpha {alpha}: wrote {path.name}  top1 max-class share {top1.iloc[0]}/{args.n_sentence_rows} "
              f"({top1.index[0]})  classes as top1: {len(top1)}  pi max {pi.max():.3f} min {pi.min():.4f}")


if __name__ == "__main__":
    main()
