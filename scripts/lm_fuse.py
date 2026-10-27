"""MEG x LLM sentence fusion for the Deep holdout (no MEG model needed).

Every one of the 868 holdout sentence rows is a primary-vocab word and the
156 sentences are 4-9 words long, i.e. the stimuli are short sentences built
only from the 50 competition words. Decoding each row independently throws
away that constraint. This script takes per-row MEG probabilities (member
CSVs, averaged in log space) and rescores every sentence with a pretrained
causal LM:

    S(i, w) = log p_MEG(w | row i) + lam * log p_LM(sentence with w at i | ctx)

where ctx is a set of beam contexts for the other positions. Coordinate
sweeps (ICM) refine the contexts; the final row score is a log-sum-exp over
the context beam, so the ranking at position i uses left *and* right context.
Isolated rows (source == word) keep their MEG scores.

    python scripts/lm_fuse.py --members a.csv b.csv c.csv --lam 0.5 \\
        --out submissions/submission_lmfuse_lam0.5.csv
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]


def render(word: str, first: bool) -> str:
    w = word.replace("’", "'")
    if w == "i":
        w = "I"
    elif w.startswith("i'") or w == "i’m":
        w = "I" + w[1:]
    if first:
        w = w[0].upper() + w[1:]
    return w


STYLE = {"prefix": "", "lower": False}

# Competition paper (arXiv 2609.03231, App. B): the vocabulary was chosen to
# form utterances like these. Used as a few-shot prefix so the LM models the
# holdout's phrase style instead of generic prose.
FEWSHOT = ("Short everyday spoken phrases:\n"
           "i think so.\nwe have time.\ni am not good at all.\ncan i have this.\nwe can do this.\n")


def sentence_text(words: list[str]) -> str:
    if STYLE["lower"]:
        return STYLE["prefix"] + " ".join(w.replace("\u2019", "'") for w in words) + "."
    return STYLE["prefix"] + " ".join(render(w, k == 0) for k, w in enumerate(words)) + "."


class LMScorer:
    def __init__(self, name: str, device: str, batch: int = 512, dtype: str = "bf16"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # LM_REVISION pins the HF snapshot (final submission: Qwen2.5-1.5B @ 8faed761d45a263340a0528343f099c05c9a4323)
        rev = os.environ.get("LM_REVISION") or None
        self.tok = AutoTokenizer.from_pretrained(name, revision=rev)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            name, revision=rev, dtype=torch.float32 if dtype == "fp32" else torch.bfloat16
        ).to(device).eval()
        self.device = device
        self.batch = batch
        bos = self.tok.bos_token_id
        if bos is None:
            bos = self.tok.eos_token_id
        self.bos = bos
        self.cache: dict[str, float] = {}
        self.prefix, self.pcache = "", None

    def set_prefix(self, prefix: str) -> None:
        """Shared text prefix (few-shot prompt): encoded once, its KV cache is
        reused for every candidate; its own log-prob is a constant and skipped.
        The prefix must end on a token boundary (it ends with a newline)."""
        self.prefix = prefix
        self.pcache = None
        if not prefix:
            return
        ids = [self.bos] + self.tok(prefix, add_special_tokens=False)["input_ids"]
        x = torch.tensor([ids], device=self.device)
        with torch.no_grad():
            out = self.model(input_ids=x, use_cache=True)
        self.pcache = out.past_key_values
        self.plen = len(ids)
        self.p_last = torch.log_softmax(out.logits[0, -1].float(), dim=-1)

    @torch.no_grad()
    def score(self, texts: list[str]) -> np.ndarray:
        """Total log p(text | BOS [+ prefix]) for each text (memoised)."""
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        for s in range(0, len(todo), self.batch):
            chunk = todo[s : s + self.batch]
            if self.pcache is None:
                ids = [[self.bos] + self.tok(t, add_special_tokens=False)["input_ids"] for t in chunk]
            else:
                ids = [self.tok(t[len(self.prefix):], add_special_tokens=False)["input_ids"] for t in chunk]
            n = max(map(len, ids))
            x = torch.full((len(ids), n), self.tok.pad_token_id, dtype=torch.long)
            m = torch.zeros((len(ids), n), dtype=torch.long)
            for k, seq in enumerate(ids):
                x[k, : len(seq)] = torch.tensor(seq)
                m[k, : len(seq)] = 1
            x, m = x.to(self.device), m.to(self.device)
            if self.pcache is None:
                logits = self.model(input_ids=x, attention_mask=m).logits.float()
                lp = torch.log_softmax(logits[:, :-1], dim=-1)
                tok_lp = lp.gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1) * m[:, 1:]
                tot = tok_lp.sum(-1)
            else:
                cache = copy.deepcopy(self.pcache)
                cache.batch_repeat_interleave(len(ids))
                attn = torch.cat([torch.ones(len(ids), self.plen, dtype=m.dtype, device=m.device), m], 1)
                pos = (self.plen + torch.arange(n, device=x.device)).unsqueeze(0).expand(len(ids), -1)
                logits = self.model(input_ids=x, attention_mask=attn, past_key_values=cache,
                                    position_ids=pos, use_cache=True).logits.float()
                lp = torch.log_softmax(logits[:, :-1], dim=-1)
                tok_lp = lp.gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1) * m[:, 1:]
                tot = tok_lp.sum(-1) + self.p_last[x[:, 0]]
            for t, v in zip(chunk, tot.cpu().numpy()):
                self.cache[t] = float(v)
        return np.array([self.cache[t] for t in texts])


class MultiLM:
    """LM ensemble: mean of the members' sentence log-probs (comma-separated --lm)."""

    def __init__(self, names: list[str], device: str, batch: int = 512, dtype: str = "bf16"):
        self.lms = [LMScorer(n, device, batch=batch, dtype=dtype) for n in names]
        self.cache: dict[str, float] = {}

    def set_prefix(self, prefix: str) -> None:
        for lm in self.lms:
            lm.set_prefix(prefix)

    def score(self, texts: list[str]) -> np.ndarray:
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if todo:
            v = np.mean([lm.score(todo) for lm in self.lms], axis=0)
            for lm in self.lms:
                lm.cache.clear()
            self.cache.update(zip(todo, v.tolist()))
        return np.array([self.cache[t] for t in texts])


def load_members(paths: list[Path]) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    frames = [pd.read_csv(p) for p in paths]
    vocab = list(frames[0].columns[1:51])
    logs = []
    for f in frames:
        p = f[vocab].values.astype(np.float64)
        p = np.clip(p, 1e-8, None)
        p = p / p.sum(1, keepdims=True)
        logs.append(np.log(p))
    return frames[0], vocab, np.mean(logs, axis=0)


def fuse_sentence(meg: np.ndarray, vocab: list[str], lm: LMScorer, lam: float,
                  beam: int, sweeps: int, prior: np.ndarray | None, beta: float) -> np.ndarray:
    """meg (L, V) log-probs -> fused (L, V) scores."""
    L, V = meg.shape
    base = meg.copy()
    if prior is not None and beta:
        base = base - beta * prior[None, :]
    # beam of contexts: start from MEG top sequences (independent top-k per position)
    order = np.argsort(-meg, axis=1)
    ctxs = [list(order[:, 0])]
    for k in range(1, beam):
        c = list(order[:, 0])
        i = k % L
        c[i] = order[i, 1 + (k // L) % (V - 1)]
        ctxs.append(c)
    ctx_w = np.zeros(len(ctxs))
    fused = base.copy()
    for _sweep in range(sweeps):
        for i in range(L):
            texts = []
            for c in ctxs:
                for w in range(V):
                    seq = list(c)
                    seq[i] = w
                    texts.append(sentence_text([vocab[j] for j in seq]))
            lmv = lm.score(texts).reshape(len(ctxs), V)
            # score of each context-with-w: base terms of the other positions are
            # constant within a context, so keep them in ctx_w for the beam mix
            other = np.array([sum(base[j, c[j]] for j in range(L) if j != i) for c in ctxs])
            tot = other[:, None] + base[i][None, :] + lam * lmv  # (C, V)
            m = tot.max()
            fused[i] = np.log(np.exp(tot - m).sum(0)) + m - np.log(len(ctxs))
            # refresh the beam: best C (context, w) pairs, distinct sequences
            flat = np.argsort(-tot, axis=None)
            new, seen = [], set()
            for f in flat:
                ci, w = divmod(int(f), V)
                seq = list(ctxs[ci])
                seq[i] = w
                key = tuple(seq)
                if key in seen:
                    continue
                seen.add(key)
                new.append((seq, tot[ci, w]))
                if len(new) >= beam:
                    break
            ctxs = [s for s, _ in new]
            ctx_w = np.array([v for _, v in new])
    return fused


def build_lattice(meg: np.ndarray, mass: float = 0.97, kmax: int = 8, cap: int = 30000):
    """Per-position candidate sets (top-k covering `mass` of the MEG row, k <= kmax),
    shrunk until the number of paths is <= cap. Returns (choices list, paths (P, L))."""
    L, V = meg.shape
    p = np.exp(meg - meg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    order = np.argsort(-p, axis=1)
    ks = []
    for i in range(L):
        c = np.cumsum(p[i, order[i]])
        ks.append(int(min(kmax, max(2, np.searchsorted(c, mass) + 1))))
    while int(np.prod(ks)) > cap:
        j = int(np.argmax(ks))
        ks[j] -= 1
    choices = [order[i, : ks[i]] for i in range(L)]
    grids = np.meshgrid(*[np.arange(k) for k in ks], indexing="ij")
    idx = np.stack([g.reshape(-1) for g in grids], 1)
    paths = np.stack([choices[i][idx[:, i]] for i in range(L)], 1)
    return choices, paths


def lattice_lm_scores(paths: np.ndarray, vocab: list[str], lm: "LMScorer") -> np.ndarray:
    return lm.score([sentence_text([vocab[j] for j in row]) for row in paths])


def lattice_marginals(meg: np.ndarray, choices, paths: np.ndarray, lmv: np.ndarray, lam: float) -> np.ndarray:
    """Exact row marginals over the pruned lattice (MBR for a per-row loss).
    Lattice words redistribute their MEG mass by the path posterior; words outside
    the lattice keep their MEG probability (LM factor treated as neutral)."""
    L, V = meg.shape
    p = np.exp(meg - meg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    score = meg[np.arange(L)[None, :], paths].sum(1) + lam * lmv
    post = np.exp(score - score.max()); post /= post.sum()
    out = np.log(np.clip(p, 1e-12, None))
    for i in range(L):
        m = np.zeros(V)
        np.add.at(m, paths[:, i], post)
        inside = choices[i]
        lat_mass = p[i, inside].sum()
        out[i, inside] = np.log(np.clip(m[inside] * lat_mass, 1e-12, None))
    return out


def lattice_variant(meg, choices, paths, lmv, lam, variant, vocab=None, lm=None,
                    T=3.0, tail_c=0.5, n_ctx=8):
    """Lattice decoders screened in the simulator (reviewer 2026-10-01).
      lat   : exact marginals (lattice_marginals)
      v1    : top-n_ctx lattice paths as contexts; ALL 50 words rescored per position
              (ICM-style logsumexp over contexts, no coordinate sweeps)
      v23   : path posterior at temperature T + tail words penalised by
              lam*(lm_min - lm_mean)*tail_c
      v5    : max-marginals for lattice words, tail = log p + lam*lm_min
    """
    L, V = meg.shape
    if variant == "lat":
        return lattice_marginals(meg, choices, paths, lmv, lam)
    p = np.exp(meg - meg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    logp = np.log(np.clip(p, 1e-8, None))
    score = meg[np.arange(L)[None, :], paths].sum(1) + lam * lmv
    if variant == "v1":
        top = np.argsort(-score)[:n_ctx]
        ctxs = [list(paths[t]) for t in top]
        out = np.empty((L, V))
        for i in range(L):
            texts = []
            for c in ctxs:
                for w in range(V):
                    seq = list(c); seq[i] = w
                    texts.append(sentence_text([vocab[j] for j in seq]))
            lmw = lm.score(texts).reshape(len(ctxs), V)
            other = np.array([sum(meg[j, c[j]] for j in range(L) if j != i) for c in ctxs])
            tot = other[:, None] + meg[i][None, :] + lam * lmw
            m = tot.max()
            out[i] = np.log(np.exp(tot - m).sum(0)) + m - np.log(len(ctxs))
        return out
    if variant == "v23":
        post = np.exp((score - score.max()) / T); post /= post.sum()
        out = logp + lam * (lmv.min() - lmv.mean()) * tail_c
        for i in range(L):
            m = np.zeros(V); np.add.at(m, paths[:, i], post)
            inside = choices[i]
            out[i, inside] = np.log(np.clip(m[inside] * p[i, inside].sum(), 1e-8, None))
        return out
    if variant == "v5":
        out = logp + lam * lmv.min()
        for i in range(L):
            best = np.full(V, -np.inf)
            np.maximum.at(best, paths[:, i], score)
            inside = choices[i]
            # put max-marginals on the MEG log-prob scale: shift so the best path's word keeps its own score
            out[i, inside] = best[inside] - score.max() + logp[i, inside].max()
        return out
    raise ValueError(variant)


def fuse_sentence_lattice(meg, vocab, lm, lam, mass=0.97, kmax=8, cap=30000):
    choices, paths = build_lattice(meg, mass, kmax, cap)
    lmv = lattice_lm_scores(paths, vocab, lm)
    return lattice_marginals(meg, choices, paths, lmv, lam)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--members", type=Path, nargs="+", required=True)
    ap.add_argument("--lm", default="Qwen/Qwen2.5-1.5B", help="HF name; comma-separated names = LM ensemble (mean log-prob)")
    ap.add_argument("--lam", type=float, nargs="+", default=[0.5])
    ap.add_argument("--beta", type=float, default=0.0,
                    help="subtract beta * log LM unigram prior (balanced-accuracy correction)")
    ap.add_argument("--gamma", type=float, nargs="+", default=[0.0],
                    help="PMI correction: subtract gamma*lam*log q(w), q = LM's mean vocab distribution over the holdout's own positions")
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--decoder", choices=["icm", "lattice"], default="icm")
    ap.add_argument("--lm-cache", type=Path, default=None,
                    help="pickle of text->log p (keyed by LM name + prompt); loaded and saved back")
    ap.add_argument("--lat-mass", type=float, default=0.97)
    ap.add_argument("--lat-kmax", type=int, default=8)
    ap.add_argument("--lat-cap", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--pre-alpha", type=float, default=0.0,
                    help="holdout-prior correction of the MEG log-probs before fusion (see prior_correct.py)")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--fewshot", action="store_true", help="few-shot phrase-style prefix, lowercase rendering")
    ap.add_argument("--prefix-file", type=Path, default=None,
                    help="LM prompt used verbatim (lowercase rendering, must end with a newline); overrides --fewshot/--fewshot-file")
    ap.add_argument("--fewshot-file", type=Path, default=None,
                    help="one example phrase per line (replaces the built-in 5 paper examples)")
    ap.add_argument("--sweeps", type=int, default=2)
    ap.add_argument("--extra-logp", type=Path, default=None, help="(rows, 50) .npy added to the MEG log-probs")
    ap.add_argument("--extra-w", type=float, default=1.0)
    ap.add_argument("--nbr-next", type=float, default=0.0, help="weight of the next row's centred log-probs")
    ap.add_argument("--nbr-prev", type=float, default=0.0, help="weight of the previous row's centred log-probs")
    ap.add_argument("--holdout-dir", type=Path, default=ROOT / "data" / "holdout")
    ap.add_argument("--out-prefix", type=str, default="submissions/submission_lmfuse")
    ap.add_argument("--limit", type=int, default=0, help="first N sentences only (smoke)")
    args = ap.parse_args()

    from pnpl.competition import LibriBrainCompetitionHoldout

    if args.fewshot or args.fewshot_file:
        prefix = FEWSHOT
        if args.fewshot_file:
            lines = [l.strip() for l in args.fewshot_file.read_text().splitlines() if l.strip()]
            prefix = "Short everyday spoken phrases:\n" + "\n".join(lines) + "\n"
        STYLE.update(prefix=prefix, lower=True)
    if args.prefix_file:
        prefix = args.prefix_file.read_text()
        assert prefix.endswith("\n"), "--prefix-file must end with a newline (token boundary for the KV cache)"
        STYLE.update(prefix=prefix, lower=True)
    h = LibriBrainCompetitionHoldout(track="deep", data_path=str(args.holdout_dir), download=False)
    md = h.metadata
    df, vocab, meg = load_members(args.members)
    if args.nbr_next or args.nbr_prev:
        # explain away the neighbouring words' responses inside each window: add
        # a * (row-centred log-probs of the next row) + b * (... of the previous row),
        # same sentence only (neighbor_stack_check.py: a -0.2, b -0.1 on every LOFO fold)
        meg0 = meg - np.logaddexp.reduce(meg, axis=1, keepdims=True)
        cen = meg0 - meg0.mean(1, keepdims=True)
        adj = np.zeros_like(meg)
        by_ep: dict[int, list[int]] = {}
        for r, m in enumerate(md):
            if m["source"] == "sentence":
                by_ep.setdefault(m["epoch"], []).append(r)
        for rows in by_ep.values():
            rows = sorted(rows, key=lambda r: md[r]["word"])
            for k, r in enumerate(rows):
                if k + 1 < len(rows):
                    adj[r] += args.nbr_next * cen[rows[k + 1]]
                if k > 0:
                    adj[r] += args.nbr_prev * cen[rows[k - 1]]
        meg = meg0 + adj
        meg = meg - np.logaddexp.reduce(meg, axis=1, keepdims=True)
        print(f"neighbour explain-away next {args.nbr_next} prev {args.nbr_prev}", flush=True)
    if args.pre_alpha:
        p = np.exp(meg - meg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        pi = p[:868].mean(0)
        pi_iso = p[868:].mean(0) if len(p) > 868 else pi
        meg = meg.copy()
        meg[:868] -= args.pre_alpha * np.log(pi)[None, :]
        # isolated rows (private-only, ~uniform class mix) get their own prior
        meg[868:] -= args.pre_alpha * np.log(pi_iso)[None, :]
        meg = meg - np.logaddexp.reduce(meg, axis=1, keepdims=True)
        print(f"pre-fusion prior correction alpha {args.pre_alpha}: pi max {pi.max():.3f} min {pi.min():.4f}", flush=True)
    if args.extra_logp:
        # extra per-row evidence (e.g. timing_prior.py log-likelihood ratios), added after the
        # pre-fusion prior correction and renormalised
        ex = np.load(args.extra_logp)
        assert ex.shape == meg.shape, (ex.shape, meg.shape)
        meg = meg + args.extra_w * ex
        meg = meg - np.logaddexp.reduce(meg, axis=1, keepdims=True)
        print(f"extra evidence {args.extra_logp} weight {args.extra_w}", flush=True)
    assert len(df) == len(md), (len(df), len(md))
    assert list(df["index"]) == [m["index"] for m in md], "row order mismatch"

    groups: dict[int, list[int]] = {}
    for r, m in enumerate(md):
        if m["source"] == "sentence":
            groups.setdefault(m["epoch"], []).append(r)
    sents = [sorted(v, key=lambda r: md[r]["word"]) for _, v in sorted(groups.items())]
    if args.limit:
        sents = sents[: args.limit]
    print(f"sentences {len(sents)} rows {sum(map(len, sents))}  members {len(args.members)}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if "," in args.lm:
        lm = MultiLM(args.lm.split(","), dev, batch=args.batch, dtype=args.dtype)
    else:
        lm = LMScorer(args.lm, dev, batch=args.batch, dtype=args.dtype)
    lm.set_prefix(STYLE["prefix"])
    cache_key = (args.lm, STYLE["prefix"], args.dtype)
    if args.lm_cache and args.lm_cache.exists():
        import pickle
        allc = pickle.loads(args.lm_cache.read_bytes())
        lm.cache.update(allc.get(cache_key, {}))
        print(f"lm cache: loaded {len(lm.cache)} entries", flush=True)
    prior = None
    if args.beta:
        prior = lm.score([sentence_text([w]) for w in vocab])
        prior = prior - np.logaddexp.reduce(prior)

    logq = None
    if any(args.gamma):
        # LM word prior at the holdout's positions: average over positions of the
        # vocab-renormalised LM distribution given the MEG top-1 context.
        acc = np.zeros(len(vocab)); cnt = 0
        for rows in sents:
            top = list(meg[rows].argmax(1))
            for i in range(len(rows)):
                texts = []
                for w in range(len(vocab)):
                    seq = list(top); seq[i] = w
                    texts.append(sentence_text([vocab[j] for j in seq]))
                v = lm.score(texts)
                acc += np.exp(v - np.logaddexp.reduce(v)); cnt += 1
        q = acc / cnt
        logq = np.log(q)
        print("LM prior q top:", {vocab[j]: round(float(q[j]), 3) for j in np.argsort(-q)[:10]}, flush=True)

    for lam in args.lam:
      for gamma in args.gamma:
        out = meg.copy()
        bias = None if logq is None else gamma * lam * logq
        for k, rows in enumerate(sents):
            m_in = meg[rows] if bias is None else meg[rows] - bias[None, :]
            if args.decoder == "lattice":
                out[rows] = fuse_sentence_lattice(m_in, vocab, lm, lam, args.lat_mass, args.lat_kmax, args.lat_cap)
            else:
                out[rows] = fuse_sentence(m_in, vocab, lm, lam, args.beam, args.sweeps, prior, args.beta)
            if k % 50 == 0:
                print(f"lam {lam} gamma {gamma} sent {k}/{len(sents)} cache {len(lm.cache)}", flush=True)
        # write probabilities (softmax per row) into the primary block; keep other columns
        sub = df.copy()
        z = out - out.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        sub[vocab] = p
        tag = "" if args.decoder == "icm" else "_lat"
        path = Path(f"{args.out_prefix}{tag}_lam{lam}_g{gamma}.csv")
        sub.to_csv(path, index=False)
        chg = (out.argmax(1) != meg.argmax(1)).mean()
        top1 = pd.Series([vocab[j] for j in out.argmax(1)]).value_counts().head(8).to_dict()
        print(f"wrote {path}  top1 changed {chg:.3f}  top1 dist {top1}", flush=True)
        # a few decoded sentences for eyeballing
        for rows in sents[:6]:
            print("   ", " ".join(vocab[j] for j in out[rows].argmax(1)),
                  "| meg:", " ".join(vocab[j] for j in meg[rows].argmax(1)))
    if args.lm_cache:
        save_lm_cache(args.lm_cache, cache_key, lm.cache)
        print(f"lm cache: saved {len(lm.cache)} entries -> {args.lm_cache}", flush=True)


def save_lm_cache(path, key, cache) -> None:
    import pickle
    allc = pickle.loads(path.read_bytes()) if path.exists() else {}
    allc.setdefault(key, {}).update(cache)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(allc))
    tmp.replace(path)


if __name__ == "__main__":
    sys.exit(main())
