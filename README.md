# PNPL Competition 2026, Deep track: Munich Logic Lab submission for verification

| | |
| --- | --- |
| Team | Munich Logic Lab (Kaggle team id 16781466) |
| Track | Deep (`pnpl-competition-2026-deep`) |
| Final submission | `submission/r6_pc1.5.csv`, Kaggle submission ref **56863120** (2026-10-05) |
| Score | private **0.74333** BAcc@10 (rank 5) / public 0.76107 |
| sha256 of the submitted CSV | `a084cb9c1a3fa67b5f1e…` (full hash in `SHA256SUMS`) |

This repository contains the checkpoints, the inference pipeline that turns them into the submitted CSV,
and the training code that produced the checkpoints.

## Method in one paragraph

Six MEG word classifiers ("hybrid TCN") are ensembled. Each classifies the 50 Deep vocabulary words from
1-s MEG windows (306 channels, 250 Hz, z-scored with a frozen training-set channel scaler, clipped at ±10).
A temporal-convolution encoder embeds every word window; a small transformer *mixer* attends across the word
windows of the same holdout sentence (sentence context is permitted for the sentence rows). Two heads, cross-entropy over the 50 words and a
SigLIP retrieval head against T5-large word embeddings, are blended 50/50. The member log-probabilities
are averaged and then decoded jointly per sentence:

1. **pre-fusion prior correction** (label-free): `log p − 0.5·log π̂`, where π̂ is the mean predicted class
   distribution over the 868 sentence rows (the 92 isolated rows use their own π̂);
2. **language-model fusion**: Qwen2.5-1.5B (revision `8faed761d45a263340a0528343f099c05c9a4323`, fp32),
   scores restricted to the 50 vocabulary words, prompted with the five example phrases from the competition
   paper's Appendix B (`i think so. / we have time. / i am not good at all. / can i have this. / we can do this.`),
   iterated conditional modes with beam 8 and 2 sweeps, weight λ = 0.25;
3. **post-fusion prior correction** (label-free, EM-refined π̂, Saerens et al. 2002) with α = 1.5.

No holdout labels are used anywhere. Steps 1 and 3 use only the unlabelled model predictions on the holdout
(transductive class-prior estimation). No information is mapped between the Deep and Broad tracks.

## Members

| checkpoint | family | params | how it was trained (seeds) |
| --- | --- | ---: | --- |
| `small_mixed_epoch=01.ckpt` | small (dim 100, 20 TCN blocks, avg pool, mixer 2×4) | 2.0 M | A: Sherlock from scratch (seed 0, best val epoch) → B: all corpora, epoch 1 (seed 0) |
| `small_s1_mixed_epoch=01.ckpt` | small | 2.0 M | A seed 1 → B seed 1 |
| `v2re12_found_ep10.ckpt` | v2re12 (dim 256, 24 TCN blocks, attn pool 4q, mixer 3×8, bf16) | 13.9 M | foundation: Sherlock from scratch, 12-epoch schedule, epoch-10 snapshot (seed 1) |
| `tcn_hybrid_v2re12_plain_nc-9-5-3_epoch=01.ckpt` | v2re12 | 13.9 M | plain adaptation of the foundation above on all corpora, epoch 1 (seed 0) |
| `v2re12s2_found_ep10.ckpt` | v2re12 | 13.9 M | foundation seed 2 |
| `tcn_hybrid_v2re12s2_plain_nc-9-3-3_epoch=01.ckpt` | v2re12 | 13.9 M | plain adaptation seed 2 |

Training data: the released LibriBrain training data only, loaded through `pnpl.datasets.LibriBrain100Word`:
Sherlock 1–9 for stage A / foundation, plus MOCHA, TIMIT and podcasts / The Moth (all stimulus families)
for stage B / plain adaptation. Validation uses the released validation splits (TIMIT ses-14 is held out as
validation). Channel scaler and bad-segment drop lists were computed from the training data
(`artifacts/data_prep/`).

The checkpoints are Lightning checkpoints with the optimizer state removed (`state_dict` + `hyper_parameters`);
they load with `LitHybridTCN.load_from_checkpoint` and give byte-identical predictions to the full checkpoints.

## Layout

```
checkpoints/            6 member checkpoints (weights + hyper-parameters)
submission/r6_pc1.5.csv the submitted CSV;  submission/members/  per-member holdout probabilities
scripts/reproduce_inference.sh   checkpoints -> submitted CSV, byte-for-byte check
scripts/build_final.py           member inference -> ensemble -> LM fusion -> prior correction
scripts/submit_tcn_hybrid_holdout.py, lm_fuse.py, prior_correct.py   the three pipeline stages
scripts/train_repro6.sh          trains the six members from scratch (exact flags + seeds)
scripts/train_tcn_hybrid.py      training script;  scripts/preload_corpus_all.py  corpus download
src/                    model (hybrid_tcn.py), data handling, T5 text embeddings
artifacts/data_prep/    frozen training-set channel scaler and drop lists
assets/t5_cache/        cached T5-large word embeddings (avoids re-embedding the vocabulary)
requirements.txt / requirements-lock.txt   pinned environment (Python 3.12, CUDA 12.8 wheels)
```

## Reproduce the submission from the checkpoints (≈15 min on one GPU)

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-lock.txt
PY=.venv/bin/python bash scripts/reproduce_inference.sh
```

The script downloads the Deep holdout (`pnpl`, into `data/holdout/`) and Qwen2.5-1.5B (HF hub, pinned
revision), runs the six members on the holdout, fuses, corrects, and compares the result with
`submission/r6_pc1.5.csv` byte for byte. Needs one CUDA GPU with ≥ 12 GB (Qwen in fp32).

## Train the members from scratch

```bash
.venv/bin/python scripts/preload_corpus_all.py        # LibriBrain training corpus (~100 GB) -> data/libribrain_word
mkdir -p data/libribrain_word/t5_cache && cp assets/t5_cache/* data/libribrain_word/t5_cache/
PY=.venv/bin/python bash scripts/train_repro6.sh small 0 0
PY=.venv/bin/python bash scripts/train_repro6.sh small 1 1
PY=.venv/bin/python bash scripts/train_repro6.sh v2re12 1 0
PY=.venv/bin/python bash scripts/train_repro6.sh v2re12 2 2
```

Checkpoints land in `ckpts_retrain/`; pass them to `scripts/build_final.py` as in `reproduce_inference.sh`.
Wall time on one RTX 5090: ≈1.5 h per call. GPU training is not bit-deterministic, so retrained members
differ slightly from the shipped checkpoints.
