#!/usr/bin/env bash
# Train the six REPRO6 members from scratch (final Deep submission r6_pc1.5,
# HANDOVER §21.1 / §25). Flags and seeds are the ones recorded in the original
# training logs (~/mirror/<host>/hybrid_*.log) and checkpoint hyper_parameters.
#
#   small trio member  = stage A (Sherlock, from scratch, best val epoch) -> stage B (corpus=all) epoch 1
#   v2re12 pair        = foundation (Sherlock, 12 epochs) epoch 10        -> plain adapt (corpus=all) epoch 1
#
#   member                      stage A / foundation seed   stage B / plain seed
#   small_mixed_epoch=01        0                           0
#   small_s1_mixed_epoch=01     1                           1
#   v2re12_found_ep10 + plain1  1                           0
#   v2re12s2_found_ep10 + plain1 2                          2
#
# Usage (one GPU, two streams in parallel fit in 32 GB):
#   bash scripts/train_repro6.sh small 0 0        # -> $OUT/small_a0_b0_mixed_epoch=01.ckpt
#   bash scripts/train_repro6.sh small 1 1
#   bash scripts/train_repro6.sh v2re12 1 0       # -> $OUT/v2re12_f1_found_ep10.ckpt, $OUT/v2re12_f1_p0_plain_epoch=01.ckpt
#   bash scripts/train_repro6.sh v2re12 2 2
# Wall time on an RTX 5090: small ~1.5 h, v2re12 pair ~1.5 h. Re-running a call skips stages whose
# log already reports "status:" (e.g. after a CUDA OOM in a later stage).
set -euo pipefail
KIND=$1; SA=$2; SB=$3
OUT=${OUT:-$PWD/ckpts_retrain}; mkdir -p "$OUT"
PY=${PY:-python}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="--full-epoch --num-workers 4 --tb-port 6008"
ckdir() { ls -d "lightning_logs/$1"/version_* | sort -V | tail -1; }
done_run() { grep -q "^status:" "hybrid_$1.log" 2>/dev/null; }   # finished stages are skipped on re-run

if [ "$KIND" = small ]; then
  ARCH="--model-dim 100 --tcn-layers 20 --mixer-depth 2 --mixer-heads 4"
  A=r6_small_a${SA}_sherlock
  done_run "$A" || $PY -u scripts/train_tcn_hybrid.py --from-scratch $ARCH \
    --corpus sherlock --val-corpus sherlock --sampler book \
    --max-epochs 35 --drop-patience 8 --es-patience 8 \
    --batch-size 16 --encode-chunk 512 --seed "$SA" \
    --save-top-k 5 --epoch-ckpt-keep 5 $COMMON --log-name "$A" > "hybrid_$A.log" 2>&1
  BEST=$(ls "$(ckdir "$A")"/checkpoints/best-epoch=*.ckpt | sort -t= -k3 -nr | head -1)
  echo "stage A best: $BEST"
  B=r6_small_a${SA}_b${SB}_mixed
  $PY -u scripts/train_tcn_hybrid.py --weights-ckpt "$BEST" $ARCH \
    --corpus all --val-corpus mixed --sampler family \
    --max-epochs 3 --drop-patience 3 --es-patience 3 \
    --batch-size 64 --encode-chunk 1024 --seed "$SB" \
    --save-top-k 3 --epoch-ckpt-keep 3 $COMMON --log-name "$B" > "hybrid_$B.log" 2>&1
  cp "$(ckdir "$B")/checkpoints/epoch=01.ckpt" "$OUT/small_a${SA}_b${SB}_mixed_epoch=01.ckpt"
elif [ "$KIND" = v2re12 ]; then
  ARCH="--model-dim 256 --tcn-layers 24 --pool attn --pool-queries 4 --mixer-depth 3 --mixer-heads 8 --precision bf16-mixed"
  F=r6_v2re12_f${SA}_sherlock
  done_run "$F" || $PY -u scripts/train_tcn_hybrid.py --from-scratch $ARCH \
    --corpus sherlock --val-corpus sherlock --sampler book --channel-dropout 0.1 \
    --max-epochs 12 --drop-patience 12 --es-patience 12 \
    --batch-size 24 --encode-chunk 128 --seed "$SA" \
    --save-top-k 3 --epoch-ckpt-keep 12 $COMMON --log-name "$F" > "hybrid_$F.log" 2>&1
  FOUND="$OUT/v2re12_f${SA}_found_ep10.ckpt"
  cp "$(ckdir "$F")/checkpoints/epoch=10.ckpt" "$FOUND"
  P=r6_v2re12_f${SA}_p${SB}_plain
  $PY -u scripts/train_tcn_hybrid.py --weights-ckpt "$FOUND" $ARCH \
    --corpus all --val-corpus mixed --sampler family --channel-dropout 0.1 --macro-families podcasts \
    --max-epochs 3 --drop-patience 3 --es-patience 3 \
    --batch-size 24 --encode-chunk 128 --seed "$SB" \
    --save-top-k 3 --epoch-ckpt-keep 3 $COMMON --log-name "$P" > "hybrid_$P.log" 2>&1
  cp "$(ckdir "$P")/checkpoints/epoch=01.ckpt" "$OUT/v2re12_f${SA}_p${SB}_plain_epoch=01.ckpt"
else
  echo "KIND must be small or v2re12" >&2; exit 2
fi
echo "done: $(ls "$OUT")"
