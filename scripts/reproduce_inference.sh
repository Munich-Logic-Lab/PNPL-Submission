#!/usr/bin/env bash
# Rebuild the final Deep submission (submission/r6_pc1.5.csv) from checkpoints/
# and check it byte-for-byte against the submitted file.
#
#   bash scripts/reproduce_inference.sh            # PY=/path/to/venv/bin/python to override
#
# Needs: one CUDA GPU (>= 12 GB for Qwen2.5-1.5B in fp32), the Deep holdout
# (downloaded to data/holdout on first run), Qwen/Qwen2.5-1.5B from the HF hub.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
export LM_REVISION=${LM_REVISION:-8faed761d45a263340a0528343f099c05c9a4323}
export CUBLAS_WORKSPACE_CONFIG=:4096:8

if [ ! -f data/holdout/COMPETITION_HOLDOUT/subj00_holdout01_sentence.npz ]; then
  $PY -c "from pnpl.competition import LibriBrainCompetitionHoldout as H; H(track='deep', data_path='data/holdout', download=True)"
fi
# T5-large word embeddings used by the retrieval head (cached; avoids re-embedding)
mkdir -p data/libribrain_word/t5_cache
[ -f data/libribrain_word/t5_cache/t5-large_layer0.5.pt ] || cp assets/t5_cache/t5-large_layer0.5.pt data/libribrain_word/t5_cache/

C=checkpoints
$PY scripts/build_final.py \
  --ckpts "$C/small_mixed_epoch=01.ckpt" "$C/tcn_hybrid_v2re12_plain_nc-9-5-3_epoch=01.ckpt" "$C/v2re12_found_ep10.ckpt" \
          "$C/small_s1_mixed_epoch=01.ckpt" "$C/tcn_hybrid_v2re12s2_plain_nc-9-3-3_epoch=01.ckpt" "$C/v2re12s2_found_ep10.ckpt" \
  --name r6_pc1.5 --lm Qwen/Qwen2.5-1.5B --fewshot --lam 0.25 --pre-alpha 0.5 --alpha 1.5 \
  --dtype fp32 --beam 8 --sweeps 2 --work out --force --no-bundle

OUT=out/r6_pc1.5/fused_lam0.25_g0.0_pc1.5.csv
sha256sum "$OUT" submission/r6_pc1.5.csv
if cmp -s "$OUT" submission/r6_pc1.5.csv; then echo "REPRODUCED: byte-identical to submission/r6_pc1.5.csv"
else echo "MISMATCH: $OUT differs from submission/r6_pc1.5.csv"; exit 1; fi
