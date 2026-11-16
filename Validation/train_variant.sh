#!/usr/bin/env bash
# Train a SHINE variant with production settings plus extra main.py arguments (e.g. --frame_num=1,
# --temporal_aug_p=0.25). Trains
#   OUT/full          on the whole stack, then denoises it (for the residual/null/temporal tests)
#   OUT/evenodd/...   on the interleaved halves (train_evenodd.sh, for the even/odd test)
#
# Usage: Validation/train_variant.sh RAW_DIR OUT_DIR [extra main.py args...]
set -euo pipefail
RAW=$1; OUT=$2; shift 2
PY=${SHINE_PYTHON:-python3}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT/full"

"$PY" "$HERE/../main.py" \
  --common_path="$OUT/full" \
  --training_path="$RAW" \
  --data_path_test="$RAW" \
  --save_folder_name=experiment \
  --version_folder_name=3x3_blind_spot \
  --model=3x3_blind \
  --img_size=256 \
  --batch_size=8 \
  --max_epochs=100 \
  --recursive_factor=1 \
  --optimizer=adam \
  --learning_rate=0.001 \
  --precision=16 \
  --loss_function=L2 \
  --train=1 --test=1 \
  --time_stamp=run \
  "$@" 2>&1 | tee "$OUT/train_full.log"

"$HERE/train_evenodd.sh" "$RAW" "$OUT/evenodd" "$@"
