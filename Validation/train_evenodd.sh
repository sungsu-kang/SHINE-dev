#!/usr/bin/env bash
# Tier 2, test 3 (training part): two SHINE models on interleaved half-stacks.
# Even frames -> model E, odd frames -> model O. Each model then denoises its own half, so the
# two outputs share no noise, neither in training nor at inference.
#
# Usage: Validation/train_evenodd.sh RAW_DIR OUT_DIR [extra main.py args...]
# Defaults match the production run of 710_80A_Insitu(8) HAADF (3x3_blind, Adam 1e-3, 100 epochs);
# recursive_factor doubles the samples per epoch so each half gets as many steps as production.
set -euo pipefail
RAW=$1; OUT=$2; shift 2
PY=${SHINE_PYTHON:-python3}
HERE=$(cd "$(dirname "$0")/.." && pwd)

mkdir -p "$OUT/split/even" "$OUT/split/odd"
i=0
while IFS= read -r f; do
  if (( i % 2 == 0 )); then half=even; else half=odd; fi
  ln -sf "$RAW/$f" "$OUT/split/$half/$f"
  i=$((i+1))
done < <(ls "$RAW" | grep -iE '\.tiff?$' | sort)

for half in even odd; do
  "$PY" "$HERE/main.py" \
    --common_path="$OUT/$half" \
    --training_path="$OUT/split/$half" \
    --data_path_test="$OUT/split/$half" \
    --save_folder_name=experiment \
    --version_folder_name=3x3_blind_spot \
    --model=3x3_blind \
    --img_size=256 \
    --batch_size=8 \
    --max_epochs=100 \
    --recursive_factor=2.2 \
    --optimizer=adam \
    --learning_rate=0.001 \
    --precision=16 \
    --loss_function=L2 \
    --train=1 --test=1 \
    --time_stamp=run \
    "$@" 2>&1 | tee "$OUT/train_$half.log"
done
