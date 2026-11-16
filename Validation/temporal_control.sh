#!/usr/bin/env bash
# Tier 3 retraining control: are the temporal responses the same when the edited events are part
# of the training data (as real events are in production)?
#   1. write an edited copy of the raw stack (blob insertions + particle removals, temporal_test.py)
#   2. train a model on it with production settings
#   3. measure the same events with the production model and the retrained model
#
# Usage: Validation/temporal_control.sh RAW_DIR DEN_DIR "PROD_CKPT [PROD_CKPT ...]" NULL_SUMMARY OUT_DIR [extra main.py args]
set -euo pipefail
RAW=$1; DEN=$2; PROD=$3; NULLS=$4; OUT=$5; shift 5
PY=${SHINE_PYTHON:-python3}
HERE=$(cd "$(dirname "$0")" && pwd)

"$PY" "$HERE/temporal_test.py" make-control "$RAW" "$DEN" "$OUT" --null_summary "$NULLS"

"$PY" "$HERE/../main.py" \
  --common_path="$OUT/model" \
  --training_path="$OUT/stack" \
  --data_path_test="$OUT/stack" \
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
  --train=1 --test=0 \
  --time_stamp=run \
  "$@" 2>&1 | tee "$OUT/train.log"

# Validation loss is flat after a few epochs, so which saved epoch is "best" is close to random.
# All saved checkpoints of both models are compared; PROD may list several (space-separated).
"$PY" "$HERE/temporal_test.py" eval-control "$RAW" "$DEN" "$OUT" \
  --prod $PROD --retrained "$OUT"/model/experimentrun/model/*.ckpt
