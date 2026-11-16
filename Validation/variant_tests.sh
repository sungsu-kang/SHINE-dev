#!/usr/bin/env bash
# Tier 2 + Tier 3 tests on a variant trained by train_variant.sh. FRAME_NUM (env, default 5) must
# match the variant.
# The temporal sweep uses the production denoised frames (PROD_DEN) only to find particles and
# register frames, and the production null summary (PROD_NULL) to set the inserted amplitudes, so
# the 1-frame and 5-frame models are tested on identical events.
#
# Usage: [FRAME_NUM=1] Validation/variant_tests.sh RAW_DIR PROD_DEN_DIR PROD_NULL_SUMMARY SF_DIR
#   SF_DIR = OUT_DIR given to train_variant.sh
set -euo pipefail
RAW=$1; PROD_DEN=$2; PROD_NULL=$3; SF=$4
PY=${SHINE_PYTHON:-python3}
FN=${FRAME_NUM:-5}
HERE=$(cd "$(dirname "$0")" && pwd)
DEN=$SF/full/experimentrun/denoised
CKPT=$("$PY" "$HERE/find_checkpoint.py" "$RAW" "$DEN" "$SF/full/experimentrun/model" --frame_num $FN \
       | tee "$SF/find_checkpoint.log" | sed -n 's/^producer: \([^ ]*\).*/\1/p')
echo "producer checkpoint: $CKPT"

"$PY" "$HERE/residual_test.py" "$RAW" "$DEN" "$SF/residual"
"$PY" "$HERE/null_test.py" "$RAW" "$DEN" "$CKPT" "$SF/null" --frame_num $FN
"$PY" "$HERE/evenodd_test.py" "$RAW" "$SF/evenodd/even/experimentrun/denoised" \
    "$SF/evenodd/odd/experimentrun/denoised" "$SF/evenodd/analysis" --null_summary "$SF/null/null_summary.json"
"$PY" "$HERE/temporal_test.py" sweep "$RAW" "$PROD_DEN" "$CKPT" "$SF/temporal" \
    --null_summary "$PROD_NULL" --frame_num $FN
