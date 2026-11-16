#!/usr/bin/env bash
# Single-frame control (--frame_num=1): same network and training as production, but each frame is
# denoised from itself alone, so there is no temporal prior.
# Usage: Validation/train_single_frame.sh RAW_DIR OUT_DIR [extra main.py args...]
set -euo pipefail
exec "$(cd "$(dirname "$0")" && pwd)/train_variant.sh" "$1" "$2" --frame_num=1 "${@:3}"
