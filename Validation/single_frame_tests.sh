#!/usr/bin/env bash
# Tier 2 + Tier 3 tests on the single-frame models (train_single_frame.sh).
# Usage: Validation/single_frame_tests.sh RAW_DIR PROD_DEN_DIR PROD_NULL_SUMMARY SF_DIR
FRAME_NUM=1 exec "$(cd "$(dirname "$0")" && pwd)/variant_tests.sh" "$@"
