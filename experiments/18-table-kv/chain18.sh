#!/usr/bin/env bash
# 18's model-bound steps in one go (the retrieval track is already done):
# reading track -> score -> attribute -> report -> posthoc. Log: logs/chain18.log, last line CHAIN18 DONE.
set -euo pipefail
cd "$(dirname "$0")/../.."
E=experiments/18-table-kv
{
  echo "start $(date '+%F %T')"
  uv run python $E/run_read.py
  uv run python $E/run_read.py --retry-only   # redoes failed calls only
  uv run python $E/score.py
  uv run python $E/attribute.py
  uv run python $E/report.py
  uv run python $E/posthoc.py
  echo "CHAIN18 DONE $(date '+%F %T')"
} >>logs/chain18.log 2>&1
