#!/usr/bin/env bash
# 17's model-bound steps in one go (the retrieval track and the sub-queries are already done):
# reading track -> score -> attribute -> report. Log: logs/chain17.log, last line CHAIN17 DONE.
set -euo pipefail
cd "$(dirname "$0")/../.."
E=experiments/17-table-retrieval
{
  echo "start $(date '+%F %T')"
  uv run python $E/decompose.py          # resumable: nothing to do when subqueries.jsonl is full
  uv run python $E/run_read.py
  uv run python $E/run_read.py --retry-only   # redoes failed calls only
  uv run python $E/score.py
  uv run python $E/attribute.py
  uv run python $E/report.py
  echo "CHAIN17 DONE $(date '+%F %T')"
} >>logs/chain17.log 2>&1
