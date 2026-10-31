#!/usr/bin/env bash
# 19's model-bound steps in one go (no retrieval track: the lookup has none, the baselines are copied):
# reading track -> score -> attribute -> report -> posthoc. Log: logs/chain19.log, last line CHAIN19 DONE.
set -euo pipefail
cd "$(dirname "$0")/../.."
E=experiments/19-lookup-tool
{
  echo "start $(date '+%F %T')"
  uv run python $E/run_read.py
  uv run python $E/run_read.py --retry-only   # redoes failed calls only
  uv run python $E/score.py
  uv run python $E/attribute.py
  uv run python $E/report.py
  uv run python $E/posthoc.py
  echo "CHAIN19 DONE $(date '+%F %T')"
} >>logs/chain19.log 2>&1
