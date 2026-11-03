#!/usr/bin/env bash
# 21 in one go: reading track (step 2 only) -> score -> attribute -> report -> posthoc.
set -euo pipefail
cd "$(dirname "$0")/../.."
E=experiments/21-lookup-segment
uv run python $E/run_read.py
uv run python $E/run_read.py --retry-only   # redoes failed calls only
uv run python $E/score.py
uv run python $E/attribute.py
uv run python $E/report.py
uv run python $E/posthoc.py
echo "CHAIN21 DONE $(date '+%F %T')"
