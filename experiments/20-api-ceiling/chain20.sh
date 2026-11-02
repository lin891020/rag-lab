#!/usr/bin/env bash
# 20: reading track (Claude API, resumable; stops past --budget) -> score -> attribute -> report -> posthoc.
set -euo pipefail
cd "$(dirname "$0")/../.."
E=experiments/20-api-ceiling
uv run python $E/run_read.py
uv run python $E/score.py
uv run python $E/attribute.py
uv run python $E/report.py
uv run python $E/posthoc.py
echo "CHAIN20 DONE"
