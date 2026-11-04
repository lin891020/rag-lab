#!/usr/bin/env bash
# 26 in one go: Claude answers (stops with exit 2 once the cost passes the budget) -> score -> report.
set -euo pipefail
cd "$(dirname "$0")/../.."
E=experiments/26-hosted-retrieval
uv run python $E/run.py
uv run python $E/score.py
uv run python $E/posthoc26.py
echo "CHAIN26 DONE $(date '+%F %T')"
