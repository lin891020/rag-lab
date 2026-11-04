#!/usr/bin/env bash
# 25 in one go: Claude answers (stops with exit 2 once the cost passes the budget) -> score -> report -> posthoc.
set -euo pipefail
cd "$(dirname "$0")/../.."
E=experiments/25-hosted-tables
uv run python $E/run.py
uv run python $E/score.py
uv run python $E/report.py
uv run python $E/posthoc25.py
echo "CHAIN25 DONE $(date '+%F %T')"
