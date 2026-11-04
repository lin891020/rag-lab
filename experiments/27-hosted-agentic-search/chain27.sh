#!/usr/bin/env bash
# 27 in one go: Claude searches and answers (stops with exit 2 once the cost passes the budget)
# -> score (with 26's rows beside) -> report.
set -euo pipefail
cd "$(dirname "$0")/../.."
E=experiments/27-hosted-agentic-search
uv run python $E/run.py
uv run python $E/score.py
uv run python $E/posthoc27.py
echo "CHAIN27 DONE $(date '+%F %T')"
