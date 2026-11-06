#!/usr/bin/env bash
# 28 in one go: Claude searches and answers (stops with exit 2 once the cost passes the budget)
# -> score (with 27's rows beside) -> report.
set -euo pipefail
cd "$(dirname "$0")/../.."
E=experiments/28-hosted-bigger-corpus
uv run python $E/run.py
uv run python $E/score.py
uv run python $E/posthoc28.py
echo "CHAIN28 DONE $(date '+%F %T')"
