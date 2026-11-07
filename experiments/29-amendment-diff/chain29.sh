#!/usr/bin/env bash
# 29 in one go: Claude answers (stops with exit 2 once the cost passes the budget) -> score ->
# report. The drift probe (27's 20 questions with 27's own run.py) is a separate step: drift.sh.
set -euo pipefail
cd "$(dirname "$0")/../.."
E=experiments/29-amendment-diff
uv run python $E/run.py
uv run python $E/score.py
uv run python $E/posthoc29.py
echo "CHAIN29 DONE $(date '+%F %T')"
