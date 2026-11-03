#!/usr/bin/env bash
# 22 in one go: (scans already made by degrade.py) OCR routes -> structure -> answers -> score -> report.
set -euo pipefail
cd "$(dirname "$0")/../.."
E=experiments/22-degraded-scan
uv run python $E/build_corpus.py
uv run python $E/structure.py
uv run python $E/run.py
uv run python $E/score.py
uv run python $E/report.py
echo "CHAIN22 DONE $(date '+%F %T')"
