#!/usr/bin/env bash
# 24 in one go: corpus from 23's saved glm-ocr replies -> answers -> score -> report -> posthoc.
set -euo pipefail
cd "$(dirname "$0")/../.."
E=experiments/24-ocr-header
uv run python $E/build_corpus.py
uv run python $E/run.py
uv run python $E/score.py
uv run python $E/report.py
uv run python $E/posthoc24.py
echo "CHAIN24 DONE $(date '+%F %T')"
