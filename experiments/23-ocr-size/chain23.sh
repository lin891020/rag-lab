#!/usr/bin/env bash
# 23 in one go: (pages already shrunk by shrink.py) glm-ocr per size -> structure -> answers -> score -> report -> posthoc.
set -euo pipefail
cd "$(dirname "$0")/../.."
E=experiments/23-ocr-size
uv run python $E/read_sizes.py
uv run python $E/structure.py
uv run python $E/run.py
uv run python $E/score.py
uv run python $E/report.py
uv run python $E/posthoc23.py
echo "CHAIN23 DONE $(date '+%F %T')"
