#!/usr/bin/env bash
# The drift probe: 27's own run.py on 20 of 27's questions (its grid: Haiku agent twice, Opus
# agent once), written to data/ (not 27's results). posthoc29 compares the Haiku agent rows with
# 27's stored answers. About US$0.8 (the Opus cell is most of it).
set -euo pipefail
cd "$(dirname "$0")/../.."
uv run python experiments/27-hosted-agentic-search/run.py --sample 20 --out data/29-amendment-diff/drift
echo "DRIFT DONE $(date '+%F %T')"
