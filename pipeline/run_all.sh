#!/usr/bin/env bash
# Full offline pipeline. Usage: bash pipeline/run_all.sh [python]
set -euo pipefail
PY=${1:-python}
cd "$(dirname "$0")"
[ -f work/raw_papers.jsonl ] || $PY ingest.py --per-year 55
[ -f work/pubmed_counts.json ] || $PY counts.py
$PY train_embed.py --epochs 3
$PY build.py
