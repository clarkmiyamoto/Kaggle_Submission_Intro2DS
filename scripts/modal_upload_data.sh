#!/usr/bin/env bash
# Upload what src/data.py reads to the kaggle-rl-data Modal volume (~3.4 GB), for
# scripts/modal_train.py. Run once from the repo root after `uv run modal setup` and
# `uv run python -m src.make_splits`. Safe to rerun: existing files are overwritten.
set -euo pipefail
cd "$(dirname "$0")/.."

uv run modal volume create kaggle-rl-data 2>/dev/null || true
for f in processed/train.parquet processed/val.parquet processed/test.parquet \
         test.csv test_dtypes.csv train_dtypes.csv sample_submission.csv; do
  echo "uploading data/$f"
  uv run modal volume put --force kaggle-rl-data "data/$f" "/$f"
done
uv run modal volume ls kaggle-rl-data /processed
