#!/usr/bin/env bash
# Download the competition data into data/.
# Requires `kaggle auth login` and accepting the rules on the competition page.
set -euo pipefail

COMP=tabular-playground-series-oct-2022
DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)/data"
FILES=(train_{0..9}.csv test.csv sample_submission.csv train_dtypes.csv test_dtypes.csv)

missing=0
for f in "${FILES[@]}"; do [[ -f "$DATA_DIR/$f" ]] || missing=1; done
if [[ $missing -eq 0 ]]; then
  echo "All files already in $DATA_DIR"
  exit 0
fi

mkdir -p "$DATA_DIR"
kaggle competitions download "$COMP" -p "$DATA_DIR"
unzip -o -q "$DATA_DIR/$COMP.zip" -d "$DATA_DIR"
rm "$DATA_DIR/$COMP.zip"
echo "Downloaded to $DATA_DIR"
