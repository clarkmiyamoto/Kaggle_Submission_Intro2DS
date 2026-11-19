#!/usr/bin/env bash
# Reproduce the final Kaggle submission end to end (REPRODUCE.md):
#   1. train the final recipe's 3 seeds on train + val (src.retrain, configs/final/),
#   2. average their Kaggle predictions into submissions/repro_<mode>_<sha>.csv (src.submit),
#   3. compare that CSV with the committed submission (src.verify_submission).
#
#   bash scripts/reproduce_final.sh local [src.retrain options]  # this machine: CUDA, Apple MPS or CPU
#   bash scripts/reproduce_final.sh modal [GPU]                   # Modal, one GPU per seed (default A10G)
#   bash scripts/reproduce_final.sh smoke                         # 20 games per split, ~10 min
#
# local trains the seeds one after another; extra options go to src.retrain, for example
#   bash scripts/reproduce_final.sh local --device cpu --max-hours none
# smoke checks the plumbing only: its tiny models are expected to FAIL the comparison.
# Runs from a git checkout with a clean tree: run ids carry the HEAD SHA, and Modal ships HEAD.
set -euo pipefail
cd "$(dirname "$0")/.."

RECIPE=final_full24_wide2_drop0.05_trval
SEEDS=(0 1 2)
mode=${1:-}
[[ $# -gt 0 ]] && shift

need() {
  for f in "$@"; do
    [[ -e $f ]] || { echo "missing $f: do the data setup in README.md first" >&2; exit 1; }
  done
}

tag=""
case $mode in
  local|smoke)
    need data/processed/train.parquet data/processed/val.parquet data/test.csv data/test_dtypes.csv \
         data/sample_submission.csv
    extra=("$@")
    if [[ $mode == smoke ]]; then
      tag="_smoke"
      extra=(--limit-games 20)
    fi
    for s in "${SEEDS[@]}"; do
      uv run python -m src.retrain "configs/final/${RECIPE}_s${s}.yaml" ${extra[@]+"${extra[@]}"}
    done
    ;;
  modal)
    need data/sample_submission.csv
    gpu=${1:-A10G}  # the submitted runs trained on A10G (fit_info.device_name)
    configs=$(printf "configs/final/${RECIPE}_s%s.yaml," "${SEEDS[@]}")
    uv run modal run scripts/modal_train.py::main --gpu "$gpu" --module src.retrain --configs "${configs%,}"
    uv run modal run scripts/modal_train.py::fetch --name "$RECIPE"
    ;;
  *)
    sed -n '2,14p' "$0" >&2
    exit 2
    ;;
esac

# The runs this checkout just made: run ids end in the HEAD SHA (src.run.git_state).
sha=$(git rev-parse --short HEAD)
runs=()
for s in "${SEEDS[@]}"; do
  id=$(ls runs 2>/dev/null | grep -E "^[0-9]{4}-[0-9]{2}-[0-9]{2}_${RECIPE}_s${s}${tag}_${sha}$" | sort | tail -1 || true)
  [[ -n $id && -f runs/$id/preds_kaggle.parquet ]] || {
    echo "no finished run of seed $s at $sha in runs/ (did training fail?)" >&2; exit 1; }
  runs+=("$id")
done

name="repro_${mode}_${sha}"
uv run python -m src.submit --name "$name" "${runs[@]}"
if [[ $mode == smoke ]]; then
  uv run python -m src.verify_submission "submissions/$name.csv" || \
    echo "smoke: FAIL is expected from 20-game models; the pipeline ran end to end"
else
  uv run python -m src.verify_submission "submissions/$name.csv"
fi
