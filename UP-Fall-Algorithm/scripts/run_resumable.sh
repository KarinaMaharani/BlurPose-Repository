#!/usr/bin/env bash
# Runs the UP-Fall benchmark and restarts it automatically if it stops (crash, OOM, lost
# connection to the machine's shell, ...). Each restart skips the experiments that already
# finished, because `run` saves every experiment as soon as it completes.
#
# usage: bash scripts/run_resumable.sh <dataset.yaml> <out_dir> [extra upfall_bench args...]
#   bash scripts/run_resumable.sh configs/datasets/urfall.yaml results/urfall/upfall2019 -s resources.n_jobs=16
#
# Run it inside tmux so it survives closing the terminal:  tmux new -s upfall
set -u
export PYTHONUNBUFFERED=1   # show progress live when the output goes through tee

if [ $# -lt 2 ]; then
  echo "usage: $0 <dataset.yaml> <out_dir> [extra args]" >&2
  exit 2
fi
DATASET="$1"; OUT="$2"; shift 2
MAX_TRIES="${MAX_TRIES:-100}"     # give up after this many restarts
WAIT="${WAIT:-60}"                # seconds between restarts

cd "$(dirname "$0")/.." || exit 1
# shellcheck disable=SC1091
[ -f venv/bin/activate ] && . venv/bin/activate
mkdir -p "$OUT"
LOG="$OUT/resumable.log"

for try in $(seq 1 "$MAX_TRIES"); do
  echo "[$(date '+%F %T')] attempt $try: python -m upfall_bench run -d $DATASET --out $OUT $*" | tee -a "$LOG"
  python -m upfall_bench run -d "$DATASET" --out "$OUT" "$@" 2>&1 | tee -a "$LOG"
  status=${PIPESTATUS[0]}
  if [ "$status" -eq 0 ]; then
    echo "[$(date '+%F %T')] FINISHED: results in $OUT/results.md" | tee -a "$LOG"
    exit 0
  fi
  # exit code 3 = the folder belongs to a different configuration: retrying cannot fix that
  if [ "$status" -eq 3 ]; then
    echo "[$(date '+%F %T')] stopped: wrong configuration for $OUT (see message above)" | tee -a "$LOG"
    exit 3
  fi
  echo "[$(date '+%F %T')] stopped with exit code $status; restarting in ${WAIT}s" | tee -a "$LOG"
  sleep "$WAIT"
done

echo "[$(date '+%F %T')] gave up after $MAX_TRIES attempts; see $LOG" | tee -a "$LOG"
exit 1
