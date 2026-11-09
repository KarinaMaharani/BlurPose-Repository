#!/usr/bin/env bash
# Full UP-Fall pipeline on the server for UR Fall and FAMS, results into results/<dataset>/.
# Safe to run again after a stop: every step skips what is already done.
#
#   tmux new -s upfall
#   bash scripts/run_server.sh            # all CPU cores by default
#   N_JOBS=8 bash scripts/run_server.sh   # or fewer
set -u
export PYTHONUNBUFFERED=1   # show progress live when the output goes through tee
cd "$(dirname "$0")/.." || exit 1

# use the Python 3.12 environment when it exists (the GPU pose model needs it)
if [ -f venv312/bin/activate ]; then . venv312/bin/activate; elif [ -f venv/bin/activate ]; then . venv/bin/activate; fi

N_JOBS="${N_JOBS:-$(nproc)}"
RES="-s resources.n_jobs=$N_JOBS"
echo "cores used: $N_JOBS"

# repeat a step until it exits cleanly; each try skips the videos already done
# (the pose model sometimes crashes on one video, and the next try carries on)
retry() {
  local name="$1"; shift
  for try in $(seq 1 100); do
    "$@" 2>&1 | tee -a "$LOG_DIR/$name.log"
    if [ "${PIPESTATUS[0]}" -eq 0 ]; then echo "$name: done"; return 0; fi
    echo "$name: try $try ended with an error, retrying"
  done
  echo "$name: still failing after 100 tries, see $LOG_DIR/$name.log"; return 1
}
LOG_DIR="logs"; mkdir -p "$LOG_DIR"

# 1. UR Fall: download once (existing files are skipped)
python -m upfall_bench download -d configs/datasets/urfall.yaml || exit 1

# 2. FAMS: extract any zips not yet extracted, with Python (no unzip needed)
if ls data/fams/*.zip > /dev/null 2>&1; then
  python - <<'EOF'
import glob, os, zipfile
for z in sorted(glob.glob("data/fams/*.zip")):
    with zipfile.ZipFile(z) as zf:
        for name in zf.namelist():
            if name.startswith("__MACOSX/") or name.endswith(".DS_Store"):
                continue
            if not os.path.exists(os.path.join("data/fams", name)):
                zf.extract(name, "data/fams")
    print("extracted", z, flush=True)
EOF
fi
python -m upfall_bench inspect -d configs/datasets/urfall.yaml > /dev/null
python -m upfall_bench inspect -d configs/datasets/fams.yaml

# 3. Optical flow for the CAM experiments
retry flow_urfall python -m upfall_bench extract-flow -d configs/datasets/urfall.yaml
retry flow_fams python -m upfall_bench extract-flow -d configs/datasets/fams.yaml

# 4. Skeleton poses for the SKELETON experiments (GPU)
retry pose_urfall python -m upfall_bench extract-pose -d configs/datasets/urfall.yaml
retry pose_fams python -m upfall_bench extract-pose -d configs/datasets/fams.yaml

# 5. The two benchmarks, each restarted automatically if it stops
bash scripts/run_resumable.sh configs/datasets/urfall.yaml results/urfall/upfall2019 $RES
bash scripts/run_resumable.sh configs/datasets/fams.yaml results/fams/upfall2019 $RES

echo "all done: results/urfall/upfall2019/results.md and results/fams/upfall2019/results.md"
