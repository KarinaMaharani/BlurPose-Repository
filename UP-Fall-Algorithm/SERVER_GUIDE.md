# Running UP-Fall-Algorithm on the GPU server (rsync + tmux)

Server: `karina@152.118.31.34`, SSH port `51022`. See `../SSH_GUIDE.md` for key login and the
`ssh gpu` shortcut. `[laptop]` commands run in **Git Bash** from the `UP-Fall-Algorithm/`
folder; `[server]` commands run after `ssh`.

## 0. rsync on the laptop (one time)

Git Bash does not include rsync. Pick one:

* **Chocolatey** (already on this laptop), in an *administrator* PowerShell:
  `choco install rsync -y`, then open a new Git Bash.
* **No rsync:** use the `scp` lines given below each rsync line. `scp` cannot resume, so
  send large data as one zip.

## 1. Send the code (re-run after every code change, only changed files go)

```bash
# [laptop]
ssh -p 51022 karina@152.118.31.34 "mkdir -p ~/upfall/data"
rsync -avP -e "ssh -p 51022" --exclude data/ --exclude cache/ --exclude results/ --exclude __pycache__/ ./ karina@152.118.31.34:~/upfall/
# without rsync:
scp -P 51022 -r upfall_bench configs tests requirements.txt README.md karina@152.118.31.34:~/upfall/
```

## 2. Send the data

The session exports are already zips, so send the zips and unzip them on the server.

```bash
# [laptop] your sessions
rsync -avP -e "ssh -p 51022" "/c/Users/Karina M/Downloads/francis_sesi_sore_1788601010155.zip" karina@152.118.31.34:~/upfall/data/
# without rsync:
scp -P 51022 "/c/Users/Karina M/Downloads/francis_sesi_sore_1788601010155.zip" karina@152.118.31.34:~/upfall/data/

# [laptop] UP-Fall skeleton release (small)
rsync -avP -e "ssh -p 51022" data/upfall_skeleton/ karina@152.118.31.34:~/upfall/data/upfall_skeleton/
```

```bash
# [server] one folder per session under data/ra_pak_edy/
cd ~/upfall/data
mkdir -p ra_pak_edy/francis_sesi_sore
unzip -q francis_sesi_sore_1788601010155.zip -d ra_pak_edy/francis_sesi_sore -x 'data/recovery/*'
rm francis_sesi_sore_1788601010155.zip
```

UR Fall is faster to download straight onto the server than to copy:
`python -m upfall_bench download -d configs/datasets/urfall.yaml` (step 4).

## 3. Python environment (one time)

```bash
# [server]
cd ~/upfall
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
nvidia-smi                       # see which GPU is free and how much memory is in use
python -m pytest tests -q        # optional check, ~2 min
```

## 4. Run inside tmux

```bash
# [server]
tmux new -s upfall
cd ~/upfall && source venv/bin/activate
export CUDA_VISIBLE_DEVICES=0    # the free GPU from nvidia-smi

# server-sized limits for every command below (it is a shared machine: don't take all cores)
R="-s resources.n_jobs=8 -s resources.torch_threads=8 -s resources.gpu_batch=16 -s resources.gpu_memory_fraction=0.8"

D=configs/datasets/ra_pak_edy.yaml
python -m upfall_bench inspect      -d $D
python -m upfall_bench extract-flow -d $D $R
python -m upfall_bench extract-pose -d $D $R -s pose.layout=body12
python -m upfall_bench run          -d $D $R                                  2>&1 | tee -a run_ra.log
python -m upfall_bench run          -d $D $R -b configs/benchmarks/subject_independent.yaml 2>&1 | tee -a run_ra.log
python -m upfall_bench cnn          -d $D $R                                  2>&1 | tee -a run_ra.log
python -m upfall_bench skeleton     -d $D $R -b configs/benchmarks/koffi2024_skeleton.yaml -s skeleton.label=fall 2>&1 | tee -a run_ra.log

D=configs/datasets/urfall.yaml
python -m upfall_bench download     -d $D
python -m upfall_bench extract-flow -d $D $R
python -m upfall_bench extract-pose -d $D $R
python -m upfall_bench run          -d $D $R                                  2>&1 | tee -a run_ur.log
python -m upfall_bench cnn          -d $D $R -s cnn.epochs=30 -s cnn.lr=0.01 2>&1 | tee -a run_ur.log
python -m upfall_bench skeleton     -d $D $R -b configs/benchmarks/koffi2024_skeleton.yaml -s skeleton.label=impact 2>&1 | tee -a run_ur.log
```

Set `fall_classes` in `configs/datasets/ra_pak_edy.yaml` **before** step 1, or the binary
task is skipped.

**Detach** with `Ctrl+b`, then `d`. The job keeps running after you close the laptop.

```bash
# [laptop] later
ssh -p 51022 karina@152.118.31.34
tmux attach -t upfall            # back into the session
tmux ls                          # list sessions
tail -f ~/upfall/run_ra.log      # follow the log without attaching
```

To run everything unattended, put the commands in a script, e.g. `nano run_all.sh`, and
start it with `bash run_all.sh 2>&1 | tee -a run_all.log` inside tmux.

## 5. Bring the results back

Results are small (tables, PNGs, CSVs). Leave `cache/` on the server.

```bash
# [laptop]
rsync -avP -e "ssh -p 51022" karina@152.118.31.34:~/upfall/results/ ./results/
# without rsync:
scp -P 51022 -r karina@152.118.31.34:~/upfall/results ./results_server
```

Open `results/<dataset>/<run>/results.md`.

## 6. Clean up on the server

```bash
# [server]
tmux kill-session -t upfall
du -sh ~/upfall/data ~/upfall/cache     # cache/ can be deleted at any time
```
