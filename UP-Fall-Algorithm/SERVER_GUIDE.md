# Running UP-Fall-Algorithm on the GPU server (rsync + tmux)

Server: `karina@152.118.31.34`, SSH port `51022`. See `../SSH_GUIDE.md` for key login.
`[laptop]` commands run in **WSL** (it has rsync; Git Bash does not). `[server]` commands run
after `ssh`.

> **Keep the `cd ... &&` at the start of each laptop command.** In rsync, `./` means "the
> current folder". Run from the wrong folder, it uploads that whole folder, e.g. your entire
> `C:\Users\Karina M`. With `&&`, nothing is sent if the `cd` fails. To preview a transfer,
> add `-n` (dry run) to the rsync command.

## Using a vast.ai machine instead

Every command in this guide works with two swaps: `ssh -p 51022` → `ssh -p 30713` and
`karina@152.118.31.34` → `root@ssh1.vast.ai`. The port and host change whenever you rent a
new instance. The `-L 8080:localhost:8080` part is only a tunnel for web pages, so rsync
doesn't need it.

* **Disk:** the instance checked on 2026-11-05 has only 16 GB. If you only use `run`, skip
  PyTorch (several GB):
  `pip install numpy pandas scipy scikit-learn matplotlib pyyaml tqdm opencv-python-headless imageio-ffmpeg`
* **You pay while it runs, and the files live on the instance:** bring `results/` back (step 5)
  before you destroy it.
* **tmux:** use `tmux new -A -s upfall`. If it says "sessions should be nested", you are
  already inside tmux, so just run your commands.

## 1. Send the code (270 KB, seconds; re-run after every code change)

```bash
# [laptop, WSL]
cd "/mnt/c/.FILES/..MY STUFF/..KARIN/.UI/.SEM 6/skripsi/..RA-Pak-Edy/UP-Fall-Algorithm" && rsync -avP -e "ssh -p 51022" --exclude data/ --exclude cache/ --exclude results/ --exclude __pycache__/ ./ karina@152.118.31.34:~/upfall/
```

## 2. Send the data

Upload from the laptop was measured at about 100 kB/s (0.36 GB/hour), so size matters. Choose
one option per session. rsync only creates the last folder of a path, so make the parents first:

```bash
# [laptop, WSL] one time
ssh -p 51022 karina@152.118.31.34 "mkdir -p ~/upfall/data ~/upfall/cache/ra_pak_edy"
```

**Option A: IMU + cached optical flow, no videos (about 115 MB, roughly 20 min).** Enough for
`run` with the IMU and CAM experiments. Not enough for `extract-pose` or `cnn`, so in step 4
use only `inspect` and `run` for this dataset.

```bash
# [laptop, WSL] IMU CSVs + metadata (-z compresses the CSVs in transit)
cd "/mnt/c/.FILES/..MY STUFF/..KARIN/.UI/.SEM 6/skripsi/..RA-Pak-Edy/UP-Fall-Algorithm" && rsync -avPz -e "ssh -p 51022" --exclude videos/ --exclude '*_sensor_data*' --exclude '[0-9]*[0-9]_consolidated.csv' data/ra_pak_edy/ karina@152.118.31.34:~/upfall/data/ra_pak_edy/
# [laptop, WSL] cached optical flow of those cameras
cd "/mnt/c/.FILES/..MY STUFF/..KARIN/.UI/.SEM 6/skripsi/..RA-Pak-Edy/UP-Fall-Algorithm" && rsync -avP -e "ssh -p 51022" cache/ra_pak_edy/derived/ karina@152.118.31.34:~/upfall/cache/ra_pak_edy/derived/
```

**Option B: the full session with videos (1.1 GB, about 3 hours at that speed).** Needed for
`extract-pose` and `cnn`. Faster route if you have a cloud link: upload the zip to Google Drive
or similar from a better connection, then download it on the server. Otherwise:

```bash
# [laptop, WSL] start inside tmux so closing the terminal doesn't stop it; if it breaks,
# run the same command again and rsync resumes the partial file
rsync -avP -e "ssh -p 51022" "/mnt/c/Users/Karina M/Downloads/francis_sesi_sore_1788601010155.zip" karina@152.118.31.34:~/upfall/data/
```

```bash
# [server] one folder per session under data/ra_pak_edy/
cd ~/upfall/data
mkdir -p ra_pak_edy/francis_sesi_sore
unzip -q francis_sesi_sore_1788601010155.zip -d ra_pak_edy/francis_sesi_sore -x 'data/recovery/*'
rm francis_sesi_sore_1788601010155.zip
```

**UP-Fall skeleton release (4 MB):**

```bash
# [laptop, WSL]
cd "/mnt/c/.FILES/..MY STUFF/..KARIN/.UI/.SEM 6/skripsi/..RA-Pak-Edy/UP-Fall-Algorithm" && rsync -avP -e "ssh -p 51022" data/upfall_skeleton/ karina@152.118.31.34:~/upfall/data/upfall_skeleton/
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
# [laptop, WSL]
cd "/mnt/c/.FILES/..MY STUFF/..KARIN/.UI/.SEM 6/skripsi/..RA-Pak-Edy/UP-Fall-Algorithm" && rsync -avP -e "ssh -p 51022" karina@152.118.31.34:~/upfall/results/ ./results/
```

Open `results/<dataset>/<run>/results.md`.

## 6. Clean up on the server

```bash
# [server]
tmux kill-session -t upfall
du -sh ~/upfall/data ~/upfall/cache     # cache/ can be deleted at any time
```
