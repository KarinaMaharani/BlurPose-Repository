# UP-Fall-Algorithm

A re-implementation of the **UP-Fall Detection benchmark** that runs on any fall dataset. The
same code runs on the original UP-Fall, on the **UR Fall Detection Dataset**, and on the
**RA-Pak-Edy** IMU + 4-camera recordings. Each dataset gets its own small adapter, and the
rest of the pipeline is shared.

It reproduces two published benchmarks:

| Track | Paper | What it benchmarks |
|---|---|---|
| **ARC** (`run`, `cnn`) | Martínez-Villaseñor *et al.*, "UP-Fall Detection Dataset: A Multimodal Approach", *Sensors* 19(9):1988, 2019 | Windowing (1/2/3 s), 18 features, 5-method feature selection, then RF / SVM / MLP / kNN on modality combinations. Also a CNN on raw frames. |
| **Skeleton** (`skeleton`) | Koffi *et al.*, "An Improved 3D Skeletons UP-Fall Dataset: Enhancing Data Quality for Efficient Impact Fall Detection", 2024 (repo `3D_skeletons-UP-Fall-Dataset`) | Impact detection from skeleton joints: SVM / MLP / RF / KNN / NB / SGD on frames, and LSTM / BiLSTM / CNN / ST-GCN on sequences. |

> **About the original repository.** The `3D_skeletons-UP-Fall-Dataset` zip contains only a
> README and `SUBJECT1-5.zip` with 82 skeleton CSVs. The `src/`, `examples/` and `notebooks/`
> that its README describes are not in the release. Everything here was therefore written
> from the two papers: Sections 3.6, 5.2–5.6 and Table 12 of the UP-Fall paper, and the
> skeleton README. Where a paper leaves a detail open, the choice made here is listed in
> [§9 Deviations from the papers](#9-deviations-from-the-papers).

---

## Contents

1. [Quick start](#1-quick-start)
2. [The pipeline, step by step](#2-the-pipeline-step-by-step)
3. [Datasets](#3-datasets)
4. [Commands](#4-commands)
5. [Configuration](#5-configuration)
6. [Adding a new dataset](#6-adding-a-new-dataset)
7. [Outputs](#7-outputs)
8. [Benchmark details](#8-benchmark-details)
9. [Deviations from the papers](#9-deviations-from-the-papers)
10. [Methodological notes for a thesis](#10-methodological-notes-for-a-thesis)
11. [Project structure](#11-project-structure)
12. [Troubleshooting](#12-troubleshooting)
13. [References](#13-references)

---

## 1. Quick start

Run every command from this folder (`UP-Fall-Algorithm/`).

```bash
pip install -r requirements.txt
```

**Check that everything works (synthetic data, about 1 minute):**

```bash
python -m upfall_bench run -d configs/datasets/synthetic.yaml -b configs/benchmarks/quick.yaml
```

**UR Fall, end to end:**

```bash
python -m upfall_bench download     -d configs/datasets/urfall.yaml   # ~110 MB into data/urfall
python -m upfall_bench inspect      -d configs/datasets/urfall.yaml   # what is in it
python -m upfall_bench extract-flow -d configs/datasets/urfall.yaml   # CAM features (~1 min)
python -m upfall_bench run          -d configs/datasets/urfall.yaml   # the paper benchmark
python -m upfall_bench cnn          -d configs/datasets/urfall.yaml   # vision CNN (Sec. 5.6)
python -m upfall_bench extract-pose -d configs/datasets/urfall.yaml   # skeletons (GPU recommended)
python -m upfall_bench skeleton     -d configs/datasets/urfall.yaml -b configs/benchmarks/koffi2024_skeleton.yaml -s skeleton.label=impact
```

**Your own sessions (RA-Pak-Edy):**

1. Unzip every session export into `data/ra_pak_edy/`, one folder per session. The folder
   names don't matter. If the sessions are already unzipped somewhere else, don't copy them;
   point `dataset.root` at that folder instead (disk space).
2. Open [configs/datasets/ra_pak_edy.yaml](configs/datasets/ra_pak_edy.yaml) and set
   `fall_classes` (which `label_id`s are falls). Optionally set `classes` (names) too.
3. Run:

```bash
python -m upfall_bench inspect      -d configs/datasets/ra_pak_edy.yaml
python -m upfall_bench extract-flow -d configs/datasets/ra_pak_edy.yaml   # optional: CAM modality
python -m upfall_bench run          -d configs/datasets/ra_pak_edy.yaml
```

The results go to `results/<dataset>/<run>/results.md`, with paper-style tables and figures.

**Laptop vs. GPU server.** By default every command stays light: 2 CPU cores, 2 torch
threads, GPU batch 4, at most half the GPU memory (see `resources:` in §5). Add
`-s resources.device=cpu` to keep the GPU free entirely. Run the full benchmarks (all
windows, 10 rounds, pose extraction over whole sessions) on the GPU server; the step-by-step
rsync + tmux commands are in **[SERVER_GUIDE.md](SERVER_GUIDE.md)**.

**Reproduce the 3D-skeleton benchmark on its own data.** Copy the `SUBJECT*.zip` files into
`data/upfall_skeleton/`, then:

```bash
python -m upfall_bench skeleton -d configs/datasets/upfall_skeleton.yaml -b configs/benchmarks/koffi2024_skeleton.yaml
```

---

## 2. The pipeline, step by step

The UP-Fall paper follows the *activity recognition chain* (ARC, Fig. 7):

```
 sensor/video files
        │  dataset adapter (one per dataset)          upfall_bench/datasets/*.py
        ▼
 Recording(subject, activity, trial, streams{IMU, CAM, ...}, label timeline, videos)
        │  [optional] extract-flow / extract-pose     upfall_bench/derive/
        ▼        (adds CAM / SKELETON streams from the videos, cached)
 Windowing: 1/2/3 s, 50 % overlap, sample-and-hold to 18 Hz        arc/windowing.py
        ▼
 Feature extraction: 12 temporal + 6 frequency per channel           arc/features.py
                     (+ mean 20×20 optical-flow vector per camera)
        ▼
 Feature selection: 5 rankers → top-100 each → vote → RF curve       arc/selection.py
        ▼
 Models: RF, SVM, MLP, kNN (Table 12)                                arc/models.py
        ▼
 Evaluation: 10 × random 70/30, acc / prec / sens / spec / F1        arc/evaluation.py
        ▼
 Report: Table 13, Table 14, Fig. 8-14 curves, confusion matrix      arc/report.py
```

All three steps are repeated for **every window size × every experiment (modality
combination) × every task** (`multiclass` = all activity labels as in the paper; `binary` =
fall vs. everything else).

**Data model.** Every adapter produces `Recording` objects
([upfall_bench/core.py](upfall_bench/core.py)):

| field | meaning |
|---|---|
| `rid, subject, activity, trial` | one UP-Fall *SubjectXActivityYTrialZ* |
| `streams: {name: Stream}` | each sensor: `t_ms` (N), `values` (N × C), `channels`, `modality` tag (`IMU`, `CAM`, `IR`, `EEG`, `DEPTH`, `SKELETON`, …), `kind` (`signal` → 18 features, `frame_vector` → mean) |
| `label_t_ms, labels` | label timeline: one label per sample, or only the change points (sample-and-hold) |
| `videos: {cam: VideoSource}` | video file or image zip plus how frame index maps to time |

Every stream of a recording uses the same millisecond clock. That is all the pipeline needs
to know about a dataset.

---

## 3. Datasets

### 3.1 UR Fall Detection Dataset (`urfall`)

<https://fenix.ur.edu.pl/mkepski/ds/uf.html>. It has 30 falls and 40 ADL sequences recorded
by 5 people, with no subject IDs per sequence. The streams are:

* a waist accelerometer, 3-axis, in g, at about 60 Hz, with no gyroscope;
* two Kinects (cam0 in front, cam1 on the ceiling; cam1 exists for falls only);
* the 8 depth-silhouette features the authors ship per frame.

The `download` command fetches, for every sequence, `*-acc.csv`, `*-data.csv` (frame times)
and `*-cam0.mp4`, plus `urfall-cam0-falls.csv` and `urfall-cam0-adls.csv`. The mp4 shows
depth and RGB side by side (640 × 240), and the adapter crops the RGB half.

| stream | modality | channels |
|---|---|---|
| `acc` | `IMU` | ax, ay, az, sv_total |
| `depth_feat` | `DEPTH` | HeightWidthRatio, MajorMinorRatio, BoundingBoxOccupancy, MaxStdXZ, HHmaxRatio, H, D, P40 |
| `cam0_flow` (after `extract-flow`) | `CAM` | 400 optical-flow cells |
| `cam0_pose` (after `extract-pose`) | `SKELETON` | COCO-17 x, y |

**Classes.** UR Fall labels every frame with a posture: −1 not lying, 0 falling, 1 lying.
That label means different things in fall and ADL sequences, so it is mapped to five classes:

| id | class | source |
|---|---|---|
| 1 | upright | −1 anywhere |
| 2 | **falling** | 0 in a fall sequence (the default `fall_classes: [2]`) |
| 3 | lying after fall | 1 in a fall sequence |
| 4 | lying down (ADL) | 0 in an ADL sequence (lying down on purpose) |
| 5 | lying (ADL) | 1 in an ADL sequence |

Set `fall_classes: [2, 3]` if being on the floor after a fall should also count as a fall.

Experiments (the UR Fall counterpart of the paper's seven): `IMU`, `DEPTH`, `CAM`, `IMU+DEPTH`,
`IMU+CAM`, `DEPTH+CAM`, `IMU+DEPTH+CAM`.

### 3.2 RA-Pak-Edy sessions (`imu_session`)

This is the `data-collector-IMU-mobile` export. Each session folder contains:

```
cameras.json                       started_at_ms of every webcam (video clock)
videos/<sid>_camN_video_sync.webm  4 × Logitech C270, VP9, 1280×720
data/manifest.json                 subject, operator, label_timeline (authoritative label change points)
data/<sid>_{waist,chest,thigh_right}_consolidated.csv
     timestamp_ms, acc_x_g, acc_y_g, acc_z_g, gyro_x_degs, gyro_y_degs, gyro_z_degs, label_id, sample_kind, …
```

The adapter does the following:

* It finds every session under `root`. Nested folders and several subjects are both fine.
* It uses `timestamp_ms` as the time axis and never assumes a fixed rate. The exports are
  irregular, at about 96 Hz true and 100 Hz per row.
* It drops the `sample_kind != 0` rows, which are held or placeholder samples.
* It cuts the session at the label change points in `manifest.json → integrity.label_timeline`.
  Each run of a non-zero label becomes one `Recording`. Its activity is the label (or
  `label_to_activity[label]`), and its trial is the occurrence count within the session.
  Label `0` (idle) is dropped unless `keep_idle: true`.
* It adds `cam1..cam4` as `VideoSource`s, with frame time = `started_at_ms + i / video_fps`.
  `cam4` is rotated, because it was filmed sideways.

| stream | modality | channels |
|---|---|---|
| `waist`, `chest`, `thigh_right` | `IMU` | acc_x, acc_y, acc_z (g), gyro_x, gyro_y, gyro_z (°/s) |
| `camN_flow` (after `extract-flow`) | `CAM` | 400 optical-flow cells |
| `camN_pose` (after `extract-pose`) | `SKELETON` | COCO-17 or body-12 x, y |

**You must set `fall_classes`.** The export only has numeric `label_id`s (1–34 in the
`francis` session). Which of them are falls is defined by your protocol, so the YAML leaves
it empty. `inspect` prints the duration and the peak |acc| of every label to help you check
your sheet. Falls usually peak above 2–3 g, but jumping does too.

Experiments: `IMU` (all three phones), `waist`, `chest`, `thigh`, `waist+chest`, `CAM`,
`cam1`, `IMU+CAM`, `SKELETON`, `IMU+SKELETON`. An experiment whose streams don't exist yet
(for example CAM before `extract-flow`) is skipped with a message.

### 3.3 Original UP-Fall (`upfall`)

To validate the implementation against the paper's own numbers, put the consolidated dataset
(<https://sites.google.com/up.edu.mx/har-up/>) in `data/upfall/` and run:

```bash
python -m upfall_bench run -d configs/datasets/upfall.yaml
```

The adapter reads `SubjectXActivityYTrialZ.csv`: timestamp, the 42 signals in Table 6
order, then Subject, Activity, Trial and Tag. The label is the last column, which inside a
fall trial switches between standing, falling and lying. Camera zips are read directly. The
experiments are the paper's seven: `IR`, `IMU`, `IMU+EEG`, `IR+IMU+EEG`, `CAM`, `IR+CAM`,
`IMU+EEG+CAM`.

### 3.4 Improved 3D-skeleton UP-Fall (`upfall_skeleton`)

The release has 82 files `C{cam}S{subj}A{act}T{trial}.csv`. Each holds 100 frames × 33
MediaPipe joints × (X, Y, Z) plus `LABEL` (impact 1/0), for fall activities 1–5 only. The
adapter reads the zips directly and tolerates the misnamed files in the release
(`C2S1_A1_T2`, `CS3A4T2`, `3S3A4T1`) and the inconsistent headers (`LLABEL`, missing header).

### 3.5 Synthetic (`synthetic`)

A fake dataset with the UP-Fall structure. Use it for tests and for trying options quickly.

---

## 4. Commands

```
python -m upfall_bench <command> -d <dataset.yaml> [-b <benchmark.yaml>] [-s key=value ...]
```

| command | does |
|---|---|
| `list` | registered adapters and the available dataset configs |
| `download` | download the dataset (UR Fall only, the others need registration or manual download) |
| `inspect` | recordings, subjects, streams and rates; per-activity duration and acceleration peak; time per label class. Writes CSVs to `results/<dataset>/` |
| `extract-flow` | Horn–Schunck optical flow → 20×20 magnitude per frame → `CAM` stream (cached in `cache/`) |
| `extract-pose` | Keypoint R-CNN (or MediaPipe) skeletons → `SKELETON` stream (cached) |
| `windows` | only build and cache the window feature tables, and print class counts |
| `run` | **the UP-Fall benchmark**: selection + RF/SVM/MLP/kNN for every window × experiment × task |
| `cnn` | the vision CNN of Sec. 5.6 on 28×28 frames |
| `skeleton` | the 3D-skeleton benchmark (ML on frames + LSTM/BiLSTM/CNN/ST-GCN on sequences) |
| `report --run <folder>` | rebuild the tables and figures of a finished run |

Useful flags: `run --rebuild` recomputes the cached windows; `--out results/x/y` sets the run
folder; `extract-* --overwrite` recomputes cached videos.

---

## 5. Configuration

The config is built from three layers, in order:

1. [configs/benchmarks/defaults.yaml](configs/benchmarks/defaults.yaml), which documents every option;
2. a **benchmark** file (`-b`, default `upfall2019.yaml`);
3. a **dataset** file (`-d`);

then any number of `-s dotted.key=value` overrides (values are parsed as YAML).

| benchmark file | purpose |
|---|---|
| `upfall2019.yaml` | faithful paper protocol (default) |
| `quick.yaml` | one window, 2 rounds, short selection curve: minutes instead of hours |
| `subject_independent.yaml` | group split by trial (or `loso`), selection inside each training split, standardised features |
| `koffi2024_skeleton.yaml` | 3D-skeleton benchmark settings |

Examples:

```bash
# only 2-s windows, binary task, 5 rounds
python -m upfall_bench run -d configs/datasets/urfall.yaml -s arc.windows=[2] -s arc.tasks=[binary] -s evaluation.rounds=5
# keep the IMUs at 100 Hz instead of the paper's 18 Hz consolidation
python -m upfall_bench run -d configs/datasets/ra_pak_edy.yaml -s arc.resample_hz=native
# add the extra models, change RF size
python -m upfall_bench run -d configs/datasets/urfall.yaml -s "models={RF: {n_estimators: 100}, SVM: {}, NB: {}, SGD: {}}"
# leave-one-subject-out once you have >= 2 subjects
python -m upfall_bench run -d configs/datasets/ra_pak_edy.yaml -b configs/benchmarks/subject_independent.yaml -s evaluation.protocol=loso
```

**Resources.** The `resources:` block keeps a run from taking over the machine:

| key | default | meaning |
|---|---|---|
| `n_jobs` | 2 | CPU cores for Random Forest, feature selection and numpy/BLAS |
| `torch_threads` | 2 | CPU threads for PyTorch |
| `device` | auto | `auto` = GPU if present, `cpu` = never touch the GPU |
| `gpu_batch` | 4 | frames per pose-model call |
| `gpu_memory_fraction` | 0.5 | share of GPU memory PyTorch may use (`null` = no cap) |

Raise them on the server (see [SERVER_GUIDE.md](SERVER_GUIDE.md)). Run one command at a
time on a laptop.

**Subsets.** `-s "dataset.include=['fall-0[1-4]', 'adl-0[1-4]']"` keeps only recordings whose
id matches one of the glob patterns. This is handy for trying `extract-pose` or `cnn` on a few
videos first. Quote the patterns, because `[` is YAML syntax.

**Experiments.** The `experiments:` section of a dataset YAML maps a name to a list of
tokens. A token matches a **modality tag** (`IMU`), a **stream name** (`waist`) or a **glob**
on stream names (`cam*_flow`):

```yaml
experiments:
  IMU: [IMU]
  waist-only: [waist]
  front-cams: ["cam1_flow", "cam2_flow"]
  IMU+CAM: [IMU, CAM]
```

---

## 6. Adding a new dataset

This is the only place where dataset-specific code lives.

1. **Copy the template**:
   [upfall_bench/datasets/_template.py](upfall_bench/datasets/_template.py) →
   `upfall_bench/datasets/my_dataset.py`. Rename the class and uncomment
   `@register("my_dataset")`.
2. **Implement `load_recordings()`**. It returns one `Recording` per trial: fill in `subject`,
   `activity`, `trial`, the label timeline, one `Stream` per sensor with a modality tag, and
   optionally `VideoSource`s. Keep timestamps in ms on one clock per recording.
3. **Register it**: add `from . import my_dataset` to
   [upfall_bench/datasets/__init__.py](upfall_bench/datasets/__init__.py). Alternatively,
   skip steps 1–3 and point the YAML at your own file:
   `adapter: "path/to/my_dataset.py:MyDatasetAdapter"`.
4. **Write the config**: copy [configs/datasets/_template.yaml](configs/datasets/_template.yaml),
   then set `root`, `fall_classes`, `classes` and `experiments`. Every key under `dataset:`
   is available in the adapter as `self.dcfg[...]`.
5. **Check it**: run `inspect` and confirm the durations, rates and label time per class.
   Then run `windows`, then `run -b configs/benchmarks/quick.yaml`, then the full `run`.

Optional overrides: `default_classes()`, `default_fall_classes()`, `download()`.

Optical flow, pose extraction, the CNN and the skeleton benchmark work for any adapter that
fills `videos`. A camera given as `VideoSource(cam, path, t0_ms=…, fps=…)` is decoded at that
constant rate with ffmpeg. One given with `frame_times_ms=[…]` uses the exact frame times.
`crop` and `rotate` are applied before anything else.

Tips:

* **Session-style data** (one long take, labels switched live) → cut it into one Recording per
  label run, as [imu_session.py](upfall_bench/datasets/imu_session.py) does.
* **Trial-style data** (one file per trial) → one Recording per file, as
  [upfall.py](upfall_bench/datasets/upfall.py) does.
* **Precomputed per-frame features** shipped with a dataset → add them as a `signal` stream
  with their own modality tag, like UR Fall's `DEPTH`.

---

## 7. Outputs

`run` writes `results/<dataset>/<run>/`:

| file | content |
|---|---|
| `results.md` | best configuration per task, Tables 13 and 14, all figures |
| `table13_best_model_per_window.csv` | per modality × window: best model by mean F1 (paper Table 13) |
| `table14_best_window_per_model.csv` | per modality × model: best window by mean F1 (paper Table 14) |
| `summary.csv` | mean and std of every metric for every task × experiment × window × model |
| `raw_metrics.csv` | one row per round, for your own statistics (e.g. Wilcoxon tests) |
| `selection/*.json` | selected features, full voted ranking, votes, incremental curve |
| `confusion/*.npz` | confusion matrix summed over the test rounds, per combination |
| `figures/selection_*.png` | incremental accuracy curves (paper Figs. 8–14) |
| `figures/f1_*.png` | F1 per modality and model |
| `figures/confusion_*.png` | confusion matrix of the best configuration (paper Fig. 16) |
| `config.yaml`, `log.txt` | the exact configuration and console log, for reproducibility |

`cnn` writes `cnn_results.json` and `cnn_confusion.npz`. `skeleton` writes `results.md`,
`skeleton_table.csv` and `raw_metrics.csv`. The `cache/` folder holds window tables, flow,
poses and CNN frames. It is safe to delete, and everything in it is recomputed when needed.

---

## 8. Benchmark details

### 8.1 Windowing (Sec. 5.2)

* Windows of 1, 2 and 3 s with 50 % overlap, over each recording's labelled span.
* Each stream is first put on a uniform grid by sample-and-hold. The paper put every device
  on the camera clock at ~18.4 Hz with "drop-sampling interpolation" (repeat the last value),
  which is `resample_hz: 18`. `native` keeps each stream at its own rate.
* A window's label is the **most frequent label** inside it, as in the paper. The binary
  label is 1 if that label is in `fall_classes`.
* A stream with samples over less than 50 % of a window gives NaN features. Those windows are
  dropped per experiment.

### 8.2 Features (Tables 10 and 11)

These are computed for every channel of every `signal` stream.

| feature | definition used |
|---|---|
| mean, std, median, max, min | usual |
| rms | √(mean x²) |
| zero_crossings | number of sign changes of the raw signal |
| skewness, kurtosis | scipy (Fisher kurtosis, biased) |
| q1, q3 | 25th and 75th percentiles |
| autocorrelation | lag-1 autocorrelation of the mean-removed signal |
| mean_freq | Σ f·P(f) / Σ P(f), with P = \|FFT\|² |
| median_freq | frequency that splits the spectral power in half |
| entropy | Shannon entropy (bits) of the normalised power spectrum |
| energy | Σ \|X_k\|² / N |
| principal_freq | frequency of the largest non-DC power peak |
| spectral_centroid | Σ f·\|X(f)\| / Σ \|X(f)\| |

**Vision (Sec. 5.2, Eq. 1).** For every pair of consecutive frames, the pipeline computes
Horn–Schunck flow (u, v), then d = √(u² + v²), resizes it to 20 × 20 and flattens it to 400
values. These vectors are **averaged** inside the window, giving 400 features per camera.

### 8.3 Feature selection (Sec. 5.3)

The paper's 8 steps, with Python versions of its Weka evaluators:

1. Drop features with missing values or zero variance.
2. Score the features with five methods: `cfs` (greedy forward CFS merit, ≈
   CfsSubsetEval), `correlation` (≈ CorrelationAttributeEval), `relieff` (≈
   ReliefFAttributeEval, k = 10), `oneattr_tree` (CV accuracy of a depth-3 tree on one
   feature, ≈ ClassifierAttributeEval + DecisionTable) and `mutual_info`.
3. Take the top 100 of each method.
4. Count how many of these subsets each feature appears in.
5. Keep the features with ≥ 2 votes.
6. Sort them by votes.
7. Run the incremental analysis: 3-fold CV accuracy of RF (10 trees) on the first k features,
   for k = 1 … 80.
8. Keep the smallest k within 0.5 accuracy points of the best k. The paper picked k by eye
   from the curve.

### 8.4 Models (Table 12)

Table 12's parameter names are scikit-learn's, so they map one to one:

| model | scikit-learn |
|---|---|
| RF | `RandomForestClassifier(n_estimators=10, min_samples_split=2, min_samples_leaf=1, bootstrap=True)` |
| SVM | `SVC(C=1.0, kernel="rbf", gamma="auto" (= 1/features), shrinking=True, tol=1e-3)` |
| MLP | `MLPClassifier((100,), relu, solver="adam", alpha=1e-4, batch_size=min(200, n), lr=1e-3, shuffle, tol=1e-4, β=(0.9, 0.999), ε=1e-8, max_iter=10)` |
| kNN | `KNeighborsClassifier(n_neighbors=5, leaf_size=30, metric="euclidean")` |
| NB, SGD | `GaussianNB()`, `SGDClassifier()` (the skeleton benchmark's extra models) |

The table lists "solver = stochastic gradient" together with Adam's β₁, β₂ and ε, so this is
sklearn's `adam` solver. No feature scaling is mentioned, so the default is `scaler: none`.

### 8.5 Evaluation (Sec. 5.4)

* There are 10 rounds of a random, stratified 70 / 30 split of the windows, and the reported
  value is mean ± std.
* The metrics are accuracy, precision, sensitivity, specificity and F1 (Eq. 2–6). For more
  than two classes, precision, sensitivity, specificity and F1 are macro-averaged one-vs-rest.
  That is why the paper reports ~99.5 % specificity next to ~70 % sensitivity. For `binary`,
  the positive class is *fall*.
* Table 13 gives the best model per modality × window, and Table 14 the best window per
  modality × model. Both pick by mean F1.
* Other protocols: `kfold`, `group_holdout` (split by trial) and `loso` (leave one subject
  out). See §10.

### 8.6 Vision CNN (Sec. 5.6, Fig. 17)

The input is a 28 × 28 × 3 frame, zero-centred. The network is three blocks of
conv 3×3 → ReLU → max-pool 2×2, with 8, 16 and 32 filters, then a fully connected layer and
softmax.

Training uses SGD with momentum 0.9 (MATLAB `sgdm`), learning rate 0.001, L2 0.004, 5
epochs and mini-batch 100. The data is split into 2/3 train and 1/3 test. Within the
training part, 5-fold CV picks the fold model with the best validation accuracy, and that
model is scored on the test set. Only one camera is used, as in the paper (`cnn.cameras`).
Frames are labelled with the label at their timestamp.

**Small datasets.** The paper trained on 140k frames. UR Fall has about 5k, so 5 epochs at
lr 0.001 are only ~160 SGD steps, and the network predicts the majority class (F1 0.16 in our
test). With `-s cnn.epochs=30 -s cnn.lr=0.01` the same network reaches F1 0.84. Report both
settings and say which one you used.

### 8.7 3D-skeleton benchmark (Koffi et al. 2024)

* **ML** classifies every frame from all joint coordinates (99 values for MediaPipe-33)
  using SVM, MLP, RF, KNN, NB and SGD, with an 80 / 20 split.
* **DL** uses sliding windows of `seq_len` frames (default 30, stride 5), labelled by the
  last frame, with these models:
  * LSTM (2 × 128);
  * BiLSTM;
  * 1D-CNN;
  * ST-GCN: 3 spatial-temporal graph blocks on the skeleton graph of
    [layouts.py](upfall_bench/skeleton/layouts.py).
  They are trained with Adam (lr 1e-3) and class-weighted cross-entropy.
* **Labels** (`skeleton.label`):
  * `class` uses the dataset label;
  * `fall` is 1 when the frame's class is in `fall_classes`;
  * `impact` applies the paper's semi-automatic rule: inside each fall, frames within ±500 ms
    of the acceleration-magnitude peak are impacts, if that peak exceeds 2 g. This works for
    any dataset with an accelerometer (UR Fall, RA-Pak-Edy).
* **Skeleton sources**:
  * native, from the UP-Fall release;
  * `extract-pose` with `torchvision` Keypoint R-CNN (COCO-17 2D; `layout: body12` drops
    the face, matching the RA-Pak-Edy privacy pipeline);
  * `extract-pose` with `mediapipe` BlazePose (33 joints, 3D, as in the paper; needs
    `pip install mediapipe`).

---

## 9. Deviations from the papers

| topic | paper | here | why / how to change |
|---|---|---|---|
| feature-selection tools | Weka CfsSubsetEval (best-first + greedy), CorrelationAttributeEval, ReliefF, ClassifierAttributeEval (ZeroR, DecisionTable) | Python equivalents (§8.3); `mutual_info` replaces ZeroR, which gives every single attribute the same score | `selection.methods` |
| choice of k | read from the curve by eye | smallest k within `tolerance` of the best | `selection.tolerance`, `selection.fixed_k` |
| selection sample | all windows | stratified subsample of 3000 windows | `selection.max_samples` |
| feature formulas | named only | definitions in §8.2 | `arc/features.py` |
| optical flow size | 640 × 480 | computed at 160 × 120, then 20 × 20 | `flow.work_size: [640, 480]` |
| optical flow parameters | Horn–Schunck, parameters not given | α = 1, 10 iterations (MATLAB `opticalFlowHS` defaults) | `flow.alpha`, `flow.iterations` |
| CNN framework | MATLAB (implied by `sgdm`, L2, "regularization coefficient") | PyTorch, same topology and hyper-parameters | `cnn.*` |
| skeleton ML/DL hyper-parameters | not published (repo code missing) | sensible defaults (§8.7), standardised features | `skeleton.*` |
| UP-Fall "unknown" class 12 | 12 classes in Fig. 16 | whatever labels the dataset has (UP-Fall tag 20 = unknown) | adapter |

---

## 10. Methodological notes for a thesis

* **Window-level random splits are optimistic.** With 50 % overlap, neighbouring windows
  share half their samples, so a random 70/30 split puts near-copies of test windows in
  training. The paper does this, so `upfall2019.yaml` does too, to stay comparable. Also
  report `group_holdout` (by trial) or `loso` (by subject) with
  `-b configs/benchmarks/subject_independent.yaml`.
* **Global feature selection leaks**, because it sees the test windows. `selection.scope:
  per_split` re-selects on every training set; it is slower but clean.
* **One subject = no subject-independent claim.** The `francis` session has 1 subject and
  each of its 34 labels occurs once. Multiclass results on it measure within-session
  separability only. With `group_holdout`, the test trials' classes are never seen in
  training, so use the binary task there. Add sessions (more subjects and repetitions) and
  use `loso`.
* **The MLP is weak by design.** Table 12 caps it at 10 epochs on unscaled features, so the
  MLP and SVM are often poor (as in the paper, where SVM never wins). Report the paper setting
  and, separately, `-s preprocessing.scaler=standard`.
* **Class imbalance.** Falls are a few percent of windows, so accuracy is meaningless for the
  binary task. Look at F1 and sensitivity. For selection on imbalanced data, try
  `-s selection.scoring=f1_macro`.
* **Label semantics differ between datasets.** UP-Fall marks only the fall moment as the fall
  activity (standing and lying around it are separate labels). UR Fall gives a "falling"
  phase. RA-Pak-Edy labels the whole activity segment (~10–20 s). This changes what "fall
  window" means. Write down which definition you use (`fall_classes`, `segment_padding_s`,
  or impact labels).
* **Synchronisation.** RA-Pak-Edy videos are mapped with `started_at_ms` from `cameras.json`.
  If you measure a constant camera offset (e.g. with the flash), shift `t0_ms` in the adapter.

---

## 11. Project structure

```
UP-Fall-Algorithm/
├── README.md
├── SERVER_GUIDE.md       rsync + tmux commands for the GPU server
├── requirements.txt
├── configs/
│   ├── benchmarks/   defaults.yaml · upfall2019.yaml · quick.yaml · subject_independent.yaml · koffi2024_skeleton.yaml
│   └── datasets/     urfall.yaml · ra_pak_edy.yaml · upfall.yaml · upfall_skeleton.yaml · synthetic.yaml · _template.yaml
├── upfall_bench/
│   ├── cli.py            commands (python -m upfall_bench …)
│   ├── config.py         YAML loading / merging / --set
│   ├── core.py           Recording, Stream, VideoSource, sample-and-hold
│   ├── video.py          ffmpeg / image-zip frame reader
│   ├── datasets/         ONE FILE PER DATASET  (+ _template.py, base.py with the registry)
│   ├── derive/           flow.py (Horn–Schunck), pose.py (Keypoint R-CNN / MediaPipe), impact.py (2 g labels)
│   ├── arc/              windowing · features · selection · models · evaluation · runner · report
│   ├── skeleton/         bench.py (ML + DL), dl.py (LSTM, BiLSTM, CNN, ST-GCN), layouts.py (joint graphs)
│   └── vision_cnn.py     Sec. 5.6 CNN
├── tests/test_smoke.py   python -m pytest tests -q
├── data/      (git-ignored) raw datasets
├── cache/     (git-ignored) derived data
└── results/   (git-ignored) runs
```

---

## 12. Troubleshooting

* **Laptop becomes slow or crashes**: lower `resources.n_jobs`, use `-s resources.device=cpu`,
  and never start several commands in parallel. Pose extraction over full sessions belongs on
  the GPU server ([SERVER_GUIDE.md](SERVER_GUIDE.md)).
* **Disk space**: `data/` holds raw datasets (UR Fall ≈ 100 MB, your sessions ≈ 1 GB each with
  videos). `cache/` holds derived data (≈ 15 MB of flow per camera per session) and can be
  deleted at any time. Results are small.
* **`ffmpeg not found`**: run `pip install imageio-ffmpeg`, or put ffmpeg on PATH. OpenCV is
  not used for video because VP9 WebM from browser recorders has no frame index.
* **`fall_classes is empty`**: set it in the dataset YAML; the binary task is skipped until
  you do.
* **`no stream matches ['CAM'], skipped`**: run `extract-flow` first. For `SKELETON`, run
  `extract-pose`.
* **Changed the data or the config but the results look identical**: window tables are cached
  by content and settings, but if in doubt run `run --rebuild` or delete `cache/<dataset>/windows`.
* **Slow `run`**: start with `-b configs/benchmarks/quick.yaml`. The expensive parts are
  `relieff` and `oneattr_tree` on large feature sets (CAM has 400 per camera). Lower
  `selection.max_samples`, or raise `selection.curve_step`.
* **Pose extraction speed**: Keypoint R-CNN runs at about 8–20 frames/s on a GPU. For the
  4 × 20-min RA-Pak-Edy videos at 18 fps, only frames inside activities are processed. Set
  `pose.fps: 10` or `pose.frame_size: [640, 360]` to speed it up.
* **MLP "did not converge" warnings**: expected (`max_iter=10` from Table 12), and silenced
  in `run`.

---

## 13. References

```bibtex
@article{martinez2019upfall,
  title   = {UP-Fall Detection Dataset: A Multimodal Approach},
  author  = {Mart{\'i}nez-Villase{\~n}or, Lourdes and Ponce, Hiram and Brieva, Jorge and Moya-Albor, Ernesto and N{\'u}{\~n}ez-Mart{\'i}nez, Jos{\'e} and Pe{\~n}afort-Asturiano, Carlos},
  journal = {Sensors}, volume = {19}, number = {9}, pages = {1988}, year = {2019},
  doi     = {10.3390/s19091988}
}
@article{koffi2024improved,
  title  = {An Improved 3D Skeletons UP-Fall Dataset: Enhancing Data Quality for Efficient Impact Fall Detection},
  author = {Koffi, Tresor Y. and Mourchid, Youssef and Hindawi, Mohammed and Dupuis, Yohan},
  year   = {2024}, note = {https://zenodo.org/records/12773013}
}
@article{kwolek2014urfall,
  title   = {Human fall detection on embedded platform using depth maps and wireless accelerometer},
  author  = {Kwolek, Bogdan and Kepski, Michal},
  journal = {Computer Methods and Programs in Biomedicine}, volume = {117}, number = {3}, pages = {489--501}, year = {2014}
}
@inproceedings{horn1981,
  title = {Determining optical flow}, author = {Horn, Berthold K. P. and Schunck, Brian G.},
  booktitle = {Artificial Intelligence}, volume = {17}, pages = {185--203}, year = {1981}
}
@inproceedings{yan2018stgcn,
  title = {Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition},
  author = {Yan, Sijie and Xiong, Yuanjun and Lin, Dahua}, booktitle = {AAAI}, year = {2018}
}
```

The UR Fall dataset is licensed CC BY-NC-SA 4.0 (non-commercial academic use).
