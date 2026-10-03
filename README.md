# Does face censoring degrade pose estimation?

Measures what happens to human pose estimation when faces are blurred or removed
for privacy. For each video it runs the pose estimator twice — once on the
original, once on the censored version — matches the two up person by person,
and reports how far the body joints moved.

Built for **action footage**, so it targets the failure modes that matter there:
people the censoring made undetectable, joints that jump, and confidence that
collapses.

---

## 1. The two ways to run it

The same pipeline exists twice. They are not redundant: one is for looking, the
other is for producing.

| | `blur.ipynb` (notebook) | `blurpose.py` (script) |
| --- | --- | --- |
| **For** | exploring, tuning, figures for your thesis | full runs, the GPU server |
| **Frames** | 120 per video by default | **the entire video** |
| **Memory** | holds censored frames in RAM | streams in chunks — constant memory, any clip length |
| **Passes over the video** | three (pose, censor, pose) | **one** (pose → censor → pose, per chunk) |
| **Pose cache** | yes, in `work/` — re-runs are instant | no, single pass |
| **Previews** | GIF + figures shown inline | GIF + figures written to disk |
| **Needs a display** | yes (Jupyter) | no — headless, `matplotlib Agg` |
| **Survives SSH disconnect** | no | yes, with `tmux` |
| **Configured by** | the `Config` dataclass in the setup cell | command-line flags |

They produce **the same output files and the same metrics**. Tune settings in
the notebook on a short clip, then pass the equivalent flags to the script for
the real run.

### Why the script streams

Holding a full video in memory is not possible: a 10-minute 720p clip is about
18,000 frames, roughly 48 GB. So the script processes a chunk of frames at a
time and does everything to that chunk before discarding it:

```
read N frames → pose (original) → censor → pose (censored) → write → discard
```

This has a second benefit that matters scientifically. The censored frames go
straight from memory into the pose estimator, so the "after" measurement never
passes through a video encoder. A write-and-re-decode round-trip shifts pixels
by about 2 on average and up to 31 at the extremes, and that error would land
directly in the displacement numbers. Both versions avoid it; the notebook by
keeping frames in RAM, the script by never materialising them.

---

## 2. Libraries

| what | library | used for |
| --- | --- | --- |
| **Censoring** | [`blurfaces`](https://github.com/AlexPasqua/blurfaces) | `apply_censor` — the `blur` / `blackout` / `pixel` / `bar` methods, **copied unchanged**. The original file is kept at `vendor/blurfaces_original.py` with its licence so the provenance is checkable. |
| **Face detection** | **OpenCV YuNet** (`cv2.FaceDetectorYN`) | finding the faces to censor. Ships with OpenCV, model downloaded on first use, no compiler needed. |
| | *optional:* [`insightface`](https://github.com/deepinsight/insightface) | only if you need `--mode target/exclude` — it produces face embeddings so you can censor one specific person. Requires a build toolchain. |
| **Pose estimation** | **torchvision Keypoint R-CNN** (`keypointrcnn_resnet50_fpn`) | the pose estimator. COCO-pretrained, weights download automatically, runs on GPU. |
| | [`AlphaPose`](https://github.com/MVIG-SJTU/AlphaPose) | vendored at `vendor/AlphaPose/` and wired up, but **cannot run** — see `ALPHAPOSE_SETUP.md`. Kept for reference. |
| **Image & video I/O** | **OpenCV** (`opencv-python`) | reading and writing video, resizing, drawing the skeleton, Gaussian blur, pixelation, and `cv2.inpaint` for face removal. |
| **Tensors / GPU** | **PyTorch** | runs the pose model. |
| **Numerics** | **NumPy** | keypoint arrays, all the metrics. |
| **Figures** | **Matplotlib** | the 4-panel metrics figure. |
| **GIF** | **Pillow** | the animated preview. |
| **Progress bars** | **tqdm** | notebook only. |

Install: `pip install opencv-python torch torchvision numpy matplotlib pillow tqdm`
(see `requirements.txt`).

### Censoring methods

`--method` / `CFG.blur_method`, roughly in order of how much they destroy:

| method | what it does | source |
| --- | --- | --- |
| `bar` | black bar across the eye line only | blurfaces |
| `blur` | Gaussian blur, sigma 30 | blurfaces |
| `pixel` | 8×8 mosaic | blurfaces |
| `fill` | flat patch in the surrounding average colour | ours |
| `blackout` | solid black rectangle | blurfaces |
| `inpaint` | face reconstructed away from surrounding pixels | ours |

`blackout` vs `inpaint` is the informative pair. Both delete the face, but
`blackout` leaves a hard-edged rectangle that is itself a conspicuous landmark,
while `inpaint` blends into the background. If `blackout` scores well and
`inpaint` collapses, the black box was doing the work.

`--box-pad` (default 0.25) controls how much of the *head* goes with the face —
the detector returns a face box, not a head box. `--miss-hold` (default 3) keeps
censoring with the last known box for a few frames when detection drops out,
which happens constantly in action footage.

---

## 3. The skeleton

**12 body joints, no face.** `--keypoints body12` slices the five facial
keypoints (nose, eyes, ears) off the model output *inside the inference
function*, so no face keypoint reaches any array, drawing or export.

```
shoulders · elbows · wrists · hips · knees · ankles
```

Grouped and colour-coded as **torso** (cyan), **arms** (green), **legs**
(periwinkle). `--keypoints coco17` restores the face group (amber) if you want
to see it get mangled.

### On "never inferred"

Keypoint R-CNN's head emits a fixed stack of 17 heatmaps from shared features.
There is no setting for 12, and masking the output is the only option short of
retraining. That is fine in practice: **the heatmaps are independent output
channels, so discarding the face ones cannot change the values of the body
ones.** Masking and never-inferring give bit-identical body keypoints.

Two skeletons in the literature genuinely have no facial joints —
**CrowdPose-14** and **MPII-16** — but the libraries serving them
(OpenPifPaf, MMPose, AlphaPose) will not install on Python 3.12 + torch 2.8,
while the ones that install easily (MediaPipe, YOLO-pose) all include facial
landmarks. Slicing torchvision's output is the best available option.

---

## 4. Metrics

Each frame, people are matched between the two runs by bounding-box IoU (greedy,
highest first), so someone moving across frame still lines up.

- **lost** — a person detected before censoring but not after. The most severe
  failure; watch this one.
- **PCK@t** — fraction of joints landing within `t ×` the person's bbox diagonal
  of where they were. Reported at 0.05, 0.10, 0.20.
- **OKS** — the COCO object keypoint similarity behind keypoint mAP.
- **displacement** — normalised distance per joint.
- **confidence delta** — how much the estimator's own certainty drops.

Only joints the *before* run was confident about (`--vis-thr`, default 0.2) are
scored. A joint the model never saw — occluded, out of frame — has no meaningful
position, and scoring it buries the real effect in noise. In action footage that
is a lot of joints.

Since the skeleton is body-only, **PCK and OKS are body-pose metrics by
construction** — exactly the number that answers *does censoring the face
disturb the rest of the pose?*

---

## 5. Folder layout

```
.
├── blur.ipynb              # notebook — exploring, 120 frames
├── blurpose.py             # script — full videos, headless
├── requirements.txt
├── README.md               # this file
├── SSH_GUIDE.md            # running it on the GPU server
├── ALPHAPOSE_SETUP.md      # why AlphaPose does not run, and what it would take
├── input/                  # ← put your videos here
├── output/                 # ← results appear here
├── work/                   # cache — see below
└── vendor/
    ├── blurfaces_original.py   # the unmodified source of apply_censor
    ├── blurfaces_LICENSE
    └── AlphaPose/              # vendored source, not runnable
```

### What `work/` is

**A cache. Nothing in it is a result, and it is always safe to delete** — the
only cost is re-downloading and recomputing.

| what lands there | written by | size | if you delete it |
| --- | --- | --- | --- |
| `face_detection_yunet_2023mar.onnx` | both | ~232 KB | re-downloaded automatically on the next run |
| `pose_<video>_<before/after>_<backend>_<keypointset>.npy` | **notebook only** | a few MB each | the pose pass re-runs (slow, but identical) |
| `alphapose_*/` | only if AlphaPose is used | varies | regenerated |

The pose cache is why the notebook feels fast on the second run. It is keyed by
video name, which pass, the backend, and the keypoint set — so changing
`keypoint_set` correctly produces a fresh entry instead of reusing stale
17-joint data. It does **not** key on the censoring method, which is deliberate:
the *before* pass does not depend on how you censor, so a sweep across methods
computes it once and reuses it.

The script does not write a pose cache at all — it streams in one pass, so there
is nothing to reuse. It only uses `work/` for the YuNet model.

**Delete `work/` whenever results look stale or you have changed something the
cache key does not cover.** It costs you time, never data.

One cache that is *not* in `work/`: the Keypoint R-CNN weights (~226 MB) go to
PyTorch's own cache at `~/.cache/torch/hub/checkpoints/` (Windows:
`C:\Users\<you>\.cache\torch\hub\checkpoints\`). That downloads once per machine.

### What lands in `output/`

One folder per *(video, method, backend)*, e.g.
`output/myclip__blur__torchvision/`:

| file | what it is |
| --- | --- |
| `*_1_blur_raw_*.mp4` | censored video, no overlay — what a privacy pipeline would ship |
| `*_2_pose_before.mp4` | original + skeleton — ground truth |
| `*_3_pose_after_*.mp4` | censored + skeleton |
| `*_4_compare_*.mp4` | the two side by side, with moved joints ringed in red |
| `*_compare_*.gif` | animated preview, renders anywhere |
| `*_metrics.png` | the 4-panel figure |
| `keypoints.csv` | **the skeleton data — cite this one** |
| `keypoints.json` | same data plus schema (notebook only) |
| `keypoints_*.npy` | pickled numpy, fast reload (notebook only) |
| `summary.json` | every metric, plus the exact settings used |

Plus `output/results__<method>__<keypointset>.csv` — one row per video, for
comparing across clips.

### The keypoint CSV

Long/tidy format, one row per joint per person per frame:

```
video, run, frame, person, keypoint, group, x, y, score
```

`run` is `before` or `after`; `person` is the per-frame detection index; `x, y`
are pixels in the original frame; `score` is the per-joint confidence (0–1).
Loads straight into pandas, Excel, R or SPSS. **This is the durable record** —
the `.npy` files need `allow_pickle=True` and a matching numpy version, so treat
them as a convenience only.

---

## 6. Running it

### Notebook

Put videos in `input/`, open `blur.ipynb`, run every cell top to bottom.
Processes 120 frames per video by default (`CFG.max_frames`). Everything is
configured in the `Config` dataclass in the setup cell.

> If you edit the notebook while it is open elsewhere, the editor can save its
> in-memory copy over changes made on disk. After any external edit: close the
> tab **without saving**, reopen, then **restart the kernel** — reloading the
> file alone does not replace functions already defined in memory.

### Script

```bash
python blurpose.py --input input --output output --method blur
```

Processes every video in `input/`, in full. Useful flags:

```bash
--max-frames 200       # cap, for a quick test (0 = entire video)
--method inpaint       # bar | blur | pixel | fill | blackout | inpaint
--keypoints coco17     # put the face keypoints back
--batch 8              # frames per GPU batch; raise on a big GPU, lower on OOM
--videos none          # skip rendering (barely faster — the GPU is the bottleneck)
--device cuda:1        # pick a GPU
```

Speed is dominated by the pose model, not by rendering or censoring: measured on
an RTX 3050, roughly **1.3–2.4 frames/sec**, and turning rendering off changed
almost nothing. Two pose passes per frame is the cost. Budget accordingly, and
see `SSH_GUIDE.md` for running long jobs on the server.

### Sweeping the methods

The real experiment is comparing censoring methods:

```bash
for m in bar blur pixel fill blackout inpaint; do
    python blurpose.py --method "$m"
done
```

Each writes its own `output/<video>__<method>__torchvision/` and
`results__<method>__body12.csv`. Compare them:

```python
import pandas as pd, glob
df = pd.concat([pd.read_csv(f) for f in glob.glob("output/results__*.csv")])
df.pivot_table(index="method", values=["pck_005", "mean_oks", "lost_rate"])
```

---

## 7. A result worth not misreading

`summary.json` records `face_detections`. **If it is 0, nothing was censored** —
the before and after videos are identical, so PCK comes out at 1.000 and OKS at
1.000. That is not a finding, it is a no-op, and the script prints a warning for
any video where it happens. It already occurred once on `cam1.webm`, where the
subject was too distant or turned away for the detector.

Always check `face_detections` before quoting a score.
