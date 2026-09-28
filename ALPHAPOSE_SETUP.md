# AlphaPose — what is here, and what it still needs

The AlphaPose source is vendored at `vendor/AlphaPose/` and the notebook's
`alphapose` backend is wired to it. **It cannot run yet**, and the two reasons are
both outside the notebook's control. This file is the honest account of what is
missing, so you can decide whether it is worth the effort for your thesis.

The notebook runs fine without it — `CFG.pose_backend = "torchvision"` is the
default and needs nothing.

## Blocker 1 — the weights are not in the zip

`vendor/AlphaPose/pretrained_models/` contains exactly one file: an **empty**
`get_models.sh`. The repository ships source only. You need two downloads:

| file | size | goes in |
| --- | --- | --- |
| `halpe26_fast_res50_256x192.pth` | ~140 MB | `vendor/AlphaPose/pretrained_models/` |
| `yolov3-spp.weights` | ~240 MB | `vendor/AlphaPose/detector/yolo/data/` |

Links are in `vendor/AlphaPose/docs/MODEL_ZOO.md`. They point at Google Drive and
Baidu, which gate large files behind a browser confirmation, so these have to be
fetched by hand — a script cannot pull them reliably.

## Blocker 2 — four CUDA extensions must be compiled

`vendor/AlphaPose/setup.py` builds `nms_cpu`, `nms_cuda`, `roi_align_cuda` and
`deform_conv_cuda` against the PyTorch C++ API, using headers (`THC/THC.h`) that
PyTorch removed after the 1.x series. This kernel runs **Python 3.12 + torch
2.8.0+cu128**, and the build fails against it. There is no flag for this; the
sources would have to be ported.

### The combination that does build

```
Python 3.9
torch 1.13.1 + torchvision 0.14.1  (cu117)
numpy 1.23.x
CUDA toolkit 11.7  +  Visual Studio 2019 Build Tools (C++ workload)
```

The CUDA toolkit and MSVC build tools are genuinely required — the extensions
compile at install time.

```powershell
conda create -n alphapose python=3.9 -y
conda activate alphapose
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --index-url https://download.pytorch.org/whl/cu117
pip install numpy==1.23.5 cython
cd "<project>\vendor\AlphaPose"
pip install -r requirements.txt
python setup.py build develop
```

If it fails on a missing compiler, install the **Desktop development with C++**
workload from the VS 2019 Build Tools and re-run from an *x64 Native Tools
Command Prompt*.

## Pointing the notebook at it

Once it builds and the weights are in place, in the config cell:

```python
CFG.pose_backend     = "alphapose"
CFG.alphapose_python = r"C:\Users\<you>\miniconda3\envs\alphapose\python.exe"
```

`CFG.alphapose_root` already points at `vendor/AlphaPose`. Nothing else changes —
this kernel never imports AlphaPose, it only runs `scripts/demo_inference.py` as a
subprocess and reads `alphapose-results.json` back. The notebook calls
`alphapose_preflight()` first, which names whichever piece is still missing rather
than failing somewhere inside the subprocess.

Sanity-check the subprocess on its own before running the whole notebook:

```powershell
C:\...\envs\alphapose\python.exe scripts\demo_inference.py `
  --cfg configs\halpe_26\resnet\256x192_res50_lr1e-3_1x.yaml `
  --checkpoint pretrained_models\halpe26_fast_res50_256x192.pth `
  --video <a test clip> --outdir out --sp
```

`out\alphapose-results.json` should appear.

## Is it worth it?

For a blur-vs-pose ablation, probably not on its own merits. The comparison is
*before against after* with the estimator held fixed, so the estimator's absolute
accuracy cancels out — Keypoint R-CNN answers the research question just as
validly. AlphaPose is worth the setup if your supervisor wants it specifically, or
if you need Halpe-26's foot and head keypoints, which COCO-17 does not have.

## Notes for action footage, once it runs

- `--sp` (single process) is required on Windows; the multiprocessing loader
  deadlocks there.
- AlphaPose outputs **Halpe-26**; the notebook slices the first 17 joints, which
  are already in COCO order, so both backends compare on the same skeleton.
- If people are lost during fast motion, the bottleneck is usually the YOLO
  detector rather than the pose head. `--detbatch 1 --posebatch 8` also helps on
  a 6 GB card.
