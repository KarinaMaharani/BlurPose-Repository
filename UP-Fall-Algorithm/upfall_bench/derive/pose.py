"""2D / 3D skeletons from video, cached as a ``SKELETON`` stream per camera.

Backends (``pose.backend``):
  * ``torchvision`` (default): Keypoint R-CNN ResNet-50 FPN, COCO-17 2D joints. Same model as
    the RA-Pak-Edy blur/pose pipeline. ``pose.layout: body12`` drops the 5 face joints.
  * ``mediapipe``: BlazePose, 33 joints with x, y, z, as used by Koffi et al. 2024
    (``pip install mediapipe``).

Per frame the most confident person is kept. Coordinates are normalised by frame width/height
to [0, 1] (MediaPipe convention). Frames without a person are NaN and get forward-filled
later, at windowing time.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..config import torch_device
from ..datasets.base import save_derived
from ..skeleton.layouts import BODY12, COCO17, MEDIAPIPE33
from ..video import in_spans, iter_frames, unique_videos


class TorchvisionPose:
    def __init__(self, layout="coco17", min_score=0.5, batch=4, dev="cpu"):
        import torch
        import torchvision
        self.torch = torch
        self.dev = dev
        w = torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=w).eval().to(self.dev)
        self.keep = list(range(5, 17)) if layout == "body12" else list(range(17))
        self.names = BODY12 if layout == "body12" else COCO17
        self.dims, self.min_score, self.batch = 2, min_score, batch

    def __call__(self, frames: list[np.ndarray]) -> np.ndarray:
        t = self.torch
        out = []
        with t.no_grad():
            for i in range(0, len(frames), self.batch):
                ims = [t.from_numpy(np.array(f)).permute(2, 0, 1).float().div(255).to(self.dev) for f in frames[i:i + self.batch]]
                for f, det in zip(frames[i:i + self.batch], self.model(ims)):
                    h, w = f.shape[:2]
                    row = np.full((len(self.keep), 2), np.nan, np.float32)
                    if len(det["scores"]) and float(det["scores"][0]) >= self.min_score:
                        k = det["keypoints"][0].cpu().numpy()[self.keep, :2]
                        row = k / [w, h]
                    out.append(row.ravel())
        return np.stack(out)


class MediaPipePose:
    def __init__(self, **_):
        import mediapipe as mp
        self.pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=2)
        self.names, self.dims = MEDIAPIPE33, 3

    def __call__(self, frames):
        out = []
        for f in frames:
            r = self.pose.process(f)
            if r.pose_landmarks:
                out.append(np.array([[p.x, p.y, p.z] for p in r.pose_landmarks.landmark], np.float32).ravel())
            else:
                out.append(np.full(33 * 3, np.nan, np.float32))
        return np.stack(out)


def extract_pose(adapter, recs, cfg: dict, overwrite: bool = False):
    pcfg = cfg.get("pose", {})
    backend = pcfg.get("backend", "torchvision")
    model = (MediaPipePose() if backend == "mediapipe" else
             TorchvisionPose(pcfg.get("layout", "coco17"), float(pcfg.get("min_score", .5)),
                             int(cfg.get("resources", {}).get("gpu_batch", 4)), torch_device(cfg)))
    axes = "xyz"[:model.dims]
    channels = [f"{n}_{a}" for n in model.names for a in axes]
    margin = float(pcfg.get("margin_s", 1.0)) * 1000
    fps = pcfg.get("fps")
    for (cam, _), (src, spans) in unique_videos(recs).items():
        out = adapter.derived_path("pose", cam, src.path)
        if out.exists() and not overwrite:
            print(f"  pose cached: {out.name}")
            continue
        # Progress is saved in small chunks, so a crash (the GPU model sometimes segfaults on long
        # videos) only loses the last chunk: the next try resumes after the last saved frame.
        chunks = adapter.cache_dir / "derived" / "pose_chunks" / f"{cam}__{Path(src.path).stem}"
        chunks.mkdir(parents=True, exist_ok=True)
        def last_t(c):
            with np.load(c) as z:
                return float(z["t"][-1])
        done_t = max((last_t(c) for c in chunks.glob("*.npz")), default=-np.inf)
        if done_t > -np.inf:
            print(f"  resuming {src.path.name} after {done_t / 1000:.1f} s", flush=True)
        size = tuple(pcfg["frame_size"]) if pcfg.get("frame_size") else None
        buf, bt = [], []

        def flush():
            if not buf:
                return
            v = model(buf)
            name = f"{len(list(chunks.glob('*.npz'))):06d}.npz"
            tmp = chunks / f"{name}.tmp"              # not *.npz until complete
            with open(tmp, "wb") as f:
                np.savez(f, t=np.array(bt, np.float64), v=v)
            os.replace(tmp, chunks / name)
            buf.clear(); bt.clear()

        for t, frame in tqdm(iter_frames(src, size=size, fps=fps), desc=f"pose {cam} {src.path.name}",
                             unit="f", mininterval=2):
            if t <= done_t or not in_spans(t, spans, margin):
                continue
            buf.append(frame)
            bt.append(t)
            if len(buf) >= 64:
                flush()
        flush()
        files = sorted(chunks.glob("*.npz"))
        if not files:
            continue
        ts, vals = [], []
        for c in files:
            with np.load(c) as z:
                ts.append(z["t"]); vals.append(z["v"])
        ts, vals = np.concatenate(ts), np.concatenate(vals)
        save_derived(out, ts, vals, channels, "SKELETON", "signal")
        shutil.rmtree(chunks)
        print(f"  saved {len(ts)} skeletons -> {out}")
