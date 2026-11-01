"""Vision features of the UP-Fall paper (Sec. 3.6.2 and 5.2).

For every pair of consecutive frames: Horn-Schunck optical flow (u, v) -> magnitude
d = sqrt(u^2 + v^2) -> resized to 20 x 20 -> a 400-vector. Inside a window these vectors are
averaged (``kind="frame_vector"``), giving 400 features per camera.

Differences to the paper, all configurable under ``flow:``:
  * flow is computed at ``work_size`` (default 160x120) instead of 640x480, ~16x faster; set
    ``work_size: [640, 480]`` for full fidelity.
  * ``method: farneback`` swaps in OpenCV's Farneback flow (more robust, not the paper's).
"""
from __future__ import annotations

import numpy as np
from tqdm import tqdm

from ..datasets.base import save_derived
from ..video import in_spans, iter_frames, unique_videos

_KX = np.array([[-1, 1], [-1, 1]], np.float32) * .25
_KY = np.array([[-1, -1], [1, 1]], np.float32) * .25
_KT = np.ones((2, 2), np.float32) * .25
_KAVG = np.array([[1 / 12, 1 / 6, 1 / 12], [1 / 6, 0, 1 / 6], [1 / 12, 1 / 6, 1 / 12]], np.float32)


def horn_schunck(im1: np.ndarray, im2: np.ndarray, alpha: float = 1.0, iters: int = 10):
    """Classic Horn & Schunck (1981). Images as float32 in [0, 1]. Defaults = MATLAB opticalFlowHS."""
    import cv2
    f = lambda img, k: cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REPLICATE)
    fx = f(im1, _KX) + f(im2, _KX)
    fy = f(im1, _KY) + f(im2, _KY)
    ft = f(im2, _KT) - f(im1, _KT)
    u = np.zeros_like(im1)
    v = np.zeros_like(im1)
    den = alpha ** 2 + fx ** 2 + fy ** 2
    for _ in range(iters):
        ua, va = f(u, _KAVG), f(v, _KAVG)
        d = (fx * ua + fy * va + ft) / den
        u, v = ua - fx * d, va - fy * d
    return u, v


def flow_vector(prev: np.ndarray, cur: np.ndarray, cfg: dict) -> np.ndarray:
    import cv2
    grid = tuple(cfg.get("grid", [20, 20]))
    if cfg.get("method", "horn_schunck") == "farneback":
        fl = cv2.calcOpticalFlowFarneback(prev, cur, None, .5, 3, 15, 3, 5, 1.2, 0)
        u, v = fl[..., 0], fl[..., 1]
    else:
        u, v = horn_schunck(prev.astype(np.float32) / 255, cur.astype(np.float32) / 255,
                            float(cfg.get("alpha", 1.0)), int(cfg.get("iterations", 10)))
    mag = np.sqrt(u * u + v * v)
    return cv2.resize(mag, grid, interpolation=cv2.INTER_AREA).ravel()


def extract_flow(adapter, recs, cfg: dict, overwrite: bool = False):
    fcfg = cfg.get("flow", {})
    size = tuple(fcfg.get("work_size", [160, 120]))
    margin = float(fcfg.get("margin_s", 2.0)) * 1000
    gw, gh = fcfg.get("grid", [20, 20])
    channels = [f"px{r:02d}_{c:02d}" for r in range(gh) for c in range(gw)]
    for (cam, _), (src, spans) in unique_videos(recs).items():
        out = adapter.derived_path("flow", cam, src.path)
        if out.exists() and not overwrite:
            print(f"  flow cached: {out.name}")
            continue
        ts, vecs, prev = [], [], None
        for t, frame in tqdm(iter_frames(src, size=size, gray=True), desc=f"flow {cam} {src.path.name}",
                             unit="f", mininterval=2):
            if prev is not None and in_spans(t, spans, margin):
                ts.append(t)
                vecs.append(flow_vector(prev, frame, fcfg))
            prev = frame
        if not ts:
            print(f"  no frames inside labelled spans for {src.path.name}")
            continue
        save_derived(out, ts, np.stack(vecs), channels, "CAM", "frame_vector")
        print(f"  saved {len(ts)} flow vectors -> {out}")
