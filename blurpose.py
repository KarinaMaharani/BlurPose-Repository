#!/usr/bin/env python3
"""blurpose — does face censoring degrade pose estimation?

Batch version of blur.ipynb, for running headless on a GPU box.

For each video in --input:
  1. estimate pose on the original frame          (ground truth)
  2. censor the faces in that frame
  3. estimate pose again on the censored frame    (same pixels, never re-encoded)
  4. compare, render, export

Everything is done in ONE streaming pass over the video, a chunk of frames at a
time, so memory does not depend on clip length and full-length videos are fine.
The censored frames go to the pose estimator directly from memory, so no lossy
video round-trip ever enters the measurement.

  python blurpose.py --input input --output output --method blur

Run `python blurpose.py --help` for the options.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

# Headless: no display on a server.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

# Turn sideways footage upright on the fly, so you do not have to pre-encode a
# rotated copy and upload it. Applied to every frame before anything else sees
# it, so pose, censoring and all renders stay consistent.
ROTATIONS = {"none": None,
             "cw": cv2.ROTATE_90_CLOCKWISE,
             "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
             "180": cv2.ROTATE_180}

# --------------------------------------------------------------- keypoints --
COCO17 = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
          "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
          "left_wrist", "right_wrist", "left_hip", "right_hip",
          "left_knee", "right_knee", "left_ankle", "right_ankle"]
COCO17_GROUP = {**{k: "face" for k in (0, 1, 2, 3, 4)},
                **{k: "torso" for k in (5, 6, 11, 12)},
                **{k: "arms" for k in (7, 8, 9, 10)},
                **{k: "legs" for k in (13, 14, 15, 16)}}
COCO17_BONES = [((0,1),"face"), ((0,2),"face"), ((1,3),"face"), ((2,4),"face"),
                ((0,5),"torso"), ((0,6),"torso"), ((5,6),"torso"), ((11,12),"torso"),
                ((5,11),"torso"), ((6,12),"torso"),
                ((5,7),"arms"), ((7,9),"arms"), ((6,8),"arms"), ((8,10),"arms"),
                ((11,13),"legs"), ((13,15),"legs"), ((12,14),"legs"), ((14,16),"legs")]
COCO17_SIGMAS = np.array([.026,.025,.025,.035,.035,.079,.079,.072,.072,
                          .062,.062,.107,.107,.087,.087,.089,.089], np.float32)
KEYPOINT_SETS = {"coco17": list(range(17)), "body12": list(range(5, 17))}

GROUP_BGR = {"face": (0, 190, 255), "torso": (255, 200, 40),
             "arms": (90, 230, 90), "legs": (255, 120, 120)}
GROUP_HEX = {"face": "#ffbe00", "torso": "#28c8ff",
             "arms": "#5ae65a", "legs": "#7878ff"}
MOVED_BGR = (0, 0, 255)


class Schema:
    """Which joints survive, and everything re-indexed to them."""

    def __init__(self, keypoint_set: str, score_groups=None, draw_groups=None):
        self.name = keypoint_set
        self.active = KEYPOINT_SETS[keypoint_set]
        remap = {old: new for new, old in enumerate(self.active)}
        self.names = [COCO17[i] for i in self.active]
        self.K = len(self.names)
        self.group_of = {remap[i]: COCO17_GROUP[i] for i in self.active}
        self.bones = [((remap[i], remap[j]), g) for (i, j), g in COCO17_BONES
                      if i in remap and j in remap]
        self.sigmas = COCO17_SIGMAS[self.active]
        present = [g for g in ("face", "torso", "arms", "legs")
                   if g in set(self.group_of.values())]
        self.groups = present
        self.kpt_groups = {g: [k for k in range(self.K) if self.group_of[k] == g]
                           for g in present}

        def resolve(v):
            if not v:
                return tuple(g for g in present if g != "face")
            bad = [g for g in v if g not in present]
            if bad:
                raise SystemExit(f"unknown/absent group(s) {bad}; present: {present}")
            return tuple(v)

        self.score_groups = resolve(score_groups)
        self.draw_groups = resolve(draw_groups)
        self.scored = np.array([k for k in range(self.K)
                                if self.group_of[k] in self.score_groups], dtype=int)
        self.scored_mask = np.zeros(self.K, bool)
        self.scored_mask[self.scored] = True


# ------------------------------------------------------------------- video --
FPS_FALLBACK, FPS_RANGE = 30.0, (1.0, 240.0)


def video_meta(path, warn=True):
    """(w, h, nframes, fps) with the container's lies filtered out.

    WebM and some streamed files carry no frame index, so OpenCV falls back to
    the timebase and reports nonsense (1000 fps, ~900k frames). Writing output
    at that fps yields a video that plays instantly.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    raw = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    fps = raw if (raw and FPS_RANGE[0] <= raw <= FPS_RANGE[1]) else FPS_FALLBACK
    if warn and fps != raw:
        print(f"    ! {path.name}: container reports {raw:g} fps / {n} frames — "
              f"unreliable; using {fps:g} fps", flush=True)
    return w, h, n, fps


def frame_reader(path, limit=None, rotation=None):
    cap = cv2.VideoCapture(str(path))
    try:
        i = 0
        while limit is None or i < limit:
            ok, frame = cap.read()
            if not ok:
                break
            yield cv2.rotate(frame, rotation) if rotation is not None else frame
            i += 1
    finally:
        cap.release()


def chunked(it, size):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == size:
            yield buf
            buf = []
    if buf:
        yield buf


@contextlib.contextmanager
def quiet_stderr():
    """Mute fd 2 while OpenCV probes codecs (it writes failures straight there)."""
    try:
        saved = os.dup(2)
    except Exception:
        yield
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        try:
            sys.stderr.flush()
        except Exception:
            pass
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


_FOURCC = None


def _try_codec(tag, verify):
    probe = Path(tempfile.gettempdir()) / f"_bp_probe_{tag}_{os.getpid()}.mp4"
    blank = np.zeros((64, 64, 3), np.uint8)
    try:
        with quiet_stderr():
            vw = cv2.VideoWriter(str(probe), cv2.VideoWriter_fourcc(*tag), 25, (64, 64))
            opened = vw.isOpened()
            if opened:
                for _ in range(3):
                    vw.write(blank)
            vw.release()
            if not opened:
                return False
            if not verify:
                return True
            cap = cv2.VideoCapture(str(probe))
            ok = cap.read()[0] and int(cap.get(cv2.CAP_PROP_FOURCC)) != 0
            cap.release()
        return ok
    except Exception:
        return False
    finally:
        probe.unlink(missing_ok=True)


def probe_fourcc():
    """isOpened() lies when the H.264 encoder fails, so verify by round-trip."""
    global _FOURCC
    if _FOURCC:
        return _FOURCC
    for tag in ("avc1", "mp4v"):
        if _try_codec(tag, verify=True):
            _FOURCC = tag
            break
    else:
        for tag in ("mp4v", "avc1", "MJPG"):
            if _try_codec(tag, verify=False):
                _FOURCC = tag
                print(f"    warning: no mp4 codec verified; using '{tag}' unchecked")
                break
    if _FOURCC is None:
        raise IOError("OpenCV cannot write video in this environment")
    return _FOURCC


def frame_writer(path, w, h, fps):
    with quiet_stderr():
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*probe_fourcc()),
                             fps, (w, h))
    if not vw.isOpened():
        raise IOError(f"cannot open writer for {path}")
    return vw


# ---------------------------------------------------------------- censoring --
def apply_censor(frame, bbox, method):
    """Verbatim from blurfaces.py (github.com/AlexPasqua/blurfaces)."""
    x1, y1, x2, y2 = (max(0, int(v)) for v in bbox)
    fh, fw = frame.shape[:2]
    x2, y2 = min(fw, x2), min(fh, y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    if method == "blackout":
        frame[y1:y2, x1:x2] = 0
    elif method == "pixel":
        rh, rw = roi.shape[:2]
        small = cv2.resize(roi, (8, 8), interpolation=cv2.INTER_AREA)
        frame[y1:y2, x1:x2] = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_AREA)
    elif method == "bar":
        rh = y2 - y1
        frame[y1 + int(rh * 0.3):y1 + int(rh * 0.55), x1:x2] = 0
    else:
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (0, 0), 30)


def remove_face(frame, bbox, method, inpaint_res=64):
    """Ours: take the face out rather than obscure it in place."""
    x1, y1, x2, y2 = (max(0, int(v)) for v in bbox)
    fh, fw = frame.shape[:2]
    x2, y2 = min(fw, x2), min(fh, y2)
    if x2 <= x1 or y2 <= y1:
        return
    if method == "inpaint":
        m = max(8, int(0.15 * max(x2 - x1, y2 - y1)))
        X1, Y1 = max(0, x1 - m), max(0, y1 - m)
        X2, Y2 = min(fw, x2 + m), min(fh, y2 + m)
        patch = frame[Y1:Y2, X1:X2]
        mask = np.zeros(patch.shape[:2], np.uint8)
        mask[y1 - Y1:y2 - Y1, x1 - X1:x2 - X1] = 255
        ph, pw = patch.shape[:2]
        s = inpaint_res / max(ph, pw)
        if s < 1.0:
            sp = cv2.resize(patch, (max(1, int(pw*s)), max(1, int(ph*s))),
                            interpolation=cv2.INTER_AREA)
            sm = cv2.resize(mask, (sp.shape[1], sp.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
            out = cv2.resize(cv2.inpaint(sp, sm, 3, cv2.INPAINT_TELEA), (pw, ph),
                             interpolation=cv2.INTER_LINEAR)
        else:
            out = cv2.inpaint(patch, mask, 3, cv2.INPAINT_TELEA)
        patch[mask > 0] = out[mask > 0]
    elif method == "fill":
        m = max(6, int(0.12 * max(x2 - x1, y2 - y1)))
        X1, Y1 = max(0, x1 - m), max(0, y1 - m)
        X2, Y2 = min(fw, x2 + m), min(fh, y2 + m)
        ring = frame[Y1:Y2, X1:X2].reshape(-1, 3).astype(np.float32)
        inner = frame[y1:y2, x1:x2].reshape(-1, 3).astype(np.float32)
        n = max(1, len(ring) - len(inner))
        frame[y1:y2, x1:x2] = ((ring.sum(0) - inner.sum(0)) / n
                               ).clip(0, 255).astype(np.uint8)


CENSOR_METHODS = ("bar", "blur", "pixel", "fill", "blackout", "inpaint")


def censor(frame, bbox, method, inpaint_res=64):
    if method in ("inpaint", "fill"):
        remove_face(frame, bbox, method, inpaint_res)
    else:
        apply_censor(frame, bbox, method)


def pad_box(box, pad, w, h):
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
    return [max(0, x1 - dx), max(0, y1 - dy), min(w, x2 + dx), min(h, y2 + dy)]


# ------------------------------------------------------------ face detector --
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
YUNET_MIN_BYTES = 200_000


def fetch_yunet(dst, tries=3):
    """Download atomically: a truncated file passes exists() and then fails to load."""
    if dst.exists() and dst.stat().st_size >= YUNET_MIN_BYTES:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for attempt in range(1, tries + 1):
        tmp = dst.with_suffix(f".part{attempt}")
        try:
            print(f"  downloading YuNet ({attempt}/{tries})...", flush=True)
            urllib.request.urlretrieve(YUNET_URL, tmp)
            if tmp.stat().st_size < YUNET_MIN_BYTES:
                raise IOError(f"truncated ({tmp.stat().st_size} bytes)")
            tmp.replace(dst)
            return dst
        except Exception as exc:
            last = exc
            print(f"    failed: {type(exc).__name__}: {exc}")
        finally:
            if tmp.exists():
                tmp.unlink()
    raise IOError(f"could not download YuNet ({last}).\nSave it manually as {dst}\n"
                  f"from {YUNET_URL}")


class FaceDetector:
    def __init__(self, backend, work_dir, model_pack="buffalo_l", det_size=640):
        self.backend = backend
        self._impl = None
        self.work_dir = work_dir
        self.model_pack = model_pack
        self.det_size = det_size

    def __call__(self, frame):
        h, w = frame.shape[:2]
        if self.backend == "insightface":
            if self._impl is None:
                import insightface
                self._impl = insightface.app.FaceAnalysis(name=self.model_pack)
                self._impl.prepare(ctx_id=0, det_size=(self.det_size, self.det_size))
            return [(f.bbox, f.normed_embedding) for f in self._impl.get(frame)]
        if self._impl is None:
            mp = fetch_yunet(self.work_dir / "face_detection_yunet_2023mar.onnx")
            self._impl = cv2.FaceDetectorYN.create(str(mp), "", (w, h), 0.6, 0.3, 5000)
        self._impl.setInputSize((w, h))
        _, faces = self._impl.detect(frame)
        if faces is None:
            return []
        return [(np.array([x, y, x + fw, y + fh], np.float32), None)
                for x, y, fw, fh, *_ in faces]


# -------------------------------------------------------------------- pose --
class PoseModel:
    """torchvision Keypoint R-CNN, batched, face keypoints sliced at the source."""

    def __init__(self, schema: Schema, device=None, score_thr=0.5):
        import torch
        from torchvision.models.detection import (
            keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights)
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.schema = schema
        self.score_thr = score_thr
        self.model = keypointrcnn_resnet50_fpn(
            weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT).eval().to(self.device)

    def __call__(self, frames):
        """frames: list of BGR arrays -> list (per frame) of person dicts."""
        torch = self.torch
        with torch.inference_mode():
            tensors = [torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                       .permute(2, 0, 1).float().div(255).to(self.device)
                       for f in frames]
            outs = self.model(tensors)
        results = []
        for out in outs:
            people = []
            for box, sc, kp, ksc in zip(out["boxes"], out["scores"],
                                        out["keypoints"], out["keypoints_scores"]):
                if float(sc) < self.score_thr:
                    continue
                k = np.concatenate(
                    [kp[:, :2].cpu().numpy(),
                     torch.sigmoid(ksc).cpu().numpy()[:, None]], 1).astype(np.float32)
                people.append({"kpts": k[self.schema.active],   # face dropped here
                               "box": box.cpu().numpy().astype(np.float32),
                               "score": float(sc)})
            results.append(people)
        return results


# ----------------------------------------------------------------- metrics --
def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match_frame(before, after, min_iou=0.3):
    pairs, used, taken = [], set(), set()
    cand = sorted(((iou(b["box"], a["box"]), i, j)
                   for i, b in enumerate(before) for j, a in enumerate(after)),
                  key=lambda z: -z[0])
    for v, i, j in cand:
        if v < min_iou or i in taken or j in used:
            continue
        taken.add(i)
        used.add(j)
        pairs.append((before[i], after[j]))
    return pairs, len(before) - len(taken), len(after) - len(used)


class Metrics:
    """Accumulates the comparison as frames stream past."""

    def __init__(self, schema: Schema, vis_thr, pck_thresholds):
        self.s = schema
        self.vis_thr = vis_thr
        self.pck_thresholds = pck_thresholds
        self.disp = [[] for _ in range(schema.K)]
        self.conf_b = [[] for _ in range(schema.K)]
        self.conf_a = [[] for _ in range(schema.K)]
        self.per_frame = []
        self.matched = self.lost = self.spurious = 0

    def oks(self, kb, ka, box):
        area = max(1.0, (box[2]-box[0]) * (box[3]-box[1]))
        d2 = (kb[:, 0]-ka[:, 0])**2 + (kb[:, 1]-ka[:, 1])**2
        e = d2 / (2 * (self.s.sigmas**2) * area)
        vis = (kb[:, 2] > self.vis_thr) & self.s.scored_mask
        return float(np.exp(-e[vis]).mean()) if vis.any() else np.nan

    def add(self, t, before, after):
        pairs, nl, ns = match_frame(before, after)
        self.matched += len(pairs)
        self.lost += nl
        self.spurious += ns
        fd, fo = [], []
        for pb, pa in pairs:
            kb, ka, box = pb["kpts"], pa["kpts"], pb["box"]
            diag = np.hypot(box[2]-box[0], box[3]-box[1]) or 1.0
            d = np.hypot(kb[:, 0]-ka[:, 0], kb[:, 1]-ka[:, 1]) / diag
            vis = kb[:, 2] > self.vis_thr
            for k in np.where(vis)[0]:
                self.disp[k].append(float(d[k]))
                self.conf_b[k].append(float(kb[k, 2]))
                self.conf_a[k].append(float(ka[k, 2]))
            sv = vis & self.s.scored_mask
            if sv.any():
                fd.append(float(d[sv].mean()))
                fo.append(self.oks(kb, ka, box))
        self.per_frame.append((t, float(np.nanmean(fd)) if fd else np.nan,
                               float(np.nanmean(fo)) if fo else np.nan, nl))
        return pairs

    def finish(self):
        disp = [np.array(x, np.float32) for x in self.disp]
        all_d = (np.concatenate([disp[k] for k in self.s.scored if len(disp[k])])
                 if any(len(disp[k]) for k in self.s.scored) else np.array([]))
        rows = [(self.s.names[k], self.s.group_of[k], float(disp[k].mean()),
                 float(np.median(disp[k])), float((disp[k] <= 0.05).mean()),
                 float(np.mean(self.conf_b[k])), float(np.mean(self.conf_a[k])))
                for k in range(self.s.K) if len(disp[k])]
        group_d = {g: np.concatenate([disp[k] for k in ks if len(disp[k])])
                   for g, ks in self.s.kpt_groups.items()
                   if any(len(disp[k]) for k in ks)}
        oks_vals = [o for _, _, o, _ in self.per_frame]
        return dict(
            disp=disp, all_d=all_d, rows=rows, group_d=group_d,
            per_frame=self.per_frame, matched=self.matched, lost=self.lost,
            spurious=self.spurious,
            pck={f"@{t:.2f}": (float((all_d <= t).mean()) if all_d.size else float("nan"))
                 for t in self.pck_thresholds},
            mean_oks=float(np.nanmean(oks_vals)) if oks_vals else float("nan"))


# ---------------------------------------------------------------- drawing ---
def draw_pose(img, people, schema, ref=None, mark_thr=0.10, kpt_thr=0.35):
    shown = set(schema.draw_groups)
    img = img.copy()
    for p in people:
        k = p["kpts"]
        for (i, j), grp in schema.bones:
            if schema.group_of[i] not in shown or schema.group_of[j] not in shown:
                continue
            if k[i, 2] > kpt_thr and k[j, 2] > kpt_thr:
                cv2.line(img, tuple(k[i, :2].astype(int)), tuple(k[j, :2].astype(int)),
                         GROUP_BGR[grp], 2, cv2.LINE_AA)
        for n in range(len(k)):
            if schema.group_of[n] in shown and k[n, 2] > kpt_thr:
                c = tuple(k[n, :2].astype(int))
                cv2.circle(img, c, 4, (30, 30, 30), -1, cv2.LINE_AA)
                cv2.circle(img, c, 3, GROUP_BGR[schema.group_of[n]], -1, cv2.LINE_AA)
    if ref is not None:
        for pb, pa in ref:
            diag = np.hypot(pb["box"][2]-pb["box"][0], pb["box"][3]-pb["box"][1]) or 1.0
            d = np.hypot(pb["kpts"][:, 0]-pa["kpts"][:, 0],
                         pb["kpts"][:, 1]-pa["kpts"][:, 1]) / diag
            for n in np.where((d > mark_thr) & (pb["kpts"][:, 2] > kpt_thr)
                              & schema.scored_mask)[0]:
                cv2.circle(img, tuple(pa["kpts"][n, :2].astype(int)), 11,
                           MOVED_BGR, 2, cv2.LINE_AA)
    return img


def label(img, text):
    cv2.putText(img, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
    return img


def draw_legend(img, schema, show_moved=False):
    entries = [(g, GROUP_BGR[g]) for g in schema.groups if g in schema.draw_groups]
    if show_moved:
        entries.append(("moved", MOVED_BGR))
    w = img.shape[1]
    fs = max(0.42, min(1.0, w / 1400))
    th = 1 if w < 1200 else 2
    sw = max(12, int(w / 55))
    bh = int(22 * fs / 0.5)
    x, y = int(w / 90) + 6, img.shape[0] - int(bh * 0.5)
    gap = int(sw * 0.7)

    def tw(s):
        return cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, fs, th)[0][0]

    width = sum(sw + 6 + tw(n) + gap for n, _ in entries)
    cv2.rectangle(img, (x - 6, y - bh), (x + width, y + int(bh * 0.35)), (0, 0, 0), -1)
    cx = x
    for name, bgr in entries:
        cv2.rectangle(img, (cx, y - int(bh*0.72)), (cx + sw, y - int(bh*0.18)), bgr, -1)
        cx += sw + 6
        cv2.putText(img, name, (cx, y - int(bh*0.2)), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (255, 255, 255), th, cv2.LINE_AA)
        cx += tw(name) + gap
    return img


def plot_report(R, schema, stem, method, backend, outdir):
    rows, group_d, per_frame = R["rows"], R["group_d"], R["per_frame"]
    if not rows:
        return None
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    names = [r[0] for r in rows]
    cols = [GROUP_HEX[r[1]] for r in rows]
    ax[0,0].barh(names, [r[2] for r in rows], color=cols, edgecolor="#333", linewidth=.4)
    ax[0,0].invert_yaxis(); ax[0,0].set_xlabel("mean displacement (/ bbox diagonal)")
    ax[0,0].set_title("(a) Which joints move"); ax[0,0].grid(axis="x", alpha=.3)

    gk = [g for g in schema.groups if g in group_d]
    if gk:
        ax[0,1].hist([group_d[g] for g in gk], bins=40, range=(0, 0.4), label=gk,
                     color=[GROUP_HEX[g] for g in gk], density=True)
        ax[0,1].legend()
    ax[0,1].set_xlabel("normalised displacement")
    ax[0,1].set_title("(b) Error distribution")

    t = [p[0] for p in per_frame]
    ax[1,0].plot(t, [p[1] for p in per_frame], lw=.6, c="#d1495b")
    ax[1,0].set_xlabel("frame"); ax[1,0].set_ylabel("displacement", color="#d1495b")
    a2 = ax[1,0].twinx(); a2.plot(t, [p[2] for p in per_frame], lw=.6, c="#3d8bbf")
    a2.set_ylabel("OKS", color="#3d8bbf"); a2.set_ylim(0, 1)
    ax[1,0].set_title("(c) Per-frame agreement")

    x = np.arange(len(rows)); wdt = .38
    ax[1,1].bar(x - wdt/2, [r[5] for r in rows], wdt, label="before", color="#b0b0b0")
    ax[1,1].bar(x + wdt/2, [r[6] for r in rows], wdt, label="after", color=cols,
                edgecolor="#333", linewidth=.4)
    ax[1,1].set_xticks(x); ax[1,1].set_xticklabels(names, rotation=90, fontsize=8)
    ax[1,1].set_ylabel("mean keypoint confidence"); ax[1,1].legend()
    ax[1,1].set_title("(d) Confidence shift")

    plt.suptitle(f"{stem} — {method}, {backend}, {schema.name}", y=1.01)
    plt.tight_layout()
    path = outdir / f"{stem}_{method}_metrics.png"
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def write_gif(frames, path, fps, colors):
    if not frames:
        return None
    try:
        from PIL import Image as PILImage
    except ImportError:
        return None
    pil = [PILImage.fromarray(f).quantize(colors=colors,
                                          method=PILImage.Quantize.FASTOCTREE)
           for f in frames]
    pil[0].save(path, save_all=True, append_images=pil[1:],
                duration=int(1000 / fps), loop=0, optimize=True)
    return path


def print_report(R, schema):
    hdr = ("keypoint".ljust(16) + "group".ljust(7) + "mean disp".rjust(10)
           + "median".rjust(9) + "PCK@.05".rjust(9) + "conf b".rjust(8)
           + "conf a".rjust(8) + "d conf".rjust(8))
    tot = max(1, R["matched"] + R["lost"])
    print(f"  matched={R['matched']}  lost={R['lost']} ({R['lost']/tot:.1%})  "
          f"spurious={R['spurious']}")
    print(f"  BODY POSE ({'+'.join(schema.score_groups)}): ", end="")
    print("  ".join(f"PCK{k}={v:.3f}" for k, v in R["pck"].items()),
          f"  OKS={R['mean_oks']:.4f}")
    if not R["rows"]:
        print("  (no scored keypoints — nothing matched)")
        return
    print("  " + hdr)
    print("  " + "-" * len(hdr))
    for n, g, m, md, p, cb, ca in R["rows"]:
        print(f"  {n:<16}{g:<7}{m:>10.4f}{md:>9.4f}{p:>9.3f}{cb:>8.3f}{ca:>8.3f}{ca-cb:>+8.3f}")
    print("  " + "-" * len(hdr))
    for g in schema.groups:
        if g not in R["group_d"]:
            continue
        d = R["group_d"][g]
        dc = np.mean([r[6]-r[5] for r in R["rows"] if r[1] == g])
        print(f"  {g.upper():<16}{len(d):>7}{d.mean():>10.4f}{np.median(d):>9.4f}"
              f"{(d <= 0.05).mean():>9.3f}{'':>16}{dc:>+8.3f}")


# ------------------------------------------------------------- the pipeline --
def process_video(video, args, schema, pose, detector):
    stem = video.stem
    vw_, vh_, vn, fps = video_meta(video)
    rotation = ROTATIONS[args.rotate]
    if args.rotate in ("cw", "ccw"):
        vw_, vh_ = vh_, vw_          # a quarter turn swaps the frame dimensions
    limit = args.max_frames if args.max_frames > 0 else None
    suffix = f"__rot{args.rotate}" if args.rotate != "none" else ""
    run_dir = Path(args.output) / f"{stem}{suffix}__{args.method}__torchvision"
    run_dir.mkdir(parents=True, exist_ok=True)

    est_total = min(vn, limit) if (vn and limit) else (limit or vn or 0)
    print(f"\n{'='*70}\n{video.name}  {vw_}x{vh_} @ {fps:g} fps"
          + (f"  [rotated {args.rotate}]" if rotation is not None else "")
          + f"  -> {est_total or 'all'} frames\n{'='*70}", flush=True)

    writers = {}
    if args.videos != "none":
        writers["blur_raw"] = frame_writer(
            run_dir / f"{stem}_1_blur_raw_{args.method}.mp4", vw_, vh_, fps)
        if args.videos == "all":
            writers["pose_before"] = frame_writer(
                run_dir / f"{stem}_2_pose_before.mp4", vw_, vh_, fps)
            writers["pose_after"] = frame_writer(
                run_dir / f"{stem}_3_pose_after_{args.method}.mp4", vw_, vh_, fps)
        writers["compare"] = frame_writer(
            run_dir / f"{stem}_4_compare_{args.method}.mp4", vw_ * 2, vh_, fps)

    csv_path = run_dir / "keypoints.csv"
    csv_fh = open(csv_path, "w", newline="", encoding="utf-8")
    csv_wr = csv.writer(csv_fh)
    csv_wr.writerow(["video", "run", "frame", "person", "keypoint", "group",
                     "x", "y", "score"])

    M = Metrics(schema, args.vis_thr, tuple(args.pck))
    gif_every = max(1, (est_total or 1) // max(1, args.gif_frames))
    still_every = max(1, (est_total or 1) // 4)
    gif_frames, stills, n_hits, n_held, held = [], [], 0, 0, []
    # Keeping every pose for the JSON/npy exports costs memory on a long clip,
    # so only hold them when those exports were actually asked for.
    keep_poses = args.export_json or args.export_npy
    pose_log = {"before": [], "after": []} if keep_poses else None
    t = 0
    t0 = time.time()

    try:
        for chunk in chunked(frame_reader(video, limit, rotation), args.batch):
            before = pose(chunk)
            censored = []
            for frame in chunk:
                f = frame.copy()
                boxes = [pad_box(b, args.box_pad, vw_, vh_)
                         for b, emb in detector(f)]
                if boxes:
                    held = [[b, args.miss_hold] for b in boxes]
                    n_hits += len(boxes)
                else:
                    # Action footage drops detections for a frame or two; reuse
                    # the last box rather than exposing the face.
                    held = [[b, k - 1] for b, k in held if k > 1]
                    boxes = [b for b, _ in held]
                    n_held += len(boxes)
                for b in boxes:
                    censor(f, b, args.method, args.inpaint_res)
                censored.append(f)
            after = pose(censored)

            for i, (orig, cen) in enumerate(zip(chunk, censored)):
                pairs = M.add(t, before[i], after[i])
                if pose_log is not None:
                    pose_log["before"].append(before[i])
                    pose_log["after"].append(after[i])
                for run, people in (("before", before[i]), ("after", after[i])):
                    for pid, p in enumerate(people):
                        for k in range(schema.K):
                            x, y, s = p["kpts"][k]
                            csv_wr.writerow([video.name, run, t, pid,
                                             schema.names[k], schema.group_of[k],
                                             round(float(x), 2), round(float(y), 2),
                                             round(float(s), 4)])
                if writers:
                    bl = label(draw_pose(orig, before[i], schema), "BEFORE — original")
                    br = label(draw_pose(cen, after[i], schema, ref=pairs),
                               f"AFTER — {args.method}")
                    left = draw_legend(bl.copy(), schema)
                    right = draw_legend(br.copy(), schema, show_moved=True)
                    combo = np.hstack([left, right])
                    writers["blur_raw"].write(cen)
                    if "pose_before" in writers:
                        writers["pose_before"].write(left)
                        writers["pose_after"].write(right)
                    writers["compare"].write(combo)
                    if args.gif_width and t % gif_every == 0 \
                            and len(gif_frames) < args.gif_frames:
                        gw = args.gif_width
                        bare = np.hstack([bl, br])
                        small = cv2.resize(
                            bare, (gw, int(gw * bare.shape[0] / bare.shape[1])),
                            interpolation=cv2.INTER_AREA)
                        draw_legend(small, schema, show_moved=True)
                        gif_frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
                    if t % still_every == 0 and len(stills) < 4:
                        stills.append((t, combo.copy()))
                t += 1

            if t % max(args.batch, args.progress) < args.batch:
                el = time.time() - t0
                rate = t / el if el else 0
                eta = ((est_total - t) / rate) if (rate and est_total) else 0
                print(f"    {t}{'/' + str(est_total) if est_total else ''} frames  "
                      f"{rate:.1f} fps  elapsed {el/60:.1f}m"
                      + (f"  eta {eta/60:.1f}m" if eta else ""), flush=True)
    finally:
        csv_fh.close()
        for w in writers.values():
            w.release()

    if t == 0:
        print("  no frames decoded — skipped")
        return None

    R = M.finish()
    print(f"  done: {t} frames in {(time.time()-t0)/60:.1f} min   "
          f"face detections={n_hits}  carried-over={n_held}")
    print_report(R, schema)

    gif = write_gif(gif_frames, run_dir / f"{stem}_compare_{args.method}.gif",
                    args.gif_fps, args.gif_colors)
    plot_report(R, schema, stem, args.method, "torchvision", run_dir)

    # sample stills — easier to read closely than the GIF
    if stills:
        fig, axes = plt.subplots(len(stills), 1,
                                 figsize=(14, 14 * vh_ / (vw_ * 2) * len(stills)))
        for a, (ft, im) in zip(np.atleast_1d(axes), stills):
            a.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); a.axis("off")
            a.set_title(f"frame {ft}", fontsize=8)
        plt.tight_layout()
        plt.savefig(run_dir / f"{stem}_{args.method}_stills.png", dpi=110,
                    bbox_inches="tight")
        plt.close(fig)

    # optional heavier keypoint exports (CSV is always written, streaming)
    if pose_log is not None:
        if args.export_json:
            doc = {"video": video.name,
                   "schema": {"keypoints": schema.names,
                              "groups": [schema.group_of[k] for k in range(schema.K)],
                              "skeleton": [[int(i), int(j)] for (i, j), _ in schema.bones],
                              "keypoint_set": schema.name,
                              "rotation": args.rotate,
                              "coords": "pixels in the (rotated) frame",
                              "kpts_layout": "[x, y, score] per keypoint, in 'keypoints' order"},
                   "runs": {run: [[{"person": pid,
                                    "box": [round(float(v), 2) for v in q["box"]],
                                    "score": round(float(q["score"]), 4),
                                    "kpts": [[round(float(x), 2), round(float(y), 2),
                                              round(float(s), 4)] for x, y, s in q["kpts"]]}
                                   for pid, q in enumerate(people)]
                                  for people in frames]
                            for run, frames in pose_log.items()}}
            (run_dir / "keypoints.json").write_text(json.dumps(doc), encoding="utf-8")
        if args.export_npy:
            for run, frames in pose_log.items():
                np.save(run_dir / f"keypoints_{run}.npy",
                        np.array(frames, dtype=object), allow_pickle=True)

    summary = {
        "video": video.name, "frames": t, "resolution": [vw_, vh_], "fps": fps,
        "method": args.method, "keypoint_set": schema.name, "rotate": args.rotate,
        "keypoints": schema.names, "scored_groups": list(schema.score_groups),
        "face_detections": n_hits, "carried_over_boxes": n_held,
        "detections": {"matched": R["matched"], "lost": R["lost"],
                       "spurious": R["spurious"],
                       "lost_rate": R["lost"] / max(1, R["matched"] + R["lost"])},
        "pck": R["pck"], "mean_oks": R["mean_oks"],
        "mean_norm_displacement": {
            "all": float(R["all_d"].mean()) if R["all_d"].size else None,
            **{g: float(d.mean()) for g, d in R["group_d"].items()}},
        "per_group": {g: {"n_samples": int(d.size), "mean_disp": float(d.mean()),
                          "median_disp": float(np.median(d)),
                          "pck_005": float((d <= 0.05).mean())}
                      for g, d in R["group_d"].items()},
        "per_keypoint": {r[0]: {"group": r[1], "mean_disp": r[2], "median_disp": r[3],
                                "pck_005": r[4], "conf_before": r[5], "conf_after": r[6]}
                         for r in R["rows"]},
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  -> {run_dir}")
    return summary


def main():
    ap = argparse.ArgumentParser(
        description="Measure how face censoring affects pose estimation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--input", default="input", help="folder of videos")
    ap.add_argument("--output", default="output", help="where results go")
    ap.add_argument("--work", default="work", help="model cache")
    ap.add_argument("--method", default="blur", choices=CENSOR_METHODS)
    ap.add_argument("--keypoints", default="body12", choices=list(KEYPOINT_SETS),
                    help="body12 drops nose/eyes/ears at inference")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="0 = the entire video")
    ap.add_argument("--batch", type=int, default=4,
                    help="frames per GPU batch; lower it if you hit OOM")
    ap.add_argument("--device", default=None, help="cuda, cuda:1, cpu")
    ap.add_argument("--face-backend", default="yunet", choices=["yunet", "insightface"])
    ap.add_argument("--videos", default="all", choices=["all", "compare", "none"],
                    help="which renders to write; 'none' is fastest")
    ap.add_argument("--box-pad", type=float, default=0.25)
    ap.add_argument("--miss-hold", type=int, default=3)
    ap.add_argument("--inpaint-res", type=int, default=64)
    ap.add_argument("--score-thr", type=float, default=0.5)
    ap.add_argument("--vis-thr", type=float, default=0.2)
    ap.add_argument("--pck", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    ap.add_argument("--gif-width", type=int, default=800, help="0 disables")
    ap.add_argument("--gif-frames", type=int, default=40)
    ap.add_argument("--gif-fps", type=int, default=8)
    ap.add_argument("--gif-colors", type=int, default=128)
    ap.add_argument("--progress", type=int, default=100, help="print every N frames")
    ap.add_argument("--rotate", default="none", choices=list(ROTATIONS),
                    help="turn every frame before processing; 'cw' is 90 to the "
                         "right — use it for the sideways cam clips")
    ap.add_argument("--only", nargs="+", default=None, metavar="TEXT",
                    help="process only videos whose filename contains any of these")
    ap.add_argument("--skip", nargs="+", default=None, metavar="TEXT",
                    help="skip videos whose filename contains any of these")
    ap.add_argument("--export-json", action="store_true",
                    help="also write keypoints.json (large on long clips)")
    ap.add_argument("--export-npy", action="store_true",
                    help="also write keypoints_{before,after}.npy")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.input), Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    if not in_dir.is_dir():
        raise SystemExit(f"input folder not found: {in_dir.resolve()}")
    found = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    if not found:
        raise SystemExit(f"no videos in {in_dir.resolve()}")

    # The script's equivalent of the notebook's checklist.
    def selected(p):
        if args.only and not any(s.lower() in p.name.lower() for s in args.only):
            return False
        if args.skip and any(s.lower() in p.name.lower() for s in args.skip):
            return False
        return True

    videos = [p for p in found if selected(p)]
    if not videos:
        raise SystemExit("--only / --skip excluded every video")

    schema = Schema(args.keypoints)
    print(f"blurpose | method={args.method} | {schema.name} -> {schema.K} joints: "
          f"{', '.join(schema.names)}")
    print(f"scored groups: {schema.score_groups}")

    pose = PoseModel(schema, args.device, args.score_thr)
    print(f"device: {pose.device}")
    detector = FaceDetector(args.face_backend, work)

    print(f"\n{len(found)} video(s) in {in_dir.resolve()}, {len(videos)} selected:")
    for p in found:
        on = p in videos
        w, h, n, fps = video_meta(p, warn=on)
        if args.rotate in ("cw", "ccw"):
            w, h = h, w
        print(f"  [{'x' if on else ' '}] {p.name[:46]:<48} {w}x{h} {fps:6.2f}fps "
              f"{n or '?':>8} frames" + ("" if on else "   (skipped)"))
    if args.rotate != "none":
        print(f"  rotation: {args.rotate} (applied to every frame before processing)")

    summaries = []
    started = time.time()
    for v in videos:
        try:
            s = process_video(v, args, schema, pose, detector)
        except Exception as exc:
            import traceback
            print(f"  !! {v.name} FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            continue
        if s:
            summaries.append(s)

    if not summaries:
        raise SystemExit("nothing processed")

    print(f"\n\n{'='*70}\nCOMBINED  ({len(summaries)} videos, "
          f"{(time.time()-started)/60:.1f} min total)\n{'='*70}")
    hdr = (f"{'video':<34}{'frames':>8}{'faces':>7}{'lost%':>8}"
           f"{'PCK@.05':>9}{'PCK@.10':>9}{'OKS':>8}")
    print(hdr); print("-" * len(hdr))
    for s in summaries:
        print(f"{s['video'][:33]:<34}{s['frames']:>8}{s['face_detections']:>7}"
              f"{s['detections']['lost_rate']*100:>7.1f}%"
              f"{s['pck']['@0.05']:>9.3f}{s['pck']['@0.10']:>9.3f}"
              f"{s['mean_oks']:>8.3f}")

    combined = out_dir / f"results__{args.method}__{schema.name}.csv"
    with open(combined, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["video", "frames", "method", "keypoint_set", "face_detections",
                     "matched", "lost", "lost_rate", "spurious",
                     "pck_005", "pck_010", "pck_020", "mean_oks", "mean_disp"])
        for s in summaries:
            wr.writerow([s["video"], s["frames"], s["method"], s["keypoint_set"],
                         s["face_detections"], s["detections"]["matched"],
                         s["detections"]["lost"], round(s["detections"]["lost_rate"], 4),
                         s["detections"]["spurious"],
                         round(s["pck"]["@0.05"], 4), round(s["pck"]["@0.10"], 4),
                         round(s["pck"]["@0.20"], 4), round(s["mean_oks"], 4),
                         round(s["mean_norm_displacement"]["all"] or float("nan"), 4)])
    print(f"\nwrote {combined}")

    warn = [s["video"] for s in summaries if s["face_detections"] == 0]
    if warn:
        print("\n!! no faces were detected in: " + ", ".join(warn))
        print("   Nothing was censored there, so before/after are identical and "
              "the scores are meaningless. Check the footage or the detector.")


if __name__ == "__main__":
    main()
