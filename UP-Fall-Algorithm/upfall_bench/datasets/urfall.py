"""UR Fall Detection Dataset (Kwolek & Kepski 2014), https://fenix.ur.edu.pl/mkepski/ds/uf.html

Layout expected under ``root`` (exactly what ``python -m upfall_bench download`` produces)::

    fall-01-acc.csv   fall-01-data.csv   fall-01-cam0.mp4   [fall-01-cam1.mp4]
    adl-01-acc.csv    adl-01-data.csv    adl-01-cam0.mp4
    urfall-cam0-falls.csv  urfall-cam0-adls.csv

* ``*-acc.csv``  : time_ms, SV_total, Ax, Az, Ay   (g, ~60 Hz; PS Move / x-IMU)
* ``*-data.csv`` : frame, time_ms, SV_total-at-frame   (camera clock, 30 fps)
* ``*-cam0.mp4`` : 640x240, depth (left half) | RGB (right half), one frame per data.csv row
* ``urfall-cam0-*.csv``: per depth frame: seq, frame, posture label (-1 not lying, 0 falling,
  1 lying) and 8 depth-silhouette features.

Classes. UR Fall only gives posture per frame, so we derive five classes that keep falls and
intentional lying-down (an ADL) apart:

    1 upright            (-1 in any sequence)
    2 falling            ( 0 in a fall sequence)   <- fall class by default
    3 lying after fall   ( 1 in a fall sequence)
    4 lying down (ADL)   ( 0 in an ADL sequence)
    5 lying (ADL)        ( 1 in an ADL sequence)

Use ``fall_classes: [2, 3]`` in the YAML if "on the floor after a fall" should also count.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from ..core import Recording, Stream, VideoSource
from .base import DatasetAdapter, register

BASE_URL = "https://fenix.ur.edu.pl/mkepski/ds/data/"
N_FALL, N_ADL = 30, 40
DEPTH_FEATURES = ["HeightWidthRatio", "MajorMinorRatio", "BoundingBoxOccupancy", "MaxStdXZ",
                  "HHmaxRatio", "H", "D", "P40"]


@register("urfall")
class URFallAdapter(DatasetAdapter):

    def default_classes(self):
        return {1: "upright", 2: "falling", 3: "lying after fall", 4: "lying down (ADL)", 5: "lying (ADL)"}

    def default_fall_classes(self):
        return {2}

    # ------------------------------------------------------------------------------------
    def download(self):
        self.root.mkdir(parents=True, exist_ok=True)
        want = self.dcfg.get("download", {})
        cams = want.get("cameras", ["cam0"])
        names = ["urfall-cam0-falls.csv", "urfall-cam0-adls.csv"]
        for kind, n in (("fall", N_FALL), ("adl", N_ADL)):
            for i in range(1, n + 1):
                seq = f"{kind}-{i:02d}"
                names += [f"{seq}-acc.csv", f"{seq}-data.csv"]
                names += [f"{seq}-{c}.mp4" for c in cams if not (kind == "adl" and c == "cam1")]
                if want.get("rgb_zip"):
                    names += [f"{seq}-{c}-rgb.zip" for c in cams if not (kind == "adl" and c == "cam1")]
        for name in names:
            dst = self.root / name
            if dst.exists() and dst.stat().st_size > 0:
                continue
            print("  downloading", name)
            tmp = dst.with_suffix(dst.suffix + ".part")
            urllib.request.urlretrieve(BASE_URL + name, tmp)
            tmp.replace(dst)
        print(f"UR Fall ready in {self.root}")

    # ------------------------------------------------------------------------------------
    def load_recordings(self):
        feats = []
        for f in ("urfall-cam0-falls.csv", "urfall-cam0-adls.csv"):
            p = self.root / f
            if p.exists():
                feats.append(pd.read_csv(p, header=None, names=["seq", "frame", "posture"] + DEPTH_FEATURES))
        if not feats:
            raise FileNotFoundError(f"{self.root}: urfall-cam0-*.csv not found (run the download command)")
        feats = pd.concat(feats)
        by_seq = dict(tuple(feats.groupby("seq")))

        cameras = self.dcfg.get("cameras", ["cam0"])
        crop = tuple(self.dcfg.get("rgb_crop", [320, 0, 640, 240]))
        recs = []
        for seq in sorted(by_seq):
            sync_p, acc_p = self.root / f"{seq}-data.csv", self.root / f"{seq}-acc.csv"
            if not sync_p.exists():
                continue
            is_fall = seq.startswith("fall")
            sync = pd.read_csv(sync_p, header=None, names=["frame", "t_ms", "sv"])
            frame_t = dict(zip(sync.frame, sync.t_ms.astype(float)))

            f = by_seq[seq]
            f = f[f.frame.isin(frame_t)]
            t_lab = f.frame.map(frame_t).to_numpy(float)
            code = {(-1, True): 1, (0, True): 2, (1, True): 3, (-1, False): 1, (0, False): 4, (1, False): 5}
            labels = np.array([code[(int(p), is_fall)] for p in f.posture])

            rec = Recording(rid=seq, subject=self.dcfg.get("subject_name", "urfd"),
                            activity=1 if is_fall else 2, trial=int(seq.split("-")[1]),
                            label_t_ms=t_lab, labels=labels, meta={"sequence": seq, "is_fall": is_fall})
            if acc_p.exists():
                acc = pd.read_csv(acc_p, header=None, names=["t_ms", "sv_total", "ax", "az", "ay"])
                rec.add(Stream("acc", "IMU", acc.t_ms, acc[["ax", "ay", "az", "sv_total"]].to_numpy(),
                               ["ax", "ay", "az", "sv_total"]))
            rec.add(Stream("depth_feat", "DEPTH", t_lab, f[DEPTH_FEATURES].to_numpy(), DEPTH_FEATURES))
            for cam in cameras:
                vp = self.root / f"{seq}-{cam}.mp4"
                if vp.exists():
                    rec.videos[cam] = VideoSource(cam, vp, frame_times_ms=sync.t_ms.to_numpy(float), crop=crop)
            recs.append(rec)
        return recs
