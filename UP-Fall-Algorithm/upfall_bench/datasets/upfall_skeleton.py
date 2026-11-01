"""Improved 3D-skeleton UP-Fall (Koffi et al. 2024), the ``3D_skeletons-UP-Fall-Dataset`` repo.

``root`` holds the ``SUBJECT*.zip`` files from the repo (or their extracted CSVs). Each CSV
``C{camera}S{subject}A{activity}T{trial}.csv`` has 100 frames x (33 MediaPipe joints x X,Y,Z)
plus a per-frame ``LABEL`` (1 = impact, 0 = no impact). A few files in the release are
misnamed (``C2S1_A1_T2``, ``CS3A4T2``, ``3S3A4T1``); the parser tolerates them.

Only fall activities (1-5) are in the release. Each file becomes one Recording with a
``SKELETON`` stream; the frame rate is not stored, ``fps`` in the YAML (default 18, the UP-Fall
camera rate) sets the time axis.
"""
from __future__ import annotations

import io
import re
import zipfile

import numpy as np
import pandas as pd

from ..core import Recording, Stream
from .base import DatasetAdapter, register

NAME_RE = re.compile(r"C?(\d*)S(\d+)_?A(\d+)_?T(\d+)\.csv$", re.I)


def _is_number(x) -> bool:
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False


@register("upfall_skeleton")
class UPFallSkeletonAdapter(DatasetAdapter):

    def default_classes(self):
        return {0: "no impact", 1: "impact"}

    def default_fall_classes(self):
        return {1}

    def _csv_sources(self):
        for z in sorted(self.root.glob("*.zip")):
            with zipfile.ZipFile(z) as zf:
                for n in sorted(zf.namelist()):
                    if n.lower().endswith(".csv"):
                        yield n.split("/")[-1], zf.read(n)
        for p in sorted(self.root.rglob("*.csv")):
            yield p.name, p.read_bytes()

    def load_recordings(self):
        fps = float(self.dcfg.get("fps", 18))
        recs, seen = [], set()
        for name, blob in self._csv_sources():
            m = NAME_RE.search(name)
            if not m or name in seen:
                continue
            seen.add(name)
            cam, s, a, t = m.groups()
            cam = cam or "x"
            # headers in the release vary ("LABEL", "LLABEL", or no header row at all), so the
            # layout is positional: 99 coordinates, then the label
            df = pd.read_csv(io.BytesIO(blob), header=None)
            if not _is_number(df.iat[0, 0]):
                df = df.iloc[1:]
            df = df.apply(pd.to_numeric, errors="coerce").dropna()
            joints = [f"Joint{j}_{ax}" for j in range(1, 34) for ax in "XYZ"]
            t_ms = np.arange(len(df)) * 1000.0 / fps
            rec = Recording(rid=f"C{cam}S{s}A{a}T{t}", subject=s, activity=int(a), trial=int(t),
                            label_t_ms=t_ms, labels=df.iloc[:, 99].astype(int).to_numpy(),
                            meta={"camera": cam, "file": name})
            rec.add(Stream("skeleton", "SKELETON", t_ms, df.iloc[:, :99].to_numpy(np.float32), joints))
            recs.append(rec)
        return recs
