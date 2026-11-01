"""TEMPLATE: copy this file to add a new dataset.

1. Copy to ``upfall_bench/datasets/my_dataset.py`` and rename the class + ``@register`` name.
2. Add ``from . import my_dataset`` to ``upfall_bench/datasets/__init__.py``
   (or skip both steps and set ``adapter: "path/to/my_dataset.py:MyDatasetAdapter"`` in YAML).
3. Copy ``configs/datasets/_template.yaml`` and fill it in.
4. ``python -m upfall_bench inspect --dataset configs/datasets/my_dataset.yaml``

The ONLY job of an adapter is to return ``Recording`` objects:
  * one Recording per trial (subject x activity x repetition),
  * every sensor as a ``Stream`` (timestamps in ms, values N x C, channel names, modality tag),
  * a label timeline (``label_t_ms``, ``labels``): change points or one label per sample,
  * optionally cameras as ``VideoSource`` (for optical flow, pose and the CNN).
Everything else (windowing, the 18 features, selection, models, metrics, reports) is shared.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import Recording, Stream, VideoSource
from .base import DatasetAdapter  # , register


# @register("my_dataset")
class MyDatasetAdapter(DatasetAdapter):

    def default_classes(self):
        # label id -> readable name (used in reports). Can also be given in the YAML `classes:`.
        return {1: "fall", 2: "walk", 3: "sit"}

    def default_fall_classes(self):
        # which label ids count as "fall" for the binary task. YAML `fall_classes:` overrides.
        return {1}

    def load_recordings(self):
        recs = []
        for csv in sorted(self.root.glob("*.csv")):            # self.root = dataset.root from YAML
            df = pd.read_csv(csv)
            subject, activity, trial = csv.stem.split("_")        # adapt to your naming scheme
            t = df["timestamp_ms"].to_numpy(float)
            rec = Recording(rid=csv.stem, subject=subject, activity=int(activity), trial=int(trial),
                            label_t_ms=t, labels=df["label"].to_numpy(int))
            rec.add(Stream("waist", "IMU", t, df[["ax", "ay", "az", "gx", "gy", "gz"]].to_numpy(),
                           ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]))
            video = csv.with_suffix(".mp4")
            if video.exists():
                # constant-rate video whose first frame is at t0_ms on the sensor clock:
                rec.videos["cam1"] = VideoSource("cam1", video, t0_ms=float(t[0]), fps=18)
            recs.append(rec)
        return recs
