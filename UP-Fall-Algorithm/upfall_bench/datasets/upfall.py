"""Original UP-Fall Detection consolidated dataset (Martinez-Villasenor et al., Sensors 2019).

https://sites.google.com/up.edu.mx/har-up/ -> "Consolidated dataset". Expected layout (Fig. 3)::

    root/Subject1/Activity1/Trial1/Subject1Activity1Trial1.csv
                                  /Subject1Activity1Trial1Camera1.zip   (PNG frames named by timestamp)
                                  /Subject1Activity1Trial1Camera2.zip

Any nesting works, files are found by name. Each CSV: timestamp, the 42 signals of Table 6
(in that order), then Subject, Activity, Trial and (in most releases) Tag. The per-sample
activity used as label is the LAST column (Tag), which inside fall trials switches between
standing, the fall and lying, as described in Sec. 3.6.1.

Having this adapter lets you run the *same* code on the original UP-Fall and compare against
Tables 13-14 of the paper before trusting the numbers on a new dataset.
"""
from __future__ import annotations

import re
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..core import Recording, Stream, VideoSource
from .base import DatasetAdapter, register

DEVICES = ["ankle", "pocket", "waist", "neck", "wrist"]           # Table 6, device IDs 1-5
IMU_CH = ["acc_x", "acc_y", "acc_z", "gyro_roll", "gyro_pitch", "gyro_yaw", "lux"]
ACTIVITIES = {1: "Falling forward using hands", 2: "Falling forward using knees", 3: "Falling backwards",
              4: "Falling sideward", 5: "Falling sitting in empty chair", 6: "Walking", 7: "Standing",
              8: "Sitting", 9: "Picking up an object", 10: "Jumping", 11: "Laying", 20: "Unknown"}
NAME_RE = re.compile(r"Subject(\d+)Activity(\d+)Trial(\d+)\.csv$", re.I)


def _ts_ms(s: str) -> float:
    s = str(s).strip().replace("_", ":")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).timestamp() * 1000
        except ValueError:
            pass
    return float("nan")


@register("upfall")
class UPFallAdapter(DatasetAdapter):

    def default_classes(self):
        return dict(ACTIVITIES)

    def default_fall_classes(self):
        return {1, 2, 3, 4, 5}

    def load_recordings(self):
        files = sorted(p for p in self.root.rglob("*.csv") if NAME_RE.search(p.name))
        if not files:
            raise FileNotFoundError(f"no SubjectXActivityYTrialZ.csv under {self.root}")
        lux_in_imu = self.dcfg.get("imu_include_luminosity", True)
        recs = []
        for p in files:
            s, a, t = map(int, NAME_RE.search(p.name).groups())
            raw = pd.read_csv(p, header=None, low_memory=False)
            ts = raw.iloc[:, 0].map(_ts_ms)
            raw = raw[ts.notna()]                                  # drops the 1-2 header rows
            t_ms = ts[ts.notna()].to_numpy(float)
            sig = raw.iloc[:, 1:43].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
            label = pd.to_numeric(raw.iloc[:, -1], errors="coerce").fillna(a).astype(int).to_numpy()

            rec = Recording(rid=f"S{s}A{a}T{t}", subject=str(s), activity=a, trial=t,
                            label_t_ms=t_ms, labels=label)
            for i, dev in enumerate(DEVICES):
                block = sig[:, 7 * i: 7 * i + 7]
                if lux_in_imu:
                    rec.add(Stream(dev, "IMU", t_ms, block, IMU_CH))
                else:
                    rec.add(Stream(dev, "IMU", t_ms, block[:, :6], IMU_CH[:6]))
                    rec.add(Stream(f"{dev}_lux", "LUX", t_ms, block[:, 6:], ["lux"]))
            rec.add(Stream("eeg", "EEG", t_ms, sig[:, 35:36], ["raw"]))
            rec.add(Stream("ir", "IR", t_ms, sig[:, 36:42], [f"ir{k}" for k in range(1, 7)]))
            for cam in (1, 2):
                z = p.with_name(p.stem + f"Camera{cam}.zip")
                if z.exists():
                    with zipfile.ZipFile(z) as zf:
                        names = sorted(n for n in zf.namelist() if n.lower().endswith(".png"))
                    times = np.array([_ts_ms(Path(n).stem) for n in names])
                    rec.videos[f"cam{cam}"] = VideoSource(f"cam{cam}", z, frame_times_ms=times)
            recs.append(rec)
        return recs
