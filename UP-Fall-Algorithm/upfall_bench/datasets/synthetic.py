"""A small fake dataset with the UP-Fall structure, for smoke tests and for trying the
pipeline before any real data is downloaded. Falls are a 2-4 g spike followed by a
change of gravity axis (lying); ADLs are periodic or still signals with noise."""
from __future__ import annotations

import numpy as np

from ..core import Recording, Stream
from .base import DatasetAdapter, register


@register("synthetic")
class SyntheticAdapter(DatasetAdapter):

    def default_classes(self):
        return {1: "fall", 6: "walking", 7: "standing", 11: "laying"}

    def default_fall_classes(self):
        return {1}

    def load_recordings(self):
        rng = np.random.default_rng(int(self.dcfg.get("seed", 0)))
        n_subj, n_trials = int(self.dcfg.get("subjects_n", 4)), int(self.dcfg.get("trials", 3))
        recs = []
        for s in range(1, n_subj + 1):
            for act in (1, 6, 7, 11):
                for tr in range(1, n_trials + 1):
                    dur = 10.0 if act == 1 else 20.0
                    t = np.arange(0, dur * 1000, 10.0) + rng.uniform(0, 3, int(dur * 100))
                    t.sort()
                    labels = np.full(len(t), act)
                    acc = np.tile([0.0, 0.0, 1.0], (len(t), 1)) + rng.normal(0, .03, (len(t), 3))
                    gyr = rng.normal(0, 3, (len(t), 3))
                    if act == 6:
                        acc[:, 2] += .3 * np.sin(2 * np.pi * 1.8 * t / 1000)
                        gyr[:, 0] += 30 * np.sin(2 * np.pi * 1.8 * t / 1000)
                    if act == 11:
                        acc = np.tile([1.0, 0.0, 0.0], (len(t), 1)) + rng.normal(0, .02, (len(t), 3))
                    if act == 1:
                        t_imp = rng.uniform(3500, 5500)
                        pre, imp = (t >= t_imp - 1000) & (t < t_imp), (t >= t_imp) & (t < t_imp + 300)
                        post = t >= t_imp + 300
                        acc[pre] += rng.normal(0, .4, (pre.sum(), 3))
                        acc[imp] += rng.normal(0, 1.5, (imp.sum(), 3)) + [0, 0, rng.uniform(1, 3)]
                        acc[post] = [1.0, 0, 0] + rng.normal(0, .02, (post.sum(), 3))
                        gyr[pre | imp] += rng.normal(0, 150, ((pre | imp).sum(), 3))
                        labels = np.where(t < t_imp - 1000, 7, np.where(post, 11, 1))
                    rec = Recording(rid=f"S{s}A{act}T{tr}", subject=str(s), activity=act, trial=tr,
                                    label_t_ms=t, labels=labels)
                    rec.add(Stream("waist", "IMU", t, np.c_[acc, gyr],
                                   ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]))
                    ts = np.arange(0, dur * 1000, 1000 / 18)
                    lab18 = rec.label_at(ts)
                    skel = rng.normal(0, .01, (len(ts), 12 * 3)) + np.where(lab18[:, None] == 11, .5, 0)
                    rec.add(Stream("skeleton", "SKELETON", ts, skel, [f"j{j}_{c}" for j in range(12) for c in "xyz"]))
                    recs.append(rec)
        return recs
