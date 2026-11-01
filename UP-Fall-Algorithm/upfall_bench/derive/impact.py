"""Semi-automatic impact labelling (Koffi et al. 2024, Sec. "Impact labeling").

Inside every fall-labelled stretch, the instant of maximum acceleration magnitude (SMV) is the
impact if it exceeds ``threshold_g`` (paper: 2 g instead of UP-Fall's 1 g). Frames within
``[peak - before_ms, peak + after_ms]`` get label 1, everything else 0. The paper then checked
every label by eye; ``report`` prints the peaks so you can do the same.
"""
from __future__ import annotations

import re

import numpy as np

from ..core import Recording


def acc_magnitude(rec: Recording, stream: str | None = None):
    cands = [rec.streams[stream]] if stream else [s for s in rec.streams.values() if s.modality in ("IMU", "ACC")]
    for s in cands:
        idx = [i for i, c in enumerate(s.channels) if re.fullmatch(r"(acc_)?a?[xyz](_g)?", c.lower())]
        if len(idx) == 3:
            return s.t_ms, np.linalg.norm(s.values[:, idx], axis=1)
    return None, None


def impact_labels(rec: Recording, fall_classes: set[int], threshold_g=2.0, before_ms=500.0, after_ms=500.0,
                  stream: str | None = None, t_query: np.ndarray | None = None):
    """Return (t, label) with label in {0, 1}, at ``t_query`` (default: the recording's label times)."""
    t_query = rec.label_t_ms if t_query is None else t_query
    out = np.zeros(len(t_query), int)
    t, mag = acc_magnitude(rec, stream)
    if t is None:
        return t_query, out, None
    is_fall = np.isin(rec.label_at(t), list(fall_classes))
    if not is_fall.any():
        return t_query, out, None
    m = np.where(is_fall, mag, -np.inf)
    k = int(np.argmax(m))
    if mag[k] < threshold_g:
        return t_query, out, float(mag[k])
    out[(t_query >= t[k] - before_ms) & (t_query <= t[k] + after_ms)] = 1
    return t_query, out, float(mag[k])
