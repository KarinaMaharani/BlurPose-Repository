"""Dataset-independent data model.

Every dataset adapter turns its files into a list of ``Recording`` objects. Everything
downstream (windowing, features, models, evaluation) only ever sees these classes, which
is what makes adding a new dataset a matter of writing one adapter.

Time is always **milliseconds** (float64) on a clock shared by all streams of one
recording. The absolute origin does not matter (epoch ms, or ms since trial start).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# Modality tags used to compose experiments ("IMU", "IMU+CAM", ...). Adapters may use any
# string, these are just the conventional ones from the UP-Fall paper plus a few extras.
MODALITIES = ("IMU", "ACC", "EEG", "IR", "CAM", "DEPTH", "SKELETON")


@dataclass
class Stream:
    """One time series of a recording, e.g. the waist IMU or the cam1 optical-flow vector.

    ``kind`` decides how a window of it becomes features:
      * ``"signal"``        – 18 temporal + frequency statistics per channel (paper, Tables 10-11)
      * ``"frame_vector"``  – mean of every channel over the window (paper, Sec. 5.2, images)
    """
    name: str
    modality: str
    t_ms: np.ndarray
    values: np.ndarray
    channels: list[str]
    kind: str = "signal"

    def __post_init__(self):
        self.t_ms = np.asarray(self.t_ms, dtype=np.float64)
        v = np.asarray(self.values, dtype=np.float32)
        self.values = v[:, None] if v.ndim == 1 else v
        if len(self.t_ms) != len(self.values):
            raise ValueError(f"stream {self.name}: {len(self.t_ms)} timestamps vs {len(self.values)} rows")
        if self.values.shape[1] != len(self.channels):
            raise ValueError(f"stream {self.name}: {self.values.shape[1]} columns vs {len(self.channels)} channel names")
        order = np.argsort(self.t_ms, kind="stable")
        if np.any(order != np.arange(len(order))):
            self.t_ms, self.values = self.t_ms[order], self.values[order]

    @property
    def rate_hz(self) -> float:
        if len(self.t_ms) < 2:
            return float("nan")
        return 1000.0 / float(np.median(np.diff(self.t_ms)))

    def slice(self, t0: float, t1: float) -> np.ndarray:
        """Rows with t0 <= t < t1."""
        i0, i1 = np.searchsorted(self.t_ms, [t0, t1], side="left")
        return self.values[i0:i1]


@dataclass
class VideoSource:
    """Where the frames of one camera live and how frame index maps to time.

    Either ``frame_times_ms`` (one entry per frame, exact) or ``t0_ms`` + ``fps`` (constant
    rate; the reader then resamples the video to that rate with ffmpeg).
    """
    camera: str
    path: Path
    t0_ms: float = 0.0
    fps: Optional[float] = None
    frame_times_ms: Optional[np.ndarray] = None
    crop: Optional[tuple[int, int, int, int]] = None   # x0, y0, x1, y1 in source pixels
    rotate: Optional[str] = None                        # "cw" | "ccw" | "180"


@dataclass
class Recording:
    """One trial: a subject doing one activity once (UP-Fall's SubjectXActivityYTrialZ).

    ``labels`` is a per-time class label (int) at ``label_t_ms``. Inside a fall trial it may
    change (UP-Fall: standing -> falling -> lying), so window labels are taken as the most
    frequent value in the window, exactly as in the paper.
    """
    rid: str
    subject: str
    activity: int
    trial: int
    streams: dict[str, Stream] = field(default_factory=dict)
    label_t_ms: np.ndarray = field(default_factory=lambda: np.zeros(0))
    labels: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    videos: dict[str, VideoSource] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def add(self, stream: Stream):
        self.streams[stream.name] = stream

    @property
    def span_ms(self) -> tuple[float, float]:
        if len(self.label_t_ms):
            return float(self.label_t_ms[0]), float(self.label_t_ms[-1])
        ts = [s.t_ms for s in self.streams.values() if len(s.t_ms)]
        return float(min(t[0] for t in ts)), float(max(t[-1] for t in ts))

    def label_at(self, t_ms: np.ndarray) -> np.ndarray:
        """Label of the most recent label sample at each time (sample-and-hold)."""
        idx = np.searchsorted(self.label_t_ms, t_ms, side="right") - 1
        return self.labels[np.clip(idx, 0, len(self.labels) - 1)]


def sample_and_hold(t_src: np.ndarray, v_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """Resample by repeating the last known sample, the 'drop-sampling interpolation'
    the UP-Fall authors used to put every device on the camera clock (Sec. 3.5)."""
    idx = np.searchsorted(t_src, t_dst, side="right") - 1
    out = v_src[np.clip(idx, 0, len(v_src) - 1)].astype(np.float32, copy=True)
    out[idx < 0] = np.nan
    return out


def resample_stream(s: Stream, rate_hz: float, t0: float, t1: float) -> Stream:
    grid = np.arange(t0, t1, 1000.0 / rate_hz)
    return Stream(s.name, s.modality, grid, sample_and_hold(s.t_ms, s.values, grid), s.channels, s.kind)
