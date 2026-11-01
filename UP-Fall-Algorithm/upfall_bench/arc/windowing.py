"""Windowing step of the activity recognition chain (Sec. 5.2): fixed windows of 1/2/3 s with
50 % overlap, 18 features per channel of every ``signal`` stream, the mean of every channel of
every ``frame_vector`` stream (optical flow), and the window label = most frequent label.

Every stream is first put on a uniform grid by sample-and-hold. ``resample_hz: 18`` reproduces
the paper (everything on the ~18 Hz camera clock); ``resample_hz: native`` keeps each stream at
its own rate (better for 100 Hz IMUs, not the paper).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..core import Recording, sample_and_hold
from . import features as F

META = ["rid", "subject", "activity", "trial", "t0_ms", "label", "fall"]


def _uniform(stream, rate):
    grid = np.arange(stream.t_ms[0], stream.t_ms[-1] + 1e-6, 1000.0 / rate)
    v = sample_and_hold(stream.t_ms, stream.values, grid)
    if np.isnan(v).any():                               # e.g. frames without a detected person
        v = pd.DataFrame(v).ffill().bfill().to_numpy(np.float32)
    return grid, v


def recording_windows(rec: Recording, win_s: float, overlap: float, resample, fall_classes: set,
                      which_feats=None, min_coverage=0.5) -> pd.DataFrame:
    win = win_s * 1000.0
    step = win * (1 - overlap)
    t0, t1 = rec.span_ms
    starts = np.arange(t0, t1 - win + 1e-6, step)
    if len(starts) == 0:
        return pd.DataFrame()
    cols: dict[str, np.ndarray] = {}
    for s in rec.streams.values():
        if len(s.t_ms) < 2:
            continue
        rate = s.rate_hz if resample in (None, "native") else float(resample)
        rate = float(np.clip(round(rate), 1, 2000))
        grid, v = _uniform(s, rate)
        need = max(2, int(min_coverage * win_s * rate))
        idx = np.searchsorted(grid, np.c_[starts, starts + win])
        if s.kind == "frame_vector":
            names = [f"{s.name}.{c}.mean" for c in s.channels]
            block = np.full((len(starts), len(s.channels)), np.nan, np.float32)
            for i, (a, b) in enumerate(idx):
                if b - a >= max(1, need // 2):
                    block[i] = v[a:b].mean(0)
            cols.update(dict(zip(names, block.T)))
            continue
        feats = which_feats or F.ALL
        block = np.full((len(starts), len(feats), len(s.channels)), np.nan, np.float32)
        for i, (a, b) in enumerate(idx):
            if b - a >= need:
                d = F.signal_features(v[a:b].astype(np.float64), rate, feats)
                block[i] = np.stack([d[f] for f in feats])
        for j, f in enumerate(feats):
            for k, c in enumerate(s.channels):
                cols[f"{s.name}.{c}.{f}"] = block[:, j, k]
    labels = np.array([pd.Series(rec.label_at(np.arange(a, a + win, 10.0))).mode().iloc[0] for a in starts])
    meta = pd.DataFrame({"rid": rec.rid, "subject": str(rec.subject), "activity": rec.activity,
                         "trial": rec.trial, "t0_ms": starts, "label": labels.astype(int),
                         "fall": np.isin(labels, list(fall_classes)).astype(int)})
    return pd.concat([meta, pd.DataFrame(cols)], axis=1)


def stream_modalities(recs) -> dict[str, str]:
    out = {}
    for r in recs:
        for s in r.streams.values():
            out[s.name] = s.modality
    return out


def build_windows(recs, adapter, cfg: dict, win_s: float, rebuild: bool = False) -> pd.DataFrame:
    a = cfg.get("arc", {})
    overlap, resample = float(a.get("overlap", .5)), a.get("resample_hz", 18)
    feats, cov = a.get("features"), float(a.get("min_coverage", .5))
    sig = json.dumps(dict(win=win_s, ov=overlap, rs=resample, f=feats, cov=cov, fc=sorted(adapter.fall_classes),
                          recs=[(r.rid, sorted((s.name, len(s.t_ms)) for s in r.streams.values())) for r in recs]),
                     default=str)
    key = hashlib.md5(sig.encode()).hexdigest()[:10]
    path = Path(adapter.cache_dir) / "windows" / f"win{win_s:g}s_ov{overlap:g}_rs{resample}_{key}.pkl"
    if path.exists() and not rebuild:
        return pd.read_pickle(path)
    parts = [recording_windows(r, win_s, overlap, resample, adapter.fall_classes, feats, cov)
             for r in tqdm(recs, desc=f"windows {win_s:g}s", unit="rec", mininterval=2)]
    df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    df.attrs["stream_modality"] = stream_modalities(recs)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META]
