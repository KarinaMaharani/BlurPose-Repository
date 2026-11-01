"""RA-Pak-Edy IMU + multi-camera sessions (the ``data-collector-IMU-mobile`` export format).

One session folder (an unzipped export) looks like::

    <session>/
      cameras.json                      started_at_ms per camera -> video clock
      videos/<sid>_cam1_video_sync.webm ...
      data/manifest.json                subject, label_timeline (authoritative label change points)
      data/<sid>_<role>_consolidated.csv  timestamp_ms, acc_x_g.., gyro_x_degs.., label_id, sample_kind

Put any number of session folders under ``root`` (nested folders are fine). Every session is
one long take in which the operator switched ``label_id`` while the subject performed the
protocol; ``0`` is the idle/"between activities" label.

Each contiguous non-idle label run becomes one ``Recording`` (= one UP-Fall trial): activity =
the label (or ``label_to_activity[label]``), trial = how many times that activity has been seen
so far in the session. All IMU rows with ``sample_kind != 0`` (held/placeholder rows) are
dropped, and ``timestamp_ms`` is used as-is: the export is irregularly sampled.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..core import Recording, Stream, VideoSource
from .base import DatasetAdapter, register

IMU_COLS = ["acc_x_g", "acc_y_g", "acc_z_g", "gyro_x_degs", "gyro_y_degs", "gyro_z_degs"]
IMU_NAMES = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]


def find_sessions(root: Path) -> list[Path]:
    out = set()
    for p in root.rglob("*_consolidated.csv"):
        if p.parent.name == "data" and not p.name.endswith("_sensor_data.csv"):
            out.add(p.parent.parent)
    return sorted(out)


@register("imu_session")
class IMUSessionAdapter(DatasetAdapter):

    def default_classes(self):
        return {}            # numeric labels are used as names unless `classes:` is given

    def load_recordings(self):
        sessions = find_sessions(self.root)
        if not sessions:
            raise FileNotFoundError(f"no session folders (with data/*_consolidated.csv) under {self.root}")
        recs = []
        for s in sessions:
            recs += self._load_session(s)
        return recs

    # ------------------------------------------------------------------------------------
    def _load_session(self, folder: Path) -> list[Recording]:
        d = self.dcfg
        roles = d.get("roles", ["waist", "chest", "thigh_right"])
        idle = int(d.get("idle_label", 0))
        pad = float(d.get("segment_padding_s", 0.0)) * 1000
        l2a = {int(k): int(v) for k, v in (d.get("label_to_activity") or {}).items()}
        skip = {int(x) for x in d.get("skip_labels", [])}

        manifest = _read_json(folder / "data" / "manifest.json") or _read_json(folder / "manifest.json") or {}
        cams_json = _read_json(folder / "cameras.json") or {}
        sid = str(manifest.get("session_id") or cams_json.get("session_id") or
                  next((folder / "data").glob("*_consolidated.csv")).name.split("_")[0])
        subject = str(manifest.get("subject") or folder.name) if d.get("subject_from", "manifest") == "manifest" \
            else folder.name

        # IMU streams, whole session
        streams = {}
        for role in roles:
            p = folder / "data" / f"{sid}_{role}_consolidated.csv"
            if not p.exists():
                print(f"  [{folder.name}] missing {p.name}, skipped")
                continue
            df = pd.read_csv(p, comment="#")
            if "sample_kind" in df and d.get("drop_held_rows", True):
                df = df[df.sample_kind == 0]
            df = df.drop_duplicates("timestamp_ms").sort_values("timestamp_ms")
            streams[role] = (df.timestamp_ms.to_numpy(float), df[IMU_COLS].to_numpy(np.float32), df)

        # label change points: manifest timeline if present, else from the first IMU file
        tl = (manifest.get("integrity") or {}).get("label_timeline")
        if tl:
            lt = np.array([e["timestamp_ms"] for e in tl], float)
            lv = np.array([int(e["label_id"]) for e in tl])
        else:
            df = next(iter(streams.values()))[2]
            ch = np.r_[True, df.label_id.to_numpy()[1:] != df.label_id.to_numpy()[:-1]]
            lt, lv = df.timestamp_ms.to_numpy(float)[ch], df.label_id.to_numpy()[ch]
        t_end = max(t[-1] for t, _, _ in streams.values())
        lt_end = np.r_[lt[1:], t_end]

        videos = {}
        for c in cams_json.get("cameras", []):
            if d.get("cameras") and c["cam_id"] not in d["cameras"]:
                continue
            vp = folder / c["file"]
            if vp.exists():
                videos[c["cam_id"]] = VideoSource(c["cam_id"], vp, t0_ms=float(c["started_at_ms"]),
                                                  fps=float(d.get("video_fps", 18)),
                                                  rotate=(d.get("camera_rotate") or {}).get(c["cam_id"]))

        recs, seen = [], defaultdict(int)
        for t0, t1, lab in zip(lt, lt_end, lv):
            lab = int(lab)
            if lab in skip or (lab == idle and not d.get("keep_idle", False)) or t1 - t0 < 500:
                continue
            act = l2a.get(lab, lab)
            seen[act] += 1
            a, b = t0 - pad, t1 + pad
            m = (lt >= a) & (lt < b)
            rec = Recording(rid=f"{subject}_{sid}_A{act}_T{seen[act]}", subject=subject, activity=act,
                            trial=seen[act],
                            label_t_ms=np.r_[a, lt[m], b] if pad else np.array([t0, t1]),
                            labels=np.r_[_label_at(lt, lv, a), lv[m], idle] if pad else np.array([lab, lab]),
                            videos=dict(videos), meta={"session": sid, "folder": str(folder), "raw_label": lab})
            if l2a:
                rec.labels = np.array([l2a.get(int(x), int(x)) for x in rec.labels])
            for role, (t, v, _) in streams.items():
                i0, i1 = np.searchsorted(t, [a, b])
                rec.add(Stream(role, "IMU", t[i0:i1], v[i0:i1], [f"{n}" for n in IMU_NAMES]))
            recs.append(rec)
        print(f"  session {sid} ({subject}): {len(recs)} labelled segments, roles={list(streams)}, "
              f"cameras={list(videos)}")
        return recs


def _label_at(lt, lv, t):
    i = np.searchsorted(lt, t, side="right") - 1
    return int(lv[max(i, 0)])


def _read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
