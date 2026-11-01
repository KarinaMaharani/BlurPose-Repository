"""Adapter base class and registry.

To support a new dataset you write ONE subclass of ``DatasetAdapter`` that implements
``load_recordings()`` and register it with ``@register("name")``. See ``_template.py``.
Alternatively point ``dataset.adapter`` in the YAML at ``"some_module:SomeClass"`` and the
class is imported from your own file, no edit of this package needed.
"""
from __future__ import annotations

import fnmatch
import importlib
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import resolve
from ..core import Recording, Stream

_REGISTRY: dict[str, type["DatasetAdapter"]] = {}


def register(name: str):
    def deco(cls):
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return deco


def available() -> list[str]:
    return sorted(_REGISTRY)


def get_adapter(cfg: dict) -> "DatasetAdapter":
    key = cfg["dataset"]["adapter"]
    if ":" in key:                                     # "path/or.module:ClassName"
        mod, cls = key.split(":", 1)
        if mod.endswith(".py") or "/" in mod or "\\" in mod:
            spec = importlib.util.spec_from_file_location("user_adapter", resolve(mod))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            module = importlib.import_module(mod)
        return getattr(module, cls)(cfg)
    if key not in _REGISTRY:
        raise KeyError(f"unknown dataset adapter '{key}'. Registered: {available()}")
    return _REGISTRY[key](cfg)


class DatasetAdapter:
    """Turns raw files into ``Recording`` objects.

    Subclasses MUST implement ``load_recordings``. They MAY override ``default_classes``,
    ``default_fall_classes`` and ``download``. Everything in ``dataset:`` of the YAML is
    available as ``self.dcfg``.
    """
    name = "base"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.dcfg = cfg["dataset"]
        self.root = resolve(self.dcfg.get("root", f"data/{self.dcfg.get('name', self.name)}"))
        self.cache_dir = resolve(cfg.get("cache_dir", "cache")) / self.dcfg.get("name", self.name)

    # ---- to implement -------------------------------------------------------------------
    def load_recordings(self) -> list[Recording]:
        raise NotImplementedError

    def default_classes(self) -> dict[int, str]:
        return {}

    def default_fall_classes(self) -> set[int]:
        return set()

    def download(self):
        raise NotImplementedError(f"{self.name}: no automatic download, place the files in {self.root}")

    # ---- shared behaviour ---------------------------------------------------------------
    @property
    def classes(self) -> dict[int, str]:
        cls = self.dcfg.get("classes") or self.default_classes()
        return {int(k): str(v) for k, v in cls.items()}

    @property
    def fall_classes(self) -> set[int]:
        fc = self.dcfg.get("fall_classes")
        return {int(x) for x in fc} if fc is not None else set(self.default_fall_classes())

    def recordings(self, with_derived: bool = True) -> list[Recording]:
        recs = self.load_recordings()
        keep_subj = self.dcfg.get("subjects")
        if keep_subj:
            recs = [r for r in recs if str(r.subject) in {str(s) for s in keep_subj}]
        include = self.dcfg.get("include")             # glob patterns on recording ids
        if include:
            recs = [r for r in recs if any(fnmatch.fnmatch(r.rid, p) for p in include)]
        drop_streams = set(self.dcfg.get("drop_streams") or [])
        for r in recs:
            for s in drop_streams:
                r.streams.pop(s, None)
        if with_derived:
            self.attach_derived(recs)
        if not recs:
            raise RuntimeError(f"{self.name}: no recordings found under {self.root}")
        return recs

    # ---- derived streams (optical flow, pose) computed from videos and cached ----------
    def derived_path(self, kind: str, camera: str, video_path: Path) -> Path:
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(video_path).stem)
        return self.cache_dir / "derived" / kind / f"{camera}__{stem}.npz"

    def attach_derived(self, recs: list[Recording]):
        kinds = [k for k in ("flow", "pose") if (self.cache_dir / "derived" / k).exists()]
        for r in recs:
            t0, t1 = r.span_ms
            for cam, vid in r.videos.items():
                for kind in kinds:
                    p = self.derived_path(kind, cam, vid.path)
                    if not p.exists():
                        continue
                    t, v, ch, modality, skind = _load_npz(str(p))
                    m = (t >= t0 - 1000) & (t <= t1 + 1000)
                    if m.sum() == 0:
                        continue
                    r.add(Stream(f"{cam}_{kind}", modality, t[m], v[m], ch, skind))

    def summary(self, recs: list[Recording]) -> pd.DataFrame:
        rows = []
        for r in recs:
            t0, t1 = r.span_ms
            rows.append(dict(rid=r.rid, subject=r.subject, activity=r.activity, trial=r.trial,
                             duration_s=round((t1 - t0) / 1000, 2),
                             streams=", ".join(f"{s.name}[{s.modality},{len(s.channels)}ch,{s.rate_hz:.0f}Hz]"
                                               for s in r.streams.values()),
                             cameras=", ".join(r.videos)))
        return pd.DataFrame(rows)


@lru_cache(maxsize=64)
def _load_npz(path: str):
    z = np.load(path, allow_pickle=False)
    return (z["t_ms"], z["values"], [str(c) for c in z["channels"]],
            str(z["modality"]), str(z["kind"]))


def save_derived(path: Path, t_ms, values, channels, modality: str, kind: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, t_ms=np.asarray(t_ms, np.float64), values=np.asarray(values, np.float32),
                        channels=np.asarray(channels), modality=modality, kind=kind)
    _load_npz.cache_clear()
