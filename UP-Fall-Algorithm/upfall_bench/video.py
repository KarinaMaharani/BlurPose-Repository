"""Frame reader for every ``VideoSource``: video files through an ffmpeg pipe, or a ZIP of
images (UP-Fall's camera zips). ffmpeg is used instead of OpenCV because VP9 WebM files from
browser recorders have no frame index (OpenCV reports 1000 fps and cannot seek)."""
from __future__ import annotations

import re
import shutil
import subprocess
import zipfile
from typing import Iterator, Optional

import numpy as np

from .core import VideoSource


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as e:
        raise RuntimeError("ffmpeg not found: install it or `pip install imageio-ffmpeg`") from e


def probe_size(path) -> tuple[int, int]:
    r = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    m = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", r.stderr)
    if not m:
        raise RuntimeError(f"cannot read video size of {path}:\n{r.stderr[-500:]}")
    return int(m.group(1)), int(m.group(2))


def _out_size(src: VideoSource, size: Optional[tuple[int, int]]) -> tuple[int, int]:
    if size:
        return size
    w, h = probe_size(src.path)
    if src.crop:
        w, h = src.crop[2] - src.crop[0], src.crop[3] - src.crop[1]
    if src.rotate in ("cw", "ccw"):
        w, h = h, w
    return w, h


def iter_frames(src: VideoSource, size: Optional[tuple[int, int]] = None, gray: bool = False,
                fps: Optional[float] = None) -> Iterator[tuple[float, np.ndarray]]:
    """Yield ``(t_ms, frame)``; frame is HxWx3 RGB uint8 (or HxW if ``gray``).

    ``size=(w, h)`` resizes; ``fps`` overrides the source rate (constant-rate sources only).
    """
    if str(src.path).lower().endswith(".zip"):
        yield from _iter_zip(src, size, gray)
        return
    w, h = _out_size(src, size)
    vf = []
    if src.crop:
        x0, y0, x1, y1 = src.crop
        vf.append(f"crop={x1 - x0}:{y1 - y0}:{x0}:{y0}")
    vf += {"cw": ["transpose=1"], "ccw": ["transpose=2"], "180": ["transpose=1,transpose=1"]}.get(src.rotate, [])
    rate = fps or src.fps
    if src.frame_times_ms is None:
        if not rate:
            raise ValueError(f"{src.path}: VideoSource needs fps or frame_times_ms")
        vf.append(f"fps={rate}")
    vf.append(f"scale={w}:{h}:flags=area")
    pix, ch = ("gray", 1) if gray else ("rgb24", 3)
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-i", str(src.path), "-vf", ",".join(vf),
           "-f", "rawvideo", "-pix_fmt", pix, "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10 ** 7)
    nbytes, i = w * h * ch, 0
    try:
        while True:
            buf = proc.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            if src.frame_times_ms is not None:
                if i >= len(src.frame_times_ms):
                    break
                t = float(src.frame_times_ms[i])
            else:
                t = src.t0_ms + i * 1000.0 / rate
            f = np.frombuffer(buf, np.uint8).reshape((h, w) if gray else (h, w, 3))
            yield t, f
            i += 1
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


def _iter_zip(src: VideoSource, size, gray):
    import cv2
    with zipfile.ZipFile(src.path) as zf:
        names = sorted(n for n in zf.namelist() if n.lower().endswith((".png", ".jpg", ".jpeg")))
        for i, n in enumerate(names):
            img = cv2.imdecode(np.frombuffer(zf.read(n), np.uint8), cv2.IMREAD_COLOR)
            if src.crop:
                x0, y0, x1, y1 = src.crop
                img = img[y0:y1, x0:x1]
            if src.rotate:
                img = cv2.rotate(img, {"cw": cv2.ROTATE_90_CLOCKWISE, "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
                                       "180": cv2.ROTATE_180}[src.rotate])
            if size:
                img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY if gray else cv2.COLOR_BGR2RGB)
            t = float(src.frame_times_ms[i]) if src.frame_times_ms is not None else src.t0_ms + i * 1000 / src.fps
            yield t, img


def unique_videos(recs) -> dict:
    """Map (camera, path) -> (VideoSource, [(t0, t1) spans of recordings that use it])."""
    out = {}
    for r in recs:
        for cam, v in r.videos.items():
            key = (cam, str(v.path))
            out.setdefault(key, (v, []))[1].append(r.span_ms)
    return out


def in_spans(t: float, spans, margin_ms: float = 0.0) -> bool:
    return any(a - margin_ms <= t <= b + margin_ms for a, b in spans)
