"""The 18 window features of UP-Fall (Table 10: 12 temporal, Table 11: 6 frequency).

All functions take a window ``x`` of shape (n_samples, n_channels) and return (n_channels,).
The paper names the features but does not give formulas; the definitions used here are the
standard ones and are listed in the README ("Feature definitions").
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import stats

TEMPORAL = ["mean", "std", "rms", "max", "min", "median", "zero_crossings", "skewness", "kurtosis",
            "q1", "q3", "autocorrelation"]
FREQUENCY = ["mean_freq", "median_freq", "entropy", "energy", "principal_freq", "spectral_centroid"]
ALL = TEMPORAL + FREQUENCY


def temporal(x: np.ndarray) -> dict[str, np.ndarray]:
    m = x.mean(0)
    c = x - m
    var = (c ** 2).sum(0)
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        sk = np.nan_to_num(stats.skew(x, axis=0))
        ku = np.nan_to_num(stats.kurtosis(x, axis=0))
        ac = np.where(var > 0, (c[:-1] * c[1:]).sum(0) / np.where(var > 0, var, 1), 0.0)
    s = np.signbit(x)
    return {
        "mean": m, "std": x.std(0), "rms": np.sqrt((x ** 2).mean(0)), "max": x.max(0), "min": x.min(0),
        "median": np.median(x, 0), "zero_crossings": (s[1:] != s[:-1]).sum(0).astype(np.float32),
        "skewness": sk, "kurtosis": ku, "q1": np.percentile(x, 25, 0), "q3": np.percentile(x, 75, 0),
        "autocorrelation": ac,
    }


def frequency(x: np.ndarray, fs: float) -> dict[str, np.ndarray]:
    n = len(x)
    X = np.fft.rfft(x, axis=0)
    f = np.fft.rfftfreq(n, d=1.0 / fs)[:, None]
    mag = np.abs(X)
    p = mag ** 2
    ps = p.sum(0)
    safe = np.where(ps > 0, ps, 1)
    pn = p / safe
    cum = np.cumsum(p, 0)
    med = f[np.argmax(cum >= 0.5 * cum[-1:], axis=0), 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(pn > 0, pn * np.log2(pn), 0), 0)
    ms = mag.sum(0)
    p_ac = p[1:] if n > 2 else p
    return {
        "mean_freq": (f * p).sum(0) / safe,                          # power-weighted mean frequency
        "median_freq": med,                                         # frequency splitting power in half
        "entropy": ent,                                             # spectral (Shannon) entropy, bits
        "energy": ps / n,                                           # sum |X_k|^2 / N
        "principal_freq": f[np.argmax(p_ac, 0) + (1 if n > 2 else 0), 0],   # dominant non-DC frequency
        "spectral_centroid": (f * mag).sum(0) / np.where(ms > 0, ms, 1),   # magnitude-weighted centroid
    }


def signal_features(x: np.ndarray, fs: float, which=None) -> dict[str, np.ndarray]:
    which = which or ALL
    out = {}
    if any(w in TEMPORAL for w in which):
        out.update(temporal(x))
    if any(w in FREQUENCY for w in which):
        out.update(frequency(x, fs))
    return {k: out[k] for k in which}
