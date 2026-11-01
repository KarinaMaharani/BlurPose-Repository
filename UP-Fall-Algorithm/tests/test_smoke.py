"""End-to-end checks on the synthetic dataset:  python -m pytest tests -q"""
import numpy as np
import pandas as pd

from upfall_bench.arc import features as F
from upfall_bench.arc.evaluation import metrics
from upfall_bench.arc.runner import run_arc
from upfall_bench.arc.windowing import build_windows, feature_columns
from upfall_bench.config import load_config
from upfall_bench.core import Stream, sample_and_hold
from upfall_bench.datasets import get_adapter

QUICK = ["arc.windows=[2]", "evaluation.rounds=1", "selection.max_features=10", "selection.max_samples=500",
         "dataset.subjects_n=2", "dataset.trials=2"]


def _cfg(tmp_path, extra=()):
    return load_config("configs/datasets/synthetic.yaml", "configs/benchmarks/quick.yaml",
                       QUICK + [f"cache_dir={tmp_path / 'cache'}", f"results_dir={tmp_path / 'results'}", *extra])


def test_features_known_values():
    t = np.arange(200) / 100
    x = np.sin(2 * np.pi * 5 * t)[:, None]
    f = F.signal_features(x, 100.0)
    assert set(f) == set(F.ALL)
    assert abs(f["principal_freq"][0] - 5) < 0.6
    assert abs(f["rms"][0] - np.sqrt(.5)) < 1e-2
    assert f["zero_crossings"][0] >= 18


def test_sample_and_hold():
    v = sample_and_hold(np.array([0., 10, 20]), np.array([[1.], [2.], [3.]]), np.array([-1., 0, 5, 15, 25]))
    assert np.isnan(v[0, 0]) and list(v[1:, 0]) == [1, 1, 2, 3]


def test_stream_sorts_and_validates():
    s = Stream("a", "IMU", [2, 1, 3], np.array([[2.], [1.], [3.]]), ["x"])
    assert list(s.t_ms) == [1, 2, 3] and list(s.values[:, 0]) == [1, 2, 3]


def test_metrics_binary_and_macro():
    m = metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), positive=1)
    assert m["accuracy"] == .75 and m["sensitivity"] == 1 and m["specificity"] == .5
    mm = metrics(np.array([0, 1, 2]), np.array([0, 1, 2]))
    assert mm["f1"] == 1


def test_windows_and_full_run(tmp_path):
    cfg = _cfg(tmp_path)
    ad = get_adapter(cfg)
    recs = ad.recordings()
    W = build_windows(recs, ad, cfg, 2.0)
    assert len(W) > 0 and W.fall.sum() > 0
    assert len(feature_columns(W)) == 6 * 18 + 36 * 18
    out = run_arc(ad, recs, cfg, tmp_path / "run")
    raw = pd.read_csv(out / "raw_metrics.csv")
    assert set(raw.model) == {"RF", "SVM", "MLP", "KNN"}
    assert (out / "results.md").exists()
    rf = raw[(raw.model == "RF") & (raw.experiment == "IMU") & (raw.task == "multiclass")]
    assert rf.f1.mean() > .7          # synthetic falls are easy
