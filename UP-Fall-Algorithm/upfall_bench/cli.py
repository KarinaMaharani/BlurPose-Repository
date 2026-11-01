"""Command line: ``python -m upfall_bench <command> --dataset configs/datasets/<name>.yaml``"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .config import apply_resources, load_config, resolve

DEFAULT_BENCH = "configs/benchmarks/upfall2019.yaml"


def _setup(args):
    from .datasets import get_adapter
    cfg = load_config(args.dataset, getattr(args, "benchmark", None) or DEFAULT_BENCH, args.set)
    apply_resources(cfg)
    return cfg, get_adapter(cfg)


def cmd_list(args):
    from .datasets import available
    print("registered dataset adapters:", ", ".join(available()))
    print("dataset configs:", ", ".join(p.stem for p in sorted((resolve("configs/datasets")).glob("*.yaml"))))


def cmd_download(args):
    cfg, ad = _setup(args)
    ad.download()


def cmd_inspect(args):
    from .derive.impact import acc_magnitude
    cfg, ad = _setup(args)
    recs = ad.recordings()
    summ = ad.summary(recs)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 90)
    print(f"\n{len(recs)} recordings, {summ.subject.nunique()} subjects")
    print(summ.head(int(args.rows)).to_string(index=False))
    # time per label and acceleration peak per activity: helps decide `fall_classes`
    rows = []
    for r in recs:
        t, mag = acc_magnitude(r)
        t0, t1 = r.span_ms
        rows.append(dict(activity=r.activity, subject=r.subject, duration_s=(t1 - t0) / 1000,
                         peak_g=float(np.nanmax(mag)) if mag is not None else np.nan))
    act = pd.DataFrame(rows).groupby("activity").agg(trials=("subject", "size"), subjects=("subject", "nunique"),
                                                      mean_duration_s=("duration_s", "mean"),
                                                      max_peak_g=("peak_g", "max"), mean_peak_g=("peak_g", "mean"))
    print("\nper recording activity (peak_g = max |acc| in g; falls usually > 2 g):")
    print(act.round(2).to_string())
    # labels as they are used for windows (inside a fall trial UP-Fall switches standing/fall/lying)
    names, falls = ad.classes, ad.fall_classes
    secs = {}
    for r in recs:
        t0, t1 = r.span_ms
        lab = r.label_at(np.arange(t0, t1, 10.0))
        for v, n in zip(*np.unique(lab, return_counts=True)):
            secs[int(v)] = secs.get(int(v), 0) + n / 100
    lab = pd.DataFrame({"seconds": secs}).sort_index()
    lab["name"] = [names.get(i, "") for i in lab.index]
    lab["is_fall"] = [i in falls for i in lab.index]
    print("\nper label class (what the classifiers predict):")
    print(lab.round(1).to_string())
    if not falls:
        print("\n!! fall_classes is empty: set `dataset.fall_classes` in the YAML for the binary task.")
    out = resolve(cfg.get("results_dir", "results")) / cfg["dataset"].get("name", ad.name)
    out.mkdir(parents=True, exist_ok=True)
    summ.to_csv(out / "inspect_recordings.csv", index=False)
    act.to_csv(out / "inspect_activities.csv")
    lab.to_csv(out / "inspect_labels.csv")
    print(f"\nsaved -> {out}")


def cmd_extract_flow(args):
    from .derive.flow import extract_flow
    cfg, ad = _setup(args)
    extract_flow(ad, ad.recordings(with_derived=False), cfg, overwrite=args.overwrite)


def cmd_extract_pose(args):
    from .derive.pose import extract_pose
    cfg, ad = _setup(args)
    extract_pose(ad, ad.recordings(with_derived=False), cfg, overwrite=args.overwrite)


def cmd_windows(args):
    from .arc.windowing import build_windows, feature_columns
    cfg, ad = _setup(args)
    recs = ad.recordings()
    for w in cfg["arc"]["windows"]:
        W = build_windows(recs, ad, cfg, float(w), rebuild=args.rebuild)
        print(f"{w}s: {len(W)} windows x {len(feature_columns(W))} features; "
              f"labels {W.label.value_counts().sort_index().to_dict()}; falls {int(W.fall.sum())}")


def cmd_run(args):
    from .arc.runner import run_arc
    cfg, ad = _setup(args)
    out = resolve(args.out) if args.out else None
    run_arc(ad, ad.recordings(), cfg, out, rebuild=args.rebuild)


def cmd_cnn(args):
    from .vision_cnn import run_cnn
    cfg, ad = _setup(args)
    run_cnn(ad, ad.recordings(with_derived=False), cfg, resolve(args.out) if args.out else None)


def cmd_skeleton(args):
    from .skeleton.bench import run_skeleton
    cfg, ad = _setup(args)
    run_skeleton(ad, ad.recordings(), cfg, resolve(args.out) if args.out else None)


def cmd_report(args):
    import yaml
    from .arc.report import make_report
    run = resolve(args.run)
    classes = None
    if (run / "config.yaml").exists():
        c = yaml.safe_load(open(run / "config.yaml", encoding="utf-8"))
        classes = {int(k): v for k, v in (c.get("dataset", {}).get("classes") or {}).items()}
    make_report(run, classes)


def main(argv=None):
    # joblib probes physical cores with a Windows tool that is often missing -> noisy traceback;
    # also a hard cap for any joblib worker pool (resources.n_jobs is the per-call setting)
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
    warnings.filterwarnings("ignore", message="Could not find the number of physical cores")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(prog="upfall_bench", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_, dataset=True, bench=False):
        sp = sub.add_parser(name, help=help_)
        if dataset:
            sp.add_argument("--dataset", "-d", required=True, help="dataset YAML, e.g. configs/datasets/urfall.yaml")
            sp.add_argument("--set", "-s", action="append", default=[], metavar="KEY=VALUE",
                            help="override any config value, e.g. -s arc.windows=[1] -s evaluation.protocol=loso")
        if bench:
            sp.add_argument("--benchmark", "-b", default=DEFAULT_BENCH, help="benchmark YAML")
        sp.set_defaults(fn=fn)
        return sp

    add("list", cmd_list, "list adapters and dataset configs", dataset=False)
    add("download", cmd_download, "download the dataset (UR Fall)")
    add("inspect", cmd_inspect, "summarise recordings, activities, acceleration peaks", bench=True) \
        .add_argument("--rows", default=40)
    for name, fn, h in (("extract-flow", cmd_extract_flow, "optical-flow features from videos (CAM modality)"),
                        ("extract-pose", cmd_extract_pose, "skeletons from videos (SKELETON modality)")):
        add(name, fn, h, bench=True).add_argument("--overwrite", action="store_true")
    add("windows", cmd_windows, "build & cache the window feature tables only", bench=True) \
        .add_argument("--rebuild", action="store_true")
    sp = add("run", cmd_run, "UP-Fall ML benchmark (RF/SVM/MLP/KNN x modalities x windows)", bench=True)
    sp.add_argument("--rebuild", action="store_true")
    sp.add_argument("--out")
    add("cnn", cmd_cnn, "UP-Fall vision CNN on raw frames", bench=True).add_argument("--out")
    add("skeleton", cmd_skeleton, "3D-skeleton impact benchmark (ML + LSTM/BiLSTM/CNN/STGCN)", bench=True) \
        .add_argument("--out")
    sp = add("report", cmd_report, "rebuild tables/figures of a run folder", dataset=False)
    sp.add_argument("--run", required=True)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
