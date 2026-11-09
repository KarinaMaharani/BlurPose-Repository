"""The full UP-Fall benchmark: for every window size x experiment (modality combination) x task,
select features, then train/test every model under the chosen protocol. Writes everything a
report needs into one run folder.

Resumable: every finished (window, task, experiment) is written to ``parts/`` straight away.
If the run is stopped, running the same command again with the same ``--out`` folder skips
the finished parts and continues where it stopped.
"""
from __future__ import annotations

import fnmatch
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix

# Table 12 caps the MLP at 10 epochs, so it never "converges"; the warning is expected noise.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from ..config import n_jobs, resolve  # noqa: E402
from .evaluation import metrics, splits  # noqa: E402
from .models import build_model  # noqa: E402
from .selection import select_features  # noqa: E402
from .windowing import build_windows, feature_columns  # noqa: E402


def experiment_columns(df: pd.DataFrame, tokens: list[str]) -> tuple[list[str], list[str]]:
    """Feature columns whose stream matches any token (modality tag, stream name or glob),
    plus the tokens that matched no stream (an experiment with a missing part is skipped)."""
    s2m = df.attrs.get("stream_modality", {})
    hit = lambda t, s, m: t == m or t == s or fnmatch.fnmatch(s, t)
    missing = [t for t in tokens if not any(hit(t, s, m) for s, m in s2m.items())]
    streams = {s for s, m in s2m.items() if any(hit(t, s, m) for t in tokens)}
    return [c for c in feature_columns(df) if c.split(".", 1)[0] in streams], missing


def _part_name(task, exp, win) -> str:
    return f"{task}__{exp}__{win:g}s"


def _write_atomic(path: Path, write):
    """Write to a temporary file, then rename: a crash never leaves a half-written part."""
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def _check_same_config(out_dir: Path, cfg: dict):
    """Refuse to resume into a folder that was made with a different benchmark setup.
    ``resources`` is ignored, so you can change the number of cores between restarts."""
    saved = out_dir / "config.yaml"
    if not saved.exists():
        return
    old = yaml.safe_load(saved.read_text(encoding="utf-8")) or {}
    # experiments may change (e.g. skeleton added later): finished parts stay valid, so they're allowed
    strip = lambda c: {k: v for k, v in c.items() if k not in ("resources", "experiments")}
    if strip(old) != strip(cfg):
        print(f"{out_dir} was started with a different configuration. "
              f"Use a new --out folder, or delete this folder to start over.", flush=True)
        raise SystemExit(3)


def run_arc(adapter, recs, cfg: dict, out_dir: Path | None = None, rebuild: bool = False) -> Path:
    a, ev = cfg.get("arc", {}), cfg.get("evaluation", {})
    name = cfg["dataset"].get("name", adapter.name)
    if out_dir is None:
        out_dir = resolve(cfg.get("results_dir", "results")) / name / "upfall2019"   # fixed: resume finds it
    out_dir.mkdir(parents=True, exist_ok=True)
    _check_same_config(out_dir, cfg)
    (out_dir / "selection").mkdir(exist_ok=True)
    (out_dir / "confusion").mkdir(exist_ok=True)
    (out_dir / "parts").mkdir(exist_ok=True)
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:    # the latest settings used
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    log_f = open(out_dir / "log.txt", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    experiments = cfg.get("experiments") or {"ALL": sorted({s.modality for r in recs for s in r.streams.values()})}
    tasks = a.get("tasks", ["multiclass"])
    models = cfg.get("models") or {"RF": {}, "SVM": {}, "MLP": {}, "KNN": {}}
    protocol, seed = ev.get("protocol", "holdout"), int(ev.get("seed", 0))
    scope = cfg.get("selection", {}).get("scope", "global")
    windows = [float(w) for w in a.get("windows", [1, 2, 3])]
    done = sorted(p.stem for p in (out_dir / "parts").glob("*.csv"))
    log(f"\n##### run started {time.strftime('%Y-%m-%d %H:%M:%S')} -> {out_dir} "
        f"({len(done)} parts already finished) #####")

    for win in windows:
        W = None
        for task in tasks:
            for exp, tokens in experiments.items():
                part = out_dir / "parts" / f"{_part_name(task, exp, win)}.csv"
                if part.exists():
                    log(f"  [{task}] {exp} {win:g}s: already done, skipped")
                    continue
                if W is None:
                    W = build_windows(recs, adapter, cfg, win, rebuild=rebuild)
                    log(f"\n=== window {win:g}s: {len(W)} windows, {len(feature_columns(W))} features ===")
                rows = _run_one(W, adapter, cfg, out_dir, task, exp, tokens, win, models, protocol, seed,
                                scope, ev, log)
                if rows is None:
                    continue
                _write_atomic(part, lambda p: pd.DataFrame(rows).to_csv(p, index=False))
                log(f"      part saved: {part.name}")

    log_f.close()
    rows = []
    for part in sorted((out_dir / "parts").glob("*.csv")):
        rows += pd.read_csv(part).to_dict("records")
    if not rows:
        raise RuntimeError("no experiment produced results; see the messages above")
    pd.DataFrame(rows).to_csv(out_dir / "raw_metrics.csv", index=False)
    from .report import make_report
    make_report(out_dir, class_names=adapter.classes)
    return out_dir


def _run_one(W, adapter, cfg, out_dir, task, exp, tokens, win, models, protocol, seed, scope, ev, log):
    """One (task, experiment, window): returns its metric rows, or None if it can't be run."""
    y_all = (W["fall"] if task == "binary" else W["label"]).to_numpy()
    cols, missing = experiment_columns(W, tokens)
    if missing or not cols:
        log(f"  [{task}] {exp}: no stream matches {missing or tokens}, skipped")
        return None
    D = W[cols]
    usable = D.columns[D.notna().mean() > 0.5]
    ok = D[usable].notna().all(1).to_numpy()
    X_df, y = D.loc[ok, usable], y_all[ok]
    g_rec, g_subj = W.loc[ok, "rid"].to_numpy(), W.loc[ok, "subject"].to_numpy()
    if len(np.unique(y)) < 2:
        log(f"  [{task}] {exp}: only one class present, skipped")
        return None
    log(f"  [{task}] {exp} ({'+'.join(tokens)}) {win:g}s: {ok.sum()} windows, {len(usable)} features, "
        f"{len(np.unique(y))} classes")
    labels = np.unique(y)
    positive = 1 if task == "binary" else None
    sel = None
    if scope == "global":
        try:
            sel = select_features(X_df, y, cfg, seed, log)
        except ValueError as e:      # e.g. every feature is constant for this experiment
            log(f"  [{task}] {exp} {win:g}s: {e}, skipped")
            return None
        _save_sel(out_dir, task, exp, win, sel)
    rows = []
    cms = {m: np.zeros((len(labels), len(labels)), int) for m in models}
    for r, tr, te in splits(protocol, y, g_rec, g_subj, int(ev.get("rounds", 10)),
                            float(ev.get("test_size", .3)), int(ev.get("folds", 10)), seed):
        if scope == "per_split":
            sel = select_features(X_df.iloc[tr], y[tr], cfg, seed + r, log)
            if r == 0:
                _save_sel(out_dir, task, exp, win, sel)
        Xtr = X_df.iloc[tr][sel["selected"]].to_numpy(np.float32)
        Xte = X_df.iloc[te][sel["selected"]].to_numpy(np.float32)
        for m, params in models.items():
            clf = build_model(m, params, cfg.get("preprocessing", {}).get("scaler"), seed + r, n_jobs(cfg))
            t0 = time.time()
            clf.fit(Xtr, y[tr])
            pred = clf.predict(Xte)
            res = metrics(y[te], pred, labels, positive)
            cms[m] += confusion_matrix(y[te], pred, labels=labels)
            rows.append(dict(task=task, experiment=exp, window_s=win, model=m, round=r,
                             n_features=len(sel["selected"]), n_train=len(tr), n_test=len(te),
                             fit_s=round(time.time() - t0, 3), **res))
    for m, cm in cms.items():
        np.savez(out_dir / "confusion" / f"{task}__{exp}__{win:g}s__{m}.npz", cm=cm, labels=labels)
    summ = pd.DataFrame(rows).groupby("model")["f1"].mean()
    log("      F1: " + "  ".join(f"{k}={v:.3f}" for k, v in summ.items()))
    return rows


def _save_sel(out_dir, task, exp, win, sel):
    with open(out_dir / "selection" / f"{task}__{exp}__{win:g}s.json", "w") as f:
        json.dump(sel, f, indent=1)
