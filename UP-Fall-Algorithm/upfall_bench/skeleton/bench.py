"""Impact-fall-detection benchmark on skeletons (Koffi et al. 2024, the repo in the zip).

* ML: every frame is one sample (all joint coordinates flattened, 33 x 3 = 99 for MediaPipe),
  label = impact / no impact; SVM, MLP, RF, KNN, NB, SGD; 80/20 split.
* DL: sliding sequences of ``seq_len`` frames; LSTM, BiLSTM, 1D-CNN, ST-GCN.

Frame labels (``skeleton.label``):
  class   the dataset's own per-frame label (UP-Fall skeleton release: already impact 0/1)
  fall    1 if the frame's class is in ``fall_classes``
  impact  semi-automatic impact labels from the accelerometer (SMV peak > 2 g, +-``impact_ms``)
"""
from __future__ import annotations

import fnmatch
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..arc.evaluation import metrics
from ..arc.models import build_model
from ..arc.report import _md
from ..config import n_jobs, resolve, torch_device
from ..derive.impact import impact_labels
from .layouts import adjacency, guess_layout


def _hip_center(F, J, D):
    x = F.reshape(len(F), J, D)
    layout = guess_layout(J)
    hips = {"coco17": (11, 12), "body12": (6, 7), "mediapipe33": (23, 24)}.get(layout)
    if hips is None:
        return F
    c = x[:, list(hips)].mean(1, keepdims=True)
    return (x - c).reshape(len(F), -1)


def collect(adapter, recs, cfg):
    s = cfg.get("skeleton", {})
    pattern = s.get("stream", "*")
    mode = s.get("label", "class")
    rows = []
    for r in recs:
        for st in r.streams.values():
            if st.modality != "SKELETON" or not fnmatch.fnmatch(st.name, pattern):
                continue
            F = pd.DataFrame(st.values).ffill().bfill().to_numpy(np.float32)
            if np.isnan(F).any():
                continue
            if mode == "impact":
                _, lab, _ = impact_labels(r, adapter.fall_classes, float(s.get("threshold_g", 2.0)),
                                          float(s.get("impact_ms", 500)), float(s.get("impact_ms", 500)),
                                          t_query=st.t_ms)
            elif mode == "fall":
                lab = np.isin(r.label_at(st.t_ms), list(adapter.fall_classes)).astype(int)
            else:
                lab = r.label_at(st.t_ms).astype(int)
            rows.append((r.rid + "/" + st.name, str(r.subject), F, lab, st.channels))
    if not rows:
        raise RuntimeError("no SKELETON streams found: run `extract-pose` first or check skeleton.stream")
    return rows


def run_skeleton(adapter, recs, cfg, out_dir: Path | None = None):
    s, seed = cfg.get("skeleton", {}), int(cfg.get("evaluation", {}).get("seed", 0))
    name = cfg["dataset"].get("name", adapter.name)
    out_dir = out_dir or resolve(cfg.get("results_dir", "results")) / name / time.strftime("skeleton_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    data = collect(adapter, recs, cfg)
    ch = data[0][4]
    dims = 3 if ch[0].endswith(("_X", "_x")) and any(c.endswith(("_Z", "_z")) for c in ch) else 2
    J = len(ch) // dims
    if s.get("normalize", "none") == "hip_center":
        data = [(a, b, _hip_center(F, J, dims), l, c) for a, b, F, l, c in data]
    test_size = float(s.get("test_size", .2))
    rounds = int(s.get("rounds", 5))
    split_by = s.get("split", "frames")          # frames | recording | subject
    rows = []

    # ---------- ML, one frame = one sample ----------
    X = np.concatenate([d[2] for d in data])
    y = np.concatenate([d[3] for d in data])
    grp = np.concatenate([[d[0] if split_by == "recording" else d[1]] * len(d[3]) for d in data])
    labels = np.unique(y)
    pos = 1 if set(labels) <= {0, 1} else None
    print(f"skeleton ML: {len(X)} frames, {X.shape[1]} coords, classes {dict(zip(*np.unique(y, return_counts=True)))}")
    for r, tr, te in _splits(y, grp, split_by, rounds, test_size, seed):
        for m in s.get("ml_models", ["SVM", "MLP", "RF", "KNN", "NB", "SGD"]):
            params = (cfg.get("skeleton_model_params") or {}).get(m, {})
            clf = build_model(m, params, s.get("scaler", "none"), seed + r, n_jobs(cfg))
            clf.fit(X[tr], y[tr])
            rows.append(dict(family="ML", model=m, round=r, **metrics(y[te], clf.predict(X[te]), labels, pos)))
        print(f"  round {r}: " + "  ".join(f"{x['model']}={x['f1']:.3f}" for x in rows if x["round"] == r
                                           and x["family"] == "ML"))

    # ---------- DL, sliding sequences ----------
    dl_models = s.get("dl_models", ["LSTM", "BiLSTM", "CNN", "STGCN"])
    if dl_models:
        rows += _run_dl(data, dl_models, J, dims, labels, pos, s, split_by, rounds, test_size, seed,
                        torch_device(cfg))

    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / "raw_metrics.csv", index=False)
    g = raw.groupby(["family", "model"], sort=False)[["accuracy", "precision", "sensitivity", "specificity", "f1"]]
    summ = g.agg(["mean", "std"])
    table = pd.DataFrame([{"Family": f, "Model": m, **{k.capitalize() + " (%)": f"{100 * summ.loc[(f, m), (k, 'mean')]:.2f}"
                                                       f" ± {100 * np.nan_to_num(summ.loc[(f, m), (k, 'std')]):.2f}"
                                                       for k in ["accuracy", "precision", "sensitivity", "specificity", "f1"]}}
                          for f, m in summ.index])
    table.to_csv(out_dir / "skeleton_table.csv", index=False)
    (out_dir / "results.md").write_text(f"# Skeleton benchmark – {name}\n\nlabels: `{s.get('label', 'class')}`, "
                                        f"split: `{split_by}`, {rounds} rounds, test {test_size:.0%}\n\n" + _md(table),
                                        encoding="utf-8")
    print(table.to_string(index=False))
    print(f"-> {out_dir}")
    return out_dir


def _splits(y, grp, split_by, rounds, test_size, seed):
    from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
    idx = np.arange(len(y))
    for r in range(rounds):
        if split_by == "frames":
            sp = StratifiedShuffleSplit(1, test_size=test_size, random_state=seed + r)
            tr, te = next(sp.split(idx, y))
        else:
            tr, te = next(GroupShuffleSplit(1, test_size=test_size, random_state=seed + r).split(idx, y, grp))
        yield r, tr, te


def _run_dl(data, models, J, dims, labels, pos, s, split_by, rounds, test_size, seed, dev):
    import torch
    from .dl import build
    T, stride = int(s.get("seq_len", 30)), int(s.get("seq_stride", 5))
    how = s.get("seq_label", "last")
    Xs, ys, gs = [], [], []
    for rid, subj, F, lab, _ in data:
        for a in range(0, len(F) - T + 1, stride):
            l = lab[a:a + T]
            Xs.append(F[a:a + T])
            ys.append(int(l[-1]) if how == "last" else int(l.max()) if how == "any" else int(np.bincount(l).argmax()))
            gs.append(rid if split_by == "recording" else subj)
    X, y, g = np.stack(Xs).astype(np.float32), np.array(ys), np.array(gs)
    yi = np.searchsorted(labels, y)
    A = adjacency(J)
    print(f"skeleton DL: {len(X)} sequences of {T} frames on {dev}")
    rows = []
    for r, tr, te in _splits(y, g, split_by, rounds, test_size, seed):
        mu, sd = X[tr].mean((0, 1)), X[tr].std((0, 1)) + 1e-6
        Xt = torch.from_numpy((X - mu) / sd)
        for m in models:
            torch.manual_seed(seed + r)
            net = build(m, X.shape[2], len(labels), J, dims, A).to(dev)
            opt = torch.optim.Adam(net.parameters(), lr=float(s.get("lr", 1e-3)))
            w = torch.tensor(len(tr) / (len(labels) * np.maximum(np.bincount(yi[tr], minlength=len(labels)), 1)),
                             dtype=torch.float32, device=dev) if s.get("class_weights", True) else None
            lossf = torch.nn.CrossEntropyLoss(weight=w)
            ytr = torch.from_numpy(yi[tr])
            for _ in range(int(s.get("epochs", 30))):
                net.train()
                for b in torch.randperm(len(tr)).split(int(s.get("batch_size", 32))):
                    opt.zero_grad()
                    lossf(net(Xt[tr[b.numpy()]].to(dev)), ytr[b].to(dev)).backward()
                    opt.step()
            net.eval()
            with torch.no_grad():
                pred = torch.cat([net(Xt[te[i:i + 256]].to(dev)).argmax(1).cpu()
                                  for i in range(0, len(te), 256)]).numpy()
            rows.append(dict(family="DL", model=m, round=r, **metrics(y[te], labels[pred], labels, pos)))
        print(f"  DL round {r}: " + "  ".join(f"{x['model']}={x['f1']:.3f}" for x in rows if x["round"] == r))
    return rows
