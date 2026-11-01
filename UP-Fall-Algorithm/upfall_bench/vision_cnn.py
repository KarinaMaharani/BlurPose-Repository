"""CNN on raw frames, Sec. 5.6 / Fig. 17 of the UP-Fall paper.

input 28x28x3 (zero-centred) -> [conv 3x3 x8 -> ReLU -> maxpool 2] -> [x16] -> [x32]
-> fully connected (n classes) -> softmax.  SGD with momentum 0.9 (MATLAB ``sgdm``), learning
rate 0.001, L2 0.004, 5 epochs, mini-batch 100. One camera. 2/3 train, 1/3 test; 5-fold CV on
the training part, the fold model with the best validation accuracy is evaluated on the test set.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .arc.evaluation import metrics
from .config import resolve, torch_device
from .video import iter_frames


def build_frames(adapter, recs, cfg):
    c = cfg.get("cnn", {})
    size = tuple(c.get("input_size", [28, 28]))
    cams = c.get("cameras") or sorted({cam for r in recs for cam in r.videos})[:1]
    tag = "_".join(cams) + f"_{size[0]}x{size[1]}_{c.get('fps', 'native')}fps"
    cache = Path(adapter.cache_dir) / "cnn" / f"frames_{tag}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return {k: z[k] for k in z.files}
    by_video = {}
    for r in recs:
        for cam in cams:
            if cam in r.videos:
                by_video.setdefault((cam, str(r.videos[cam].path)), (r.videos[cam], []))[1].append(r)
    X, y, rid, subj, fall = [], [], [], [], []
    fc = adapter.fall_classes
    for (cam, _), (src, rs) in by_video.items():
        rs = sorted(rs, key=lambda r: r.span_ms[0])
        spans = np.array([r.span_ms for r in rs])
        for t, f in tqdm(iter_frames(src, size=size, fps=c.get("fps")), desc=f"frames {cam} {src.path.name}",
                         unit="f", mininterval=2):
            k = np.flatnonzero((spans[:, 0] <= t) & (t < spans[:, 1]))
            if not len(k):
                continue
            r = rs[k[0]]
            lab = int(r.label_at(np.array([t]))[0])
            X.append(f); y.append(lab); rid.append(r.rid); subj.append(str(r.subject)); fall.append(int(lab in fc))
    out = dict(X=np.stack(X), y=np.array(y), rid=np.array(rid), subject=np.array(subj), fall=np.array(fall))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


def make_net(n_classes, size):
    import torch.nn as nn
    blocks, ch = [], 3
    for out in (8, 16, 32):
        blocks += [nn.Conv2d(ch, out, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
        ch = out
    h, w = size[1] // 8, size[0] // 8
    return nn.Sequential(*blocks, nn.Flatten(), nn.Linear(32 * h * w, n_classes))


def _train(net, X, y, c, dev, seed):
    import torch
    torch.manual_seed(seed)
    opt = torch.optim.SGD(net.parameters(), lr=float(c.get("lr", 1e-3)), momentum=float(c.get("momentum", .9)),
                          weight_decay=float(c.get("l2", 4e-3)))
    lossf = torch.nn.CrossEntropyLoss()
    bs = int(c.get("batch_size", 100))
    g = torch.Generator().manual_seed(seed)
    for _ in range(int(c.get("epochs", 5))):
        net.train()
        for idx in torch.randperm(len(X), generator=g).split(bs):
            opt.zero_grad()
            lossf(net(X[idx].to(dev)), y[idx].to(dev)).backward()
            opt.step()
    return net


def _predict(net, X, dev):
    import torch
    net.eval()
    with torch.no_grad():
        return torch.cat([net(X[i:i + 256].to(dev)).argmax(1).cpu() for i in range(0, len(X), 256)]).numpy()


def run_cnn(adapter, recs, cfg, out_dir: Path | None = None):
    import torch
    from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, train_test_split
    c, seed = cfg.get("cnn", {}), int(cfg.get("evaluation", {}).get("seed", 0))
    name = cfg["dataset"].get("name", adapter.name)
    out_dir = out_dir or resolve(cfg.get("results_dir", "results")) / name / time.strftime("cnn_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    d = build_frames(adapter, recs, cfg)
    task = c.get("task", "multiclass")
    y_raw = d["fall"] if task == "binary" else d["y"]
    classes = np.unique(y_raw)
    y = np.searchsorted(classes, y_raw)
    X = d["X"].astype(np.float32) / 255
    print(f"CNN: {len(X)} frames, {len(classes)} classes, task={task}")
    idx = np.arange(len(y))
    if c.get("split", "holdout") == "group_holdout":
        tr, te = next(GroupShuffleSplit(1, test_size=1 / 3, random_state=seed).split(idx, y, d["rid"]))
    else:
        strat = y if np.bincount(y).min() >= 2 else None
        tr, te = train_test_split(idx, test_size=1 / 3, random_state=seed, stratify=strat)
    mean = X[tr].mean(0, keepdims=True)                         # zero-centre normalisation
    T = lambda a: torch.from_numpy(np.ascontiguousarray((X[a] - mean).transpose(0, 3, 1, 2)))
    dev = torch_device(cfg)
    size = tuple(c.get("input_size", [28, 28]))
    best, best_acc = None, -1
    n_min = int(np.bincount(y[tr]).min())
    kf = StratifiedKFold(min(int(c.get("folds", 5)), max(2, n_min)), shuffle=True, random_state=seed)
    for k, (a, b) in enumerate(kf.split(tr, y[tr])):
        net = _train(make_net(len(classes), size).to(dev), T(tr[a]), torch.from_numpy(y[tr[a]]), c, dev, seed + k)
        acc = float((_predict(net, T(tr[b]), dev) == y[tr[b]]).mean())
        print(f"  fold {k + 1}: val acc {acc:.4f}")
        if acc > best_acc:
            best, best_acc = net, acc
    pred = _predict(best, T(te), dev)
    res = metrics(y[te], pred, np.arange(len(classes)), 1 if task == "binary" else None)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y[te], pred, labels=np.arange(len(classes)))
    np.savez(out_dir / "cnn_confusion.npz", cm=cm, labels=classes)
    json.dump({"task": task, "n_frames": int(len(X)), "n_test": int(len(te)), "best_val_acc": best_acc,
               "classes": classes.tolist(), **res}, open(out_dir / "cnn_results.json", "w"), indent=1)
    print("CNN test: " + "  ".join(f"{k}={v:.4f}" for k, v in res.items()))
    print(f"-> {out_dir}")
    return out_dir
