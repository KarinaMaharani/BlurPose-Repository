"""Metrics (Eq. 2-6 of the paper) and train/test protocols.

Metrics: accuracy, precision, sensitivity (recall), specificity, F1. For more than two classes
precision/sensitivity/specificity/F1 are macro-averaged over classes (one-vs-rest), which is how
the paper's 12-class numbers (e.g. 99.5 % specificity next to 70 % sensitivity) come about.
For the binary task the positive class is "fall".

Protocols (``evaluation.protocol``):
  holdout        R rounds of a random stratified 70/30 split of the WINDOWS  (paper, Sec. 5.4)
  kfold          stratified k-fold over windows
  group_holdout  R rounds of 70/30 split by RECORDING (trial): no window of a test trial in train
  loso           leave-one-subject-out
Overlapping windows of one trial are nearly identical, so window-level splits (the paper's)
are optimistic; report a group-level protocol next to it in a thesis.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (GroupShuffleSplit, LeaveOneGroupOut, StratifiedKFold,
                                     StratifiedShuffleSplit)


def metrics(y_true, y_pred, labels=None, positive=None) -> dict:
    labels = np.unique(np.r_[y_true, y_pred]) if labels is None else np.asarray(labels)
    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(float)
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    tn = cm.sum() - tp - fp - fn
    div = lambda a, b: np.divide(a, b, out=np.zeros_like(a), where=b > 0)
    prec, sens, spec = div(tp, tp + fp), div(tp, tp + fn), div(tn, tn + fp)
    f1 = div(2 * prec * sens, prec + sens)
    if positive is not None and positive in labels:
        k = int(np.flatnonzero(labels == positive)[0])
        sel = slice(k, k + 1)
    else:
        present = np.isin(labels, np.unique(y_true))       # macro over classes present in the test set
        sel = present
    return {"accuracy": float(tp.sum() / cm.sum()), "precision": float(np.mean(prec[sel])),
            "sensitivity": float(np.mean(sens[sel])), "specificity": float(np.mean(spec[sel])),
            "f1": float(np.mean(f1[sel]))}


def splits(protocol: str, y, groups_rec, groups_subj, rounds=10, test_size=0.3, folds=10, seed=0):
    """Yield (round, train_idx, test_idx)."""
    idx = np.arange(len(y))
    if protocol == "holdout":
        for r in range(rounds):
            try:
                sss = StratifiedShuffleSplit(1, test_size=test_size, random_state=seed + r)
                tr, te = next(sss.split(idx, y))
            except ValueError:                                  # a class with a single window
                rng = np.random.default_rng(seed + r)
                perm = rng.permutation(len(y))
                n_te = int(round(test_size * len(y)))
                tr, te = perm[n_te:], perm[:n_te]
            yield r, tr, te
    elif protocol == "kfold":
        n_min = int(np.bincount(np.unique(y, return_inverse=True)[1]).min())
        for r, (tr, te) in enumerate(StratifiedKFold(min(folds, max(2, n_min)), shuffle=True,
                                                     random_state=seed).split(idx, y)):
            yield r, tr, te
    elif protocol == "group_holdout":
        for r in range(rounds):
            tr, te = next(GroupShuffleSplit(1, test_size=test_size, random_state=seed + r).split(idx, y, groups_rec))
            yield r, tr, te
    elif protocol == "loso":
        if len(np.unique(groups_subj)) < 2:
            raise ValueError("loso needs at least two subjects")
        for r, (tr, te) in enumerate(LeaveOneGroupOut().split(idx, y, groups_subj)):
            yield r, tr, te
    else:
        raise ValueError(f"unknown protocol {protocol}")
