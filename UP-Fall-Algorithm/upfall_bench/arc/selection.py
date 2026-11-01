"""Feature selection of the UP-Fall paper (Sec. 5.3), re-implemented in Python.

The paper used Weka: CfsSubsetEval (best-first / greedy stepwise) and three rankers
(CorrelationAttributeEval, ReliefFAttributeEval, ClassifierAttributeEval with ZeroR and
DecisionTable). Its 8 steps, as done here:

 1. drop features with missing values / no variance
 2. score the features with every method in ``methods``
 3. take the best ``top_k`` (100) of each method
 4-6. count in how many of those subsets each feature appears, keep the ones that appear
      at least ``min_votes`` (2) times, sort by that count (ties: better mean rank first)
 7. incremental analysis: Random-Forest accuracy using the first 1, 2, ... features
 8. keep the smallest prefix whose accuracy is within ``tolerance`` of the best one

Python stand-ins for the Weka evaluators:
  cfs           greedy forward CFS, merit = k*r_cf / sqrt(k + k(k-1)*r_ff)   (= CfsSubsetEval greedy)
  correlation   class-prior-weighted |Pearson r| of feature vs one-vs-rest class   (= CorrelationAttributeEval)
  relieff       ReliefF, k=10 nearest hits/misses, range-normalised Manhattan    (= ReliefFAttributeEval)
  oneattr_tree  3-fold CV accuracy of a depth-3 tree on that feature alone      (~ ClassifierAttributeEval/DecisionTable)
  mutual_info   sklearn mutual information (replaces ClassifierAttributeEval/ZeroR, which cannot rank)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier


def _subsample(X, y, n, seed):
    if len(y) <= n:
        return X, y
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(y):
        ci = np.flatnonzero(y == c)
        take = max(1, int(round(n * len(ci) / len(y))))
        idx.append(rng.choice(ci, min(take, len(ci)), replace=False))
    idx = np.concatenate(idx)
    return X[idx], y[idx]


def _zscore(X):
    sd = X.std(0)
    return (X - X.mean(0)) / np.where(sd > 0, sd, 1)


def class_correlation(X, y):
    Z = _zscore(X)
    score = np.zeros(X.shape[1])
    for c in np.unique(y):
        t = (y == c).astype(float)
        if t.std() == 0:
            continue
        t = (t - t.mean()) / t.std()
        score += (y == c).mean() * np.abs((Z * t[:, None]).mean(0))
    return score


def relieff(X, y, k=10, m=600, seed=0):
    rng = np.random.default_rng(seed)
    rngx = X.max(0) - X.min(0)
    Xn = (X - X.min(0)) / np.where(rngx > 0, rngx, 1)
    classes, counts = np.unique(y, return_counts=True)
    prior = dict(zip(classes, counts / len(y)))
    w = np.zeros(X.shape[1])
    sample = rng.choice(len(y), min(m, len(y)), replace=False)
    for i in sample:
        d = np.abs(Xn - Xn[i]).sum(1)
        d[i] = np.inf
        for c in classes:
            ci = np.flatnonzero((y == c) & np.isfinite(d))
            if len(ci) == 0:
                continue
            nn = ci[np.argsort(d[ci])[:k]]
            diff = np.abs(Xn[nn] - Xn[i]).mean(0)
            if c == y[i]:
                w -= diff
            else:
                w += prior[c] / (1 - prior[y[i]]) * diff
    return w / len(sample)


def cfs_greedy(X, y, max_features=100):
    rcf = class_correlation(X, y)
    R = np.abs(np.nan_to_num(np.corrcoef(X, rowvar=False)))
    chosen, best = [], -np.inf
    remaining = list(np.argsort(-rcf))
    while remaining and len(chosen) < max_features:
        k = len(chosen) + 1
        sum_cf = rcf[chosen].sum() if chosen else 0.0
        sum_ff = R[np.ix_(chosen, chosen)].sum() - k + 1 if chosen else 0.0
        cand = np.array(remaining)
        add_ff = 2 * R[np.ix_(cand, chosen)].sum(1) if chosen else np.zeros(len(cand))
        merit = (sum_cf + rcf[cand]) / np.sqrt(k + sum_ff + add_ff)
        j = int(np.argmax(merit))
        if merit[j] <= best:
            break
        best = merit[j]
        chosen.append(int(cand[j]))
        remaining.remove(int(cand[j]))
    score = np.zeros(X.shape[1])
    score[chosen] = np.linspace(1, 0.5, len(chosen)) if chosen else 0
    return score


def oneattr_tree(X, y, seed=0):
    cv = StratifiedKFold(3, shuffle=True, random_state=seed)
    out = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        out[j] = cross_val_score(DecisionTreeClassifier(max_depth=3, random_state=seed), X[:, [j]], y, cv=cv).mean()
    return out


METHODS = {
    "cfs": lambda X, y, s: cfs_greedy(X, y),
    "correlation": lambda X, y, s: class_correlation(X, y),
    "relieff": lambda X, y, s: relieff(X, y, seed=s),
    "oneattr_tree": lambda X, y, s: oneattr_tree(X, y, s),
    "mutual_info": lambda X, y, s: mutual_info_classif(X, y, random_state=s),
}


def select_features(df: pd.DataFrame, y: np.ndarray, cfg: dict, seed: int = 0, log=print) -> dict:
    """Return {"selected": [...], "ranking": [...], "curve": [(k, acc)], "votes": {...}}."""
    sc = cfg.get("selection", {})
    if not sc.get("enabled", True):
        return {"selected": list(df.columns), "ranking": list(df.columns), "curve": [], "votes": {}}
    X_all = df.to_numpy(np.float64)
    keep = np.isfinite(X_all).all(0) & (np.nanstd(X_all, 0) > 0)              # step 1
    cols = np.array(df.columns)[keep]
    if len(cols) == 0:
        raise ValueError("no usable features")
    X, yy = _subsample(X_all[:, keep], np.asarray(y), int(sc.get("max_samples", 3000)), seed)

    top_k = int(sc.get("top_k", 100))
    methods = sc.get("methods", list(METHODS))
    ranks, subsets = {}, []
    for m in methods:                                                          # steps 2-3
        s = np.nan_to_num(METHODS[m](X, yy, seed), nan=-np.inf)
        order = np.argsort(-s, kind="stable")
        if m == "cfs":
            order = order[s[order] > 0]
        subsets.append(set(order[:top_k]))
        ranks[m] = np.full(len(cols), len(cols))
        ranks[m][order] = np.arange(len(order))
    votes = np.zeros(len(cols), int)
    for sub in subsets:
        for j in sub:
            votes[j] += 1
    min_votes = int(sc.get("min_votes", 2))
    cand = np.flatnonzero(votes >= min(min_votes, len(subsets)))              # step 5
    if len(cand) == 0:
        cand = np.flatnonzero(votes >= 1)
    mean_rank = np.mean(list(ranks.values()), 0)
    cand = sorted(cand, key=lambda j: (-votes[j], mean_rank[j]))              # step 6

    if sc.get("fixed_k"):
        k_best = min(int(sc["fixed_k"]), len(cand))
        curve = []
    else:                                                                      # step 7
        max_k = min(int(sc.get("max_features", 80)), len(cand))
        n_min = int(np.bincount(np.unique(yy, return_inverse=True)[1]).min())
        folds = int(sc.get("cv_folds", 3))
        cv = (StratifiedKFold(min(folds, n_min), shuffle=True, random_state=seed) if n_min >= 2
              else KFold(folds, shuffle=True, random_state=seed))
        rf = RandomForestClassifier(n_estimators=10, random_state=seed,
                                    n_jobs=int(cfg.get("resources", {}).get("n_jobs", 2)))
        ks = list(range(1, max_k + 1, int(sc.get("curve_step", 1))))
        scoring = sc.get("scoring", "accuracy")                                # paper: accuracy
        curve = [(k, float(cross_val_score(rf, X[:, cand[:k]], yy, cv=cv, scoring=scoring).mean())) for k in ks]
        best = max(a for _, a in curve)
        tol = float(sc.get("tolerance", 0.005))
        k_best = next(k for k, a in curve if a >= best - tol)                  # step 8
    sel = [str(cols[j]) for j in cand[:k_best]]
    log(f"      selection: {len(cols)} usable -> {len(cand)} voted -> {len(sel)} kept"
        + (f" (RF cv {sc.get('scoring', 'accuracy')} {dict(curve)[k_best]:.3f})" if curve else ""))
    return {"selected": sel, "ranking": [str(cols[j]) for j in cand], "curve": curve,
            "votes": {str(cols[j]): int(votes[j]) for j in cand}}
