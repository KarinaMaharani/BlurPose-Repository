"""Classifiers. Defaults = Table 12 of the UP-Fall paper (which match scikit-learn's parameter
names one to one, so the authors most likely used scikit-learn), plus the two extra ML models
of the 3D-skeleton benchmark (NB, SGD). Any parameter can be overridden in YAML:

    models:
      RF:  {n_estimators: 100}
      SVM: {}
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

PAPER_DEFAULTS = {
    "RF": (RandomForestClassifier, dict(n_estimators=10, min_samples_split=2, min_samples_leaf=1, bootstrap=True)),
    "SVM": (SVC, dict(C=1.0, kernel="rbf", gamma="auto", shrinking=True, tol=1e-3)),   # gamma auto = 1/features
    "MLP": (MLPClassifier, dict(hidden_layer_sizes=(100,), activation="relu", solver="adam", alpha=1e-4,
                                batch_size="auto", learning_rate_init=1e-3, shuffle=True, tol=1e-4,
                                beta_1=0.9, beta_2=0.999, epsilon=1e-8, max_iter=10)),
    "KNN": (KNeighborsClassifier, dict(n_neighbors=5, leaf_size=30, metric="euclidean")),
    "NB": (GaussianNB, dict()),
    "SGD": (SGDClassifier, dict()),
}
SEEDED = {"RF", "MLP", "SGD"}


def build_model(name: str, params: dict | None = None, scaler: str | None = None, seed: int = 0,
                n_jobs: int = 2):
    cls, base = PAPER_DEFAULTS[name]
    p = {**base, **(params or {})}
    if "hidden_layer_sizes" in p:
        p["hidden_layer_sizes"] = tuple(p["hidden_layer_sizes"]) if not isinstance(p["hidden_layer_sizes"], int) \
            else (p["hidden_layer_sizes"],)
    if name in SEEDED:
        p.setdefault("random_state", seed)
    if name == "RF":
        p.setdefault("n_jobs", n_jobs)
    model = cls(**p)
    if scaler in (None, "none", False):
        return model
    return make_pipeline({"standard": StandardScaler(), "minmax": MinMaxScaler()}[scaler], model)
