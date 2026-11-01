"""Paper-style outputs from a run folder: Table 13 (best model per modality x window),
Table 14 (best window per modality x model), Fig. 8-14 (incremental feature curves),
Fig. 16 (confusion matrix of the best configuration) and an F1 overview chart."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

METRICS = ["accuracy", "precision", "sensitivity", "specificity", "f1"]
NICE = {"accuracy": "Accuracy (%)", "precision": "Precision (%)", "sensitivity": "Sensitivity (%)",
        "specificity": "Specificity (%)", "f1": "F1-score (%)"}


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    g = raw.groupby(["task", "experiment", "window_s", "model"], sort=False)
    s = g[METRICS].agg(["mean", "std"])
    s.columns = [f"{m}_{a}" for m, a in s.columns]
    s["n_features"] = g["n_features"].first()
    s["rounds"] = g.size()
    return s.reset_index()


def _fmt(row):
    return {NICE[m]: f"{100 * row[f'{m}_mean']:.2f} ± {100 * np.nan_to_num(row[f'{m}_std']):.2f}" for m in METRICS}


def table13(summ):
    best = summ.loc[summ.groupby(["task", "experiment", "window_s"], sort=False)["f1_mean"].idxmax()]
    return pd.DataFrame([{"Task": r.task, "Modality": r.experiment, "Window": f"{r.window_s:g} s ({r.model})",
                          **_fmt(r)} for _, r in best.iterrows()])


def table14(summ):
    best = summ.loc[summ.groupby(["task", "experiment", "model"], sort=False)["f1_mean"].idxmax()]
    return pd.DataFrame([{"Task": r.task, "Modality": r.experiment, "Model": f"{r.model} ({r.window_s:g} s)",
                          **_fmt(r)} for _, r in best.iterrows()])


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def plot_curves(run: Path):
    files = sorted((run / "selection").glob("*.json"))
    groups = {}
    for f in files:
        task, exp, win = f.stem.split("__")
        groups.setdefault((task, exp), []).append((float(win[:-1]), json.loads(f.read_text())))
    for (task, exp), items in groups.items():
        items.sort()
        fig, axes = plt.subplots(1, len(items), figsize=(4.2 * len(items), 3), squeeze=False)
        for ax, (win, sel) in zip(axes[0], items):
            if sel["curve"]:
                k, acc = zip(*sel["curve"])
                ax.plot(k, 100 * np.array(acc), "k-", lw=1.4)
                ax.axvline(len(sel["selected"]), color="k", ls="--", lw=1)
            ax.set(title=f"{win:g}-sec window", xlabel="cumulative features", ylabel="accuracy")
            ax.grid(alpha=.3)
        fig.suptitle(f"Incremental feature analysis – {exp} ({task})", fontsize=10)
        fig.tight_layout()
        fig.savefig(run / "figures" / f"selection_{task}__{exp}.png", dpi=130)
        plt.close(fig)


def plot_f1(summ: pd.DataFrame, run: Path):
    for task, s in summ.groupby("task"):
        best = s.loc[s.groupby(["experiment", "model"], sort=False)["f1_mean"].idxmax()]
        exps, models = list(dict.fromkeys(best.experiment)), list(dict.fromkeys(best.model))
        w = 0.8 / len(models)
        fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(exps) * len(models) / 2), 3.6))
        for i, m in enumerate(models):
            b = best[best.model == m].set_index("experiment").reindex(exps)
            ax.bar(np.arange(len(exps)) + i * w - 0.4 + w / 2, 100 * b.f1_mean, w, yerr=100 * b.f1_std.fillna(0),
                   label=m, capsize=2)
        ax.set_xticks(range(len(exps)), exps, rotation=20, ha="right")
        ax.set(ylabel="F1-score (%)", ylim=(0, 100), title=f"Best window per modality and model ({task})")
        ax.legend(ncol=len(models), fontsize=8)
        ax.grid(axis="y", alpha=.3)
        fig.tight_layout()
        fig.savefig(run / "figures" / f"f1_{task}.png", dpi=130)
        plt.close(fig)


def plot_confusion(run: Path, summ: pd.DataFrame, class_names: dict | None):
    out = []
    for task, s in summ.groupby("task"):
        r = s.loc[s.f1_mean.idxmax()]
        f = run / "confusion" / f"{task}__{r.experiment}__{r.window_s:g}s__{r.model}.npz"
        if not f.exists():
            f = run / "confusion" / f"{task}__{r.experiment}__{r.window_s}s__{r.model}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        cm, labels = z["cm"], z["labels"]
        names = (["no fall", "fall"] if task == "binary" else
                 [f"{l}" + (f" {class_names.get(int(l), '')}" if class_names else "") for l in labels])
        n = len(labels)
        fig, ax = plt.subplots(figsize=(1.0 + .55 * n + 2, .55 * n + 1.8))
        ax.imshow(cm.T, cmap="Greens")
        for i in range(n):
            for j in range(n):
                ax.text(i, j, int(cm[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > cm.max() * .6 else "black")
        ax.set_xticks(range(n), names, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(n), names, fontsize=7)
        ax.set(xlabel="Target class", ylabel="Output class",
               title=f"{task}: {r.experiment}, {r.model}, {r.window_s:g} s (summed over test rounds)")
        ax.title.set_fontsize(8)
        fig.tight_layout()
        p = run / "figures" / f"confusion_{task}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        out.append((task, r))
    return out


def make_report(run: Path, class_names: dict | None = None) -> Path:
    run = Path(run)
    (run / "figures").mkdir(exist_ok=True)
    raw = pd.read_csv(run / "raw_metrics.csv")
    summ = summarize(raw)
    summ.to_csv(run / "summary.csv", index=False)
    t13, t14 = table13(summ), table14(summ)
    t13.to_csv(run / "table13_best_model_per_window.csv", index=False)
    t14.to_csv(run / "table14_best_window_per_model.csv", index=False)
    plot_curves(run)
    plot_f1(summ, run)
    best = plot_confusion(run, summ, class_names)
    md = [f"# Benchmark results – `{run.name}`", "",
          "Mean ± standard deviation over the test rounds. Model/window chosen by mean F1, as in the paper.", ""]
    for task, r in best:
        md.append(f"**Best {task}:** {r.experiment} · {r.model} · {r.window_s:g} s → "
                  f"F1 {100 * r.f1_mean:.2f} %, accuracy {100 * r.accuracy_mean:.2f} %")
    md += ["", "## Table 13 – best model for each modality and window size", "", _md(t13), "",
           "## Table 14 – best window size for each modality and model", "", _md(t14), "",
           "## Figures", ""]
    md += [f"![{p.stem}](figures/{p.name})" for p in sorted((run / "figures").glob("*.png"))]
    (run / "results.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nreport -> {run / 'results.md'}")
    return run / "results.md"
