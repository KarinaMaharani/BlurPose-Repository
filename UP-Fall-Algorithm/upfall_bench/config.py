"""YAML configs: a dataset file + a benchmark file, merged, with ``--set a.b=c`` overrides.

Relative paths inside configs are resolved against the project root (the folder that
contains ``upfall_bench/``), so commands work from any working directory.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = PROJECT_ROOT / p
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """``["arc.windows=[1,2]", "dataset.root=D:/x"]`` -> nested assignment (values parsed as YAML)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def load_config(dataset: str | Path, benchmark: str | Path | None = None,
                overrides: list[str] | None = None) -> dict:
    """defaults <- benchmark <- dataset (after its ``extends`` base) <- --set overrides.

    A dataset file may start with ``extends: other.yaml`` to reuse another dataset file
    (same protocol, labels and experiments) and change only what differs, e.g. the root.
    """
    cfg = load_yaml(PROJECT_ROOT / "configs" / "benchmarks" / "defaults.yaml")
    if benchmark:
        cfg = deep_merge(cfg, load_yaml(benchmark))
    ds = load_yaml(dataset)
    base = ds.pop("extends", None)
    if base:
        base_path = Path(dataset).parent / base
        if not base_path.exists():
            base_path = PROJECT_ROOT / "configs" / "datasets" / base
        cfg = deep_merge(cfg, load_yaml(base_path))
    cfg = deep_merge(cfg, ds)                      # dataset file may override benchmark defaults
    return apply_overrides(cfg, overrides)


def resolve(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    p = Path(path).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def n_jobs(cfg: dict) -> int:
    return int(get(cfg, "resources.n_jobs", 2))


def apply_resources(cfg: dict):
    """Cap CPU threads of numpy/BLAS/OpenMP (and torch, if it gets imported) so a run never
    takes the whole machine. Set ``resources.n_jobs`` higher on a dedicated server."""
    n = n_jobs(cfg)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(n)
    except ImportError:
        pass


def torch_device(cfg: dict) -> str:
    """Pick the torch device under the ``resources`` limits (called by every torch user)."""
    import torch
    torch.set_num_threads(int(get(cfg, "resources.torch_threads", n_jobs(cfg))))
    want = get(cfg, "resources.device", "auto")
    if want == "cpu" or not torch.cuda.is_available():
        return "cpu"
    frac = get(cfg, "resources.gpu_memory_fraction")
    if frac:
        torch.cuda.set_per_process_memory_fraction(float(frac))
    return "cuda"


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    node = cfg
    for p in dotted.split("."):
        if not isinstance(node, dict) or p not in node:
            return default
        node = node[p]
    return node
