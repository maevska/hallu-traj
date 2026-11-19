from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from .classifier import TrajectoryDetector


def auroc(y: Sequence[int], s: Sequence[float]) -> float:
    return float(roc_auc_score(y, s))


def auprc(y: Sequence[int], s: Sequence[float]) -> float:
    return float(average_precision_score(y, s))


def fpr_at_tpr(y: Sequence[int], s: Sequence[float], tpr_level: float = 0.95) -> float:
    fpr, tpr, _ = roc_curve(y, s)
    return float(fpr[np.searchsorted(tpr, tpr_level, side="left")])


def roc_points(y: Sequence[int], s: Sequence[float], max_points: int = 200) -> dict[str, list[float]]:
    fpr, tpr, _ = roc_curve(y, s)
    if len(fpr) > max_points:
        idx = np.unique(np.linspace(0, len(fpr) - 1, max_points).astype(int))
        fpr, tpr = fpr[idx], tpr[idx]
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist()}


def _cluster_resample(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    ug = np.unique(groups)
    pick = rng.choice(ug, size=ug.size, replace=True)
    members = {g: np.flatnonzero(groups == g) for g in ug}
    return np.concatenate([members[g] for g in pick])


def bootstrap_ci(
    y: np.ndarray,
    s: np.ndarray,
    metric: Callable = auroc,
    groups: np.ndarray | None = None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    y, s = np.asarray(y), np.asarray(s)
    g = np.arange(len(y)) if groups is None else np.asarray(groups)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = _cluster_resample(g, rng)
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(metric(y[idx], s[idx]))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_bootstrap_diff(
    y: np.ndarray,
    s_a: np.ndarray,
    s_b: np.ndarray,
    metric: Callable = auroc,
    groups: np.ndarray | None = None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    y, s_a, s_b = np.asarray(y), np.asarray(s_a), np.asarray(s_b)
    g = np.arange(len(y)) if groups is None else np.asarray(groups)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = _cluster_resample(g, rng)
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(metric(y[idx], s_a[idx]) - metric(y[idx], s_b[idx]))
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return float(metric(y, s_a) - metric(y, s_b)), float(lo), float(hi)


def grouped_folds(groups: Sequence, n_splits: int, seed: int = 0) -> list[np.ndarray]:
    groups = np.asarray(groups)
    ug = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(ug)
    parts = np.array_split(ug, n_splits)
    return [np.flatnonzero(np.isin(groups, p)) for p in parts]


def oof_scores(
    feats: Sequence[np.ndarray],
    y: Sequence[int],
    groups: Sequence,
    make_detector: Callable[[], TrajectoryDetector],
    n_splits: int = 5,
    seed: int = 0,
    how: str | None = None,
) -> np.ndarray:
    y = np.asarray(y)
    out = np.full(len(y), np.nan)
    for test_idx in grouped_folds(groups, n_splits, seed):
        train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
        if len(np.unique(y[train_idx])) < 2:
            raise ValueError("в обучающей части остался один класс")
        det = make_detector().fit([feats[i] for i in train_idx], y[train_idx])
        out[test_idx] = det.answer_scores([feats[i] for i in test_idx], how)
    return out
