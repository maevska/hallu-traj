from __future__ import annotations

import re
from collections.abc import Callable, Sequence

import numpy as np

_ARTICLES = {"a", "an", "the"}


def normalize_tokens(s: str) -> list[str]:
    return [w for w in re.findall(r"\w+", s.lower()) if w not in _ARTICLES]


def token_f1(a: str, b: str) -> float:
    ta, tb = normalize_tokens(a), normalize_tokens(b)
    if not ta or not tb:
        return float(ta == tb)
    common: dict[str, int] = {}
    for w in ta:
        common[w] = common.get(w, 0) + 1
    overlap = 0
    for w in tb:
        if common.get(w, 0) > 0:
            overlap += 1
            common[w] -= 1
    if overlap == 0:
        return 0.0
    p, r = overlap / len(ta), overlap / len(tb)
    return 2 * p * r / (p + r)


def mean_entropy(entropy: Sequence[float]) -> float:
    return float(np.mean(entropy))


def max_entropy(entropy: Sequence[float]) -> float:
    return float(np.max(entropy))


def mean_nll(logprob: Sequence[float]) -> float:
    return float(-np.mean(logprob))


def self_consistency_score(
    answer: str,
    samples: Sequence[str],
    sim: Callable[[str, str], float] = token_f1,
    threshold: float = 0.5,
    soft: bool = False,
) -> float:
    if not samples:
        raise ValueError("нужна хотя бы одна выборка")
    sims = np.array([sim(answer, s) for s in samples], dtype=float)
    agree = sims.mean() if soft else (sims >= threshold).mean()
    return float(1.0 - agree)
