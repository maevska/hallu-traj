from __future__ import annotations

from collections.abc import Callable, Sequence

from .baselines import normalize_tokens


def rouge_l_f1(a: str, b: str) -> float:
    ta, tb = normalize_tokens(a), normalize_tokens(b)
    if not ta or not tb:
        return float(ta == tb)
    prev = [0] * (len(tb) + 1)
    for x in ta:
        cur = [0]
        for j, y in enumerate(tb, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(ta), lcs / len(tb)
    return 2 * p * r / (p + r)


def label_answer(
    answer: str,
    correct: Sequence[str],
    incorrect: Sequence[str],
    sim: Callable[[str, str], float] = rouge_l_f1,
    margin: float = 0.0,
    min_sim: float = 0.15,
) -> int | None:
    s_c = max((sim(answer, c) for c in correct), default=0.0)
    s_i = max((sim(answer, i) for i in incorrect), default=0.0)
    if max(s_c, s_i) < min_sim:
        return None
    if s_i > s_c + margin:
        return 1
    if s_c > s_i + margin:
        return 0
    return None
