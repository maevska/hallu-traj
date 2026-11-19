from __future__ import annotations

import html
from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np


@dataclass
class Span:
    start: int  
    end: int  
    text: str
    mean_score: float
    max_score: float


def merge_spans(
    tokens: Sequence[str], scores: Sequence[float], threshold: float, max_gap: int = 0
) -> list[Span]:
    if len(tokens) != len(scores):
        raise ValueError("число токенов и оценок различается")
    flags = [s >= threshold for s in scores]
    spans: list[Span] = []
    i, n = 0, len(tokens)
    while i < n:
        if not flags[i]:
            i += 1
            continue
        end = i + 1  
        while True:
            nxt = next((h for h in range(end, min(n, end + max_gap + 1)) if flags[h]), None)
            if nxt is None:
                break
            end = nxt + 1
        seg = np.asarray(scores[i:end], dtype=float)
        spans.append(Span(i, end, "".join(tokens[i:end]), float(seg.mean()), float(seg.max())))
        i = end
    return spans


def to_gradio_items(
    tokens: Sequence[str], scores: Sequence[float], threshold: float
) -> list[tuple[str, float | None]]:
    return [(t, float(s) if s >= threshold else None) for t, s in zip(tokens, scores)]


def render_html(tokens: Sequence[str], scores: Sequence[float], threshold: float) -> str:
    out = []
    for t, s in zip(tokens, scores):
        text = html.escape(t).replace("\n", "<br>")
        if s >= threshold:
            alpha = 0.25 + 0.65 * float(np.clip((s - threshold) / (1.0 - threshold + 1e-9), 0, 1))
            out.append(
                f'<span title="score={s:.2f}" style="background:rgba(220,38,38,{alpha:.2f})">{text}</span>'
            )
        else:
            out.append(text)
    return "".join(out)
