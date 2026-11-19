from __future__ import annotations

import zlib
from collections.abc import Callable

import numpy as np

from .records import Record


def default_regime(question: str, answer: str) -> bool:
    return "halluc" in (question + " " + answer).lower()


class MockExtractor:
    def __init__(
        self,
        n_layers: int = 25,
        d: int = 48,
        seed: int = 0,
        regime_fn: Callable[[str, str], bool] = default_regime,
        hot_fraction: float = 0.4,
    ) -> None:
        self.n_layers, self.d, self.seed = n_layers, d, seed
        self.regime_fn = regime_fn
        self.hot_fraction = hot_fraction

    def _rng(self, *parts: str) -> np.random.Generator:
        h = zlib.crc32("||".join(parts).encode("utf-8")) ^ (self.seed * 2654435761 % 2**32)
        return np.random.default_rng(h)

    def _trajectory(self, rng: np.random.Generator, hot: bool) -> np.ndarray:
        rot, jitter = (0.55, 0.60) if hot else (0.06, 0.10)
        x = rng.normal(size=self.d) * 3.0
        u = rng.normal(size=self.d)
        u /= np.linalg.norm(u)
        out = [x.copy()]
        for _ in range(self.n_layers - 1):
            u = u + rot * rng.normal(size=self.d) / np.sqrt(self.d) * np.sqrt(self.d) / 3.0
            u /= np.linalg.norm(u)
            step = 0.8 * np.exp(jitter * rng.normal())
            x = x + step * u
            out.append(x.copy())
        return np.stack(out)  # (n_layers, d)

    def analyze(self, question: str, answer: str | None = None) -> Record:
        if answer is None:
            answer = "Mock answer to: " + question
        tokens = [w + " " for w in answer.split()] or [answer]
        rng = self._rng(question, answer)
        hot_answer = self.regime_fn(question, answer)
        T = len(tokens)
        hot_idx = set(rng.choice(T, size=max(1, int(round(self.hot_fraction * T))), replace=False)) if hot_answer else set()
        states = np.stack([self._trajectory(rng, t in hot_idx) for t in range(T)]).astype(np.float32)
        entropy = rng.gamma(2.0, 0.5 + (0.15 if hot_answer else 0.0), size=T)
        logprob = -rng.gamma(2.0, 0.4, size=T)
        return Record(question, answer, tokens, states, entropy, logprob)

    def sample(self, question: str, k: int, temperature: float = 1.0) -> list[str]:
        rng = self._rng(question, "sample", str(k))
        base = "Mock answer to: " + question
        if self.regime_fn(question, base):
            return [f"alt{int(rng.integers(10**6))} {question[::-1][:20]}" for _ in range(k)]
        return [base for _ in range(k)]
