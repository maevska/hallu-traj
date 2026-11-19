from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class Record:
    question: str
    answer: str
    tokens: list[str]
    states: np.ndarray
    entropy: np.ndarray
    logprob: np.ndarray
    timings: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        t = len(self.tokens)
        if not (self.states.shape[0] == self.entropy.shape[0] == self.logprob.shape[0] == t):
            raise ValueError("число токенов не согласовано с размерами массивов")

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)


@runtime_checkable
class Extractor(Protocol):

    def analyze(self, question: str, answer: str | None = None) -> Record: ...

    def sample(self, question: str, k: int, temperature: float = 1.0) -> list[str]: ...
