from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .classifier import TrajectoryDetector
from .features import series_layers, states_to_features, trajectory_series
from .highlight import Span, merge_spans
from .records import Extractor


@dataclass
class AnalysisResult:
    question: str
    answer: str
    tokens: list[str]
    token_scores: np.ndarray
    answer_score: float
    threshold: float
    spans: list[Span]
    series: dict[str, np.ndarray]  # speed/curv/stretch: (T, число слоёв)
    layers: dict[str, np.ndarray]  # номера слоёв для осей графиков
    entropy: np.ndarray
    timings: dict = field(default_factory=dict)

    def profile(self, name: str, token: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        arr = self.series[name]
        y = arr.mean(axis=0) if token is None else arr[token]
        return self.layers[name], y


class HallucinationService:
    def __init__(
        self,
        extractor: Extractor,
        detector: TrajectoryDetector,
        skip_first: int = 1,
        center: np.ndarray | None = None,
        max_gap: int = 0,
    ) -> None:
        self.extractor = extractor
        self.detector = detector
        self.skip_first = skip_first
        self.center = center
        self.max_gap = max_gap

    def analyze(self, question: str, answer: str | None = None) -> AnalysisResult:
        import time

        rec = self.extractor.analyze(question, answer)
        t0 = time.perf_counter()
        F = states_to_features(rec.states, self.skip_first, self.center)
        scores = self.detector.token_scores(F)
        a_score = self.detector.answer_score(F)
        series = trajectory_series(rec.states, self.skip_first, self.center)
        layers = series_layers(rec.states.shape[1], self.skip_first)
        thr = self.detector.token_threshold_
        spans = merge_spans(rec.tokens, scores, thr, self.max_gap)
        timings = dict(rec.timings)
        timings["features+classifier"] = time.perf_counter() - t0
        return AnalysisResult(
            question=question,
            answer=rec.answer,
            tokens=rec.tokens,
            token_scores=scores,
            answer_score=a_score,
            threshold=thr,
            spans=spans,
            series=series,
            layers=layers,
            entropy=rec.entropy,
            timings=timings,
        )
