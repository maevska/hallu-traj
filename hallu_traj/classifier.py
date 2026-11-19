from __future__ import annotations

from collections.abc import Sequence

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_GROUPS, FEATURE_NAMES, pool_answer, pooled_names


class TrajectoryDetector:
    def __init__(
        self,
        model: str = "logreg",
        answer_scoring: str = "pooled",
        top_frac: float = 0.2,
        feature_groups: Sequence[str] | None = None,
        feature_names: Sequence[str] | None = None,
        C: float = 0.1,
        token_fpr: float = 0.10,
        random_state: int = 0,
    ) -> None:
        if model not in ("logreg", "gboost"):
            raise ValueError("model: 'logreg' или 'gboost'")
        if answer_scoring not in ("pooled", "token"):
            raise ValueError("answer_scoring: 'pooled' или 'token'")
        self.model = model
        self.answer_scoring = answer_scoring
        self.top_frac = top_frac
        self.C = C
        self.token_fpr = token_fpr
        self.random_state = random_state
        self.feature_names = list(feature_names) if feature_names is not None else list(FEATURE_NAMES)
        if feature_groups is None:
            self.cols = list(range(len(self.feature_names)))
        else:
            allowed = {n for g in feature_groups for n in FEATURE_GROUPS[g]}
            self.cols = [i for i, n in enumerate(self.feature_names) if n in allowed]
            if not self.cols:
                raise ValueError("выбранные группы признаков не содержат ни одного столбца")
        self.selected_names = [self.feature_names[i] for i in self.cols]
        self.token_model_ = None
        self.pooled_model_ = None
        self.token_threshold_ = 0.5


    def _make(self) -> Pipeline:
        if self.model == "logreg":
            clf = LogisticRegression(
                C=self.C, class_weight="balanced", max_iter=5000, random_state=self.random_state
            )
            return Pipeline([("scale", StandardScaler()), ("clf", clf)])
        clf = HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.05,
            max_iter=150,
            min_samples_leaf=20,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=self.random_state,
        )
        return Pipeline([("clf", clf)])

    def _pool(self, F: np.ndarray) -> np.ndarray:
        return pool_answer(F[:, self.cols], self.top_frac)


    def fit(self, feats: Sequence[np.ndarray], labels: Sequence[int]) -> "TrajectoryDetector":
        y = np.asarray(labels, dtype=int)
        if len(feats) != len(y):
            raise ValueError("число ответов и меток различается")
        if len(np.unique(y)) < 2:
            raise ValueError("для обучения нужны оба класса")
        Fs = [np.asarray(f, dtype=np.float64)[:, self.cols] for f in feats]

        Xt = np.vstack(Fs)
        yt = np.concatenate([np.full(len(f), l, dtype=int) for f, l in zip(Fs, y)])
        wt = np.concatenate([np.full(len(f), 1.0 / len(f)) for f in Fs])
        wt *= len(wt) / wt.sum()
        self.token_model_ = self._make()
        self.token_model_.fit(Xt, yt, clf__sample_weight=wt)

        Xp = np.vstack([pool_answer(f, self.top_frac) for f in Fs])
        self.pooled_model_ = self._make()
        self.pooled_model_.fit(Xp, y)

        truthful = Xt[yt == 0]
        ts = self.token_model_.predict_proba(truthful)[:, 1]
        self.token_threshold_ = float(np.quantile(ts, 1.0 - self.token_fpr)) if ts.size else 0.5
        return self

    def _check(self) -> None:
        if self.token_model_ is None:
            raise RuntimeError("детектор не обучен: вызовите fit()")

    def token_scores(self, F: np.ndarray) -> np.ndarray:
        self._check()
        return self.token_model_.predict_proba(np.asarray(F)[:, self.cols])[:, 1]

    def answer_score(self, F: np.ndarray, how: str | None = None) -> float:
        self._check()
        how = how or self.answer_scoring
        if how == "pooled":
            return float(self.pooled_model_.predict_proba(self._pool(np.asarray(F))[None, :])[0, 1])
        ts = np.sort(self.token_scores(F))
        k = max(1, int(np.ceil(self.top_frac * ts.size)))
        return float(ts[-k:].mean())

    def answer_scores(self, feats: Sequence[np.ndarray], how: str | None = None) -> np.ndarray:
        return np.array([self.answer_score(f, how) for f in feats])

    def explain(self, top: int = 10) -> list[tuple[str, float]]:
        self._check()
        if self.model != "logreg":
            raise RuntimeError("explain() доступен только для logreg")
        w = self.pooled_model_.named_steps["clf"].coef_[0]
        names = pooled_names(self.selected_names)
        order = np.argsort(-np.abs(w))[:top]
        return [(names[i], float(w[i])) for i in order]

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "TrajectoryDetector":
        obj = joblib.load(path)
        if not isinstance(obj, TrajectoryDetector):
            raise TypeError("файл не содержит TrajectoryDetector")
        return obj
