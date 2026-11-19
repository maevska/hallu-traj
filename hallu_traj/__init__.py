from .baselines import mean_entropy, mean_nll, self_consistency_score, token_f1
from .classifier import TrajectoryDetector
from .features import (
    FEATURE_GROUPS,
    FEATURE_NAMES,
    pool_answer,
    states_to_features,
    token_features,
    trajectory_series,
)
from .highlight import merge_spans, render_html
from .records import Extractor, Record
from .service import AnalysisResult, HallucinationService

__all__ = [
    "AnalysisResult",
    "Extractor",
    "FEATURE_GROUPS",
    "FEATURE_NAMES",
    "HallucinationService",
    "Record",
    "TrajectoryDetector",
    "mean_entropy",
    "mean_nll",
    "merge_spans",
    "pool_answer",
    "render_html",
    "self_consistency_score",
    "states_to_features",
    "token_f1",
    "token_features",
    "trajectory_series",
]
__version__ = "0.1.0"
