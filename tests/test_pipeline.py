"""Сквозная проверка конвейера на синтетическом источнике состояний
Метрики здесь проверяют корректность кода, а не качество метода на реальных LLM
"""
import numpy as np
import pytest

from hallu_traj import (
    HallucinationService,
    TrajectoryDetector,
    mean_entropy,
    self_consistency_score,
    states_to_features,
    token_f1,
)
from hallu_traj.evaluate import (
    auroc,
    bootstrap_ci,
    grouped_folds,
    oof_scores,
    paired_bootstrap_diff,
    roc_points,
)
from hallu_traj.highlight import merge_spans, render_html, to_gradio_items
from hallu_traj.labeling import label_answer, rouge_l_f1
from hallu_traj.mock import MockExtractor


@pytest.fixture(scope="module")
def dataset():
    ext = MockExtractor(n_layers=25, d=32, seed=1)
    feats, labels, groups, recs = [], [], [], []
    for i in range(80):
        halluc = i % 2
        q = f"{'halluc ' if halluc else ''}question {i}"
        rec = ext.analyze(q)
        feats.append(states_to_features(rec.states))
        labels.append(halluc)
        groups.append(i)
        recs.append(rec)
    return ext, feats, np.array(labels), np.array(groups), recs


def test_mock_is_deterministic():
    a = MockExtractor(seed=5).analyze("halluc q")
    b = MockExtractor(seed=5).analyze("halluc q")
    assert np.array_equal(a.states, b.states)


@pytest.mark.parametrize("model", ["logreg", "gboost"])
def test_detector_learns_synthetic_signal(dataset, model):
    _, feats, y, groups, _ = dataset
    oof = oof_scores(feats, y, groups, lambda: TrajectoryDetector(model=model), n_splits=4, seed=0, how="pooled")
    assert not np.isnan(oof).any()
    assert auroc(y, oof) > 0.9  


def test_token_level_scoring_and_threshold(dataset):
    _, feats, y, _, _ = dataset
    det = TrajectoryDetector().fit(feats, y)
    ts = det.token_scores(feats[1])  # ответ с меткой 1
    assert ts.shape == (feats[1].shape[0],) and np.all((0 <= ts) & (ts <= 1))
    truthful_scores = np.concatenate([det.token_scores(f) for f, l in zip(feats, y) if l == 0])
    frac = (truthful_scores >= det.token_threshold_).mean()
    assert frac == pytest.approx(det.token_fpr, abs=0.04)
    assert det.answer_score(feats[1], "pooled") > det.answer_score(feats[0], "pooled")


def test_save_load_roundtrip(tmp_path, dataset):
    _, feats, y, _, _ = dataset
    det = TrajectoryDetector().fit(feats, y)
    path = str(tmp_path / "det.joblib")
    det.save(path)
    det2 = TrajectoryDetector.load(path)
    assert np.allclose(det.answer_scores(feats[:10]), det2.answer_scores(feats[:10]))


def test_feature_groups_and_explain(dataset):
    _, feats, y, _, _ = dataset
    det = TrajectoryDetector(feature_groups=["curvature"]).fit(feats, y)
    assert all(n.startswith("curv_") for n in det.selected_names)
    top = TrajectoryDetector().fit(feats, y).explain(5)
    assert len(top) == 5 and isinstance(top[0][0], str)
    with pytest.raises(ValueError):
        TrajectoryDetector(feature_groups=["entropy"])  


def test_requires_both_classes(dataset):
    _, feats, _, _, _ = dataset
    with pytest.raises(ValueError):
        TrajectoryDetector().fit(feats[:4], [0, 0, 0, 0])


def test_service_end_to_end(dataset):
    ext, feats, y, _, _ = dataset
    det = TrajectoryDetector().fit(feats, y)
    svc = HallucinationService(ext, det)
    res = svc.analyze("halluc brand new question here")
    assert len(res.tokens) == len(res.token_scores)
    assert res.series["speed"].shape[0] == len(res.tokens)
    layers, prof = res.profile("curv", token=0)
    assert layers.shape == prof.shape
    assert 0.0 <= res.answer_score <= 1.0
    items = to_gradio_items(res.tokens, res.token_scores, res.threshold)
    assert len(items) == len(res.tokens)
    assert res.timings["features+classifier"] >= 0


def test_grouped_folds_have_no_leakage():
    groups = np.repeat(np.arange(20), 3)
    folds = grouped_folds(groups, 5, seed=0)
    seen = set()
    for f in folds:
        g = set(groups[f])
        assert not (g & seen)
        seen |= g
    assert seen == set(range(20))


def test_bootstrap_helpers():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    good = y + rng.normal(scale=0.8, size=300)
    bad = rng.normal(size=300)
    lo, hi = bootstrap_ci(y, good, n_boot=200)
    assert lo <= auroc(y, good) <= hi
    d, dlo, dhi = paired_bootstrap_diff(y, good, bad, n_boot=200)
    assert d > 0 and dlo > 0
    pts = roc_points(y, good, max_points=50)
    assert len(pts["fpr"]) == len(pts["tpr"]) <= 51


def test_baselines():
    assert token_f1("The cat sat", "a cat sat") == pytest.approx(1.0)
    assert token_f1("dogs", "cats") == 0.0
    same = ["Paris is the capital"] * 4
    assert self_consistency_score("Paris is the capital", same) == 0.0
    diff = ["Rome", "Berlin", "Madrid", "Oslo"]
    assert self_consistency_score("Paris is the capital", diff) == 1.0
    assert 0.0 <= self_consistency_score("Paris is the capital", same + diff, soft=True) <= 1.0
    assert mean_entropy([1.0, 3.0]) == 2.0


def test_labeling():
    assert rouge_l_f1("nothing happens", "nothing happens") == pytest.approx(1.0)
    assert label_answer("Nothing happens", ["Nothing happens"], ["You will go blind"]) == 0
    assert label_answer("You will go blind", ["Nothing happens"], ["You will go blind"]) == 1
    assert label_answer("Bananas", ["Nothing happens"], ["You will go blind"]) is None


def test_merge_spans():
    toks = list("abcdef")
    sc = [0.1, 0.9, 0.8, 0.1, 0.7, 0.1]
    spans = merge_spans(toks, sc, 0.5)
    assert [(s.start, s.end, s.text) for s in spans] == [(1, 3, "bc"), (4, 5, "e")]
    merged = merge_spans(toks, sc, 0.5, max_gap=1)
    assert [(s.start, s.end, s.text) for s in merged] == [(1, 5, "bcde")]
    assert merge_spans(toks, [0.0] * 6, 0.5) == []
    with pytest.raises(ValueError):
        merge_spans(toks, [0.1], 0.5)


def test_render_html_escapes():
    html = render_html(["<b>", "x"], [0.9, 0.0], 0.5)
    assert "&lt;b&gt;" in html and "<b>" not in html


def test_app_module_compiles():
    import py_compile

    import hallu_traj.app as app

    py_compile.compile(app.__file__, doraise=True)
