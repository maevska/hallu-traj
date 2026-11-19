"""Проверки признаков на траекториях с известными аналитическими свойствами"""
import numpy as np
import pytest

from hallu_traj.features import (
    FEATURE_NAMES,
    N_FEATURES,
    curvature_series,
    fit_layer_mean,
    lyapunov_rosenstein,
    pool_answer,
    pooled_names,
    series_layers,
    speed_series,
    states_to_features,
    stretch_series,
    token_features,
    tortuosity,
    trajectory_series,
)


def circle(n=30, phi=0.3, r=5.0, d=16):
    t = phi * np.arange(n)
    X = np.zeros((n, d))
    X[:, 0], X[:, 1] = r * np.cos(t), r * np.sin(t)
    return X


def test_straight_line_has_zero_curvature_and_unit_tortuosity():
    rng = np.random.default_rng(0)
    u = rng.normal(size=32)
    u /= np.linalg.norm(u)
    X = 10 * rng.normal(size=32) + np.outer(np.arange(20), u) * 0.7
    assert np.allclose(curvature_series(X), 0.0, atol=1e-6)
    assert tortuosity(X) == pytest.approx(1.0, abs=1e-9)
    assert np.allclose(stretch_series(X), 0.0, atol=1e-9)


def test_circle_turning_angle_equals_central_angle():
    phi = 0.3
    X = circle(phi=phi)
    assert np.allclose(curvature_series(X), phi, atol=1e-9)
    assert np.allclose(speed_series(X), 2 * np.sin(phi / 2), atol=1e-9)


def test_scale_invariance_of_relative_features():
    X = circle() + np.random.default_rng(1).normal(scale=0.05, size=(30, 16))
    f1 = token_features(X, skip_first=0)
    f2 = token_features(7.3 * X, skip_first=0)
    names = FEATURE_NAMES
    for n in names:
        if n.startswith("speed_") or n.startswith("curv_") or n.startswith("stretch_") or n in ("tortuosity", "lyap_R"):
            i = names.index(n)
            assert f1[i] == pytest.approx(f2[i], rel=1e-6, abs=1e-9), n


def test_stretch_recovers_geometric_growth_rate():
    a = 1.15
    u = np.ones(8) / np.sqrt(8)
    steps = a ** np.arange(15)
    X = np.cumsum(np.outer(steps, u), axis=0)
    assert np.allclose(stretch_series(X), np.log(a), atol=1e-6)


def test_rosenstein_zero_for_rigid_rotation():
    X = circle(n=500, phi=2 * np.pi / 97, r=1.0, d=2)
    assert lyapunov_rosenstein(X, theiler=5) == pytest.approx(0.0, abs=1e-6)


def henon(n, seed=0, a=1.4, b=0.3, burn=1000):
    rng = np.random.default_rng(seed)
    x, y = 0.1 + 0.05 * rng.random(), 0.1 + 0.05 * rng.random()
    out = []
    for i in range(n + burn):
        x, y = 1 - a * x * x + y, b * x
        if i >= burn:
            out.append((x, y))
    return np.array(out)


def test_rosenstein_close_to_known_exponent_on_henon_map():
    lam = lyapunov_rosenstein(henon(1500, seed=3))
    assert 0.30 < lam < 0.50


def test_token_features_shape_and_finite():
    rng = np.random.default_rng(2)
    X = np.cumsum(rng.normal(size=(25, 64)), axis=0)
    f = token_features(X)
    assert f.shape == (N_FEATURES,) and np.all(np.isfinite(f))
    assert len(FEATURE_NAMES) == N_FEATURES == len(set(FEATURE_NAMES))


def test_too_short_trajectory_raises():
    with pytest.raises(ValueError):
        token_features(np.zeros((5, 4)))


def test_states_to_features_and_series_shapes():
    rng = np.random.default_rng(3)
    S = np.cumsum(rng.normal(size=(7, 20, 32)), axis=1)
    F = states_to_features(S)
    assert F.shape == (7, N_FEATURES)
    ser = trajectory_series(S)
    n = 20 - 1
    assert ser["speed"].shape == (7, n - 1)
    assert ser["curv"].shape == (7, n - 2) == ser["stretch"].shape
    lay = series_layers(20)
    assert len(lay["speed"]) == n - 1 and len(lay["curv"]) == n - 2
    assert lay["speed"][0] == 2 


def test_pool_answer_layout():
    F = np.arange(12, dtype=float).reshape(4, 3)
    p = pool_answer(F, top_frac=0.5)
    assert p.shape == (9,)
    assert np.allclose(p[:3], F.mean(0)) and np.allclose(p[3:6], F.max(0))
    assert np.allclose(p[6:], np.sort(F, 0)[-2:].mean(0))
    assert len(pooled_names(["a", "b", "c"])) == 9


def test_fit_layer_mean():
    S1 = np.ones((2, 3, 4))
    S2 = 3 * np.ones((2, 3, 4))
    assert np.allclose(fit_layer_mean([S1, S2]), 2.0)
