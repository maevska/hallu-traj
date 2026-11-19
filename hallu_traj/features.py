from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

EPS = 1e-12
MIN_LAYERS = 8  

SERIES = ("speed", "curv", "stretch")
STATS = ("mean", "std", "max", "min", "argmax", "slope", "early", "mid", "late")

FEATURE_NAMES: list[str] = (
    [f"{s}_{t}" for s in SERIES for t in STATS] + ["tortuosity", "lyap_R"]
)
N_FEATURES = len(FEATURE_NAMES)

FEATURE_GROUPS: dict[str, list[str]] = {
    "speed": [n for n in FEATURE_NAMES if n.startswith("speed_")],
    "curvature": [n for n in FEATURE_NAMES if n.startswith("curv_")],
    "lyapunov": ["lyap_R"] + [n for n in FEATURE_NAMES if n.startswith("stretch_")],
    "global": ["tortuosity"],
    "entropy": ["token_entropy"], 
}


def _prep(X, skip_first: int = 1, center: np.ndarray | None = None) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"ожидается массив (слои, d), получено {X.shape}")
    if center is not None:
        X = X - np.asarray(center, dtype=np.float64)
    X = X[skip_first:]
    if X.shape[0] < MIN_LAYERS:
        raise ValueError(
            f"слишком короткая траектория: {X.shape[0]} < {MIN_LAYERS} состояний"
        )
    return X


def _norm(a: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a, axis=-1)


def speed_series(X: np.ndarray, relative: bool = True) -> np.ndarray:
    V = np.diff(X, axis=0)
    s = _norm(V)
    if relative:
        s = s / (_norm(X[:-1]) + EPS)
    return s


def curvature_series(X: np.ndarray) -> np.ndarray:
    V = np.diff(X, axis=0)
    a, b = V[:-1], V[1:]
    an = a / (_norm(a)[:, None] + EPS)
    bn = b / (_norm(b)[:, None] + EPS)
    # устойчивая формула угла (Кахан): не теряет точность у 0 и π, в отличие от arccos
    return 2.0 * np.arctan2(_norm(an - bn), _norm(an + bn))


def stretch_series(X: np.ndarray) -> np.ndarray:
    s = speed_series(X, relative=False)
    return np.log((s[1:] + EPS) / (s[:-1] + EPS))


def tortuosity(X: np.ndarray) -> float:
    path = speed_series(X, relative=False).sum()
    chord = float(_norm(X[-1] - X[0]))
    return float(path / (chord + EPS))


def lyapunov_rosenstein(
    X: np.ndarray, horizon: int = 4, theiler: int = 2, min_pairs: int = 3
) -> float:
    n = X.shape[0]
    m = n - horizon
    if m <= theiler + 1:
        return 0.0
    D = cdist(X, X)
    idx = np.arange(m)
    near = np.abs(idx[:, None] - idx[None, :]) <= theiler
    Dm = np.where(near, np.inf, D[:m, :m])
    j = Dm.argmin(axis=1)
    dmin = Dm[idx, j]
    ok = np.isfinite(dmin) & (dmin > EPS)
    ks = np.arange(horizon + 1)
    d = np.stack([D[idx + k, j + k] for k in ks], axis=1)  # (m, horizon+1)
    ok &= (d > EPS).all(axis=1)
    if ok.sum() < min_pairs:
        return 0.0
    y = np.log(d[ok]).mean(axis=0)
    return float(np.polyfit(ks, y, 1)[0])


def _summarize(x: np.ndarray, name: str) -> dict[str, float]:
    n = x.size
    t = np.linspace(0.0, 1.0, n)
    tc = t - t.mean()
    slope = float((tc * (x - x.mean())).sum() / (tc @ tc))
    th = np.array_split(x, 3)
    return {
        f"{name}_mean": float(x.mean()),
        f"{name}_std": float(x.std()),
        f"{name}_max": float(x.max()),
        f"{name}_min": float(x.min()),
        f"{name}_argmax": float(np.argmax(x)) / max(n - 1, 1),
        f"{name}_slope": slope,
        f"{name}_early": float(th[0].mean()),
        f"{name}_mid": float(th[1].mean()),
        f"{name}_late": float(th[2].mean()),
    }


def token_features(
    X,
    skip_first: int = 1,
    center: np.ndarray | None = None,
    horizon: int = 4,
    theiler: int = 2,
) -> np.ndarray:
    Xp = _prep(X, skip_first, center)
    feats: dict[str, float] = {}
    feats.update(_summarize(speed_series(Xp), "speed"))
    feats.update(_summarize(curvature_series(Xp), "curv"))
    feats.update(_summarize(stretch_series(Xp), "stretch"))
    feats["tortuosity"] = tortuosity(Xp)
    feats["lyap_R"] = lyapunov_rosenstein(Xp, horizon=horizon, theiler=theiler)
    return np.array([feats[n] for n in FEATURE_NAMES], dtype=np.float64)


def states_to_features(S, skip_first: int = 1, center: np.ndarray | None = None, **kw) -> np.ndarray:
    S = np.asarray(S)
    if S.ndim != 3:
        raise ValueError(f"ожидается массив (T, слои, d), получено {S.shape}")
    return np.vstack([token_features(S[t], skip_first, center, **kw) for t in range(S.shape[0])])


def trajectory_series(S, skip_first: int = 1, center: np.ndarray | None = None) -> dict[str, np.ndarray]:
    S = np.asarray(S)
    out: dict[str, list[np.ndarray]] = {"speed": [], "curv": [], "stretch": []}
    for t in range(S.shape[0]):
        Xp = _prep(S[t], skip_first, center)
        out["speed"].append(speed_series(Xp))
        out["curv"].append(curvature_series(Xp))
        out["stretch"].append(stretch_series(Xp))
    return {k: np.vstack(v) for k, v in out.items()}


def series_layers(n_states: int, skip_first: int = 1) -> dict[str, np.ndarray]:
    n = n_states - skip_first
    return {
        "speed": skip_first + 1 + np.arange(n - 1),
        "curv": skip_first + 1 + np.arange(n - 2),
        "stretch": skip_first + 1 + np.arange(n - 2),
    }


def fit_layer_mean(states_list) -> np.ndarray:
    acc, cnt = None, 0
    for S in states_list:
        S = np.asarray(S, dtype=np.float64)
        s = S.sum(axis=0)
        acc = s if acc is None else acc + s
        cnt += S.shape[0]
    if acc is None:
        raise ValueError("пустой список состояний")
    return acc / cnt


def pool_answer(F: np.ndarray, top_frac: float = 0.2) -> np.ndarray:
    F = np.asarray(F, dtype=np.float64)
    k = max(1, int(np.ceil(top_frac * F.shape[0])))
    top = np.sort(F, axis=0)[-k:].mean(axis=0)
    return np.concatenate([F.mean(axis=0), F.max(axis=0), top])


def pooled_names(names: list[str]) -> list[str]:
    return [f"{agg}[{n}]" for agg in ("mean", "max", "topk") for n in names]
