from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

from ..baselines import max_entropy, mean_entropy, mean_nll, self_consistency_score
from ..classifier import TrajectoryDetector
from ..evaluate import (
    auprc,
    auroc,
    bootstrap_ci,
    fpr_at_tpr,
    oof_scores,
    paired_bootstrap_diff,
    roc_points,
)
from ..features import FEATURE_NAMES, states_to_features
from ..labeling import label_answer



def load_truthfulqa(n: int | None, seed: int):
    from datasets import load_dataset

    try:
        ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    except Exception:
        ds = load_dataset("truthful_qa", "generation", split="validation")
    rows = [dict(r) for r in ds]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    rows = [rows[i] for i in order]
    return rows[:n] if n else rows


def fake_rows(n: int):
    rows = []
    for i in range(n):
        bad = i % 2
        rows.append(
            dict(
                question=f"synthetic question {i} about topic {i % 7}",
                best_answer=f"truthful statement {i}",
                correct_answers=[f"truthful statement {i}"],
                incorrect_answers=[f"halluc false claim {i}"],
            )
        )
    return rows



def _load_cache(path: Path, header: dict) -> dict:
    done: dict = {}
    with open(path, "rb") as f:
        try:
            first = pickle.load(f)
        except Exception:
            first = None
        if first != ("header", header):
            raise SystemExit(
                f"Кэш {path} создан с другими параметрами (модель, --n, --seed, --k-samples и т.д.).\n"
                "Удалите его, укажите другой --out или запустите с --fresh."
            )
        good = f.tell()
        while True:
            try:
                rec = pickle.load(f)
            except EOFError:
                break
            except Exception:  
                break
            done[rec[0]] = rec[1]
            good = f.tell()
    if good < path.stat().st_size:
        with open(path, "r+b") as f:
            f.truncate(good)
    return done


def _append_cache(path: Path, record) -> None:
    with open(path, "ab") as f:
        pickle.dump(record, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())


def _process_question(extractor, qi, row, protocol, k_samples, refs_per_class):
    q = row["question"]
    samples, t_sample = [], 0.0
    if k_samples > 0:
        t0 = time.perf_counter()
        samples = extractor.sample(q, k_samples)
        t_sample = time.perf_counter() - t0

    if protocol == "generated":
        try:
            rec = extractor.analyze(q)
        except ValueError:
            return []
        lab = label_answer(rec.answer, row["correct_answers"], row["incorrect_answers"])
        cands = [(rec, lab)]
    else:
        cands = []
        good = [row["best_answer"]] + [a for a in row["correct_answers"] if a != row["best_answer"]]
        for ans, lab in [(a, 0) for a in good[:refs_per_class]] + [
            (a, 1) for a in row["incorrect_answers"][:refs_per_class]
        ]:
            try:
                cands.append((extractor.analyze(q, ans), lab))
            except ValueError:
                continue

    items = []
    for rec, lab in cands:
        if lab is None:
            items.append(dict(qi=qi, skipped=True))
            continue
        t0 = time.perf_counter()
        F = states_to_features(rec.states)
        t_feat = time.perf_counter() - t0
        items.append(
            dict(
                qi=qi,
                skipped=False,
                question=q,
                answer=rec.answer,
                label=int(lab),
                F=F,
                entropy=rec.entropy,
                logprob=rec.logprob,
                sc_hard=self_consistency_score(rec.answer, samples) if samples else None,
                sc_soft=self_consistency_score(rec.answer, samples, soft=True) if samples else None,
                t_generate=rec.timings.get("generate", 0.0),
                t_forward=rec.timings.get("forward", 0.0),
                t_feat=t_feat,
                t_sample=t_sample,
            )
        )
    return items


def collect(extractor, rows, protocol: str, k_samples: int, refs_per_class: int, log,
            cache_path: Path | None = None, header: dict | None = None):
    done: dict = {}
    if cache_path is not None:
        header = header or {}
        if cache_path.exists():
            done = _load_cache(cache_path, header)
            log(f"  найден кэш: уже обработано {len(done)} из {len(rows)} вопросов, продолжаю")
        else:
            _append_cache(cache_path, ("header", header))

    items = []
    t_start, n_new = time.time(), 0
    for qi, row in enumerate(rows):
        if qi in done:
            items.extend(done[qi])
            continue
        its = _process_question(extractor, qi, row, protocol, k_samples, refs_per_class)
        items.extend(its)
        if cache_path is not None:
            _append_cache(cache_path, (qi, its))
        n_new += 1
        if n_new % 5 == 0:
            el = time.time() - t_start
            left = (len(rows) - qi - 1) * el / n_new
            log(f"  {qi + 1}/{len(rows)} вопросов, прошло {el / 60:.0f} мин, осталось примерно {left / 60:.0f} мин")
    return items



def analyse(items, k_samples: int, n_splits: int, n_boot: int, seed: int, log):
    kept = [it for it in items if not it["skipped"]]
    skipped = len(items) - len(kept)
    y = np.array([it["label"] for it in kept])
    groups = np.array([it["qi"] for it in kept])
    feats = [it["F"] for it in kept]

    scores: dict[str, np.ndarray] = {
        "token_entropy_mean": np.array([mean_entropy(it["entropy"]) for it in kept]),
        "token_entropy_max": np.array([max_entropy(it["entropy"]) for it in kept]),
        "neg_logprob_mean": np.array([mean_nll(it["logprob"]) for it in kept]),
    }
    if k_samples > 0:
        scores["self_consistency_hard"] = np.array([it["sc_hard"] for it in kept])
        scores["self_consistency_soft"] = np.array([it["sc_soft"] for it in kept])

    for model in ("logreg", "gboost"):
        for how in ("pooled", "token"):
            log(f"  CV: traj_{model}_{how}")
            scores[f"traj_{model}_{how}"] = oof_scores(
                feats, y, groups, lambda m=model: TrajectoryDetector(model=m), n_splits, seed, how
            )

    ablation: dict[str, np.ndarray] = {}
    for name, grp in {
        "only_speed": ["speed"],
        "only_curvature": ["curvature"],
        "only_lyapunov": ["lyapunov"],
        "no_lyapunov": ["speed", "curvature", "global"],
        "no_curvature": ["speed", "lyapunov", "global"],
        "no_speed": ["curvature", "lyapunov", "global"],
    }.items():
        ablation[name] = oof_scores(
            feats, y, groups, lambda g=grp: TrajectoryDetector(feature_groups=g), n_splits, seed, "pooled"
        )
    # комбинированная модель: траектории + энтропия токена
    names_ext = list(FEATURE_NAMES) + ["token_entropy"]
    feats_ext = [np.hstack([it["F"], it["entropy"][:, None]]) for it in kept]
    ablation["traj_plus_entropy"] = oof_scores(
        feats_ext, y, groups, lambda: TrajectoryDetector(feature_names=names_ext), n_splits, seed, "pooled"
    )

    def metrics(s):
        lo, hi = bootstrap_ci(y, s, auroc, groups, n_boot, seed=seed)
        return dict(auroc=auroc(y, s), auroc_lo=lo, auroc_hi=hi, auprc=auprc(y, s), fpr_at_tpr95=fpr_at_tpr(y, s))

    result = dict(
        n=int(len(y)),
        n_questions=int(len(np.unique(groups))),
        skipped_undetermined=int(skipped),
        prevalence=float(y.mean()),
        metrics={k: metrics(v) for k, v in scores.items()},
        ablation={k: metrics(v) for k, v in ablation.items()},
        roc={k: roc_points(y, v) for k, v in scores.items()},
    )
    paired = {}
    main = "traj_logreg_pooled"
    for base in [k for k in scores if not k.startswith("traj_")]:
        d, lo, hi = paired_bootstrap_diff(y, scores[main], scores[base], auroc, groups, n_boot, seed=seed)
        paired[f"{main}__minus__{base}"] = dict(diff=d, lo=lo, hi=hi)
    result["paired"] = paired

    med = lambda key: float(np.median([it[key] for it in kept]))
    result["timing"] = dict(
        median_generate_s=med("t_generate"),
        median_forward_s=med("t_forward"),
        median_features_s=med("t_feat"),
        median_sampling_k_s=med("t_sample") if k_samples > 0 else None,
        k_samples=k_samples,
    )
    return result, kept, scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="имя модели HF или 'mock'")
    ap.add_argument("--protocol", choices=["generated", "reference"], default="generated")
    ap.add_argument("--n", type=int, default=None, help="число вопросов (по умолчанию все 817)")
    ap.add_argument("--k-samples", type=int, default=5, help="выборок для self-consistency (0 — отключить)")
    ap.add_argument("--refs-per-class", type=int, default=1, help="эталонов каждого класса (протокол reference)")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-detector", action="store_true", help="обучить и сохранить детектор на всех данных")
    ap.add_argument("--fresh", action="store_true", help="удалить кэш в --out и начать с нуля")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = lambda s: print(s, flush=True)

    synthetic = args.model == "mock"
    if synthetic:
        from ..mock import MockExtractor

        extractor = MockExtractor(seed=args.seed)
        rows = fake_rows(args.n or 80)
        if args.protocol == "generated":
            log("mock: протокол переключён на reference (у синтетики нет генерации)")
            args.protocol = "reference"
    else:
        from ..extract import HiddenStateExtractor

        extractor = HiddenStateExtractor(
            args.model, device=args.device, dtype=args.dtype, max_new_tokens=args.max_new_tokens
        )
        rows = load_truthfulqa(args.n, args.seed)

    log(f"Сбор данных: {len(rows)} вопросов, протокол {args.protocol}")
    cache_path = out / "cache.pkl"
    if args.fresh and cache_path.exists():
        cache_path.unlink()
    header = dict(model=args.model, protocol=args.protocol, n=args.n, seed=args.seed, k_samples=args.k_samples,
                  refs_per_class=args.refs_per_class, max_new_tokens=args.max_new_tokens, dtype=args.dtype)
    items = collect(extractor, rows, args.protocol, args.k_samples, args.refs_per_class, log, cache_path, header)
    log("Анализ и кросс-валидация")
    result, kept, scores = analyse(items, args.k_samples, args.splits, args.boot, args.seed, log)
    result["config"] = {**vars(args), "synthetic": synthetic}

    (out / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out / "scores.csv", "w", encoding="utf-8") as f:
        names = list(scores)
        f.write("qi,label," + ",".join(names) + "\n")
        for i, it in enumerate(kept):
            f.write(f"{it['qi']},{it['label']}," + ",".join(f"{scores[n][i]:.6f}" for n in names) + "\n")

    if args.save_detector:
        det = TrajectoryDetector().fit([it["F"] for it in kept], [it["label"] for it in kept])
        det.save(str(out / "detector.joblib"))
        log(f"Детектор сохранён: {out / 'detector.joblib'}")

    log(f"\nn={result['n']} (вопросов {result['n_questions']}), доля галлюцинаций {result['prevalence']:.2f}")
    log("| метод | AUROC | 95% ДИ | AUPRC |")
    log("|---|---|---|---|")
    for k, m in result["metrics"].items():
        log(f"| {k} | {m['auroc']:.3f} | [{m['auroc_lo']:.3f}; {m['auroc_hi']:.3f}] | {m['auprc']:.3f} |")
    if synthetic:
        log("\n данные синтетические; цифры проверяют только конвейер.")


if __name__ == "__main__":
    main()
