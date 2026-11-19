"""Продолжение сбора данных после обрыва(уже обработанные вопросы не пересчитываются)"""
import numpy as np
import pytest

from hallu_traj.experiments import truthfulqa as tq
from hallu_traj.mock import MockExtractor


class Counting(MockExtractor):
    def __init__(self, *a, fail_after=None, **k):
        super().__init__(*a, **k)
        self.calls, self.fail_after = 0, fail_after

    def analyze(self, question, answer=None):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise KeyboardInterrupt  
        return super().analyze(question, answer)


HEADER = dict(model="mock", protocol="reference", n=12, seed=0, k_samples=0, refs_per_class=1, max_new_tokens=64, dtype="auto")


def run(ext, rows, path):
    return tq.collect(ext, rows, "reference", 0, 1, lambda s: None, path, HEADER)


def key(items):
    return [(it["qi"], it.get("label"), None if it["skipped"] else float(np.round(it["F"].sum(), 6))) for it in items]


def test_resume_matches_clean_run_and_skips_done(tmp_path):
    rows = tq.fake_rows(12)
    clean = run(Counting(seed=1), rows, tmp_path / "clean.pkl")

    path = tmp_path / "c.pkl"
    with pytest.raises(KeyboardInterrupt):
        run(Counting(seed=1, fail_after=14), rows, path)  
    ext2 = Counting(seed=1)
    resumed = run(ext2, rows, path)
    assert key(resumed) == key(clean)
    assert ext2.calls < 2 * len(rows) 


def test_truncated_last_record_is_dropped(tmp_path):
    rows = tq.fake_rows(6)
    path = tmp_path / "t.pkl"
    run(Counting(seed=1), rows, path)
    size = path.stat().st_size
    with open(path, "r+b") as f:
        f.truncate(size - 7)  
    again = run(Counting(seed=1), rows, path)
    assert len(again) == len(run(Counting(seed=1), rows, tmp_path / "x.pkl"))


def test_header_mismatch_refuses(tmp_path):
    rows = tq.fake_rows(4)
    path = tmp_path / "h.pkl"
    run(Counting(seed=1), rows, path)
    with pytest.raises(SystemExit):
        tq.collect(Counting(seed=1), rows, "reference", 0, 1, lambda s: None, path, dict(HEADER, n=999))
