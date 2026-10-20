"""The memory guard stops a run before it exhausts memory (experiments/memory_guard.py)."""
import time

import numpy as np
import pytest
from joblib import Parallel, delayed

from experiments import memory_guard as mg


def _hog(mb, seconds):
    block = np.ones(mb * 2 ** 20 // 8)  # touched pages, so they count as resident
    time.sleep(seconds)
    return float(block[0])


def _budget_above_now(extra_gb):
    total, _ = mg.tree_memory()
    return total / mg.GB + extra_gb


def test_workers_over_budget_are_stopped():
    budget = _budget_above_now(0.3)  # two workers of 400 MB cannot fit
    t0 = time.perf_counter()
    with pytest.raises(mg.MemoryBudgetExceeded, match="memory budget exceeded"):
        with mg.MemoryWatchdog(budget, reserve_gb=0, context="test", interval=0.1) as wd:
            wd.stage = "two hungry workers"
            Parallel(n_jobs=2)(delayed(_hog)(400, 30) for _ in range(2))
    assert time.perf_counter() - t0 < 25, "the guard should stop the run long before the workers finish"


def test_main_process_over_budget_is_interrupted():
    budget = _budget_above_now(0.2)
    held = []
    with pytest.raises(mg.MemoryBudgetExceeded, match=r"\[growing main\]"):
        with mg.MemoryWatchdog(budget, reserve_gb=0, context="test", interval=0.1) as wd:
            wd.stage = "growing main"
            for _ in range(200):  # up to 2 GB in 10 MB steps
                held.append(np.ones(10 * 2 ** 20 // 8))
                time.sleep(0.02)
    held.clear()


def test_within_budget_runs_normally_and_records_peak():
    with mg.MemoryWatchdog(_budget_above_now(2.0), reserve_gb=0, context="test", interval=0.1) as wd:
        out = Parallel(n_jobs=2)(delayed(_hog)(100, 1) for _ in range(2))
    assert out == [1.0, 1.0] and wd.peak > 0 and wd.tripped is None


def test_safe_n_jobs_caps_to_free_memory(monkeypatch):
    monkeypatch.setattr(mg, "available_bytes", lambda: 10 * mg.GB)
    monkeypatch.setattr(mg, "tree_memory", lambda root=None: (0, []))
    assert mg.safe_n_jobs(18, per_worker_gb=1.0, main_gb=2.0, reserve_gb=2.0) == 6
    assert mg.safe_n_jobs(4, per_worker_gb=1.0, main_gb=2.0, reserve_gb=2.0) == 4
    with pytest.raises(mg.MemoryBudgetExceeded, match="not enough free memory"):
        mg.safe_n_jobs(4, per_worker_gb=1.0, main_gb=8.0, reserve_gb=2.0)


def test_safe_n_jobs_counts_own_idle_memory(monkeypatch):
    """Idle workers of this run are reused, so they do not shrink the next fold's cap."""
    monkeypatch.setattr(mg, "available_bytes", lambda: 7 * mg.GB)
    monkeypatch.setattr(mg, "tree_memory", lambda root=None: (3 * mg.GB, []))
    assert mg.safe_n_jobs(18, per_worker_gb=1.0, main_gb=2.0, reserve_gb=2.0) == 6


def test_default_budget_counts_own_memory(monkeypatch):
    """From the second experiment on, the run's idle workers are part of the budget, not outside it."""
    from experiments import protocol
    monkeypatch.setattr(protocol, "available_bytes", lambda: 7 * mg.GB)
    monkeypatch.setattr(protocol, "tree_memory", lambda root=None: (4 * mg.GB, []))
    assert protocol.memory_settings({"memory": {"reserve_gb": 2.0}})["budget_gb"] == 9.0
