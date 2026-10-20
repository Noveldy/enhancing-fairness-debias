"""Phase 7 timing harness: the meter, the task graph, and one task end to end."""
import importlib.util
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import yaml

from experiments.cost import Meter, run_task

REPO = Path(__file__).resolve().parents[1]


def _cost_run():
    spec = importlib.util.spec_from_file_location("cost_run", REPO / "scripts" / "cost_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_meter_times_and_tracks_memory():
    with Meter() as m:
        time.sleep(0.2)
        block = np.ones(50_000_000 // 8)   # ~50 MB
        time.sleep(0.05)
    del block
    r = m.record()
    assert 0.2 <= r["seconds"] < 0.6
    assert r["cpu_seconds"] < r["seconds"]          # sleeping uses no CPU
    assert r["rss_increase_gb"] > 0.04


def test_task_graph():
    cfg = yaml.safe_load((REPO / "configs" / "cost.yaml").read_text())
    tasks = _cost_run().build_tasks(cfg)
    assert len(tasks) == len({t["task_id"] for t in tasks.values()})
    for t in tasks.values():
        assert all(d in tasks for d in t["deps"])
    res = {k: t for k, t in tasks.items() if t["task"] == "resampler" and t["dataset"] == "adult"
           and t["resampler"] == "CondensedNearestNeighbour"}
    full = [t for t in res.values() if t["frac"] == 1.0]
    assert len(full) == 3 and len(res) == 12
    first_full = next(t for t in full if t["fold"] == 0)
    for t in full:
        # needs its cached data, the first 50% subsample, and (except the first) the first full-size task
        assert any(tasks[d]["task"] == "dbim_fit" for d in t["deps"])
        assert any(tasks[d].get("frac") == 0.5 for d in t["deps"])
        assert (t is first_full) or first_full["task_id"] in t["deps"]
    # the 100% scaling points reuse the 7.2 cost fits instead of duplicating them
    assert sum(t["task"] == "dbim_fit" and t["frac"] == 1.0 for t in tasks.values()) == 6


def test_run_task_in_child_process(tmp_path):
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    spec = {"task_id": "t", "part": "cost", "task": "ts_cleaning", "fold": 0, "scenario": "linear"}
    p = ctx.Process(target=run_task, args=(spec, str(tmp_path), None, child))
    p.start()
    child.close()
    status, rows = parent.recv()
    p.join()
    assert status == "ok", rows
    assert rows[0]["stage"] == "cleaning" and rows[0]["method"] == "linear" and rows[0]["seconds"] > 0
    assert rows[0]["cells_filled"] > 1000          # the artificial gap plus the natural gaps
