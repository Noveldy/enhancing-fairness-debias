"""Keep an experiment run from exhausting the machine's memory (Linux only).

Why: the first full run of configs/main.yaml (2026-10-18 14:26) was killed by
the kernel's OOM killer 19 s after it started. The kernel log shows 18 inner-CV
worker processes using 0.4-0.8 GB each (~10.5 GB together) on a 14.6 GB
machine that also ran a browser and an editor; the kernel killed the desktop
applications first. See REVISION_LOG.md (Phase 2, memory safeguard).

Three layers:

1. ``safe_n_jobs``: before a search starts, cap the number of parallel workers
   so that ``n_jobs * per_worker_gb + main_process_gb`` fits in the memory that
   is free right now minus a reserve. Parallelism does not change results
   (every estimator has a fixed random_state and runs single-threaded), so the
   cap only changes wall-clock time. It is logged.
2. ``MemoryWatchdog``: a background thread that every ``interval`` seconds sums
   the proportional set size (PSS) of this process and all its descendants. If
   that exceeds the budget, or the system's MemAvailable drops below the
   reserve, it kills the worker processes and the main thread receives
   ``MemoryBudgetExceeded`` naming the experiment, the fold, the stage and the
   largest processes.
3. ``deprioritize_for_oom_killer``: raises this process's oom_score_adj (which
   children inherit), so if memory still runs out the kernel kills this run
   before the user's desktop applications.

Reads /proc directly, so it needs no extra dependency.
"""
import logging
import os
import signal
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)
GB = 1024 ** 3


class MemoryBudgetExceeded(MemoryError):
    """Raised in the main thread when the watchdog stops a run."""


# ------------------------------------------------------------------ /proc helpers
def meminfo_bytes(field):
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(field + ":"):
            return int(line.split()[1]) * 1024
    raise KeyError(field)


def available_bytes():
    return meminfo_bytes("MemAvailable")


def _children(pid):
    out = []
    task_dir = Path(f"/proc/{pid}/task")
    try:
        for tid in task_dir.iterdir():
            text = (tid / "children").read_text().split()
            out.extend(int(c) for c in text)
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
    return out


def process_tree(root=None):
    root = root or os.getpid()
    seen, stack = [], [root]
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.append(pid)
        stack.extend(_children(pid))
    return seen


def process_memory_bytes(pid):
    """PSS if available (shared pages split fairly between processes), else RSS."""
    try:
        for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
            if line.startswith("Pss:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        pass
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        pass
    return 0


def _is_helper(pid):
    """joblib/multiprocessing resource trackers hold almost no memory; killing them leaks semaphores."""
    try:
        return b"resource_tracker" in Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False


def tree_memory(root=None):
    """(total bytes, [(pid, bytes), ...] largest first) for this process tree."""
    per = [(pid, process_memory_bytes(pid)) for pid in process_tree(root)]
    per.sort(key=lambda t: -t[1])
    return sum(b for _, b in per), per


# ------------------------------------------------------------------ layer 1
def safe_n_jobs(requested, per_worker_gb, main_gb, reserve_gb, label=""):
    """Largest n_jobs <= requested whose estimated peak fits in free memory minus the reserve."""
    # This run's own memory (idle joblib workers kept for reuse, the previous fold's
    # model) is given back before the next search, so count it as available.
    free_gb = available_bytes() / GB
    own_gb = tree_memory()[0] / GB
    usable = free_gb + own_gb - reserve_gb - main_gb
    fit = int(usable // per_worker_gb) if per_worker_gb > 0 else requested
    n = max(1, min(requested, fit))
    if fit < 1:
        raise MemoryBudgetExceeded(
            f"{label}: not enough free memory to start even one worker "
            f"(free {free_gb:.1f} GB, reserve {reserve_gb} GB, main process {main_gb} GB, "
            f"per worker {per_worker_gb} GB). Close other applications or lower memory.reserve_gb.")
    if n < requested:
        log.warning("%s: n_jobs capped %d -> %d (free %.1f GB + this run %.1f GB, reserve %.1f GB, "
                    "est. %.2f GB/worker + %.1f GB main)",
                    label, requested, n, free_gb, own_gb, reserve_gb, per_worker_gb, main_gb)
    return n


# ------------------------------------------------------------------ layer 3
def deprioritize_for_oom_killer(score=1000):
    """Make the kernel pick this run (and its children) first if memory runs out."""
    try:
        Path("/proc/self/oom_score_adj").write_text(str(score))
        return True
    except OSError:
        return False


# ------------------------------------------------------------------ layer 2
class MemoryWatchdog:
    """Context manager: stop the run and raise MemoryBudgetExceeded before the machine runs out.

    with MemoryWatchdog(budget_gb=9, reserve_gb=2, context="adult r0 f1") as wd:
        wd.stage = "inner search"
        search.fit(...)

    If the limit is hit while workers run, the workers are killed; joblib then raises
    in the main thread and ``__exit__`` turns that into MemoryBudgetExceeded. If the
    main process itself grows past the limit, the main thread is interrupted with
    SIGUSR1 (handled when it next runs Python code).
    """

    def __init__(self, budget_gb, reserve_gb, context="", interval=0.5):
        self.budget = budget_gb * GB
        self.reserve = reserve_gb * GB
        self.context = context
        self.interval = interval
        self.stage = ""
        self.peak = 0
        self.tripped = None
        self._stop = threading.Event()
        self._thread = None
        self._old_handler = None

    def _message(self, total, per, avail, why):
        top = ", ".join(f"pid {p}: {b / GB:.2f} GB{' (main)' if p == os.getpid() else ''}" for p, b in per[:5])
        return (f"{self.context} [{self.stage or 'unknown stage'}]: {why}. "
                f"This run used {total / GB:.2f} GB in {len(per)} processes "
                f"(budget {self.budget / GB:.1f} GB); system MemAvailable {avail / GB:.2f} GB "
                f"(reserve {self.reserve / GB:.1f} GB). Largest: {top}. "
                f"Lower n_jobs or memory.per_worker_gb in the config, or free memory.")

    def _loop(self):
        me = os.getpid()
        while True:  # sample at start, every interval, and once more when stopped
            stopping = self._stop.wait(self.interval)
            total, per = tree_memory(me)
            avail = available_bytes()
            self.peak = max(self.peak, total)
            why = None
            if total > self.budget:
                why = "memory budget exceeded"
            elif avail < self.reserve:
                why = "system free memory fell below the reserve"
            if why is None:
                if stopping:
                    return
                continue
            self.tripped = self._message(total, per, avail, why)
            log.error("MEMORY GUARD: %s", self.tripped)
            workers = [p for p, _ in per if p != me and not _is_helper(p)]
            for p in workers:
                try:
                    os.kill(p, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if not workers or dict(per).get(me, 0) > self.budget / 2:
                os.kill(me, signal.SIGUSR1)
            return

    def _on_signal(self, signum, frame):
        raise MemoryBudgetExceeded(self.tripped or f"{self.context}: stopped by the memory guard")

    def __enter__(self):
        if threading.current_thread() is threading.main_thread():
            self._old_handler = signal.signal(signal.SIGUSR1, self._on_signal)
        self._thread = threading.Thread(target=self._loop, name="memory-watchdog", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join()
        if self._old_handler is not None:
            signal.signal(signal.SIGUSR1, self._old_handler)
        if self.tripped and not isinstance(exc, MemoryBudgetExceeded):
            # Usually joblib's TerminatedWorkerError after the workers were killed.
            raise MemoryBudgetExceeded(self.tripped) from exc
        return False
