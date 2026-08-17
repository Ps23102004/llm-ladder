"""Tiny thread-based background job runner.

Public API:
    start_job(fn) -> str   # returns job_id immediately, fn runs in a thread
    get_status(job_id) -> dict

The callable receives a ``report_progress`` function that accepts a single
dict; the last reported dict is surfaced via ``get_status``. Pure stdlib.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Callable

# job_id -> job record
_JOBS: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()

# How long a finished job's record is kept around before it's swept.
_TTL_S = 3600.0


def _sweep() -> None:
    """Drop finished jobs older than _TTL_S. Caller must hold _LOCK.

    Lazy: only runs when start_job/get_status are called, not on a timer.
    An idle server never reclaims memory, but it's bounded by past activity.
    """
    now = time.monotonic()
    expired = [
        job_id
        for job_id, record in _JOBS.items()
        if record["finished_at"] is not None and now - record["finished_at"] > _TTL_S
    ]
    for job_id in expired:
        del _JOBS[job_id]


def start_job(fn: Callable[[Callable[[dict], None]], Any]) -> str:
    """Run ``fn(report_progress)`` in a background thread.

    Returns the job id immediately.
    """
    job_id = str(uuid.uuid4())
    with _LOCK:
        _sweep()
        _JOBS[job_id] = {
            "state": "running",
            "progress": None,
            "result": None,
            "error": None,
            "finished_at": None,
        }

    def report_progress(payload: dict) -> None:
        with _LOCK:
            record = _JOBS.get(job_id)
            if record is not None:
                record["progress"] = payload

    def _run() -> None:
        record = _JOBS[job_id]
        try:
            result = fn(report_progress)
        except BaseException as exc:  # noqa: BLE001 - surface any failure to the caller
            with _LOCK:
                record["state"] = "error"
                record["error"] = f"{type(exc).__name__}: {exc}"
                record["finished_at"] = time.monotonic()
        else:
            with _LOCK:
                record["state"] = "done"
                record["result"] = result
                record["finished_at"] = time.monotonic()

    threading.Thread(target=_run, daemon=True, name=f"job-{job_id[:8]}").start()
    return job_id


def get_status(job_id: str) -> dict:
    """Return {'state', 'progress', 'result', 'error'} for a job id.

    Unknown ids return state 'error' with a descriptive message.
    """
    with _LOCK:
        _sweep()
        record = _JOBS.get(job_id)
        if record is None:
            return {
                "state": "error",
                "progress": None,
                "result": None,
                "error": f"unknown or expired job_id: {job_id}",
            }
        return {
            "state": record["state"],
            "progress": record["progress"],
            "result": record["result"],
            "error": record["error"],
        }
