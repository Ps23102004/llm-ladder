import threading
import time

import pytest

from llm_ladder import jobs


@pytest.fixture(autouse=True)
def _clear_jobs():
    jobs._JOBS.clear()
    yield
    jobs._JOBS.clear()


def _wait_done(job_id, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = jobs.get_status(job_id)
        if status["state"] != "running":
            return status
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


def test_job_runs_and_returns_result():
    job_id = jobs.start_job(lambda report_progress: 42)
    status = _wait_done(job_id)
    assert status == {"state": "done", "progress": None, "result": 42, "error": None}


def test_job_reports_progress():
    def fn(report_progress):
        report_progress({"step": "working"})
        return "ok"

    job_id = jobs.start_job(fn)
    _wait_done(job_id)
    assert jobs.get_status(job_id)["progress"] == {"step": "working"}


def test_job_surfaces_exception():
    def fn(report_progress):
        raise ValueError("boom")

    job_id = jobs.start_job(fn)
    status = _wait_done(job_id)
    assert status["state"] == "error"
    assert "boom" in status["error"]


def test_unknown_job_id_returns_error():
    status = jobs.get_status("does-not-exist")
    assert status["state"] == "error"
    assert "does-not-exist" in status["error"]


def test_finished_job_is_swept_after_ttl(monkeypatch):
    job_id = jobs.start_job(lambda report_progress: "done")
    _wait_done(job_id)
    assert jobs.get_status(job_id)["state"] == "done"

    monkeypatch.setattr(jobs, "_TTL_S", 0.0)
    time.sleep(0.01)
    assert jobs.get_status(job_id)["error"] == f"unknown or expired job_id: {job_id}"


def test_errored_job_is_swept_after_ttl(monkeypatch):
    def fn(report_progress):
        raise ValueError("boom")

    job_id = jobs.start_job(fn)
    _wait_done(job_id)
    assert jobs.get_status(job_id)["state"] == "error"

    monkeypatch.setattr(jobs, "_TTL_S", 0.0)
    time.sleep(0.01)
    assert jobs.get_status(job_id)["error"] == f"unknown or expired job_id: {job_id}"


def test_sweep_also_triggered_by_start_job(monkeypatch):
    old_job_id = jobs.start_job(lambda report_progress: "done")
    _wait_done(old_job_id)

    monkeypatch.setattr(jobs, "_TTL_S", 0.0)
    time.sleep(0.01)
    jobs.start_job(lambda report_progress: "new")

    assert jobs.get_status(old_job_id)["error"] == f"unknown or expired job_id: {old_job_id}"


def test_running_job_survives_a_sweep(monkeypatch):
    release = threading.Event()
    job_id = jobs.start_job(lambda report_progress: release.wait(timeout=2.0))

    monkeypatch.setattr(jobs, "_TTL_S", 0.0)
    # Trigger a sweep pass while the job is still running.
    jobs.get_status(job_id)
    assert jobs.get_status(job_id)["state"] == "running"

    release.set()
    _wait_done(job_id)
