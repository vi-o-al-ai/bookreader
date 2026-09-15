"""Boot-time recovery: jobs left ``running`` (crash) or ``queued`` (clean shutdown with work
waiting) are handed back to the queue when the app starts, without touching finished jobs."""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi.testclient import TestClient

from bookreader.api.app import _resubmit_pending, create_app
from bookreader.jobs.db import JobStore
from bookreader.jobs.paths import JobPaths
from bookreader.settings import Settings
from bookreader.types import JobStatus, Stage


def _settings(root: Path) -> Settings:
    return Settings.from_env({}).with_overrides(data_dir=root / "data", worker_mode="inline", mock_ms_per_char=4, warmup=False)


def _upload(client: TestClient, sample_book_path: Path) -> str:
    with sample_book_path.open("rb") as fh:
        response = client.post("/api/jobs", files={"file": ("sample_book.txt", fh, "text/plain")}, data={"options": '{"chapters":[1],"music":false,"sfx":false}'})
    assert response.status_code == 202, response.text
    return response.json()["job_id"]


def test_resubmit_pending_hands_running_and_queued_jobs_to_the_queue(tmp_path: Path, sample_book_path: Path) -> None:
    settings = _settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        crashed, waiting, finished = (_upload(client, sample_book_path) for _ in range(3))
    store = JobStore(JobPaths(settings.data_dir, "").db_path)
    store.set_status(crashed, JobStatus.running, stage=Stage.render)
    store.requeue(waiting, from_stage=Stage.finalize)

    submitted: list[str] = []
    assert sorted(_resubmit_pending(store, submitted.append)) == sorted([crashed, waiting])
    assert sorted(submitted) == sorted([crashed, waiting]) and finished not in submitted
    assert store.get_job(crashed).status == JobStatus.queued                 # the orphan is queued, not running
    store.close()


def test_app_boot_resumes_jobs_left_running_or_queued(tmp_path: Path, sample_book_path: Path, caplog) -> None:
    settings = _settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        crashed, waiting = _upload(client, sample_book_path), _upload(client, sample_book_path)
    store = JobStore(JobPaths(settings.data_dir, "").db_path)
    store.set_status(crashed, JobStatus.running, stage=Stage.render)         # process died mid-render
    store.requeue(waiting, from_stage=Stage.finalize)                        # shut down with a retry waiting
    store.close()
    JobPaths(settings.data_dir, waiting).manifest.unlink()

    with caplog.at_level(logging.INFO, logger="bookreader.api.app"):
        with TestClient(create_app(settings)) as client:                     # inline queue: resumed before requests are served
            for job_id in (crashed, waiting):
                body = client.get(f"/api/jobs/{job_id}").json()
                assert body["status"] == "done", body
                assert client.get(f"/api/jobs/{job_id}/manifest").status_code == 200
            stages = {s["stage"]: s for s in client.get(f"/api/jobs/{waiting}").json()["stages"]}
            assert stages["finalize"]["attempts"] == 2 and stages["render"]["attempts"] == 1
    assert "resubmitted 2 pending job(s)" in caplog.text
