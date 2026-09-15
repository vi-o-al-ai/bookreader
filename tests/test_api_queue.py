"""HTTP API on the threaded queue: cancel / retry / re-cast / delete of jobs that are waiting in
the queue, and the upload size guard that must fire before the body is read."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from bookreader.api.app import UPLOAD_FORM_OVERHEAD_BYTES, UploadLimitMiddleware, create_app
from bookreader.settings import Settings

pytestmark = pytest.mark.timeout(60)


def _settings(root: Path, **overrides: object) -> Settings:
    return Settings.from_env({}).with_overrides(
        data_dir=root / "data", worker_mode="thread", workers=2, mock_ms_per_char=4, warmup=False, **overrides,
    )


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    with TestClient(create_app(_settings(tmp_path))) as client:
        yield client


def _upload(client: TestClient, sample_book_path: Path) -> str:
    with sample_book_path.open("rb") as fh:
        response = client.post("/api/jobs", files={"file": ("sample_book.txt", fh, "text/plain")})
    assert response.status_code == 202, response.text
    return response.json()["job_id"]


def _wait_terminal(client: TestClient, job_id: str, timeout: float = 30) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] in ("done", "failed", "cancelled"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish: {client.get(f'/api/jobs/{job_id}').json()['status']}")


def _events(client: TestClient, job_id: str) -> list[str]:
    return [e["message"] for e in client.get(f"/api/jobs/{job_id}/events", params={"limit": 1000}).json()["events"]]


def test_cancel_then_retry_of_a_queued_job_runs_it_exactly_once(client: TestClient, sample_book_path: Path) -> None:
    queue = client.app.state.queue
    queue.stop()                                          # park the workers: the upload stays queued
    job = _upload(client, sample_book_path)
    assert client.get(f"/api/jobs/{job}").json()["status"] == "queued" and queue.depth() == 1
    assert client.post(f"/api/jobs/{job}/cancel").json()["status"] == "cancelled"
    assert queue.depth() == 0
    assert client.post(f"/api/jobs/{job}/retry").json()["status"] == "queued"
    assert queue.depth() == 1                             # not two entries for one id
    queue.start()
    body = _wait_terminal(client, job)
    assert body["status"] == "done", body["error"]
    assert all(s["attempts"] == 1 for s in body["stages"])
    messages = _events(client, job)
    assert messages.count("cancelled while queued") == 1
    assert sum(m.startswith("run started") for m in messages) == 1
    assert messages.count("run finished: done") == 1
    assert client.get("/api/health").json()["queue"] == {"backend": "thread", "depth": 0, "running": []}


def test_same_id_submitted_twice_to_two_workers_runs_once(client: TestClient, sample_book_path: Path) -> None:
    """Even if a backend hands the same id to two workers, run_job's atomic claim lets only one run it."""
    queue = client.app.state.queue
    queue.stop()
    job = _upload(client, sample_book_path)
    queue._pending.discard(job)                           # simulate a backend without dedupe: two raw entries
    queue._queue.put(job)
    queue._pending.add(job)
    queue.start()
    body = _wait_terminal(client, job)
    assert body["status"] == "done", body["error"]
    assert all(s["attempts"] == 1 for s in body["stages"])
    assert sum(m.startswith("run started") for m in _events(client, job)) == 1


def test_delete_of_a_queued_job_is_skipped_by_the_worker(client: TestClient, sample_book_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    queue = client.app.state.queue
    queue.stop()
    job = _upload(client, sample_book_path)
    other = _upload(client, sample_book_path)
    assert client.delete(f"/api/jobs/{job}").status_code == 204
    assert queue.depth() == 1
    queue.start()
    assert _wait_terminal(client, other)["status"] == "done"
    assert client.get(f"/api/jobs/{job}").status_code == 404
    assert not any("crashed the worker loop" in record.message for record in caplog.records)


def test_put_cast_on_a_queued_job_whose_cast_stage_is_done_is_honoured(client: TestClient, sample_book_path: Path) -> None:
    """A job retried from finalize while other jobs are ahead of it is 'queued' with its cast stage
    already done; the pin must still be applied when the job runs."""
    queue = client.app.state.queue
    job = _upload(client, sample_book_path)
    assert _wait_terminal(client, job)["status"] == "done"
    before = client.get(f"/api/jobs/{job}/cast").json()["cast"]
    assert next(a for a in before["characters"] if a["character"] == "Tobias")["voice"]["id"] != "mock-f-teen"

    queue.stop()
    assert client.post(f"/api/jobs/{job}/retry", json={"from_stage": "finalize"}).status_code == 202
    assert client.get(f"/api/jobs/{job}").json()["status"] == "queued"
    response = client.put(f"/api/jobs/{job}/cast", json={"overrides": {"Tobias": "mock-f-teen"}})
    assert response.status_code == 202 and response.json()["status"] == "queued"
    assert queue.depth() == 1                             # still one queue entry
    queue.start()
    body = _wait_terminal(client, job)
    assert body["status"] == "done", body["error"]
    stages = {s["stage"]: s for s in body["stages"]}
    assert stages["cast"]["attempts"] == 2 and stages["render"]["attempts"] == 2 and stages["ingest"]["attempts"] == 1
    tobias = next(a for a in client.get(f"/api/jobs/{job}/cast").json()["cast"]["characters"] if a["character"] == "Tobias")
    assert tobias["voice"]["id"] == "mock-f-teen" and tobias["source"] == "override"
    assert any(m.startswith("re-cast requested: Tobias -> mock-f-teen") for m in _events(client, job))


# --------------------------------------------------------------------------- upload guard
def test_upload_limit_middleware_rejects_before_reading_the_body() -> None:
    import asyncio

    consumed: list[int] = []
    entered: list[bool] = []

    async def inner(scope: dict, receive, send) -> None:  # drains the body like the multipart parser
        entered.append(True)
        while True:
            message = await receive()
            consumed.append(len(message.get("body", b"")))
            if not message.get("more_body", False):
                break
        await send({"type": "http.response.start", "status": 202, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    limit = 2 * 1024 * 1024
    app = UploadLimitMiddleware(inner, limit_bytes=limit)

    def scope(headers: list[tuple[bytes, bytes]], path: str = "/api/jobs", method: str = "POST") -> dict:
        return {"type": "http", "method": method, "path": path, "root_path": "", "headers": headers}

    def body_source(chunk: bytes, chunks: int):
        sent = 0

        async def receive() -> dict:
            nonlocal sent
            sent += 1
            return {"type": "http.request", "body": chunk, "more_body": sent < chunks}
        return receive

    async def collect(app_, scope_, receive):
        messages: list[dict] = []

        async def send(message: dict) -> None:
            messages.append(message)
        await app_(scope_, receive, send)
        return messages

    # 1. an honest Content-Length above the limit: 413 without calling the app or reading a byte
    messages = asyncio.run(collect(app, scope([(b"content-length", str(500_000_000).encode())]), body_source(b"x" * 1024, 10)))
    assert messages[0]["status"] == 413 and b"BOOKREADER_MAX_UPLOAD_MB" in messages[1]["body"]
    assert not entered and not consumed

    # 2. no Content-Length (chunked) and a body that keeps growing: reception stops at the limit
    chunk = b"x" * (1024 * 1024)
    with pytest.raises(Exception) as excinfo:
        asyncio.run(collect(app, scope([]), body_source(chunk, 80)))
    assert getattr(excinfo.value, "status_code", None) == 413
    assert entered and sum(consumed) <= limit + UPLOAD_FORM_OVERHEAD_BYTES + len(chunk) < 80 * len(chunk)

    # 3. a body under the limit and any other route pass through untouched
    consumed.clear()
    messages = asyncio.run(collect(app, scope([(b"content-length", b"2048")]), body_source(b"x" * 1024, 2)))
    assert messages[0]["status"] == 202 and sum(consumed) == 2048
    messages = asyncio.run(collect(app, scope([(b"content-length", b"999999999")], path="/api/jobs/x/cast", method="PUT"), body_source(b"x", 1)))
    assert messages[0]["status"] == 202


def test_upload_over_the_limit_is_413_end_to_end_and_under_it_is_accepted(tmp_path: Path, sample_book_path: Path) -> None:
    with TestClient(create_app(_settings(tmp_path, max_upload_mb=1))) as client:
        big = b"Chapter 1\n\n" + (b"She said, \"hello there.\" " * 40 + b"\n\n") * 1500   # ~1.5 MB of text
        assert len(big) > 1024 * 1024 + UPLOAD_FORM_OVERHEAD_BYTES
        response = client.post("/api/jobs", files={"file": ("big.txt", big, "text/plain")})
        assert response.status_code == 413 and "BOOKREADER_MAX_UPLOAD_MB" in response.json()["detail"]
        assert client.get("/api/jobs").json()["jobs"] == []
        job = _upload(client, sample_book_path)
        assert _wait_terminal(client, job)["status"] == "done"
