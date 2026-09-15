"""HTTP API tests: one inline-mode app shared by the module, every request completes synchronously."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from bookreader.api.app import create_app
from bookreader.settings import Settings
from bookreader.types import NARRATOR, STAGE_ORDER

pytestmark = pytest.mark.timeout(40)


def _settings(root: Path, **overrides: object) -> Settings:
    return Settings.from_env({}).with_overrides(
        data_dir=root / "data", worker_mode="inline", mock_ms_per_char=4, warmup=False, **overrides,
    )


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    settings = _settings(tmp_path_factory.mktemp("api"))
    with TestClient(create_app(settings)) as client:
        yield client


def _upload(client: TestClient, sample_book_path: Path, **form: str) -> dict:
    with sample_book_path.open("rb") as fh:
        response = client.post("/api/jobs", files={"file": ("sample_book.txt", fh, "text/plain")}, data=form)
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["status"] in ("queued", "running", "done") and body["status_url"].endswith(f"/api/jobs/{body['job_id']}")
    return body


@pytest.fixture(scope="module")
def job_id(client: TestClient, sample_book_path: Path) -> str:
    return _upload(client, sample_book_path, title="The Lighthouse at Gull Point")["job_id"]


# --------------------------------------------------------------------------- status & listing
def test_status_done_with_every_stage_in_order(client: TestClient, job_id: str) -> None:
    body = client.get(f"/api/jobs/{job_id}").json()
    assert body["status"] == "done" and body["error"] is None and body["overall_pct"] == 100
    assert [s["stage"] for s in body["stages"]] == [s.value for s in STAGE_ORDER]
    assert all(s["state"] == "done" for s in body["stages"])
    assert body["title"] == "The Lighthouse at Gull Point" and body["filename"] == "sample_book.txt"
    assert body["estimate"]["chapters"] == 3 and body["estimate"]["chunks"] == 3
    assert body["usage"]["calls"] > 0 and body["usage"]["cache_hits"] == 0
    assert [c["index"] for c in body["chapters"]] == [1, 2, 3]
    assert all(c["ready"] and c["duration_ms"] > 0 for c in body["chapters"])
    assert body["providers"]["tts"] == {"family": "mock", "cache_version": "1"}
    assert body["started_at"] and body["finished_at"]


def test_list_jobs_contains_the_job(client: TestClient, job_id: str) -> None:
    jobs = client.get("/api/jobs").json()["jobs"]
    match = [j for j in jobs if j["id"] == job_id]
    assert len(match) == 1 and match[0]["status"] == "done" and match[0]["overall_pct"] == 100


def test_unknown_job_is_404(client: TestClient) -> None:
    for path in ("", "/cast", "/manifest", "/events", "/usage", "/log", "/artifacts", "/artifacts/manifest.json"):
        assert client.get(f"/api/jobs/nope{path}").status_code == 404


# --------------------------------------------------------------------------- cast, manifest, artifacts
def test_cast_has_five_assignments_and_voices(client: TestClient, job_id: str) -> None:
    body = client.get(f"/api/jobs/{job_id}/cast").json()
    cast = body["cast"]
    assert cast["narrator"]["character"] == NARRATOR
    assert len(cast["characters"]) == 4
    assert {a["character"] for a in cast["characters"]} == {"Mara Quill", "Tobias", "Ansel Vey", "Hetta"}
    assert len(body["voices"]) == 16 and all(v["family"] == "mock" for v in body["voices"])


def test_manifest_has_three_chapters(client: TestClient, job_id: str) -> None:
    manifest = client.get(f"/api/jobs/{job_id}/manifest").json()
    assert manifest["job_id"] == job_id and manifest["warnings"] == []
    assert [c["index"] for c in manifest["chapters"]] == [1, 2, 3]
    assert manifest["chapters"][0]["files"]["mix"] == "chapters/01/mix.wav"
    assert manifest["total_duration_ms"] > 0


def test_artifacts_listing(client: TestClient, job_id: str) -> None:
    files = client.get(f"/api/jobs/{job_id}/artifacts").json()["files"]
    paths = {f["path"] for f in files}
    assert {"chapters/01/mix.wav", "manifest.json", "cast.json", "usage.json", "job.log"} <= paths
    assert not any(p.startswith("source.") for p in paths)
    mix = next(f for f in files if f["path"] == "chapters/01/mix.wav")
    assert mix["bytes"] > 44 and mix["url"] == f"/api/jobs/{job_id}/artifacts/chapters/01/mix.wav"


def test_artifact_range_request(client: TestClient, job_id: str) -> None:
    full = client.get(f"/api/jobs/{job_id}/artifacts/chapters/01/mix.wav")
    assert full.status_code == 200 and full.headers["content-type"].startswith("audio/wav")
    assert full.headers.get("accept-ranges") == "bytes" and full.content[:4] == b"RIFF"
    partial = client.get(f"/api/jobs/{job_id}/artifacts/chapters/01/mix.wav", headers={"Range": "bytes=0-99"})
    assert partial.status_code == 206
    assert partial.headers["content-range"] == f"bytes 0-99/{len(full.content)}"
    assert partial.content == full.content[:100]
    manifest = client.get(f"/api/jobs/{job_id}/artifacts/manifest.json")
    assert manifest.headers["content-type"].startswith("application/json") and manifest.json()["job_id"] == job_id


def test_artifact_path_never_escapes(client: TestClient, job_id: str) -> None:
    for path in ("../../bookreader.db", "..%2F..%2Fbookreader.db", "chapters/..%2F..%2F..%2Fbookreader.db", "/etc/passwd"):
        response = client.get(f"/api/jobs/{job_id}/artifacts/{path}")
        assert response.status_code in (400, 404), path
        assert b"SQLite" not in response.content and b"root:" not in response.content
    assert client.get(f"/api/jobs/{job_id}/artifacts/does/not/exist.wav").status_code == 404


# --------------------------------------------------------------------------- events, usage, log
def test_events_pagination(client: TestClient, job_id: str) -> None:
    first = client.get(f"/api/jobs/{job_id}/events", params={"limit": 3}).json()
    assert len(first["events"]) == 3 and first["last_id"] == first["events"][-1]["id"]
    ids = [e["id"] for e in first["events"]]
    assert ids == sorted(ids)
    rest = client.get(f"/api/jobs/{job_id}/events", params={"after": first["last_id"]}).json()
    assert rest["events"] and all(e["id"] > first["last_id"] for e in rest["events"])
    messages = [e["message"] for e in first["events"] + rest["events"]]
    assert any(m == "stage ingest started" for m in messages) and any(m.startswith("run finished") for m in messages)
    tail = client.get(f"/api/jobs/{job_id}/events", params={"after": rest["last_id"]}).json()
    assert tail == {"events": [], "last_id": rest["last_id"]}


def test_usage(client: TestClient, job_id: str) -> None:
    usage = client.get(f"/api/jobs/{job_id}/usage").json()
    assert usage["calls"] > 0 and usage["cache_hits"] == 0 and usage["cost_usd"] == 0.0
    assert usage["events_count"] == usage["calls"] + usage["cache_hits"]
    assert usage["by_capability"]["tts"]["characters"] > 0


def test_log_tail(client: TestClient, job_id: str) -> None:
    response = client.get(f"/api/jobs/{job_id}/log", params={"lines": 5})
    assert response.status_code == 200 and response.headers["content-type"].startswith("text/plain")
    lines = response.text.strip().splitlines()
    assert len(lines) == 5 and job_id in response.text
    assert "run finished" in lines[-1]


# --------------------------------------------------------------------------- re-cast
def test_put_cast_recasts_only_that_character(client: TestClient, sample_book_path: Path) -> None:
    job = _upload(client, sample_book_path)["job_id"]
    before = client.get(f"/api/jobs/{job}/cast").json()["cast"]
    tobias_before = next(a for a in before["characters"] if a["character"] == "Tobias")["voice"]["id"]
    assert tobias_before != "mock-m-teen"
    usage_before = client.get(f"/api/jobs/{job}/usage").json()

    response = client.put(f"/api/jobs/{job}/cast", json={"overrides": {"tobias": "mock-m-teen"}})
    assert response.status_code == 202, response.text
    assert response.json()["job_id"] == job

    status = client.get(f"/api/jobs/{job}").json()
    assert status["status"] == "done", status["error"]
    after = client.get(f"/api/jobs/{job}/cast").json()["cast"]
    tobias = next(a for a in after["characters"] if a["character"] == "Tobias")
    assert tobias["voice"]["id"] == "mock-m-teen" and tobias["source"] == "override"
    assert after["narrator"]["voice"]["id"] == before["narrator"]["voice"]["id"]
    manifest = client.get(f"/api/jobs/{job}/manifest").json()
    assert manifest["warnings"] == []
    others = 0
    for chapter in manifest["chapters"]:
        for segment in chapter["segments"]:
            if segment["speaker"] == "Tobias":
                assert segment["voice_id"] == "mock-m-teen"
            else:
                others += 1
    usage = client.get(f"/api/jobs/{job}/usage").json()
    # Every other speaker's clip (narrator included) came from the cache; the new calls are
    # Tobias's four lines plus at most a handful of re-timed music beds.
    assert usage["cache_hits"] - usage_before["cache_hits"] >= others > 4
    assert 4 <= usage["calls"] - usage_before["calls"] < others
    stages = {s["stage"]: s for s in status["stages"]}
    assert stages["analyze"]["attempts"] == 1 and stages["render"]["attempts"] == 2
    assert stages["cast"]["attempts"] == 2


def test_put_cast_rejects_unknown_voice_and_character(client: TestClient, job_id: str) -> None:
    assert client.put(f"/api/jobs/{job_id}/cast", json={"overrides": {"Tobias": "no-such-voice"}}).status_code == 400
    assert client.put(f"/api/jobs/{job_id}/cast", json={"overrides": {"Nobody": "mock-m-teen"}}).status_code == 400
    assert client.put(f"/api/jobs/{job_id}/cast", json={"overrides": {"Tobias": "mock-m-teen"}, "extra": 1}).status_code == 422


# --------------------------------------------------------------------------- actions
def test_cancel_on_terminal_is_409(client: TestClient, job_id: str) -> None:
    assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 409


def test_retry_done_without_from_stage_is_409(client: TestClient, job_id: str) -> None:
    assert client.post(f"/api/jobs/{job_id}/retry").status_code == 409


def test_retry_from_finalize_rewrites_manifest(client: TestClient, sample_book_path: Path) -> None:
    job = _upload(client, sample_book_path)["job_id"]
    response = client.post(f"/api/jobs/{job}/retry", json={"from_stage": "finalize"})
    assert response.status_code == 202
    status = client.get(f"/api/jobs/{job}").json()
    assert status["status"] == "done"
    stages = {s["stage"]: s for s in status["stages"]}
    assert stages["finalize"]["attempts"] == 2 and stages["render"]["attempts"] == 1
    assert client.get(f"/api/jobs/{job}/manifest").status_code == 200


def test_delete_then_404(client: TestClient, sample_book_path: Path) -> None:
    job = _upload(client, sample_book_path)["job_id"]
    job_dir = client.app.state.settings.data_dir / "jobs" / job
    assert job_dir.is_dir()
    assert client.delete(f"/api/jobs/{job}").status_code == 204
    assert client.get(f"/api/jobs/{job}").status_code == 404
    assert client.delete(f"/api/jobs/{job}").status_code == 404
    assert not job_dir.exists()
    assert (client.app.state.settings.data_dir / "cache" / "tts").is_dir()


# --------------------------------------------------------------------------- upload guards
def test_upload_rejects_unsupported_extension(client: TestClient) -> None:
    response = client.post("/api/jobs", files={"file": ("virus.exe", b"MZ" * 100, "application/octet-stream")})
    assert response.status_code == 415


def test_upload_rejects_bad_options(client: TestClient, sample_book_path: Path) -> None:
    with sample_book_path.open("rb") as fh:
        response = client.post("/api/jobs", files={"file": ("book.txt", fh, "text/plain")}, data={"options": "{not json"})
    assert response.status_code == 400
    with sample_book_path.open("rb") as fh:
        response = client.post("/api/jobs", files={"file": ("book.txt", fh, "text/plain")}, data={"options": json.dumps({"bogus": 1})})
    assert response.status_code == 400


def test_upload_with_chapter_subset(client: TestClient, sample_book_path: Path) -> None:
    body = _upload(client, sample_book_path, options=json.dumps({"chapters": [2], "music": False, "sfx": False}))
    status = client.get(f"/api/jobs/{body['job_id']}").json()
    assert status["status"] == "done" and [c["index"] for c in status["chapters"]] == [2]
    assert status["options"] == {"chapters": [2], "music": False, "sfx": False, "cast_overrides": {}}
    manifest = client.get(f"/api/jobs/{body['job_id']}/manifest").json()
    assert len(manifest["chapters"]) == 1 and manifest["chapters"][0]["cues"] == []


def test_upload_too_large(tmp_path: Path, sample_book_path: Path) -> None:
    with TestClient(create_app(_settings(tmp_path, max_upload_mb=0))) as small:
        with sample_book_path.open("rb") as fh:
            response = small.post("/api/jobs", files={"file": ("book.txt", fh, "text/plain")})
        assert response.status_code == 413
        assert small.get("/api/jobs").json()["jobs"] == []


# --------------------------------------------------------------------------- health & index
def test_health(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("bookreader.api.routes.ffmpeg_path", lambda: None)
    body = client.get("/api/health").json()
    assert body["ok"] is True and body["version"] and body["ffmpeg"] is False
    assert [p["capability"] for p in body["providers"]] == ["analysis", "tts", "music", "sfx"]
    assert all(p["family"] == "mock" and p["ok"] for p in body["providers"])
    assert body["queue"] == {"backend": "inline", "depth": 0, "running": []}
    assert body["data_dir_writable"] is True


def test_index_serves_html(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200 and response.headers["content-type"].startswith("text/html")
    assert "bookreader" in response.text and "<script" in response.text


def test_provider_config_error_aborts_startup(tmp_path: Path) -> None:
    from bookreader.types import ProviderConfigError

    settings = _settings(tmp_path, tts_provider="elevenlabs")
    with pytest.raises(ProviderConfigError):
        with TestClient(create_app(settings)):
            pass
