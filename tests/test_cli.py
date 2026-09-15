"""CLI tests: run / estimate / providers / status through ``bookreader.cli.main`` on the mock family."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from bookreader import cli
from bookreader.types import JobManifest

pytestmark = pytest.mark.timeout(40)


@pytest.fixture(autouse=True)
def _fast_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI reads the environment: keep mock pacing fast and skip warmup."""
    monkeypatch.setenv("BOOKREADER_MOCK_MS_PER_CHAR", "4")
    monkeypatch.setenv("BOOKREADER_WARMUP", "0")


# --------------------------------------------------------------------------- run
def test_run_one_chapter_writes_manifest(tmp_path: Path, sample_book_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["run", str(sample_book_path), "--out", str(tmp_path), "--chapters", "1", "--no-sfx"])
    out = capsys.readouterr().out
    assert code == 0, out
    manifests = list((tmp_path / "jobs").glob("*/manifest.json"))
    assert len(manifests) == 1
    assert str(manifests[0]) in out
    manifest = JobManifest.model_validate_json(manifests[0].read_text(encoding="utf-8"))
    assert [c.index for c in manifest.chapters] == [1] and manifest.title == "The Lighthouse at Gull Point"
    assert all(cue.kind == "music" for cue in manifest.chapters[0].cues)
    assert "[render]" in out and "[analyze]" in out                     # console progress lines
    assert (tmp_path / "bookreader.db").is_file() and (tmp_path / "cache" / "tts").is_dir()


def test_run_missing_file_is_an_input_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["run", str(tmp_path / "missing.txt"), "--out", str(tmp_path)])
    err = capsys.readouterr().err
    assert code != 0 and "InputError" in err and "missing.txt" in err
    assert not (tmp_path / "bookreader.db").exists()


def test_run_unsupported_file_fails_with_input_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "book.csv"
    bad.write_text("a,b\n" * 100, encoding="utf-8")
    code = cli.main(["run", str(bad), "--out", str(tmp_path / "out")])
    err = capsys.readouterr().err
    assert code == 1 and "input" in err and "unsupported file type" in err


def test_run_with_cast_override_and_from_stage(tmp_path: Path, sample_book_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    overrides = tmp_path / "cast.json"
    overrides.write_text(json.dumps({"Hetta": "mock-f-elderly-dry"}), encoding="utf-8")
    assert cli.main(["run", str(sample_book_path), "--out", str(tmp_path / "out"), "--chapters", "2", "--no-music", "--no-sfx", "--cast", str(overrides)]) == 0
    job_dir = next((tmp_path / "out" / "jobs").glob("*"))
    cast = json.loads((job_dir / "cast.json").read_text(encoding="utf-8"))
    hetta = next(a for a in cast["characters"] if a["character"] == "Hetta")
    assert hetta["voice"]["id"] == "mock-f-elderly-dry" and hetta["source"] == "override"
    capsys.readouterr()

    assert cli.main(["run", str(sample_book_path), "--out", str(tmp_path / "out"), "--from-stage", "finalize"]) == 0
    out = capsys.readouterr().out
    assert "re-running from stage finalize" in out and str(job_dir / "manifest.json") in out
    assert len(list((tmp_path / "out" / "jobs").glob("*"))) == 1                 # reused, not a new job

    assert cli.main(["run", str(sample_book_path), "--out", str(tmp_path / "fresh"), "--from-stage", "render"]) == 2
    assert "no previous job" in capsys.readouterr().err


def test_run_rejects_bad_provider_config(tmp_path: Path, sample_book_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["run", str(sample_book_path), "--out", str(tmp_path), "--tts", "elevenlabs"])
    err = capsys.readouterr().err
    assert code == 2 and "ProviderConfigError" in err and "ELEVENLABS_API_KEY" in err


# --------------------------------------------------------------------------- estimate
def test_estimate_prints_counts(sample_book_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["estimate", str(sample_book_path)]) == 0
    out = capsys.readouterr().out
    lines = {line.split()[0]: line for line in out.splitlines() if line.strip()}
    assert "chunks" in lines and lines["chunks"].split()[-1] == "3"
    assert lines["chapters"].split()[-1] == "3"
    assert "The Lighthouse at Gull Point" in out and "estimated cost" in out and "total" in out


def test_estimate_chapter_subset_and_missing_file(tmp_path: Path, sample_book_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["estimate", str(sample_book_path), "--chapters", "2-3"]) == 0
    out = capsys.readouterr().out
    assert any(line.startswith("chapters") and line.split()[-1] == "2" for line in out.splitlines())
    assert cli.main(["estimate", str(tmp_path / "nope.txt")]) == 2
    assert "InputError" in capsys.readouterr().err


# --------------------------------------------------------------------------- providers
def test_providers_ok_under_mock(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["providers"]) == 0
    out = capsys.readouterr().out
    assert out.count("mock") >= 4 and "ERROR" not in out and "MockTTS" in out


def test_providers_fails_without_elevenlabs_key(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("BOOKREADER_TTS_PROVIDER", "elevenlabs")
    assert cli.main(["providers"]) == 2
    captured = capsys.readouterr()
    assert "elevenlabs" in captured.out and "ELEVENLABS_API_KEY" in captured.err


def test_providers_flag_overrides_env(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["providers", "--analysis", "anthropic"]) == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


# --------------------------------------------------------------------------- status & parsing
def test_status_command(tmp_path: Path, sample_book_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["run", str(sample_book_path), "--out", str(tmp_path), "--chapters", "3", "--no-music", "--no-sfx"]) == 0
    capsys.readouterr()
    job_id = next((tmp_path / "jobs").glob("*")).name
    assert cli.main(["status", job_id, "--data-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "status: done" in out and "finalize" in out and "manifest:" in out
    assert cli.main(["status", "nope", "--data-dir", str(tmp_path)]) == 1
    assert cli.main(["status", "nope", "--data-dir", str(tmp_path / "empty")]) == 2


def test_parse_chapters() -> None:
    assert cli.parse_chapters(None) is None and cli.parse_chapters(" ") is None
    assert cli.parse_chapters("1-3,5") == [1, 2, 3, 5]
    assert cli.parse_chapters("2") == [2]
    with pytest.raises(Exception):
        cli.parse_chapters("3-1")


def test_bad_env_value_is_a_config_error(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("BOOKREADER_CONCURRENCY", "lots")
    assert cli.main(["providers"]) == 2
    assert "BOOKREADER_CONCURRENCY" in capsys.readouterr().err


def test_no_command_is_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([]) == 2
    assert "usage" in capsys.readouterr().err
