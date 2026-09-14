"""Shared pytest fixtures. Tests never need API keys; everything runs on the mock family."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_BOOK = FIXTURES / "sample_book.txt"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip BOOKREADER_* and secret-looking variables so the host environment cannot leak into tests."""
    for name in list(os.environ):
        if name.startswith("BOOKREADER_") or name.endswith("_API_KEY") or name.endswith("_TOKEN"):
            monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="session")
def sample_book_path() -> Path:
    return SAMPLE_BOOK


@pytest.fixture
def mock_settings(tmp_path: Path):
    """All-mock settings pointed at a temp data dir; fast pacing; inline worker; no warmup."""
    from bookreader.settings import Settings

    return Settings.from_env({}).with_overrides(
        data_dir=tmp_path / "data",
        worker_mode="inline",
        mock_ms_per_char=4,
        warmup=False,
    )
