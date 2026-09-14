"""Settings parsing, validation, snapshots."""
from pathlib import Path

import pytest

from bookreader.settings import DEFAULT_PRICES, Settings


def test_defaults_are_all_mock():
    s = Settings.from_env({})
    assert s.provider_families() == {"analysis": "mock", "tts": "mock", "music": "mock", "sfx": "mock"}
    assert s.data_dir == Path("./data")
    assert s.sample_rate == 22050
    assert s.secrets == {}
    assert s.prices == DEFAULT_PRICES


def test_env_parsing_types_and_secrets():
    s = Settings.from_env(
        {
            "BOOKREADER_DATA_DIR": "/tmp/x",
            "BOOKREADER_TTS_PROVIDER": "elevenlabs",
            "BOOKREADER_WORKERS": "3",
            "BOOKREADER_WARMUP": "off",
            "BOOKREADER_MUSIC_GAIN_DB": "-10.5",
            "BOOKREADER_PRICES": '{"elevenlabs:characters": 0.0003}',
            "ELEVENLABS_API_KEY": "el-secret",
            "SOME_TOKEN": "tok",
            "UNRELATED": "no",
        }
    )
    assert s.data_dir == Path("/tmp/x")
    assert s.tts_provider == "elevenlabs"
    assert s.workers == 3
    assert s.warmup is False
    assert s.music_gain_db == -10.5
    assert s.prices["elevenlabs:characters"] == 0.0003
    assert s.prices["anthropic:input_tokens"] == DEFAULT_PRICES["anthropic:input_tokens"]
    assert s.secrets == {"ELEVENLABS_API_KEY": "el-secret", "SOME_TOKEN": "tok"}
    assert "el-secret" not in repr(s)
    assert "el-secret" not in s.snapshot()


@pytest.mark.parametrize(
    "var,value",
    [
        ("BOOKREADER_WORKERS", "many"),
        ("BOOKREADER_WARMUP", "maybe"),
        ("BOOKREADER_WORKER_MODE", "celery"),
        ("BOOKREADER_ANTHROPIC_EFFORT", "ultra"),
        ("BOOKREADER_PRICES", "[1,2]"),
        ("BOOKREADER_CONCURRENCY", "0"),
        ("BOOKREADER_ELEVENLABS_SFX_PROMPT_INFLUENCE", "1.5"),
    ],
)
def test_invalid_values_name_the_variable(var, value):
    with pytest.raises(ValueError) as exc:
        Settings.from_env({var: value})
    assert var in str(exc.value)


def test_with_overrides_and_snapshot_round_trip(tmp_path):
    s = Settings.from_env({"ANTHROPIC_API_KEY": "k"}).with_overrides(data_dir=tmp_path, mock_ms_per_char=4, worker_mode="inline")
    assert s.data_dir == tmp_path and s.mock_ms_per_char == 4
    snap = s.snapshot()
    back = Settings.from_snapshot(snap, secrets={"X_TOKEN": "y"})
    assert back.data_dir == tmp_path and back.worker_mode == "inline"
    assert back.secrets == {"X_TOKEN": "y"}
    with pytest.raises(ValueError):
        s.with_overrides(nonexistent=1)
    with pytest.raises(ValueError):
        s.with_overrides(worker_mode="nope")


def test_with_snapshot_keeps_current_providers_but_job_knobs():
    old = Settings.from_env({"BOOKREADER_MUSIC_GAIN_DB": "-20", "BOOKREADER_TTS_PROVIDER": "local"})
    current = Settings.from_env({"BOOKREADER_MUSIC_GAIN_DB": "-5", "BOOKREADER_TTS_PROVIDER": "mock", "X_API_KEY": "1"})
    merged = current.with_snapshot(old.snapshot())
    assert merged.music_gain_db == -20.0
    assert merged.tts_provider == "mock"
    assert merged.secrets == {"X_API_KEY": "1"}
