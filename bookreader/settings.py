"""bookreader.settings - process configuration parsed from BOOKREADER_* environment variables.

Every field maps to ``BOOKREADER_<FIELD_NAME_UPPER>``. Secrets are collected separately from
any environment variable whose name ends in ``_API_KEY`` or ``_TOKEN`` and are never included
in ``snapshot()`` or ``repr``.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Mapping

ENV_PREFIX = "BOOKREADER_"
SECRET_SUFFIXES: tuple[str, ...] = ("_API_KEY", "_TOKEN")

DEFAULT_PRICES: dict[str, float] = {
    # USD per unit; keys are "<family>:<unit_type>"
    "anthropic:input_tokens": 5e-6,
    "anthropic:output_tokens": 25e-6,
    "anthropic:cache_read_input_tokens": 0.5e-6,
    "anthropic:cache_creation_input_tokens": 6.25e-6,
    "elevenlabs:characters": 0.0,
    "elevenlabs:audio_seconds": 0.0,
}

CHOICES: dict[str, tuple[str, ...]] = {
    "worker_mode": ("thread", "inline"),
    "anthropic_effort": ("low", "medium", "high", "xhigh", "max"),
    "local_tts_engine": ("piper", "kokoro"),
    "mp3": ("auto", "off"),
}

# Knobs that a retried job takes from the snapshot stored at creation so it renders consistently.
# Provider families, paths, concurrency and secrets always come from the current process.
JOB_SNAPSHOT_FIELDS: tuple[str, ...] = (
    "analysis_chunk_chars",
    "music_gain_db",
    "sfx_gain_db",
    "mp3",
    "max_chapter_minutes",
    "max_segments",
)


def _parse_bool(name: str, raw: str) -> bool:
    v = raw.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off", ""):
        return False
    raise ValueError(f"{name}: expected a boolean (1/0/true/false/yes/no/on/off), got {raw!r}")


def _parse_int(name: str, raw: str) -> int:
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name}: expected an integer, got {raw!r}") from exc


def _parse_float(name: str, raw: str) -> float:
    try:
        return float(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name}: expected a number, got {raw!r}") from exc


def _parse_json_dict(name: str, raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name}: expected a JSON object, got {raw!r}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name}: expected a JSON object, got {type(value).__name__}")
    return value


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("./data")

    # provider families, one per capability (see bookreader.providers.base.CAPABILITIES)
    analysis_provider: str = "mock"
    tts_provider: str = "mock"
    music_provider: str = "mock"
    sfx_provider: str = "mock"

    # worker
    worker_mode: str = "thread"
    workers: int = 1
    warmup: bool = True
    concurrency: int = 4

    # analysis
    analysis_chunk_chars: int = 6000
    anthropic_model: str = "claude-opus-5"
    anthropic_effort: str = "medium"
    anthropic_max_tokens: int = 32000

    # elevenlabs
    elevenlabs_tts_model: str = "eleven_multilingual_v2"
    elevenlabs_sfx_prompt_influence: float = 0.5
    max_voices: int = 300

    # local family
    local_tts_engine: str = "piper"
    piper_voices_dir: Path = Path("./voices")
    kokoro_lang: str = "a"
    musicgen_model: str = "facebook/musicgen-small"
    audiogen_model: str = "facebook/audiogen-medium"
    local_sfx_fallback: bool = True

    # mock family
    mock_ms_per_char: int = 45

    # rendering
    mp3: str = "auto"
    music_gain_db: float = -14.0
    sfx_gain_db: float = -8.0

    # guards
    max_upload_mb: int = 50
    max_chapter_minutes: int = 90
    max_segments: int = 20000
    cache_max_mb: int = 5000

    prices: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_PRICES))
    log_level: str = "INFO"

    # never serialized, never logged
    secrets: dict[str, str] = field(default_factory=dict, repr=False, compare=False)

    # ------------------------------------------------------------------ constants
    @property
    def sample_rate(self) -> int:
        return 22050

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if environ is None else environ
        values: dict[str, Any] = {}
        for f in fields(cls):
            if f.name == "secrets":
                continue
            var = ENV_PREFIX + f.name.upper()
            if var not in env:
                continue
            raw = env[var]
            values[f.name] = cls._parse_field(f.name, f.type, var, raw)
        if "prices" in values:
            merged = dict(DEFAULT_PRICES)
            merged.update({k: float(v) for k, v in values["prices"].items()})
            values["prices"] = merged
        values["secrets"] = {
            k: v for k, v in env.items() if any(k.endswith(suffix) for suffix in SECRET_SUFFIXES) and v
        }
        settings = cls(**values)
        settings._validate()
        return settings

    @staticmethod
    def _parse_field(name: str, type_name: Any, var: str, raw: str) -> Any:
        # dataclass field types are strings because of `from __future__ import annotations`
        t = type_name if isinstance(type_name, str) else getattr(type_name, "__name__", str(type_name))
        if t == "Path":
            return Path(raw.strip())
        if t == "bool":
            return _parse_bool(var, raw)
        if t == "int":
            return _parse_int(var, raw)
        if t == "float":
            return _parse_float(var, raw)
        if t.startswith("dict"):
            return _parse_json_dict(var, raw)
        return raw.strip()

    def _validate(self) -> None:
        for name, choices in CHOICES.items():
            value = getattr(self, name)
            if value not in choices:
                raise ValueError(f"{ENV_PREFIX}{name.upper()}: expected one of {', '.join(choices)}, got {value!r}")
        for name in ("workers", "concurrency", "analysis_chunk_chars", "anthropic_max_tokens", "max_voices", "mock_ms_per_char"):
            if getattr(self, name) < 1:
                raise ValueError(f"{ENV_PREFIX}{name.upper()}: must be >= 1")
        for name in ("max_upload_mb", "max_chapter_minutes", "max_segments", "cache_max_mb"):
            if getattr(self, name) < 0:
                raise ValueError(f"{ENV_PREFIX}{name.upper()}: must be >= 0")
        if not 0.0 <= self.elevenlabs_sfx_prompt_influence <= 1.0:
            raise ValueError(f"{ENV_PREFIX}ELEVENLABS_SFX_PROMPT_INFLUENCE: must be between 0 and 1")

    def with_overrides(self, **kw: Any) -> "Settings":
        """Return a copy with the given fields replaced (values already typed). Validates."""
        unknown = [k for k in kw if k not in {f.name for f in fields(self)}]
        if unknown:
            raise ValueError(f"unknown settings field(s): {', '.join(unknown)}")
        cleaned = dict(kw)
        for key in ("data_dir", "piper_voices_dir"):
            if key in cleaned and not isinstance(cleaned[key], Path):
                cleaned[key] = Path(cleaned[key])
        new = replace(self, **cleaned)
        new._validate()
        return new

    # ------------------------------------------------------------------ persistence
    def snapshot(self) -> str:
        """JSON of every non-secret field (paths as strings). Stored per job at creation."""
        data: dict[str, Any] = {}
        for f in fields(self):
            if f.name == "secrets":
                continue
            value = getattr(self, f.name)
            data[f.name] = str(value) if isinstance(value, Path) else value
        return json.dumps(data, sort_keys=True)

    @classmethod
    def from_snapshot(cls, text: str, secrets: Mapping[str, str] | None = None) -> "Settings":
        data = json.loads(text)
        known = {f.name for f in fields(cls)}
        values = {k: v for k, v in data.items() if k in known and k != "secrets"}
        for key in ("data_dir", "piper_voices_dir"):
            if key in values:
                values[key] = Path(values[key])
        settings = cls(**values, secrets=dict(secrets or {}))
        settings._validate()
        return settings

    def with_snapshot(self, text: str) -> "Settings":
        """Current process settings with the job-consistency knobs taken from a stored snapshot."""
        snap = self.from_snapshot(text, secrets=self.secrets)
        return self.with_overrides(**{name: getattr(snap, name) for name in JOB_SNAPSHOT_FIELDS})

    # ------------------------------------------------------------------ helpers
    def provider_families(self) -> dict[str, str]:
        return {
            "analysis": self.analysis_provider,
            "tts": self.tts_provider,
            "music": self.music_provider,
            "sfx": self.sfx_provider,
        }

    def price(self, family: str, unit_type: str) -> float:
        return float(self.prices.get(f"{family}:{unit_type}", 0.0))
