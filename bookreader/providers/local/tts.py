"""bookreader.providers.local.tts - local text to speech: Piper (ONNX voices) or Kokoro.

``LocalTTS`` is the registry entry point; it dispatches on ``BOOKREADER_LOCAL_TTS_ENGINE`` to
:class:`PiperTTS` or :class:`KokoroTTS`. Neither engine's package is imported at module level:
Piper/Kokoro are pulled in only inside ``check()``, ``from_settings()`` and the default loaders,
so the mock family never drags torch or onnxruntime into the process.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ClassVar

import numpy as np

from bookreader.providers.base import NullUsage, UsageSink
from bookreader.types import AGES, GENDERS, AudioClip, ProviderConfigError, ProviderPermanentError, TTSRequest, VoiceInfo

if TYPE_CHECKING:
    from bookreader.settings import Settings

log = logging.getLogger(__name__)

FAMILY = "local"
PIPER_HINT = "pip install 'bookreader[local]'"
KOKORO_HINT = "pip install 'bookreader[local-kokoro]'"
PIPER_DOWNLOAD_HINT = (
    "download voices from https://huggingface.co/rhasspy/piper-voices (each voice is a .onnx file "
    "plus its .onnx.json config) into BOOKREADER_PIPER_VOICES_DIR"
)
ESPEAK_ERROR = "kokoro needs the espeak-ng system package (apt-get install espeak-ng)"
SIDECAR_NAME = "voices.json"
SPEAKER_SEP = "#"
KOKORO_RATE = 24000
PIPER_NOISE_SCALE = 0.667
PIPER_NOISE_W = 0.8
MIN_SPEED = 0.25

# Kokoro's bundled English voices; the second letter of the id is the gender (f/m), the first the accent (a/b).
KOKORO_VOICES: tuple[str, ...] = (
    "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore", "af_nicole", "af_nova",
    "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
)
KOKORO_LANG_NAMES: dict[str, str] = {"a": "american", "b": "british"}


def _length_scale(speed: float) -> float:
    """Piper's ``length_scale`` is the inverse of the requested speed (guarded against 0)."""
    return 1.0 / max(MIN_SPEED, float(speed))


def _as_numpy(audio: Any) -> np.ndarray:
    """Torch tensors (``.detach().cpu().numpy()``), lists or arrays -> 1-d float32 numpy array."""
    if hasattr(audio, "detach"):
        audio = audio.detach()
    if hasattr(audio, "cpu"):
        audio = audio.cpu()
    if hasattr(audio, "numpy"):
        audio = audio.numpy()
    return np.asarray(audio, dtype=np.float32).reshape(-1)


# --------------------------------------------------------------------------- Piper
def _default_piper_loader(path: Path) -> Any:
    try:
        from piper import PiperVoice
    except ImportError as exc:
        raise ProviderConfigError(f"piper-tts is not installed; {PIPER_HINT}") from exc
    return PiperVoice.load(str(path))


def _piper_models(voices_dir: Path) -> list[Path]:
    """Every ``*.onnx`` with a sibling ``*.onnx.json``, sorted by name."""
    if not voices_dir.is_dir():
        return []
    return sorted(p for p in voices_dir.glob("*.onnx") if p.with_name(p.name + ".json").is_file())


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("cannot read %s: %s", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _speaker_map(config: dict[str, Any]) -> dict[str, int]:
    """``{speaker name: id}`` for multi-speaker models (``speaker_id_map`` with > 1 entry, or
    ``num_speakers`` > 1); empty for single-speaker models or malformed configs."""
    raw = config.get("speaker_id_map")
    if not isinstance(raw, dict) or not raw:
        return {}
    try:
        speakers = {str(name): int(speaker_id) for name, speaker_id in raw.items()}
    except (TypeError, ValueError):
        log.warning("ignoring malformed speaker_id_map: %r", raw)
        return {}
    if len(speakers) > 1 or int(config.get("num_speakers") or 1) > 1:
        return speakers
    return {}


def _choice(value: Any, choices: tuple[str, ...]) -> str:
    key = str(value or "").strip().lower()
    return key if key in choices else "unknown"


class PiperTTS:
    """Piper ONNX voices from a directory (family ``local``). Implements ``VoiceSynthesizer``.

    Voice ids are file stems (``en_US-lessac-medium``); multi-speaker models expose one id per
    speaker as ``stem#<speaker_id>``. Metadata comes from an optional ``voices.json`` sidecar
    (``{stem: {gender, age, tags, description}}``); otherwise gender/age are ``unknown`` and the
    tags are inferred from the stem (language, quality).
    """

    family: ClassVar[str] = FAMILY
    max_chars: int = 2000

    def __init__(self, voices_dir: Path | str, loader: Callable[[Path], Any] | None = None, usage: UsageSink | None = None) -> None:
        self.voices_dir = Path(voices_dir)
        self.loader: Callable[[Path], Any] = loader or _default_piper_loader
        self.usage: UsageSink = usage or NullUsage()
        self._voices: dict[str, Any] = {}
        self._lock = threading.Lock()
        stems = [p.name[: -len(".onnx")] for p in _piper_models(self.voices_dir)]
        digest = hashlib.sha256("\n".join(sorted(stems)).encode("utf-8")).hexdigest()[:12]
        self.cache_version: str = f"piper:{digest}"

    @classmethod
    def check(cls, settings: "Settings") -> list[str]:
        """``import piper`` must work and the voices dir must hold >= 1 model with its config."""
        try:
            import piper  # noqa: F401 - presence check only
        except ImportError as exc:
            raise ProviderConfigError(f"piper-tts is not installed; {PIPER_HINT}") from exc
        voices_dir = Path(settings.piper_voices_dir)
        if not voices_dir.is_dir():
            raise ProviderConfigError(f"piper voices directory {voices_dir} does not exist; {PIPER_DOWNLOAD_HINT}")
        models = _piper_models(voices_dir)
        if not models:
            raise ProviderConfigError(f"no *.onnx voice with a sibling .onnx.json found in {voices_dir}; {PIPER_DOWNLOAD_HINT}")
        orphans = [p.name for p in voices_dir.glob("*.onnx") if p not in models]
        return [f"piper voice {name} has no {name}.json config and is ignored" for name in sorted(orphans)]

    @classmethod
    def from_settings(cls, settings: "Settings", usage: UsageSink | None = None) -> "PiperTTS":
        return cls(settings.piper_voices_dir, usage=usage)

    def warmup(self) -> None:
        """Load every voice so a corrupt model fails at startup."""
        for model in _piper_models(self.voices_dir):
            self._voice(model.name[: -len(".onnx")])
        log.info("piper warmup: %d voices loaded from %s", len(self._voices), self.voices_dir)

    # ------------------------------------------------------------------ catalog
    def list_voices(self) -> list[VoiceInfo]:
        """One VoiceInfo per model (per speaker for multi-speaker models)."""
        sidecar = _read_json(self.voices_dir / SIDECAR_NAME) if (self.voices_dir / SIDECAR_NAME).is_file() else {}
        out: list[VoiceInfo] = []
        for model in _piper_models(self.voices_dir):
            stem = model.name[: -len(".onnx")]
            config = _read_json(model.with_name(model.name + ".json"))
            speaker_map = _speaker_map(config)
            if speaker_map:
                for speaker_name, speaker_id in sorted(speaker_map.items(), key=lambda kv: kv[1]):
                    voice_id = f"{stem}{SPEAKER_SEP}{speaker_id}"
                    out.append(self._voice_info(voice_id, stem, config, sidecar, speaker=str(speaker_name)))
            else:
                out.append(self._voice_info(stem, stem, config, sidecar))
        return out

    def _voice_info(self, voice_id: str, stem: str, config: dict[str, Any], sidecar: dict[str, Any], speaker: str | None = None) -> VoiceInfo:
        meta = sidecar.get(voice_id) or sidecar.get(stem) or {}
        meta = meta if isinstance(meta, dict) else {}
        parts = stem.split("-")
        tags = [str(t).lower() for t in (meta.get("tags") or [])] or [p.lower() for p in parts if p]
        if speaker and speaker.lower() not in tags:
            tags.append(speaker.lower())
        rate = config.get("audio", {}).get("sample_rate") if isinstance(config.get("audio"), dict) else None
        return VoiceInfo(
            id=voice_id,
            name=str(meta.get("name") or (f"{stem} ({speaker})" if speaker else stem)),
            family=self.family,
            gender=_choice(meta.get("gender"), GENDERS),  # type: ignore[arg-type]
            age=_choice(meta.get("age"), AGES),  # type: ignore[arg-type]
            tags=tags,
            description=str(meta.get("description") or ""),
            sample_rate=int(rate) if rate else None,
            extra={"engine": "piper", "model": stem, "speaker": speaker},
        )

    # ------------------------------------------------------------------ synthesis
    def _voice(self, stem: str) -> Any:
        with self._lock:
            voice = self._voices.get(stem)
            if voice is None:
                path = self.voices_dir / f"{stem}.onnx"
                if not path.is_file():
                    raise ProviderPermanentError(f"unknown piper voice {stem!r} (no {path})")
                log.info("loading piper voice %s", path)
                voice = self.loader(path)
                self._voices[stem] = voice
            return voice

    @staticmethod
    def split_voice_id(voice_id: str) -> tuple[str, int | None]:
        """``'stem#3'`` -> ``('stem', 3)``; plain stems have no speaker id."""
        stem, sep, speaker = voice_id.partition(SPEAKER_SEP)
        if not sep:
            return voice_id, None
        try:
            return stem, int(speaker)
        except ValueError as exc:
            raise ProviderPermanentError(f"invalid piper voice id {voice_id!r}: speaker id must be an integer") from exc

    def synthesize(self, req: TTSRequest) -> AudioClip:
        """Render with the memoized voice; supports both the streaming-raw and the chunk APIs."""
        if len(req.text) > self.max_chars:
            raise ProviderPermanentError(f"piper accepts at most {self.max_chars} characters, got {len(req.text)}")
        stem, speaker_id = self.split_voice_id(req.voice_id)
        voice = self._voice(stem)
        length_scale = _length_scale(req.settings.speed)
        rate = int(voice.config.sample_rate)
        if hasattr(voice, "synthesize_stream_raw"):
            data = b"".join(
                voice.synthesize_stream_raw(
                    req.text, speaker_id=speaker_id, length_scale=length_scale,
                    noise_scale=PIPER_NOISE_SCALE, noise_w=PIPER_NOISE_W,
                )
            )
            clip = AudioClip.from_pcm16_bytes(data, rate)
        else:
            clip = self._synthesize_chunks(voice, req.text, speaker_id, length_scale, rate)
        self.usage.record("tts", self.family, "characters", float(len(req.text)), meta={"voice_id": req.voice_id, "engine": "piper"})
        log.debug("piper %s: %d chars -> %d ms @ %d Hz", req.voice_id, len(req.text), clip.duration_ms, rate)
        return clip

    @staticmethod
    def _synthesize_chunks(voice: Any, text: str, speaker_id: int | None, length_scale: float, rate: int) -> AudioClip:
        try:
            from piper.config import SynthesisConfig
        except ImportError:
            try:
                from piper import SynthesisConfig
            except ImportError as exc:
                raise ProviderConfigError(f"piper-tts is not installed or too old; {PIPER_HINT}") from exc
        config = SynthesisConfig(
            speaker_id=speaker_id, length_scale=length_scale, noise_scale=PIPER_NOISE_SCALE, noise_w_scale=PIPER_NOISE_W,
        )
        pieces: list[np.ndarray] = []
        for chunk in voice.synthesize(text, config):
            raw = getattr(chunk, "audio_int16_bytes", None)
            if raw is not None:
                pieces.append(np.frombuffer(raw, dtype="<i2"))
            else:
                pieces.append(AudioClip(_as_numpy(chunk.audio_float_array), rate).samples)
        samples = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.int16)
        return AudioClip(samples, rate)


# --------------------------------------------------------------------------- Kokoro
def _default_kokoro_factory(lang: str) -> Any:
    try:
        from kokoro import KPipeline
    except ImportError as exc:
        raise ProviderConfigError(f"kokoro is not installed; {KOKORO_HINT}") from exc
    return KPipeline(lang_code=lang)


def kokoro_voice_info(voice_id: str) -> VoiceInfo:
    """Static metadata for a Kokoro voice id: accent from the first letter, gender from the second."""
    accent = KOKORO_LANG_NAMES.get(voice_id[:1], voice_id[:1])
    gender = {"f": "female", "m": "male"}.get(voice_id[1:2], "unknown")
    name = voice_id.split("_", 1)[-1].replace("_", " ").title()
    return VoiceInfo(
        id=voice_id,
        name=f"Kokoro {name}",
        family=FAMILY,
        gender=gender,  # type: ignore[arg-type]
        age="adult",
        tags=[accent, gender, "kokoro"],
        description=f"kokoro {accent} {gender} voice",
        sample_rate=KOKORO_RATE,
        extra={"engine": "kokoro", "lang": voice_id[:1]},
    )


class KokoroTTS:
    """Kokoro-82M through ``KPipeline`` (family ``local``). Implements ``VoiceSynthesizer``."""

    family: ClassVar[str] = FAMILY
    max_chars: int = 2000

    def __init__(self, pipeline_factory: Callable[[str], Any] | None = None, lang: str = "a", usage: UsageSink | None = None) -> None:
        self.lang = lang
        self.pipeline_factory: Callable[[str], Any] = pipeline_factory or _default_kokoro_factory
        self.usage: UsageSink = usage or NullUsage()
        self.cache_version: str = f"kokoro:{lang}"
        self._pipeline: Any = None
        self._lock = threading.Lock()

    @classmethod
    def check(cls, settings: "Settings") -> list[str]:
        """``import kokoro`` must work and ``espeak-ng`` must be on PATH."""
        try:
            import kokoro  # noqa: F401 - presence check only
        except ImportError as exc:
            raise ProviderConfigError(f"kokoro is not installed; {KOKORO_HINT}") from exc
        if shutil.which("espeak-ng") is None:
            raise ProviderConfigError(ESPEAK_ERROR)
        return []

    @classmethod
    def from_settings(cls, settings: "Settings", usage: UsageSink | None = None) -> "KokoroTTS":
        return cls(lang=settings.kokoro_lang, usage=usage)

    def warmup(self) -> None:
        """Build the pipeline (downloads/loads the weights) at startup."""
        self._get_pipeline()
        log.info("kokoro warmup: pipeline ready (lang=%s)", self.lang)

    def _get_pipeline(self) -> Any:
        with self._lock:
            if self._pipeline is None:
                log.info("loading kokoro pipeline lang=%s", self.lang)
                self._pipeline = self.pipeline_factory(self.lang)
            return self._pipeline

    def list_voices(self) -> list[VoiceInfo]:
        """The static Kokoro English catalog."""
        return [kokoro_voice_info(v) for v in KOKORO_VOICES]

    def synthesize(self, req: TTSRequest) -> AudioClip:
        """Concatenate every result's float32 audio at 24 kHz."""
        if len(req.text) > self.max_chars:
            raise ProviderPermanentError(f"kokoro accepts at most {self.max_chars} characters, got {len(req.text)}")
        pipeline = self._get_pipeline()
        pieces: list[np.ndarray] = []
        for result in pipeline(req.text, voice=req.voice_id, speed=req.settings.speed):
            audio = getattr(result, "audio", None)
            if audio is None and isinstance(result, tuple):
                audio = result[-1]
            if audio is not None:
                pieces.append(_as_numpy(audio))
        samples = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
        clip = AudioClip(samples, KOKORO_RATE)
        self.usage.record("tts", self.family, "characters", float(len(req.text)), meta={"voice_id": req.voice_id, "engine": "kokoro"})
        log.debug("kokoro %s: %d chars -> %d ms", req.voice_id, len(req.text), clip.duration_ms)
        return clip


# --------------------------------------------------------------------------- dispatch
ENGINES: dict[str, type[PiperTTS] | type[KokoroTTS]] = {"piper": PiperTTS, "kokoro": KokoroTTS}


class LocalTTS:
    """Registry entry for ``tts=local``: dispatches on ``settings.local_tts_engine``."""

    family: ClassVar[str] = FAMILY
    cache_version: str = "local"
    max_chars: int = 2000

    @staticmethod
    def engine_class(settings: "Settings") -> type[PiperTTS] | type[KokoroTTS]:
        """The engine selected by ``BOOKREADER_LOCAL_TTS_ENGINE``."""
        engine = ENGINES.get(str(settings.local_tts_engine))
        if engine is None:
            raise ProviderConfigError(
                f"BOOKREADER_LOCAL_TTS_ENGINE must be one of {', '.join(ENGINES)}, got {settings.local_tts_engine!r}"
            )
        return engine

    @classmethod
    def check(cls, settings: "Settings") -> list[str]:
        """Delegate to the selected engine."""
        return cls.engine_class(settings).check(settings)

    @classmethod
    def from_settings(cls, settings: "Settings", usage: UsageSink | None = None) -> PiperTTS | KokoroTTS:
        """Build the selected engine (Piper or Kokoro)."""
        return cls.engine_class(settings).from_settings(settings, usage)
