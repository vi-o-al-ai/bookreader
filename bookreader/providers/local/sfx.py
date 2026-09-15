"""bookreader.providers.local.sfx - local sound effects: AudioGen when audiocraft is installed,
otherwise the procedural synthesizer under the ``local`` family name.

``audiocraft`` is not on PyPI's regular dependency path (manual install); with
``BOOKREADER_LOCAL_SFX_FALLBACK=1`` (default) its absence is a warning, not a startup failure.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, ClassVar

from bookreader.providers.base import NullUsage, UsageSink
from bookreader.providers.mock.sfx import ProceduralSfx
from bookreader.types import AudioClip, ProviderConfigError, SfxRequest

if TYPE_CHECKING:
    from bookreader.settings import Settings

log = logging.getLogger(__name__)

FAMILY = "local"
FALLBACK_WARNING = "audiocraft not installed; local sfx uses procedural synthesis"
AUDIOCRAFT_HINT = "install audiocraft manually (pip install audiocraft) or set BOOKREADER_LOCAL_SFX_FALLBACK=1"
MAX_SECONDS = 10.0

Loader = Callable[[str], Any]     # model name -> AudioGen model


def _audiocraft_available() -> bool:
    try:
        import audiocraft  # noqa: F401 - presence check only
    except ImportError:
        return False
    return True


def _default_loader(model_name: str) -> Any:
    try:
        from audiocraft.models import AudioGen
    except ImportError as exc:
        raise ProviderConfigError(f"audiocraft is not installed; {AUDIOCRAFT_HINT}") from exc
    return AudioGen.get_pretrained(model_name)


def _to_numpy(audio: Any) -> Any:
    if hasattr(audio, "cpu"):
        audio = audio.cpu()
    return audio.numpy() if hasattr(audio, "numpy") else audio


class ProceduralLocalSfx(ProceduralSfx):
    """The mock family's procedural effects, reported under the ``local`` family."""

    family: ClassVar[str] = FAMILY


class AudioGenSFX:
    """AudioGen text-to-sound (family ``local``). Implements ``SfxGenerator``."""

    family: ClassVar[str] = FAMILY
    max_duration_ms: int = int(MAX_SECONDS * 1000)

    def __init__(self, model_name: str = "facebook/audiogen-medium", loader: Loader | None = None, usage: UsageSink | None = None) -> None:
        self.model_name = model_name
        self.cache_version: str = model_name
        self.loader: Loader = loader or _default_loader
        self.usage: UsageSink = usage or NullUsage()
        self._model: Any = None
        self._lock = threading.Lock()

    @classmethod
    def check(cls, settings: "Settings") -> list[str]:
        """Same policy as :class:`LocalSFX` (this class is only ever built through it)."""
        return LocalSFX.check(settings)

    @classmethod
    def from_settings(cls, settings: "Settings", usage: UsageSink | None = None) -> "AudioGenSFX":
        return cls(settings.audiogen_model, usage=usage)

    def warmup(self) -> None:
        """Load the weights so missing artifacts fail at boot."""
        self._ensure_loaded()
        log.info("audiogen warmup: %s loaded", self.model_name)

    def _ensure_loaded(self) -> Any:
        with self._lock:
            if self._model is None:
                log.info("loading audiogen model %s", self.model_name)
                self._model = self.loader(self.model_name)
            return self._model

    def generate(self, req: SfxRequest) -> AudioClip:
        """Generate ``req.description`` for at most 10 s (one generation at a time)."""
        model = self._ensure_loaded()
        seconds = min(MAX_SECONDS, max(0.1, req.duration_ms / 1000.0))
        started = time.monotonic()
        with self._lock:
            model.set_generation_params(duration=seconds)
            wav = model.generate([req.description])
        clip = AudioClip(_to_numpy(wav[0, 0]), int(model.sample_rate))
        self.usage.record(
            "sfx", self.family, "audio_seconds", clip.duration_ms / 1000.0,
            duration_ms=int((time.monotonic() - started) * 1000), meta={"description": req.description, "model": self.model_name},
        )
        log.debug("audiogen %r (%.1f s requested) -> %d ms", req.description, seconds, clip.duration_ms)
        return clip


class LocalSFX:
    """Registry entry for ``sfx=local``: AudioGen when available, else procedural with a warning."""

    family: ClassVar[str] = FAMILY
    cache_version: str = "local"
    max_duration_ms: int = ProceduralSfx.max_duration_ms

    @classmethod
    def check(cls, settings: "Settings") -> list[str]:
        """Warn (or fail when the fallback is disabled) if audiocraft cannot be imported."""
        if _audiocraft_available():
            return []
        if settings.local_sfx_fallback:
            return [FALLBACK_WARNING]
        raise ProviderConfigError(f"audiocraft is not installed and BOOKREADER_LOCAL_SFX_FALLBACK is off; {AUDIOCRAFT_HINT}")

    @classmethod
    def from_settings(cls, settings: "Settings", usage: UsageSink | None = None) -> AudioGenSFX | ProceduralLocalSfx:
        """Build AudioGen or, when audiocraft is missing and the fallback is on, procedural sfx."""
        if _audiocraft_available():
            return AudioGenSFX.from_settings(settings, usage)
        if not settings.local_sfx_fallback:
            raise ProviderConfigError(f"audiocraft is not installed and BOOKREADER_LOCAL_SFX_FALLBACK is off; {AUDIOCRAFT_HINT}")
        log.warning(FALLBACK_WARNING)
        return ProceduralLocalSfx.from_settings(settings, usage)
