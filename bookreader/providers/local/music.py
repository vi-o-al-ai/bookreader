"""bookreader.providers.local.music - MusicGenMusic, mood beds from MusicGen via transformers.

``transformers``/``torch`` are imported only inside ``check()``, the default loader and
``compose()`` so the mock family never loads them. Generation is serialized by a lock: one
MusicGen model per process, one generation at a time.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, ClassVar

from bookreader.providers.base import NullUsage, UsageSink
from bookreader.types import AudioClip, MusicRequest, ProviderConfigError

if TYPE_CHECKING:
    from bookreader.settings import Settings

log = logging.getLogger(__name__)

FAMILY = "local"
LOCAL_HINT = "pip install 'bookreader[local]'"
TOKENS_PER_SECOND = 50          # MusicGen's frame rate
GUIDANCE_SCALE = 3.0

Loader = Callable[[str], tuple[Any, Any]]     # model name -> (processor, model)


def _default_loader(model_name: str) -> tuple[Any, Any]:
    try:
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
    except ImportError as exc:
        raise ProviderConfigError(f"transformers/torch are not installed; {LOCAL_HINT}") from exc
    processor = AutoProcessor.from_pretrained(model_name)
    model = MusicgenForConditionalGeneration.from_pretrained(model_name)
    return processor, model


def _to_numpy(audio: Any) -> Any:
    """``tensor[0, 0]`` -> numpy; tolerates plain arrays for tests."""
    if hasattr(audio, "cpu"):
        audio = audio.cpu()
    return audio.numpy() if hasattr(audio, "numpy") else audio


class MusicGenMusic:
    """MusicGen text-to-music (family ``local``). Implements ``MusicGenerator``."""

    family: ClassVar[str] = FAMILY
    min_duration_ms: int = 1000
    max_duration_ms: int = 30_000

    def __init__(self, model_name: str = "facebook/musicgen-small", loader: Loader | None = None, usage: UsageSink | None = None) -> None:
        self.model_name = model_name
        self.cache_version: str = model_name
        self.loader: Loader = loader or _default_loader
        self.usage: UsageSink = usage or NullUsage()
        self._processor: Any = None
        self._model: Any = None
        self._lock = threading.Lock()

    @classmethod
    def check(cls, settings: "Settings") -> list[str]:
        """``transformers`` and ``torch`` must import; weights are only loaded by ``warmup``."""
        try:
            import torch  # noqa: F401 - presence check only
            import transformers  # noqa: F401
        except ImportError as exc:
            raise ProviderConfigError(f"transformers/torch are not installed; {LOCAL_HINT}") from exc
        return []

    @classmethod
    def from_settings(cls, settings: "Settings", usage: UsageSink | None = None) -> "MusicGenMusic":
        return cls(settings.musicgen_model, usage=usage)

    def warmup(self) -> None:
        """Load processor and weights so missing artifacts fail at boot."""
        self._ensure_loaded()
        log.info("musicgen warmup: %s loaded", self.model_name)

    def _ensure_loaded(self) -> tuple[Any, Any]:
        with self._lock:
            if self._model is None:
                log.info("loading musicgen model %s", self.model_name)
                self._processor, self._model = self.loader(self.model_name)
            return self._processor, self._model

    def compose(self, req: MusicRequest) -> AudioClip:
        """Generate ``req.prompt`` for ``duration_ms`` clamped to 1 s..30 s (one generation at a time)."""
        try:
            import torch
        except ImportError as exc:
            raise ProviderConfigError(f"torch is not installed; {LOCAL_HINT}") from exc
        duration_ms = max(self.min_duration_ms, min(self.max_duration_ms, int(req.duration_ms)))
        processor, model = self._ensure_loaded()
        started = time.monotonic()
        with self._lock:
            inputs = processor(text=[req.prompt], padding=True, return_tensors="pt")
            with torch.no_grad():
                audio = model.generate(
                    **inputs, do_sample=True, guidance_scale=GUIDANCE_SCALE,
                    max_new_tokens=int(duration_ms / 1000.0 * TOKENS_PER_SECOND),
                )
        rate = int(model.config.audio_encoder.sampling_rate)
        clip = AudioClip(_to_numpy(audio[0, 0]), rate)
        self.usage.record(
            "music", self.family, "audio_seconds", clip.duration_ms / 1000.0,
            duration_ms=int((time.monotonic() - started) * 1000), meta={"mood": req.mood, "model": self.model_name},
        )
        log.debug("musicgen %s (%d ms requested) -> %d ms @ %d Hz", req.mood, duration_ms, clip.duration_ms, rate)
        return clip
