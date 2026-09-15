"""bookreader.providers.elevenlabs.sfx - ElevenLabsSFX, sound effects through ``text_to_sound_effects``."""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, ClassVar

from bookreader.providers.base import NullUsage, UsageSink
from bookreader.providers.elevenlabs.client import FAMILY, OUTPUT_FORMAT, api_key_from, check_sdk, guarded_call, make_client, pcm_to_clip
from bookreader.types import AudioClip, SfxRequest

if TYPE_CHECKING:
    from bookreader.settings import Settings

log = logging.getLogger(__name__)

SFX_MODEL = "eleven_text_to_sound_v2"
MIN_SECONDS = 0.5
MAX_SECONDS = 30.0


class ElevenLabsSFX:
    """Text to sound effects (family ``elevenlabs``). Implements ``SfxGenerator``."""

    family: ClassVar[str] = FAMILY
    cache_version: str = SFX_MODEL
    max_duration_ms: int = 30_000

    def __init__(self, client: Any, usage: UsageSink | None = None, prompt_influence: float = 0.5, concurrency: int = 4) -> None:
        self.client = client
        self.usage: UsageSink = usage or NullUsage()
        self.prompt_influence = min(1.0, max(0.0, float(prompt_influence)))
        self.concurrency = max(1, int(concurrency))

    @classmethod
    def check(cls, settings: "Settings") -> list[str]:
        """The SDK must import; the key is validated by the registry."""
        return check_sdk()

    @classmethod
    def from_settings(cls, settings: "Settings", usage: UsageSink | None = None) -> "ElevenLabsSFX":
        return cls(
            make_client(api_key_from(settings)),
            usage=usage,
            prompt_influence=settings.elevenlabs_sfx_prompt_influence,
            concurrency=settings.concurrency,
        )

    def warmup(self) -> None:
        """Nothing to preload."""
        return None

    def generate(self, req: SfxRequest) -> AudioClip:
        """Generate ``req.description`` for ``duration_ms`` clamped to 0.5 s..30 s; records ``audio_seconds``."""
        seconds = max(MIN_SECONDS, min(MAX_SECONDS, req.duration_ms / 1000.0))
        started = time.monotonic()
        clip = guarded_call(
            lambda: pcm_to_clip(
                self.client.text_to_sound_effects.convert(
                    text=req.description,
                    duration_seconds=seconds,
                    prompt_influence=self.prompt_influence,
                    loop=req.loop,
                    model_id=SFX_MODEL,
                    output_format=OUTPUT_FORMAT,
                )
            ),
            self.concurrency,
        )
        self.usage.record(
            "sfx", self.family, "audio_seconds", clip.duration_ms / 1000.0,
            duration_ms=int((time.monotonic() - started) * 1000), meta={"description": req.description, "loop": req.loop},
        )
        log.debug("elevenlabs sfx %r loop=%s (%.1f s requested) -> %d ms", req.description, req.loop, seconds, clip.duration_ms)
        return clip
