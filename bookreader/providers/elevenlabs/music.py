"""bookreader.providers.elevenlabs.music - ElevenLabsMusic, mood beds through ``client.music.compose``."""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, ClassVar

from bookreader.providers.base import NullUsage, UsageSink
from bookreader.providers.elevenlabs.client import (
    FAMILY,
    NO_SDK_RETRIES,
    OUTPUT_FORMAT,
    api_key_from,
    check_sdk,
    guarded_call,
    make_client,
    pcm_to_clip,
)
from bookreader.types import AudioClip, MusicRequest

if TYPE_CHECKING:
    from bookreader.settings import Settings

log = logging.getLogger(__name__)

MUSIC_MODEL = "music_v2"


class ElevenLabsMusic:
    """Instrumental music generation (family ``elevenlabs``). Implements ``MusicGenerator``."""

    family: ClassVar[str] = FAMILY
    cache_version: str = MUSIC_MODEL
    min_duration_ms: int = 3000
    max_duration_ms: int = 600_000

    def __init__(self, client: Any, usage: UsageSink | None = None, concurrency: int = 4) -> None:
        self.client = client
        self.usage: UsageSink = usage or NullUsage()
        self.concurrency = max(1, int(concurrency))

    @classmethod
    def check(cls, settings: "Settings") -> list[str]:
        """The SDK must import; the key is validated by the registry."""
        return check_sdk()

    @classmethod
    def from_settings(cls, settings: "Settings", usage: UsageSink | None = None) -> "ElevenLabsMusic":
        return cls(make_client(api_key_from(settings)), usage=usage, concurrency=settings.concurrency)

    def warmup(self) -> None:
        """Nothing to preload: music is generated per request."""
        return None

    def compose(self, req: MusicRequest) -> AudioClip:
        """Compose ``req.prompt`` for ``duration_ms`` clamped to 3 s..600 s; records ``audio_seconds``."""
        length_ms = max(self.min_duration_ms, min(self.max_duration_ms, int(req.duration_ms)))
        started = time.monotonic()
        clip = guarded_call(
            lambda: pcm_to_clip(
                self.client.music.compose(
                    prompt=req.prompt,
                    music_length_ms=length_ms,
                    model_id=MUSIC_MODEL,
                    force_instrumental=True,
                    output_format=OUTPUT_FORMAT,
                    request_options=NO_SDK_RETRIES,
                )
            ),
            self.concurrency,
        )
        self.usage.record(
            "music", self.family, "audio_seconds", clip.duration_ms / 1000.0,
            duration_ms=int((time.monotonic() - started) * 1000), meta={"mood": req.mood, "prompt": req.prompt},
        )
        log.debug("elevenlabs music %s (%d ms requested) -> %d ms", req.mood, length_ms, clip.duration_ms)
        return clip
