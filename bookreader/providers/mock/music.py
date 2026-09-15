"""bookreader.providers.mock.music - MockMusic, procedural mood beds for the mock family."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from bookreader.providers.base import NullUsage, UsageSink
from bookreader.providers.mock.synth import music_bed
from bookreader.types import SAMPLE_RATE, AudioClip, MusicRequest

if TYPE_CHECKING:
    from bookreader.settings import Settings

log = logging.getLogger(__name__)


class MockMusic:
    """Deterministic mood beds (family ``mock``). Implements ``MusicGenerator``."""

    family: ClassVar[str] = "mock"
    cache_version: str = "1"
    min_duration_ms: int = 500
    max_duration_ms: int = 3_600_000

    def __init__(self, usage: UsageSink | None = None) -> None:
        self.usage: UsageSink = usage or NullUsage()

    @classmethod
    def check(cls, settings: "Settings") -> list[str]:
        """Nothing to validate for the mock family."""
        return []

    @classmethod
    def from_settings(cls, settings: "Settings", usage: UsageSink | None = None) -> "MockMusic":
        return cls(usage=usage)

    def warmup(self) -> None:
        """No-op."""
        return None

    def compose(self, req: MusicRequest) -> AudioClip:
        """Render ``synth.music_bed(mood, energy, duration, seed)``; duration is clamped to the
        provider's range (the timeline already clamps, this is a safety net)."""
        duration_ms = max(self.min_duration_ms, min(self.max_duration_ms, req.duration_ms))
        samples = music_bed(req.mood, req.energy, duration_ms, req.seed)
        clip = AudioClip(samples, SAMPLE_RATE)
        self.usage.record("music", self.family, "audio_seconds", duration_ms / 1000.0, meta={"mood": req.mood})
        log.debug("mock music %s energy=%.2f -> %d ms", req.mood, req.energy, clip.duration_ms)
        return clip
