"""bookreader.providers.mock.sfx - ProceduralSfx, keyword-matched sound effects for the mock family.

The local family reuses this class (``ProceduralLocalSfx(ProceduralSfx)`` with ``family = "local"``)
as its fallback when audiocraft is not installed, so ``family`` is a plain overridable ClassVar.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from bookreader.providers.base import NullUsage, UsageSink
from bookreader.providers.mock.synth import sfx_recipe
from bookreader.types import SAMPLE_RATE, AudioClip, SfxRequest

if TYPE_CHECKING:
    from bookreader.settings import Settings

log = logging.getLogger(__name__)


class ProceduralSfx:
    """Deterministic procedural sound effects (family ``mock``). Implements ``SfxGenerator``."""

    family: ClassVar[str] = "mock"
    cache_version: str = "1"
    max_duration_ms: int = 120_000

    def __init__(self, usage: UsageSink | None = None) -> None:
        self.usage: UsageSink = usage or NullUsage()

    @classmethod
    def check(cls, settings: "Settings") -> list[str]:
        """Nothing to validate for procedural synthesis."""
        return []

    @classmethod
    def from_settings(cls, settings: "Settings", usage: UsageSink | None = None) -> "ProceduralSfx":
        return cls(usage=usage)

    def warmup(self) -> None:
        """No-op."""
        return None

    def generate(self, req: SfxRequest) -> AudioClip:
        """Render ``synth.sfx_recipe(description, duration, intensity, loop, seed)``; duration is
        clamped to 1..max_duration_ms as a safety net (the timeline already clamps)."""
        duration_ms = max(1, min(self.max_duration_ms, req.duration_ms))
        samples = sfx_recipe(req.description, duration_ms, req.intensity, req.loop, req.seed)
        clip = AudioClip(samples, SAMPLE_RATE)
        self.usage.record("sfx", self.family, "audio_seconds", duration_ms / 1000.0, meta={"description": req.description})
        log.debug("mock sfx %r loop=%s -> %d ms", req.description, req.loop, clip.duration_ms)
        return clip
