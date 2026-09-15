"""bookreader.providers.mock.tts - MockTTS, a deterministic procedural voice synthesizer.

Sixteen static voices, each with its own fundamental, brightness and tremolo so a listener can
tell characters apart. Audio comes from :func:`bookreader.providers.mock.synth.voice_burst_sequence`
seeded with ``stable_seed(voice_id, text, seed)``; no network, no model files, no keys.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar, NamedTuple

from bookreader.providers.base import NullUsage, UsageSink
from bookreader.providers.mock.synth import voice_burst_sequence
from bookreader.types import SAMPLE_RATE, AudioClip, ProviderPermanentError, TTSRequest, VoiceInfo, stable_seed

if TYPE_CHECKING:
    from bookreader.settings import Settings

log = logging.getLogger(__name__)


class VoiceParams(NamedTuple):
    """Synthesis parameters of one mock voice."""

    f0: float                 # fundamental in Hz (male 95-135, female 175-235, child 250-310; elderly -10 %)
    brightness: float         # 0..1, upper-partial and F2 emphasis
    tremolo: float            # 0..1 depth of the 6 Hz amplitude tremolo (elderly voices)
    gender: str
    age: str
    tags: tuple[str, ...]


VOICE_PARAMS: dict[str, VoiceParams] = {
    "mock-narrator-neutral": VoiceParams(150.0, 0.55, 0.00, "unknown", "adult", ("narration", "neutral", "clear")),
    "mock-f-adult-warm": VoiceParams(188.0, 0.45, 0.00, "female", "adult", ("warm", "soft", "narration")),
    "mock-f-adult-bright": VoiceParams(222.0, 0.85, 0.00, "female", "adult", ("bright", "clear")),
    "mock-f-elderly-dry": VoiceParams(176.0, 0.35, 0.06, "female", "elderly", ("dry", "elderly", "thin")),
    "mock-f-young-soft": VoiceParams(200.0, 0.40, 0.00, "female", "young_adult", ("soft", "gentle", "young")),
    "mock-m-adult-deep": VoiceParams(96.0, 0.40, 0.00, "male", "adult", ("deep", "calm", "narration")),
    "mock-m-adult-gruff": VoiceParams(112.0, 0.75, 0.00, "male", "adult", ("gruff", "rough")),
    "mock-m-elderly-rasp": VoiceParams(104.0, 0.65, 0.06, "male", "elderly", ("rasp", "elderly", "dry")),
    "mock-m-young-clear": VoiceParams(128.0, 0.70, 0.00, "male", "young_adult", ("clear", "young")),
    "mock-c-boy": VoiceParams(262.0, 0.60, 0.00, "male", "child", ("child", "boy", "bright")),
    "mock-c-girl": VoiceParams(296.0, 0.70, 0.00, "female", "child", ("child", "girl", "bright")),
    "mock-f-teen": VoiceParams(234.0, 0.65, 0.00, "female", "teen", ("teen", "bright", "young")),
    "mock-m-teen": VoiceParams(136.0, 0.60, 0.00, "male", "teen", ("teen", "clear", "young")),
    "mock-nb-adult": VoiceParams(160.0, 0.50, 0.00, "nonbinary", "adult", ("neutral", "calm")),
    "mock-f-adult-stern": VoiceParams(210.0, 0.30, 0.00, "female", "adult", ("stern", "firm", "dry")),
    "mock-m-adult-warm": VoiceParams(120.0, 0.50, 0.00, "male", "adult", ("warm", "soft", "narration")),
}


def _voice_info(voice_id: str, params: VoiceParams) -> VoiceInfo:
    name = voice_id.removeprefix("mock-").replace("-", " ").title()
    return VoiceInfo(
        id=voice_id,
        name=f"Mock {name}",
        family="mock",
        gender=params.gender,  # type: ignore[arg-type]
        age=params.age,  # type: ignore[arg-type]
        tags=list(params.tags),
        description=f"procedural voice, f0 {params.f0:.0f} Hz, {', '.join(params.tags)}",
        sample_rate=SAMPLE_RATE,
        extra={"f0": params.f0, "brightness": params.brightness, "tremolo": params.tremolo},
    )


VOICES: list[VoiceInfo] = [_voice_info(voice_id, params) for voice_id, params in VOICE_PARAMS.items()]


class MockTTS:
    """Deterministic procedural TTS (family ``mock``). Implements ``VoiceSynthesizer``."""

    family: ClassVar[str] = "mock"
    cache_version: str = "1"        # instances append the pacing knob: "1:<ms_per_char>"
    max_chars: int = 4000

    def __init__(self, ms_per_char: int = 45, usage: UsageSink | None = None) -> None:
        if ms_per_char < 1:
            raise ValueError("ms_per_char must be >= 1")
        self.ms_per_char = int(ms_per_char)
        # ms_per_char changes the rendered audio but is not part of TTSRequest, so it must be part
        # of the cache key; otherwise clips rendered at one pacing are reused for another.
        self.cache_version = f"{type(self).cache_version}:{self.ms_per_char}"
        self.usage: UsageSink = usage or NullUsage()

    @classmethod
    def check(cls, settings: "Settings") -> list[str]:
        """Nothing to validate: the mock family needs no SDK, key or model files."""
        return []

    @classmethod
    def from_settings(cls, settings: "Settings", usage: UsageSink | None = None) -> "MockTTS":
        return cls(ms_per_char=settings.mock_ms_per_char, usage=usage)

    def warmup(self) -> None:
        """No-op: there is nothing to load."""
        return None

    def list_voices(self) -> list[VoiceInfo]:
        """The 16 static mock voices (fresh copies so callers cannot mutate the catalog)."""
        return [v.model_copy(deep=True) for v in VOICES]

    def synthesize(self, req: TTSRequest) -> AudioClip:
        """Render *req* with the voice's parameters; pitch_shift is applied as a semitone factor."""
        params = VOICE_PARAMS.get(req.voice_id)
        if params is None:
            raise ProviderPermanentError(f"unknown mock voice id {req.voice_id!r}")
        if len(req.text) > self.max_chars:
            raise ProviderPermanentError(f"mock tts accepts at most {self.max_chars} characters, got {len(req.text)}")
        seed = stable_seed(req.voice_id, req.text, req.seed)
        f0 = params.f0 * (2.0 ** (req.settings.pitch_shift / 12.0))
        samples = voice_burst_sequence(
            text=req.text,
            f0=f0,
            brightness=params.brightness,
            tremolo=params.tremolo,
            ms_per_char=self.ms_per_char,
            speed=req.settings.speed,
            emotion=req.emotion,
            delivery=req.delivery,
            seed=seed,
        )
        clip = AudioClip(samples, SAMPLE_RATE)
        self.usage.record("tts", self.family, "characters", float(len(req.text)), meta={"voice_id": req.voice_id})
        log.debug("mock tts %s: %d chars -> %d ms", req.voice_id, len(req.text), clip.duration_ms)
        return clip
