"""bookreader.providers.elevenlabs.tts - ElevenLabsTTS, text to speech through the ElevenLabs API."""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, ClassVar

from bookreader.providers.base import NullUsage, UsageSink
from bookreader.providers.elevenlabs.client import (
    FAMILY,
    INSTALL_HINT,
    NO_SDK_RETRIES,
    OUTPUT_FORMAT,
    api_key_from,
    check_sdk,
    guarded_call,
    make_client,
    pcm_to_clip,
)
from bookreader.types import AGES, GENDERS, SAMPLE_RATE, AudioClip, ProviderConfigError, ProviderPermanentError, TTSRequest, VoiceInfo

if TYPE_CHECKING:
    from bookreader.settings import Settings

log = logging.getLogger(__name__)

PAGE_SIZE = 100
GENDER_ALIASES: dict[str, str] = {"neutral": "nonbinary", "non-binary": "nonbinary", "non_binary": "nonbinary"}
AGE_ALIASES: dict[str, str] = {"young": "young_adult", "middle_aged": "adult", "middle-aged": "adult", "old": "elderly"}


def _norm(value: Any, choices: tuple[str, ...], aliases: dict[str, str]) -> str:
    """Lower-case an ElevenLabs label and map it onto one of *choices* (``unknown`` otherwise)."""
    key = str(value or "").strip().lower().replace(" ", "_")
    key = aliases.get(key, key)
    return key if key in choices else "unknown"


def voice_info(voice: Any) -> VoiceInfo:
    """Convert one SDK ``Voice`` object (``labels`` may be None) into a VoiceInfo."""
    labels = dict(getattr(voice, "labels", None) or {})
    tags: list[str] = []
    for raw in [*labels.values(), getattr(voice, "use_case", None), getattr(voice, "accent", None)]:
        tag = str(raw).strip().lower() if raw else ""
        if tag and tag not in tags:
            tags.append(tag)
    return VoiceInfo(
        id=str(voice.voice_id),
        name=str(getattr(voice, "name", None) or voice.voice_id),
        family=FAMILY,
        gender=_norm(labels.get("gender"), GENDERS, GENDER_ALIASES),  # type: ignore[arg-type]
        age=_norm(labels.get("age"), AGES, AGE_ALIASES),  # type: ignore[arg-type]
        tags=tags,
        description=str(getattr(voice, "description", None) or ""),
        sample_rate=SAMPLE_RATE,
        extra={"category": getattr(voice, "category", None), "labels": labels},
    )


class ElevenLabsTTS:
    """Text to speech through ``client.text_to_speech.convert`` (family ``elevenlabs``)."""

    family: ClassVar[str] = FAMILY
    max_chars: int = 2500

    def __init__(
        self,
        client: Any,
        model_id: str = "eleven_multilingual_v2",
        usage: UsageSink | None = None,
        max_voices: int = 300,
        concurrency: int = 4,
    ) -> None:
        self.client = client
        self.model_id = model_id
        self.cache_version: str = model_id
        self.usage: UsageSink = usage or NullUsage()
        self.max_voices = max(1, int(max_voices))
        self.concurrency = max(1, int(concurrency))

    @classmethod
    def check(cls, settings: "Settings") -> list[str]:
        """The SDK must import; the key is validated by the registry."""
        return check_sdk()

    @classmethod
    def from_settings(cls, settings: "Settings", usage: UsageSink | None = None) -> "ElevenLabsTTS":
        return cls(
            make_client(api_key_from(settings)),
            model_id=settings.elevenlabs_tts_model,
            usage=usage,
            max_voices=settings.max_voices,
            concurrency=settings.concurrency,
        )

    def warmup(self) -> None:
        """Fetch the catalog once so a bad key fails at startup rather than at job time."""
        log.info("elevenlabs tts warmup: %d voices", len(self.list_voices()))

    # ------------------------------------------------------------------ catalog
    def list_voices(self) -> list[VoiceInfo]:
        """Page through ``voices.search`` (falling back to ``voices.get_all``) up to ``max_voices``."""
        try:
            raw = self._search_pages()
        except AttributeError:
            raw = list(guarded_call(lambda: self.client.voices.get_all(request_options=NO_SDK_RETRIES).voices, self.concurrency))
        voices = [voice_info(v) for v in raw[: self.max_voices]]
        log.debug("elevenlabs catalog: %d voices", len(voices))
        return voices

    def _search_pages(self) -> list[Any]:
        search = self.client.voices.search
        out: list[Any] = []
        token: str | None = None
        while True:
            page = guarded_call(lambda: search(page_size=PAGE_SIZE, next_page_token=token, request_options=NO_SDK_RETRIES), self.concurrency)
            out.extend(page.voices or [])
            token = getattr(page, "next_page_token", None)
            if not getattr(page, "has_more", False) or not token or len(out) >= self.max_voices:
                return out

    # ------------------------------------------------------------------ synthesis
    def synthesize(self, req: TTSRequest) -> AudioClip:
        """Convert ``req.text`` with ``req.voice_id`` at ``pcm_22050``; records ``characters`` usage."""
        if len(req.text) > self.max_chars:
            raise ProviderPermanentError(f"elevenlabs tts accepts at most {self.max_chars} characters, got {len(req.text)}")
        try:
            from elevenlabs import VoiceSettings
        except ImportError as exc:
            raise ProviderConfigError(f"the elevenlabs SDK is not installed; {INSTALL_HINT}") from exc
        s = req.settings
        voice_settings = VoiceSettings(stability=s.stability, similarity_boost=s.similarity_boost, style=s.style, speed=s.speed)
        started = time.monotonic()
        clip = guarded_call(
            lambda: pcm_to_clip(
                self.client.text_to_speech.convert(
                    req.voice_id,
                    text=req.text,
                    model_id=self.model_id,
                    output_format=OUTPUT_FORMAT,
                    voice_settings=voice_settings,
                    previous_text=req.previous_text,
                    next_text=req.next_text,
                    seed=req.seed,
                    request_options=NO_SDK_RETRIES,
                )
            ),
            self.concurrency,
        )
        self.usage.record(
            "tts", self.family, "characters", float(len(req.text)),
            duration_ms=int((time.monotonic() - started) * 1000), meta={"voice_id": req.voice_id, "model_id": self.model_id},
        )
        log.debug("elevenlabs tts %s: %d chars -> %d ms", req.voice_id, len(req.text), clip.duration_ms)
        return clip
