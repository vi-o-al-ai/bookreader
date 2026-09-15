"""ElevenLabs adapters against a fake client tree. No SDK client is ever constructed for real."""
from __future__ import annotations

import sys
import types
from typing import Any

import httpx
import numpy as np
import pytest

from bookreader.providers.base import MusicGenerator, SfxGenerator, VoiceSynthesizer
from bookreader.providers.elevenlabs import client as el_client
from bookreader.providers.elevenlabs.music import ElevenLabsMusic
from bookreader.providers.elevenlabs.sfx import ElevenLabsSFX
from bookreader.providers.elevenlabs.tts import ElevenLabsTTS
from bookreader.retry import FamilyLimiter
from bookreader.settings import Settings
from bookreader.types import (
    SAMPLE_RATE,
    MusicRequest,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderTransientError,
    SfxRequest,
    TTSRequest,
    VoiceSettings,
)

# --------------------------------------------------------------------------- helpers


def sine_pcm(ms: int = 200, freq: float = 440.0) -> tuple[np.ndarray, bytes]:
    """A known int16 sine at the canonical rate and its little-endian byte form."""
    n = int(SAMPLE_RATE * ms / 1000)
    t = np.arange(n) / SAMPLE_RATE
    samples = (np.sin(2 * np.pi * freq * t) * 12000).astype("<i2")
    return samples, samples.tobytes()


def chunked(data: bytes, size: int = 1000):
    """The SDK returns an iterator of byte chunks; mimic that."""
    for i in range(0, len(data), size):
        yield data[i : i + size]


class Recorder:
    """Callable that records (args, kwargs) and returns/raises a scripted sequence of results."""

    def __init__(self, *results: Any) -> None:
        self.results = list(results)
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        result = self.results.pop(0) if len(self.results) > 1 else self.results[0]
        if isinstance(result, BaseException):
            raise result
        return result() if callable(result) else result


class FakeError(Exception):
    """Stand-in for the SDK's ApiError: carries status_code and headers."""

    def __init__(self, status_code: int, headers: dict[str, str] | None = None) -> None:
        super().__init__(f"http {status_code}")
        self.status_code = status_code
        self.headers = headers or {}


class UsageSpy:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, float, dict[str, Any]]] = []

    def record(self, capability: str, family: str, unit_type: str, units: float, **kw: Any) -> None:
        self.rows.append((capability, family, unit_type, units, kw))


def voice(voice_id: str, name: str, labels: dict[str, str] | None, **extra: Any) -> types.SimpleNamespace:
    return types.SimpleNamespace(voice_id=voice_id, name=name, labels=labels, description=extra.pop("description", None), **extra)


def page(voices: list[Any], has_more: bool, token: str | None) -> types.SimpleNamespace:
    return types.SimpleNamespace(voices=voices, has_more=has_more, next_page_token=token)


def fake_client(**parts: Any) -> types.SimpleNamespace:
    """Client tree: text_to_speech.convert, music.compose, text_to_sound_effects.convert, voices.search/get_all."""
    return types.SimpleNamespace(
        text_to_speech=types.SimpleNamespace(convert=parts.get("convert", Recorder(b""))),
        music=types.SimpleNamespace(compose=parts.get("compose", Recorder(b""))),
        text_to_sound_effects=types.SimpleNamespace(convert=parts.get("sfx", Recorder(b""))),
        voices=parts.get("voices", types.SimpleNamespace()),
    )


@pytest.fixture(autouse=True)
def _no_sleep_fresh_limiter(monkeypatch: pytest.MonkeyPatch):
    """Retries must not sleep, and the family semaphore must start fresh."""
    import bookreader.retry as retry_mod

    monkeypatch.setattr(retry_mod.time, "sleep", lambda _s: None)
    FamilyLimiter.reset()
    yield
    FamilyLimiter.reset()


# --------------------------------------------------------------------------- client helpers


def test_pcm_to_clip_decodes_int16_from_iterator_and_bytes():
    samples, raw = sine_pcm(50)
    clip = el_client.pcm_to_clip(chunked(raw, 7))
    assert clip.sample_rate == SAMPLE_RATE
    assert clip.samples.dtype == np.int16
    np.testing.assert_array_equal(clip.samples, samples)
    np.testing.assert_array_equal(el_client.pcm_to_clip(raw).samples, samples)


@pytest.mark.parametrize(
    "exc, expected, retry_after",
    [
        (FakeError(429, {"Retry-After": "3"}), ProviderTransientError, 3.0),
        (FakeError(408), ProviderTransientError, None),
        (FakeError(409), ProviderTransientError, None),
        (FakeError(500), ProviderTransientError, None),
        (FakeError(503, {"retry-after": "not-a-number"}), ProviderTransientError, None),
        (httpx.ConnectError("boom"), ProviderTransientError, None),
        (httpx.ReadTimeout("slow"), ProviderTransientError, None),
        (FakeError(400), ProviderPermanentError, None),
        (FakeError(401), ProviderPermanentError, None),
        (FakeError(404), ProviderPermanentError, None),
        (ValueError("garbage"), ProviderPermanentError, None),
    ],
)
def test_map_error_classifies_by_status_and_transport(exc, expected, retry_after):
    mapped = el_client.map_error(exc)
    assert type(mapped) is expected
    assert "elevenlabs" in str(mapped)
    if expected is ProviderTransientError:
        assert mapped.retry_after == retry_after


def test_map_error_keeps_bookreader_errors_and_uses_sdk_api_error_shape():
    original = ProviderPermanentError("mine")
    assert el_client.map_error(original) is original
    from elevenlabs.core.api_error import ApiError

    mapped = el_client.map_error(ApiError(status_code=429, headers={"Retry-After": "2"}, body="slow down"))
    assert isinstance(mapped, ProviderTransientError) and mapped.retry_after == 2.0
    assert isinstance(el_client.map_error(ApiError(status_code=422, body="bad")), ProviderPermanentError)


def test_guarded_call_retries_transient_and_fails_fast_on_permanent():
    flaky = Recorder(FakeError(503), FakeError(429), "ok")
    assert el_client.guarded_call(flaky, concurrency=2) == "ok"
    assert len(flaky.calls) == 3
    assert FamilyLimiter.width("elevenlabs") == 2

    broken = Recorder(FakeError(400))
    with pytest.raises(ProviderPermanentError):
        el_client.guarded_call(broken, concurrency=2)
    assert len(broken.calls) == 1

    always_down = Recorder(FakeError(500))
    with pytest.raises(ProviderTransientError):
        el_client.guarded_call(always_down, concurrency=2)
    assert len(always_down.calls) == 4  # with_retry default attempts


def test_check_and_make_client_name_the_extra_when_sdk_is_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "elevenlabs", None)
    monkeypatch.setitem(sys.modules, "elevenlabs.client", None)
    settings = Settings.from_env({"ELEVENLABS_API_KEY": "k"})
    for cls in (ElevenLabsTTS, ElevenLabsMusic, ElevenLabsSFX):
        with pytest.raises(ProviderConfigError) as exc:
            cls.check(settings)
        assert "bookreader[elevenlabs]" in str(exc.value)
    with pytest.raises(ProviderConfigError) as exc2:
        el_client.make_client("k")
    assert "bookreader[elevenlabs]" in str(exc2.value)


def test_check_passes_with_sdk_installed_and_from_settings_needs_key(monkeypatch: pytest.MonkeyPatch):
    assert ElevenLabsTTS.check(Settings.from_env({})) == []
    with pytest.raises(ProviderConfigError) as exc:
        ElevenLabsTTS.from_settings(Settings.from_env({}))
    assert "ELEVENLABS_API_KEY" in str(exc.value)

    built: list[dict[str, Any]] = []
    stub = types.ModuleType("elevenlabs.client")

    class ElevenLabs:  # noqa: D401 - fake SDK client class
        def __init__(self, **kw: Any) -> None:
            built.append(kw)

    stub.ElevenLabs = ElevenLabs  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "elevenlabs.client", stub)
    settings = Settings.from_env(
        {"ELEVENLABS_API_KEY": "secret", "BOOKREADER_ELEVENLABS_TTS_MODEL": "eleven_turbo_v2_5",
         "BOOKREADER_MAX_VOICES": "7", "BOOKREADER_CONCURRENCY": "3", "BOOKREADER_ELEVENLABS_SFX_PROMPT_INFLUENCE": "0.9"}
    )
    tts = ElevenLabsTTS.from_settings(settings)
    music = ElevenLabsMusic.from_settings(settings)
    sfx = ElevenLabsSFX.from_settings(settings)
    assert built == [{"api_key": "secret"}] * 3
    assert isinstance(tts.client, ElevenLabs)
    assert tts.model_id == tts.cache_version == "eleven_turbo_v2_5"
    assert tts.max_voices == 7 and tts.concurrency == music.concurrency == sfx.concurrency == 3
    assert sfx.prompt_influence == 0.9


# --------------------------------------------------------------------------- tts


def test_tts_synthesize_passes_kwargs_and_decodes_pcm():
    samples, raw = sine_pcm(120)
    convert = Recorder(lambda: chunked(raw))
    usage = UsageSpy()
    tts = ElevenLabsTTS(fake_client(convert=convert), model_id="eleven_multilingual_v2", usage=usage, concurrency=2)
    assert isinstance(tts, VoiceSynthesizer)
    assert tts.family == "elevenlabs" and tts.cache_version == "eleven_multilingual_v2" and tts.max_chars == 2500
    req = TTSRequest(
        text="The wind came off the sea.", voice_id="v-mara",
        settings=VoiceSettings(stability=0.3, similarity_boost=0.9, style=0.2, speed=1.1),
        previous_text="Before.", next_text="After.", seed=12345,
    )
    clip = tts.synthesize(req)

    assert clip.sample_rate == SAMPLE_RATE and clip.samples.dtype == np.int16
    np.testing.assert_array_equal(clip.samples, samples)
    (args, kw), = convert.calls
    assert args == ("v-mara",)
    assert kw["text"] == req.text
    assert kw["model_id"] == "eleven_multilingual_v2"
    assert kw["output_format"] == "pcm_22050"
    assert kw["previous_text"] == "Before." and kw["next_text"] == "After." and kw["seed"] == 12345
    vs = kw["voice_settings"]
    assert (vs.stability, vs.similarity_boost, vs.style, vs.speed) == (0.3, 0.9, 0.2, 1.1)
    assert usage.rows == [("tts", "elevenlabs", "characters", float(len(req.text)), usage.rows[0][4])]
    assert usage.rows[0][4]["meta"]["voice_id"] == "v-mara"
    assert FamilyLimiter.width("elevenlabs") == 2


def test_tts_synthesize_defaults_pass_none_context_and_rejects_oversized_text():
    convert = Recorder(lambda: chunked(sine_pcm(10)[1]))
    tts = ElevenLabsTTS(fake_client(convert=convert))
    tts.synthesize(TTSRequest(text="Hi.", voice_id="v"))
    kw = convert.calls[0][1]
    assert kw["previous_text"] is None and kw["next_text"] is None and kw["seed"] == 0
    with pytest.raises(ProviderPermanentError):
        tts.synthesize(TTSRequest(text="x" * 2501, voice_id="v"))
    assert len(convert.calls) == 1


def test_tts_maps_api_errors_and_records_usage_only_on_success():
    usage = UsageSpy()
    convert = Recorder(FakeError(429, {"retry-after": "1"}), FakeError(429), FakeError(429), FakeError(429))
    tts = ElevenLabsTTS(fake_client(convert=convert), usage=usage)
    with pytest.raises(ProviderTransientError) as exc:
        tts.synthesize(TTSRequest(text="Hi.", voice_id="v"))
    assert exc.value.retry_after is None  # the last attempt's error carried no Retry-After header
    assert len(convert.calls) == 4 and usage.rows == []

    permanent = ElevenLabsTTS(fake_client(convert=Recorder(FakeError(422))), usage=usage)
    with pytest.raises(ProviderPermanentError):
        permanent.synthesize(TTSRequest(text="Hi.", voice_id="v"))
    assert usage.rows == []


def test_list_voices_normalizes_labels_and_tolerates_none():
    voices = [
        voice("v1", "Mara", {"gender": "Female", "age": "middle_aged", "accent": "British", "use_case": "Narration"},
              description="warm and steady"),
        voice("v2", "Tobias", None),
        voice("v3", "Hetta", {"gender": "female", "age": "old", "descriptive": "raspy"}),
        voice("v4", "Ansel", {"gender": "male", "age": "young"}, use_case="Audiobook"),
        voice("v5", "Robot", {"gender": "neutral", "age": "ancient"}),
    ]
    search = Recorder(page(voices, False, None))
    tts = ElevenLabsTTS(fake_client(voices=types.SimpleNamespace(search=search)))
    out = {v.id: v for v in tts.list_voices()}

    assert list(out) == ["v1", "v2", "v3", "v4", "v5"]
    assert all(v.family == "elevenlabs" and v.sample_rate == SAMPLE_RATE for v in out.values())
    assert (out["v1"].gender, out["v1"].age) == ("female", "adult")
    assert out["v1"].tags == ["female", "middle_aged", "british", "narration"]
    assert out["v1"].description == "warm and steady"
    assert (out["v2"].gender, out["v2"].age, out["v2"].tags, out["v2"].description) == ("unknown", "unknown", [], "")
    assert (out["v3"].gender, out["v3"].age) == ("female", "elderly") and "raspy" in out["v3"].tags
    assert (out["v4"].gender, out["v4"].age) == ("male", "young_adult") and "audiobook" in out["v4"].tags
    assert (out["v5"].gender, out["v5"].age) == ("nonbinary", "unknown")
    assert search.calls[0][1] == {"page_size": 100, "next_page_token": None}


def test_list_voices_pages_until_max_voices():
    pages = [page([voice(f"p{i}-{j}", f"V{j}", {}) for j in range(100)], True, f"tok{i + 1}") for i in range(10)]
    search = Recorder(*pages)
    tts = ElevenLabsTTS(fake_client(voices=types.SimpleNamespace(search=search)), max_voices=250)
    out = tts.list_voices()
    assert len(out) == 250
    assert len(search.calls) == 3
    assert [c[1]["next_page_token"] for c in search.calls] == [None, "tok1", "tok2"]

    # a catalog smaller than the cap stops when has_more is false
    small = Recorder(page([voice("a", "A", {})], True, "t1"), page([voice("b", "B", {})], False, None))
    assert [v.id for v in ElevenLabsTTS(fake_client(voices=types.SimpleNamespace(search=small))).list_voices()] == ["a", "b"]
    assert len(small.calls) == 2


def test_list_voices_falls_back_to_get_all_and_warmup_fetches_once():
    get_all = Recorder(types.SimpleNamespace(voices=[voice("x", "X", None), voice("y", "Y", {"gender": "male"})]))
    tts = ElevenLabsTTS(fake_client(voices=types.SimpleNamespace(get_all=get_all)), max_voices=1)
    assert [v.id for v in tts.list_voices()] == ["x"]
    tts.warmup()
    assert len(get_all.calls) == 2


# --------------------------------------------------------------------------- music


def test_music_compose_clamps_and_passes_model_kwargs():
    samples, raw = sine_pcm(300, 220.0)
    compose = Recorder(lambda: chunked(raw))
    usage = UsageSpy()
    music = ElevenLabsMusic(fake_client(compose=compose), usage=usage, concurrency=1)
    assert isinstance(music, MusicGenerator)
    assert (music.family, music.cache_version, music.min_duration_ms, music.max_duration_ms) == ("elevenlabs", "music_v2", 3000, 600_000)

    clip = music.compose(MusicRequest(prompt="slow strings, tense", mood="tense", energy=0.7, duration_ms=1000))
    np.testing.assert_array_equal(clip.samples, samples)
    kw = compose.calls[0][1]
    assert kw == {"prompt": "slow strings, tense", "music_length_ms": 3000, "model_id": "music_v2",
                  "force_instrumental": True, "output_format": "pcm_22050"}
    music.compose(MusicRequest(prompt="p", duration_ms=10_000_000))
    assert compose.calls[1][1]["music_length_ms"] == 600_000
    music.compose(MusicRequest(prompt="p", duration_ms=45_000))
    assert compose.calls[2][1]["music_length_ms"] == 45_000
    assert [r[:3] for r in usage.rows] == [("music", "elevenlabs", "audio_seconds")] * 3
    assert usage.rows[0][3] == pytest.approx(0.3, abs=0.01)
    assert usage.rows[0][4]["meta"]["mood"] == "tense"


def test_music_errors_are_mapped():
    music = ElevenLabsMusic(fake_client(compose=Recorder(FakeError(401))))
    with pytest.raises(ProviderPermanentError):
        music.compose(MusicRequest(prompt="p"))
    flaky = Recorder(FakeError(502), lambda: chunked(sine_pcm(10)[1]))
    assert ElevenLabsMusic(fake_client(compose=flaky)).compose(MusicRequest(prompt="p")).duration_ms == 10
    assert len(flaky.calls) == 2


# --------------------------------------------------------------------------- sfx


def test_sfx_generate_clamps_duration_and_passes_kwargs():
    samples, raw = sine_pcm(80, 880.0)
    convert = Recorder(lambda: chunked(raw))
    usage = UsageSpy()
    sfx = ElevenLabsSFX(fake_client(sfx=convert), usage=usage, prompt_influence=0.5, concurrency=2)
    assert isinstance(sfx, SfxGenerator)
    assert (sfx.family, sfx.cache_version, sfx.max_duration_ms) == ("elevenlabs", "eleven_text_to_sound_v2", 30_000)

    clip = sfx.generate(SfxRequest(description="a single deep thunderclap", duration_ms=3000, loop=False))
    np.testing.assert_array_equal(clip.samples, samples)
    assert convert.calls[0][1] == {"text": "a single deep thunderclap", "duration_seconds": 3.0, "prompt_influence": 0.5,
                                   "loop": False, "model_id": "eleven_text_to_sound_v2", "output_format": "pcm_22050"}
    sfx.generate(SfxRequest(description="tick", duration_ms=100))
    assert convert.calls[1][1]["duration_seconds"] == 0.5
    sfx.generate(SfxRequest(description="rain", kind="ambient", duration_ms=120_000, loop=True))
    assert convert.calls[2][1]["duration_seconds"] == 30.0 and convert.calls[2][1]["loop"] is True
    assert [r[:3] for r in usage.rows] == [("sfx", "elevenlabs", "audio_seconds")] * 3
    assert usage.rows[0][3] == pytest.approx(0.08, abs=0.01)


def test_sfx_prompt_influence_is_clamped_and_errors_mapped():
    assert ElevenLabsSFX(fake_client(), prompt_influence=7).prompt_influence == 1.0
    assert ElevenLabsSFX(fake_client(), prompt_influence=-1).prompt_influence == 0.0
    sfx = ElevenLabsSFX(fake_client(sfx=Recorder(httpx.ConnectError("down"))))
    with pytest.raises(ProviderTransientError):
        sfx.generate(SfxRequest(description="x"))
    assert ElevenLabsMusic.check(Settings.from_env({})) == [] and ElevenLabsSFX.check(Settings.from_env({})) == []
    assert ElevenLabsMusic(fake_client()).warmup() is None and ElevenLabsSFX(fake_client()).warmup() is None
