"""Tests for the mock audio family: procedural synth, MockTTS, MockMusic, ProceduralSfx.

Everything is offline and deterministic; the assertions are about audible properties
(FFT peaks, RMS, spectral centroid, envelopes) rather than exact sample values, except for the
cross-process determinism check which compares sha256 digests.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from bookreader.providers.base import MusicGenerator, NullUsage, SfxGenerator, VoiceSynthesizer
from bookreader.providers.mock import synth
from bookreader.providers.mock.music import MockMusic
from bookreader.providers.mock.sfx import ProceduralSfx
from bookreader.providers.mock.tts import VOICE_PARAMS, VOICES, MockTTS
from bookreader.settings import Settings
from bookreader.types import (
    MOODS,
    SAMPLE_RATE,
    AudioClip,
    MusicRequest,
    ProviderPermanentError,
    SfxRequest,
    TTSRequest,
    VoiceSettings,
)

REPO = Path(__file__).resolve().parents[1]
SENTENCE = (
    "The wind came off the sea like a living thing, and the old lighthouse groaned on its "
    "foundations. Mara Quill climbed the iron stairs two at a time, her lantern swinging wild "
    "shadows across the stone."
)
FIXED_REQUEST = dict(text=SENTENCE, voice_id="mock-m-adult-deep", emotion="tense", delivery="normal", seed=1234)

DETERMINISM_SCRIPT = """
import hashlib
from bookreader.providers.mock.tts import MockTTS
from bookreader.types import TTSRequest
clip = MockTTS(ms_per_char=4).synthesize(TTSRequest(**{req!r}))
print(hashlib.sha256(clip.samples.tobytes()).hexdigest())
"""


def _spectrum(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mag = np.abs(np.fft.rfft(x.astype(np.float64)))
    return np.fft.rfftfreq(len(x), 1.0 / SAMPLE_RATE), mag


def _peak_hz(x: np.ndarray) -> float:
    freqs, mag = _spectrum(x)
    return float(freqs[int(np.argmax(mag))])


def _centroid_hz(x: np.ndarray) -> float:
    freqs, mag = _spectrum(x)
    return float(np.sum(freqs * mag) / np.sum(mag))


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def _assert_canonical(clip: AudioClip) -> None:
    assert clip.sample_rate == SAMPLE_RATE
    assert clip.samples.dtype == np.int16
    assert clip.samples.ndim == 1
    assert len(clip.samples) > 0


@pytest.fixture
def tts() -> MockTTS:
    return MockTTS(ms_per_char=4)


# --------------------------------------------------------------------------- determinism
def test_tts_is_deterministic_across_processes(tts: MockTTS) -> None:
    local = hashlib.sha256(tts.synthesize(TTSRequest(**FIXED_REQUEST)).samples.tobytes()).hexdigest()
    again = hashlib.sha256(tts.synthesize(TTSRequest(**FIXED_REQUEST)).samples.tobytes()).hexdigest()
    assert local == again
    result = subprocess.run(
        [sys.executable, "-c", DETERMINISM_SCRIPT.format(req=FIXED_REQUEST)],
        cwd=REPO, capture_output=True, text=True, timeout=30, check=True,
    )
    assert result.stdout.strip() == local


def test_music_and_sfx_are_deterministic() -> None:
    a = synth.music_bed("hopeful", 0.6, 3000, 42)
    b = synth.music_bed("hopeful", 0.6, 3000, 42)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, synth.music_bed("hopeful", 0.6, 3000, 43))
    x = synth.sfx_recipe("wind howling", 1500, 0.5, True, 7)
    y = synth.sfx_recipe("wind howling", 1500, 0.5, True, 7)
    assert np.array_equal(x, y)
    assert not np.array_equal(x, synth.sfx_recipe("wind howling", 1500, 0.5, True, 8))


# --------------------------------------------------------------------------- voices
def test_catalog_has_sixteen_distinct_voices(tts: MockTTS) -> None:
    voices = tts.list_voices()
    assert len(voices) == 16
    ids = [v.id for v in voices]
    assert ids == list(VOICE_PARAMS)
    assert len(set(ids)) == 16
    assert all(v.family == "mock" for v in voices)
    assert all(v.tags for v in voices)
    assert {"mock-narrator-neutral", "mock-c-boy", "mock-f-adult-stern", "mock-m-adult-warm"} <= set(ids)
    assert any("narration" in v.tags for v in voices)
    voices[0].tags.append("mutated")
    assert "mutated" not in VOICES[0].tags


def test_each_voice_has_a_distinct_dominant_f0_and_groups_are_ordered(tts: MockTTS) -> None:
    peaks: dict[str, float] = {}
    for voice in tts.list_voices():
        clip = tts.synthesize(TTSRequest(text=SENTENCE * 2, voice_id=voice.id))
        _assert_canonical(clip)
        peaks[voice.id] = _peak_hz(clip.samples)
    ordered = sorted(peaks.values())
    assert all(b - a >= 3.0 for a, b in zip(ordered, ordered[1:])), peaks
    for voice in VOICES:
        assert abs(peaks[voice.id] - VOICE_PARAMS[voice.id].f0) < 12.0, (voice.id, peaks[voice.id])
    child = [peaks[v.id] for v in VOICES if v.age == "child"]
    female = [peaks[v.id] for v in VOICES if v.gender == "female" and v.age != "child"]
    male = [peaks[v.id] for v in VOICES if v.gender == "male" and v.age != "child"]
    assert child and female and male
    assert min(child) > max(female) > min(female) > max(male)


def test_ms_per_char_scales_duration_linearly() -> None:
    req = TTSRequest(text=SENTENCE, voice_id="mock-f-adult-warm")
    slow = MockTTS(ms_per_char=8).synthesize(req).samples
    fast = MockTTS(ms_per_char=4).synthesize(req).samples
    assert len(slow) == 2 * len(fast)
    assert len(fast) == round(len(SENTENCE) * 4 * SAMPLE_RATE / 1000)
    tiny = MockTTS(ms_per_char=4).synthesize(TTSRequest(text="Hi.", voice_id="mock-f-adult-warm"))
    assert tiny.duration_ms == 300                      # floor of 300 ms


def test_speed_and_emotion_rate_change_duration(tts: MockTTS) -> None:
    base = tts.synthesize(TTSRequest(text=SENTENCE, voice_id="mock-m-teen")).samples
    double = tts.synthesize(TTSRequest(text=SENTENCE, voice_id="mock-m-teen", settings=VoiceSettings(speed=2.0))).samples
    assert abs(len(double) - len(base) / 2) <= 1
    urgent = tts.synthesize(TTSRequest(text=SENTENCE, voice_id="mock-m-teen", emotion="urgent")).samples
    weary = tts.synthesize(TTSRequest(text=SENTENCE, voice_id="mock-m-teen", emotion="weary")).samples
    assert len(urgent) < len(base) < len(weary)


def test_whisper_has_lower_harmonic_peak_ratio_than_normal(tts: MockTTS) -> None:
    normal = tts.synthesize(TTSRequest(text=SENTENCE, voice_id="mock-f-adult-bright", delivery="normal")).samples
    whisper = tts.synthesize(TTSRequest(text=SENTENCE, voice_id="mock-f-adult-bright", delivery="whisper")).samples
    _, mag_n = _spectrum(normal)
    _, mag_w = _spectrum(whisper)
    ratio_normal = mag_n.max() / mag_n.mean()
    ratio_whisper = mag_w.max() / mag_w.mean()
    assert ratio_whisper < ratio_normal / 3
    assert _rms(whisper) < _rms(normal)                  # -12 dB


def test_delivery_levels_and_pitch_shift(tts: MockTTS) -> None:
    base = TTSRequest(text=SENTENCE, voice_id="mock-m-adult-warm")
    normal = tts.synthesize(base).samples
    shout = tts.synthesize(base.model_copy(update={"delivery": "shout"})).samples
    quiet = tts.synthesize(base.model_copy(update={"delivery": "quiet"})).samples
    assert _rms(shout) > _rms(normal) > _rms(quiet)
    assert _peak_hz(shout) > _peak_hz(normal) * 1.1     # +15 % f0
    up = tts.synthesize(base.model_copy(update={"settings": VoiceSettings(pitch_shift=12.0)})).samples
    assert abs(_peak_hz(up) / _peak_hz(normal) - 2.0) < 0.1
    strained = tts.synthesize(base.model_copy(update={"delivery": "strained"})).samples
    assert len(strained) == len(normal) and not np.array_equal(strained, normal)


def test_tts_provider_contract(mock_settings: Settings) -> None:
    provider = MockTTS.from_settings(mock_settings, NullUsage())
    assert isinstance(provider, VoiceSynthesizer)
    assert provider.ms_per_char == mock_settings.mock_ms_per_char == 4
    assert MockTTS.family == "mock" and provider.cache_version == "1" and provider.max_chars == 4000
    assert MockTTS.check(mock_settings) == []
    assert provider.warmup() is None
    default = MockTTS.from_settings(Settings.from_env({}))
    assert default.ms_per_char == 45
    with pytest.raises(ProviderPermanentError):
        provider.synthesize(TTSRequest(text="hello", voice_id="not-a-voice"))
    with pytest.raises(ProviderPermanentError):
        provider.synthesize(TTSRequest(text="x" * 4001, voice_id="mock-c-boy"))
    empty = provider.synthesize(TTSRequest(text="", voice_id="mock-c-boy"))
    _assert_canonical(empty)
    assert empty.duration_ms == 300


def test_synthesis_is_fast(tts: MockTTS) -> None:
    text = (SENTENCE + " ") * 3
    text = text[:500]
    assert len(text) == 500
    req = TTSRequest(text=text, voice_id="mock-narrator-neutral")
    tts.synthesize(req)                                  # warm caches / imports
    best = min(_timed(tts, req) for _ in range(5))
    assert best < 0.020, f"500-char synthesis took {best * 1000:.1f} ms"


def _timed(tts: MockTTS, req: TTSRequest) -> float:
    start = time.perf_counter()
    tts.synthesize(req)
    return time.perf_counter() - start


# --------------------------------------------------------------------------- music
def test_every_mood_bed_has_exact_length_and_moods_differ() -> None:
    duration_ms = 4000
    expected = round(duration_ms * SAMPLE_RATE / 1000)
    centroids: dict[str, float] = {}
    for mood in MOODS:
        bed = synth.music_bed(mood, 0.5, duration_ms, 11)
        assert bed.dtype == np.int16 and bed.ndim == 1
        assert len(bed) == expected, mood
        if mood == "none":
            assert not bed.any()
            continue
        assert bed.any(), mood
        assert bed[0] == 0 and bed[-1] == 0                          # 200 ms edge fades
        centroids[mood] = _centroid_hz(bed)
    ordered = sorted(centroids.values())
    assert all(b - a >= 5.0 for a, b in zip(ordered, ordered[1:])), centroids


def test_tense_bed_is_louder_than_calm_bed_at_same_energy() -> None:
    tense = synth.music_bed("tense", 0.5, 5000, 3)
    calm = synth.music_bed("calm", 0.5, 5000, 3)
    assert _rms(tense) > _rms(calm) * 1.2


def test_energy_scales_music_level() -> None:
    soft = synth.music_bed("warm", 0.1, 3000, 5)
    loud = synth.music_bed("warm", 0.9, 3000, 5)
    assert _rms(loud) > _rms(soft)


def test_music_odd_lengths_and_unknown_mood() -> None:
    assert len(synth.music_bed("joyful", 0.7, 733, 1)) == round(733 * SAMPLE_RATE / 1000)
    assert len(synth.music_bed("nonsense", 0.5, 500, 1)) == round(500 * SAMPLE_RATE / 1000)


def test_music_provider_contract(mock_settings: Settings) -> None:
    provider = MockMusic.from_settings(mock_settings, NullUsage())
    assert isinstance(provider, MusicGenerator)
    assert MockMusic.family == "mock" and provider.cache_version == "1"
    assert provider.min_duration_ms == 500 and provider.max_duration_ms == 3_600_000
    assert MockMusic.check(mock_settings) == [] and provider.warmup() is None
    clip = provider.compose(MusicRequest(prompt="", mood="ominous", energy=0.4, duration_ms=2500, seed=9))
    _assert_canonical(clip)
    assert clip.duration_ms == 2500
    assert np.array_equal(clip.samples, synth.music_bed("ominous", 0.4, 2500, 9))
    assert provider.compose(MusicRequest(prompt="", mood="calm", duration_ms=10)).duration_ms == 500


# --------------------------------------------------------------------------- sfx
def test_thunder_energy_is_concentrated_below_400_hz() -> None:
    thunder = synth.sfx_recipe("a single deep thunderclap, distant rumble", 3000, 0.9, False, 21)
    freqs, mag = _spectrum(thunder)
    energy = mag ** 2
    assert energy[freqs < 400.0].sum() / energy.sum() > 0.9


def test_bell_has_a_decaying_envelope() -> None:
    bell = synth.sfx_recipe("the great bronze bell rang once", 3000, 0.8, False, 4).astype(np.float64)
    tenth = len(bell) // 10
    assert _rms(bell[-tenth:]) < 0.1 * _rms(bell[:tenth])


def test_loopable_recipes_end_near_a_zero_crossing() -> None:
    for desc in ("wind howling", "a crackling fire", "rain on the roof", "waves on the shore", "gulls crying"):
        clip = synth.sfx_recipe(desc, 2000, 0.6, True, 3).astype(np.int64)
        tail = clip[-200:]
        crossings = np.nonzero(tail[:-1] * tail[1:] <= 0)[0]
        assert len(crossings) > 0, desc
        assert clip[-1] == 0, desc
        assert clip[:200].any(), desc


def test_recipe_keyword_matching() -> None:
    assert synth.sfx_recipe_name("Thunder split the sky") == "thunder"
    assert synth.sfx_recipe_name("a shutter slammed against the wall, again and again") == "slam"
    assert synth.sfx_recipe_name("a knock at the door") == "slam"
    assert synth.sfx_recipe_name("the rope parted with a snap") == "snap"
    assert synth.sfx_recipe_name("gulls cried over the rocks") == "gulls"
    assert synth.sfx_recipe_name("the kettle whistling") == "kettle"
    assert synth.sfx_recipe_name("a man whistling a tune") == "whistle"
    assert synth.sfx_recipe_name("footsteps on gravel") == "footsteps"
    assert synth.sfx_recipe_name("knitting needles clicking") == "clicks"
    assert synth.sfx_recipe_name("horse hooves on the road") == "hooves"
    assert synth.sfx_recipe_name("an old man coughing") == "cough"
    assert synth.sfx_recipe_name("a purple elephant") == "default"


def test_unknown_description_falls_back_to_default() -> None:
    clip = synth.sfx_recipe("a purple elephant humming", 1200, 0.5, False, 6)
    assert clip.dtype == np.int16 and len(clip) == round(1200 * SAMPLE_RATE / 1000)
    assert clip.any()
    assert np.array_equal(clip, synth.sfx_recipe("zorblat", 1200, 0.5, False, 6))


def test_repeated_hits_and_intensity() -> None:
    once = synth.sfx_recipe("a door slam", 3000, 0.7, False, 2).astype(np.float64)
    repeated = synth.sfx_recipe("a door slam, repeated", 3000, 0.7, False, 2).astype(np.float64)
    assert _rms(repeated[len(once) // 2:]) > 5 * _rms(once[len(once) // 2:])
    soft = synth.sfx_recipe("a door slam", 1000, 0.1, False, 2)
    loud = synth.sfx_recipe("a door slam", 1000, 1.0, False, 2)
    assert np.abs(loud).max() > 2 * np.abs(soft).max()


def test_every_recipe_renders_at_short_and_long_lengths() -> None:
    descriptions = [
        "thunder", "bell", "kettle", "slam", "snap", "wind", "gulls", "fire", "hooves",
        "footsteps", "clicks", "rain", "sea", "cough", "whistle", "nothing known",
    ]
    for desc in descriptions:
        for duration_ms, loop in ((500, False), (6000, True)):
            clip = synth.sfx_recipe(desc, duration_ms, 0.7, loop, 1)
            assert clip.dtype == np.int16 and len(clip) == round(duration_ms * SAMPLE_RATE / 1000), (desc, duration_ms)
            assert clip.any(), (desc, duration_ms)
            assert np.abs(clip.astype(np.int64)).max() <= 32767


def test_sfx_provider_contract(mock_settings: Settings) -> None:
    provider = ProceduralSfx.from_settings(mock_settings, NullUsage())
    assert isinstance(provider, SfxGenerator)
    assert ProceduralSfx.family == "mock" and provider.cache_version == "1" and provider.max_duration_ms == 120_000
    assert ProceduralSfx.check(mock_settings) == [] and provider.warmup() is None
    req = SfxRequest(description="wind over the rocks", kind="ambient", duration_ms=1500, loop=True, intensity=0.5, seed=77)
    clip = provider.generate(req)
    _assert_canonical(clip)
    assert clip.duration_ms == 1500
    assert np.array_equal(clip.samples, synth.sfx_recipe(req.description, 1500, 0.5, True, 77))
    assert provider.generate(SfxRequest(description="thunder", duration_ms=500_000)).duration_ms == 120_000


# --------------------------------------------------------------------------- helpers
def test_helpers_shapes_and_behaviour() -> None:
    rng = np.random.default_rng(0)
    env = synth.adsr(1000, attack_ms=5, decay_ms=5, sustain=0.5, release_ms=5)
    assert env.shape == (1000,) and env[0] == 0.0 and env[-1] < 0.01 and abs(env[500] - 0.5) < 1e-6
    assert synth.adsr(10, attack_ms=100, release_ms=100).shape == (10,)
    for color in ("white", "pink", "brown"):
        x = synth.noise(rng, 4096, color)
        assert x.shape == (4096,) and abs(np.abs(x).max() - 1.0) < 1e-5
    freqs, white_mag = _spectrum(synth.noise(rng, 22050, "white"))
    _, brown_mag = _spectrum(synth.noise(rng, 22050, "brown"))
    low, high = freqs < 200, freqs > 2000
    assert brown_mag[low].mean() / brown_mag[high].mean() > 10 * white_mag[low].mean() / white_mag[high].mean()
    tone = np.sin(2 * np.pi * 1000 * np.arange(22050) / SAMPLE_RATE) + np.sin(2 * np.pi * 5000 * np.arange(22050) / SAMPLE_RATE)
    passed = synth.bandpass(tone, 500, 2000)
    f, mag = _spectrum(passed)
    assert abs(f[np.argmax(mag)] - 1000) < 2 and mag[np.argmin(np.abs(f - 5000))] < 0.02 * mag.max()
    with pytest.raises(ValueError):
        synth.bandpass(tone, 3000, 1000)
    stack = synth.harmonic_stack(220.0, 22050, partials=4)
    assert stack.dtype == np.float32 and np.abs(stack).max() <= 1.0 + 1e-6
    f, mag = _spectrum(stack)
    assert abs(f[np.argmax(mag)] - 220) < 1.5
