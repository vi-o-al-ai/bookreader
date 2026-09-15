"""Local family adapters (Piper, Kokoro, MusicGen, AudioGen/procedural) against sys.modules stubs.

None of piper, kokoro, torch, transformers or audiocraft is installed; every test installs a
``types.ModuleType`` stand-in (or ``None`` for the "missing" case) with ``monkeypatch.setitem``.
"""
from __future__ import annotations

import json
import shutil
import sys
import types
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pytest

from bookreader.audio.pcm import to_canonical
from bookreader.providers.base import MusicGenerator, SfxGenerator, VoiceSynthesizer
from bookreader.providers.local.music import MusicGenMusic
from bookreader.providers.local.sfx import FALLBACK_WARNING, AudioGenSFX, LocalSFX, ProceduralLocalSfx
from bookreader.providers.local.tts import KOKORO_VOICES, KokoroTTS, LocalTTS, PiperTTS
from bookreader.providers.mock.sfx import ProceduralSfx
from bookreader.settings import Settings
from bookreader.types import (
    SAMPLE_RATE,
    MusicRequest,
    ProviderConfigError,
    ProviderPermanentError,
    SfxRequest,
    TTSRequest,
    VoiceSettings,
)

# --------------------------------------------------------------------------- helpers


def sine(rate: int, ms: int, freq: float = 440.0) -> np.ndarray:
    """float32 sine in [-1, 1] at *rate*."""
    t = np.arange(int(rate * ms / 1000)) / rate
    return (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)


def canonical_len(n_src: int, src_rate: int) -> int:
    return int(round(n_src * SAMPLE_RATE / src_rate))


class FakeTensor:
    """Just enough of a torch tensor: detach/cpu/numpy and 2-d indexing for ``audio[0, 0]``."""

    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    def detach(self) -> "FakeTensor":
        return self

    def cpu(self) -> "FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.array

    def __getitem__(self, item: Any) -> "FakeTensor":
        return FakeTensor(self.array[item])


class UsageSpy:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, float]] = []

    def record(self, capability: str, family: str, unit_type: str, units: float, **_: Any) -> None:
        self.rows.append((capability, family, unit_type, units))


def stub_module(monkeypatch: pytest.MonkeyPatch, name: str, **attrs: Any) -> types.ModuleType:
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


def settings_with(**overrides: Any) -> Settings:
    return Settings.from_env({}).with_overrides(**overrides)


# --------------------------------------------------------------------------- piper fixtures

PIPER_RATE = 16000


class StreamRawVoice:
    """Piper API shape 1: ``synthesize_stream_raw`` yielding int16 bytes."""

    def __init__(self, rate: int = PIPER_RATE) -> None:
        self.config = types.SimpleNamespace(sample_rate=rate)
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.audio = sine(rate, 100)

    def synthesize_stream_raw(self, text: str, **kw: Any) -> Iterator[bytes]:
        self.calls.append((text, kw))
        raw = (self.audio * 32767).astype("<i2").tobytes()
        yield raw[: len(raw) // 2]
        yield raw[len(raw) // 2 :]


class ChunkVoice:
    """Piper API shape 2 (piper-tts >= 1.3): ``synthesize(text, SynthesisConfig)`` yielding chunks."""

    def __init__(self, rate: int = PIPER_RATE) -> None:
        self.config = types.SimpleNamespace(sample_rate=rate)
        self.calls: list[tuple[str, Any]] = []
        self.audio = sine(rate, 100)

    def synthesize(self, text: str, config: Any) -> Iterator[Any]:
        self.calls.append((text, config))
        half = len(self.audio) // 2
        for piece in (self.audio[:half], self.audio[half:]):
            yield types.SimpleNamespace(
                sample_rate=self.config.sample_rate,
                audio_float_array=piece,
                audio_int16_bytes=(piece * 32767).astype("<i2").tobytes(),
            )


class FakeSynthesisConfig:
    def __init__(self, **kw: Any) -> None:
        self.kw = kw


@pytest.fixture
def voices_dir(tmp_path: Path) -> Path:
    """Two usable models (one multi-speaker), one orphan .onnx and a voices.json sidecar."""
    d = tmp_path / "voices"
    d.mkdir()
    (d / "en_US-lessac-medium.onnx").write_bytes(b"onnx")
    (d / "en_US-lessac-medium.onnx.json").write_text(json.dumps({"audio": {"sample_rate": 22050}, "num_speakers": 1}))
    (d / "en_GB-vctk-medium.onnx").write_bytes(b"onnx")
    (d / "en_GB-vctk-medium.onnx.json").write_text(
        json.dumps({"audio": {"sample_rate": 16000}, "num_speakers": 2, "speaker_id_map": {"p225": 0, "p226": 1}})
    )
    (d / "orphan.onnx").write_bytes(b"onnx")
    (d / "voices.json").write_text(json.dumps({
        "en_US-lessac-medium": {"gender": "female", "age": "adult", "tags": ["Narration", "warm"], "description": "steady narrator"},
        "en_GB-vctk-medium#1": {"gender": "male", "age": "elderly"},
    }))
    return d


# --------------------------------------------------------------------------- piper


def test_piper_check_names_extra_when_missing_and_validates_voices_dir(monkeypatch: pytest.MonkeyPatch, voices_dir: Path, tmp_path: Path):
    monkeypatch.setitem(sys.modules, "piper", None)
    with pytest.raises(ProviderConfigError) as exc:
        PiperTTS.check(settings_with(piper_voices_dir=voices_dir))
    assert "bookreader[local]" in str(exc.value)

    stub_module(monkeypatch, "piper", PiperVoice=object)
    with pytest.raises(ProviderConfigError) as exc2:
        PiperTTS.check(settings_with(piper_voices_dir=tmp_path / "nowhere"))
    assert "nowhere" in str(exc2.value) and "huggingface.co/rhasspy/piper-voices" in str(exc2.value)

    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "lonely.onnx").write_bytes(b"x")
    with pytest.raises(ProviderConfigError) as exc3:
        PiperTTS.check(settings_with(piper_voices_dir=empty))
    assert ".onnx.json" in str(exc3.value)

    warnings = PiperTTS.check(settings_with(piper_voices_dir=voices_dir))
    assert warnings == ["piper voice orphan.onnx has no orphan.onnx.json config and is ignored"]
    assert LocalTTS.check(settings_with(piper_voices_dir=voices_dir, local_tts_engine="piper")) == warnings


def test_piper_list_voices_uses_sidecar_and_speaker_map(voices_dir: Path):
    tts = PiperTTS(voices_dir, loader=lambda _p: StreamRawVoice())
    assert isinstance(tts, VoiceSynthesizer) and tts.family == "local"
    assert tts.cache_version.startswith("piper:") and len(tts.cache_version) == len("piper:") + 12
    assert tts.cache_version == PiperTTS(voices_dir).cache_version           # deterministic per catalog
    voices = {v.id: v for v in tts.list_voices()}
    assert list(voices) == ["en_GB-vctk-medium#0", "en_GB-vctk-medium#1", "en_US-lessac-medium"]

    lessac = voices["en_US-lessac-medium"]
    assert (lessac.gender, lessac.age, lessac.tags, lessac.description) == ("female", "adult", ["narration", "warm"], "steady narrator")
    assert lessac.sample_rate == 22050 and lessac.family == "local"

    p225 = voices["en_GB-vctk-medium#0"]
    assert (p225.gender, p225.age) == ("unknown", "unknown")       # inferred from the stem -> unknown
    assert "p225" in p225.tags and p225.sample_rate == 16000 and p225.extra["speaker"] == "p225"
    p226 = voices["en_GB-vctk-medium#1"]
    assert (p226.gender, p226.age) == ("male", "elderly")          # sidecar keyed by the speaker id


def test_piper_stream_raw_shape_resamples_to_canonical(voices_dir: Path):
    fake = StreamRawVoice(rate=16000)
    loads: list[Path] = []

    def loader(path: Path) -> StreamRawVoice:
        loads.append(path)
        return fake

    usage = UsageSpy()
    tts = PiperTTS(voices_dir, loader=loader, usage=usage)
    req = TTSRequest(text="The wind came off the sea.", voice_id="en_GB-vctk-medium#1", settings=VoiceSettings(speed=1.25))
    clip = tts.synthesize(req)
    tts.synthesize(req)

    assert loads == [voices_dir / "en_GB-vctk-medium.onnx"]      # memoized
    text, kw = fake.calls[0]
    assert text == req.text
    assert kw == {"speaker_id": 1, "length_scale": pytest.approx(0.8), "noise_scale": 0.667, "noise_w": 0.8}
    assert clip.sample_rate == 16000 and clip.samples.dtype == np.int16
    np.testing.assert_allclose(clip.samples, (fake.audio * 32767).astype(np.int16), atol=1)
    canonical = to_canonical(clip)
    assert canonical.sample_rate == SAMPLE_RATE
    assert len(canonical.samples) == canonical_len(len(fake.audio), 16000)
    assert usage.rows == [("tts", "local", "characters", float(len(req.text)))] * 2


def test_piper_chunk_shape_uses_synthesis_config(monkeypatch: pytest.MonkeyPatch, voices_dir: Path):
    stub_module(monkeypatch, "piper", PiperVoice=object)
    stub_module(monkeypatch, "piper.config", SynthesisConfig=FakeSynthesisConfig)
    fake = ChunkVoice(rate=16000)
    tts = PiperTTS(voices_dir, loader=lambda _p: fake)
    clip = tts.synthesize(TTSRequest(text="Stand back.", voice_id="en_US-lessac-medium", settings=VoiceSettings(speed=0.5)))

    text, config = fake.calls[0]
    assert text == "Stand back." and isinstance(config, FakeSynthesisConfig)
    assert config.kw == {"speaker_id": None, "length_scale": pytest.approx(2.0), "noise_scale": 0.667, "noise_w_scale": 0.8}
    assert clip.sample_rate == 16000
    np.testing.assert_allclose(clip.samples, (fake.audio * 32767).astype(np.int16), atol=1)
    assert len(to_canonical(clip).samples) == canonical_len(len(fake.audio), 16000)


def test_piper_chunk_shape_falls_back_to_top_level_synthesis_config(monkeypatch: pytest.MonkeyPatch, voices_dir: Path):
    stub_module(monkeypatch, "piper", PiperVoice=object, SynthesisConfig=FakeSynthesisConfig)
    monkeypatch.delitem(sys.modules, "piper.config", raising=False)
    fake = ChunkVoice()
    clip = PiperTTS(voices_dir, loader=lambda _p: fake).synthesize(TTSRequest(text="Hi.", voice_id="en_US-lessac-medium"))
    assert isinstance(fake.calls[0][1], FakeSynthesisConfig) and clip.duration_ms == 100


def test_piper_default_loader_and_bad_voice_ids(monkeypatch: pytest.MonkeyPatch, voices_dir: Path):
    monkeypatch.setitem(sys.modules, "piper", None)
    tts = PiperTTS.from_settings(settings_with(piper_voices_dir=voices_dir))
    with pytest.raises(ProviderConfigError) as exc:
        tts.synthesize(TTSRequest(text="Hi.", voice_id="en_US-lessac-medium"))
    assert "bookreader[local]" in str(exc.value)

    loaded: list[str] = []
    stub_module(monkeypatch, "piper", PiperVoice=types.SimpleNamespace(load=lambda p: loaded.append(p) or StreamRawVoice()))
    tts = PiperTTS.from_settings(settings_with(piper_voices_dir=voices_dir))
    tts.warmup()
    assert sorted(Path(p).name for p in loaded) == ["en_GB-vctk-medium.onnx", "en_US-lessac-medium.onnx"]
    with pytest.raises(ProviderPermanentError):
        tts.synthesize(TTSRequest(text="Hi.", voice_id="no-such-voice"))
    with pytest.raises(ProviderPermanentError):
        tts.synthesize(TTSRequest(text="Hi.", voice_id="en_US-lessac-medium#x"))
    with pytest.raises(ProviderPermanentError):
        tts.synthesize(TTSRequest(text="x" * 2001, voice_id="en_US-lessac-medium"))


# --------------------------------------------------------------------------- kokoro


class FakePipeline:
    def __init__(self, lang: str) -> None:
        self.lang = lang
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.audio = sine(24000, 150, 220.0)

    def __call__(self, text: str, **kw: Any) -> Iterator[Any]:
        self.calls.append((text, kw))
        half = len(self.audio) // 2
        yield types.SimpleNamespace(graphemes=text[:5], audio=FakeTensor(self.audio[:half]))
        yield types.SimpleNamespace(graphemes=text[5:], audio=FakeTensor(self.audio[half:]))


def test_kokoro_check_requires_package_and_espeak(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "kokoro", None)
    with pytest.raises(ProviderConfigError) as exc:
        KokoroTTS.check(settings_with(local_tts_engine="kokoro"))
    assert "bookreader[local-kokoro]" in str(exc.value)

    stub_module(monkeypatch, "kokoro", KPipeline=object)
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    with pytest.raises(ProviderConfigError) as exc2:
        LocalTTS.check(settings_with(local_tts_engine="kokoro"))
    assert "espeak-ng" in str(exc2.value) and "apt-get install espeak-ng" in str(exc2.value)

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/espeak-ng" if name == "espeak-ng" else None)
    assert KokoroTTS.check(settings_with(local_tts_engine="kokoro")) == []


def test_kokoro_synthesize_concatenates_24k_audio_and_resamples():
    pipelines: list[FakePipeline] = []

    def factory(lang: str) -> FakePipeline:
        pipelines.append(FakePipeline(lang))
        return pipelines[-1]

    usage = UsageSpy()
    tts = KokoroTTS(pipeline_factory=factory, lang="b", usage=usage)
    assert isinstance(tts, VoiceSynthesizer) and tts.family == "local" and tts.cache_version == "kokoro:b"
    req = TTSRequest(text="Only you, so far.", voice_id="bf_emma", settings=VoiceSettings(speed=0.9))
    clip = tts.synthesize(req)
    tts.synthesize(req)

    assert len(pipelines) == 1 and pipelines[0].lang == "b"        # memoized pipeline
    assert pipelines[0].calls[0] == (req.text, {"voice": "bf_emma", "speed": 0.9})
    assert clip.sample_rate == 24000 and clip.samples.dtype == np.int16
    np.testing.assert_allclose(clip.samples, (pipelines[0].audio * 32767).astype(np.int16), atol=1)
    canonical = to_canonical(clip)
    assert canonical.sample_rate == SAMPLE_RATE
    assert len(canonical.samples) == canonical_len(len(pipelines[0].audio), 24000)
    assert usage.rows == [("tts", "local", "characters", float(len(req.text)))] * 2


def test_kokoro_catalog_and_default_factory(monkeypatch: pytest.MonkeyPatch):
    tts = KokoroTTS(pipeline_factory=lambda lang: FakePipeline(lang))
    voices = tts.list_voices()
    assert [v.id for v in voices] == list(KOKORO_VOICES)
    by_id = {v.id: v for v in voices}
    assert by_id["af_heart"].gender == "female" and by_id["am_adam"].gender == "male"
    assert by_id["bf_emma"].gender == "female" and by_id["bm_george"].gender == "male"
    assert all(v.family == "local" and v.sample_rate == 24000 for v in voices)
    assert "british" in by_id["bm_george"].tags and "american" in by_id["af_heart"].tags

    monkeypatch.setitem(sys.modules, "kokoro", None)
    with pytest.raises(ProviderConfigError) as exc:
        KokoroTTS.from_settings(settings_with(local_tts_engine="kokoro", kokoro_lang="a")).warmup()
    assert "bookreader[local-kokoro]" in str(exc.value)

    built: list[dict[str, Any]] = []
    stub_module(monkeypatch, "kokoro", KPipeline=lambda **kw: built.append(kw) or FakePipeline(kw["lang_code"]))
    kokoro = KokoroTTS.from_settings(settings_with(local_tts_engine="kokoro", kokoro_lang="b"))
    kokoro.warmup()
    assert built == [{"lang_code": "b"}]
    with pytest.raises(ProviderPermanentError):
        kokoro.synthesize(TTSRequest(text="x" * 2001, voice_id="af_heart"))


def test_local_tts_dispatches_on_engine(voices_dir: Path):
    assert isinstance(LocalTTS.from_settings(settings_with(local_tts_engine="piper", piper_voices_dir=voices_dir)), PiperTTS)
    kokoro = LocalTTS.from_settings(settings_with(local_tts_engine="kokoro", kokoro_lang="b"))
    assert isinstance(kokoro, KokoroTTS) and kokoro.lang == "b"
    assert LocalTTS.family == "local"


# --------------------------------------------------------------------------- musicgen

MUSICGEN_RATE = 32000


class FakeProcessor:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kw: Any) -> dict[str, Any]:
        self.calls.append(kw)
        return {"input_ids": "ids", "attention_mask": "mask"}


class FakeMusicgen:
    def __init__(self) -> None:
        self.config = types.SimpleNamespace(audio_encoder=types.SimpleNamespace(sampling_rate=MUSICGEN_RATE))
        self.calls: list[dict[str, Any]] = []
        self.audio = sine(MUSICGEN_RATE, 250, 110.0)

    def generate(self, **kw: Any) -> FakeTensor:
        self.calls.append(kw)
        return FakeTensor(self.audio.reshape(1, 1, -1))


class NoGrad:
    entered = 0

    def __enter__(self) -> None:
        NoGrad.entered += 1

    def __exit__(self, *exc: Any) -> None:
        return None


def test_musicgen_check_names_extra_when_torch_or_transformers_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ProviderConfigError) as exc:
        MusicGenMusic.check(settings_with())
    assert "bookreader[local]" in str(exc.value)

    stub_module(monkeypatch, "torch", no_grad=NoGrad)
    with pytest.raises(ProviderConfigError):
        MusicGenMusic.check(settings_with())
    stub_module(monkeypatch, "transformers")
    assert MusicGenMusic.check(settings_with()) == []


def test_musicgen_compose_loads_once_and_generates_under_no_grad(monkeypatch: pytest.MonkeyPatch):
    stub_module(monkeypatch, "torch", no_grad=NoGrad)
    processor, model = FakeProcessor(), FakeMusicgen()
    loads: list[str] = []

    def loader(name: str) -> tuple[FakeProcessor, FakeMusicgen]:
        loads.append(name)
        return processor, model

    usage = UsageSpy()
    music = MusicGenMusic("facebook/musicgen-small", loader=loader, usage=usage)
    assert isinstance(music, MusicGenerator)
    assert (music.family, music.cache_version, music.min_duration_ms, music.max_duration_ms) == ("local", "facebook/musicgen-small", 1000, 30_000)

    music.warmup()
    before = NoGrad.entered
    clip = music.compose(MusicRequest(prompt="warm strings by a fire", mood="warm", duration_ms=12_000))
    assert loads == ["facebook/musicgen-small"]
    assert processor.calls == [{"text": ["warm strings by a fire"], "padding": True, "return_tensors": "pt"}]
    assert model.calls[0] == {"input_ids": "ids", "attention_mask": "mask", "do_sample": True, "guidance_scale": 3.0, "max_new_tokens": 600}
    assert NoGrad.entered == before + 1
    assert clip.sample_rate == MUSICGEN_RATE and clip.samples.dtype == np.int16
    np.testing.assert_allclose(clip.samples, (model.audio * 32767).astype(np.int16), atol=1)
    assert len(to_canonical(clip).samples) == canonical_len(len(model.audio), MUSICGEN_RATE)

    music.compose(MusicRequest(prompt="p", duration_ms=90_000))
    music.compose(MusicRequest(prompt="p", duration_ms=10))
    assert [c["max_new_tokens"] for c in model.calls[1:]] == [1500, 50]      # clamped to 30 s / 1 s
    assert usage.rows[0] == ("music", "local", "audio_seconds", pytest.approx(0.25, abs=0.01))
    assert len(usage.rows) == 3


def test_musicgen_default_loader_and_compose_need_the_extra(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    music = MusicGenMusic.from_settings(settings_with(musicgen_model="facebook/musicgen-medium"))
    assert music.cache_version == "facebook/musicgen-medium"
    with pytest.raises(ProviderConfigError) as exc:
        music.warmup()
    assert "bookreader[local]" in str(exc.value)
    with pytest.raises(ProviderConfigError):
        music.compose(MusicRequest(prompt="p"))

    built: list[str] = []
    stub_module(
        monkeypatch, "transformers",
        AutoProcessor=types.SimpleNamespace(from_pretrained=lambda n: built.append(f"proc:{n}") or FakeProcessor()),
        MusicgenForConditionalGeneration=types.SimpleNamespace(from_pretrained=lambda n: built.append(f"model:{n}") or FakeMusicgen()),
    )
    music.warmup()
    assert built == ["proc:facebook/musicgen-medium", "model:facebook/musicgen-medium"]


# --------------------------------------------------------------------------- local sfx


class FakeAudioGen:
    sample_rate = 16000

    def __init__(self) -> None:
        self.params: list[dict[str, Any]] = []
        self.calls: list[list[str]] = []
        self.audio = sine(16000, 500, 330.0)

    def set_generation_params(self, **kw: Any) -> None:
        self.params.append(kw)

    def generate(self, descriptions: list[str]) -> FakeTensor:
        self.calls.append(descriptions)
        return FakeTensor(self.audio.reshape(1, 1, -1))


def test_local_sfx_falls_back_to_procedural_with_warning(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
    monkeypatch.setitem(sys.modules, "audiocraft", None)
    settings = settings_with(local_sfx_fallback=True)
    assert LocalSFX.check(settings) == [FALLBACK_WARNING]
    assert "audiocraft not installed" in FALLBACK_WARNING and "procedural" in FALLBACK_WARNING

    with caplog.at_level("WARNING", logger="bookreader.providers.local.sfx"):
        sfx = LocalSFX.from_settings(settings)
    assert any(FALLBACK_WARNING in rec.getMessage() for rec in caplog.records)
    assert isinstance(sfx, ProceduralLocalSfx) and isinstance(sfx, ProceduralSfx) and isinstance(sfx, SfxGenerator)
    assert sfx.family == "local" and ProceduralLocalSfx.family == "local" and ProceduralSfx.family == "mock"
    clip = sfx.generate(SfxRequest(description="a single deep thunderclap", duration_ms=1500, seed=7))
    assert clip.sample_rate == SAMPLE_RATE and clip.duration_ms == 1500 and int(np.abs(clip.samples).max()) > 0

    strict = settings_with(local_sfx_fallback=False)
    with pytest.raises(ProviderConfigError) as exc:
        LocalSFX.check(strict)
    assert "audiocraft" in str(exc.value)
    with pytest.raises(ProviderConfigError):
        LocalSFX.from_settings(strict)


def test_local_sfx_uses_audiogen_when_audiocraft_imports(monkeypatch: pytest.MonkeyPatch):
    stub_module(monkeypatch, "audiocraft")
    settings = settings_with(audiogen_model="facebook/audiogen-medium")
    assert LocalSFX.check(settings) == [] and AudioGenSFX.check(settings) == []
    sfx = LocalSFX.from_settings(settings)
    assert isinstance(sfx, AudioGenSFX) and isinstance(sfx, SfxGenerator)
    assert (sfx.family, sfx.cache_version, sfx.max_duration_ms) == ("local", "facebook/audiogen-medium", 10_000)

    model = FakeAudioGen()
    loads: list[str] = []
    usage = UsageSpy()
    gen = AudioGenSFX("facebook/audiogen-medium", loader=lambda n: loads.append(n) or model, usage=usage)
    gen.warmup()
    clip = gen.generate(SfxRequest(description="gulls crying over the harbour", duration_ms=25_000))
    gen.generate(SfxRequest(description="a snap", duration_ms=800))
    assert loads == ["facebook/audiogen-medium"]
    assert model.params == [{"duration": 10.0}, {"duration": 0.8}]
    assert model.calls == [["gulls crying over the harbour"], ["a snap"]]
    assert clip.sample_rate == 16000
    np.testing.assert_allclose(clip.samples, (model.audio * 32767).astype(np.int16), atol=1)
    assert len(to_canonical(clip).samples) == canonical_len(len(model.audio), 16000)
    assert [r[:3] for r in usage.rows] == [("sfx", "local", "audio_seconds")] * 2


def test_audiogen_default_loader_needs_audiocraft(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "audiocraft", None)
    monkeypatch.setitem(sys.modules, "audiocraft.models", None)
    gen = AudioGenSFX.from_settings(settings_with())
    with pytest.raises(ProviderConfigError) as exc:
        gen.warmup()
    assert "audiocraft" in str(exc.value)

    fetched: list[str] = []
    stub_module(monkeypatch, "audiocraft")
    stub_module(monkeypatch, "audiocraft.models", AudioGen=types.SimpleNamespace(get_pretrained=lambda n: fetched.append(n) or FakeAudioGen()))
    gen.warmup()
    assert fetched == ["facebook/audiogen-medium"]
