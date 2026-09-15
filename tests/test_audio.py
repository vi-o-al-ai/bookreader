"""Tests for bookreader.audio.pcm and bookreader.audio.dsp (offline, deterministic)."""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from bookreader.types import SAMPLE_RATE, AudioClip

from bookreader.audio import dsp, pcm


def tone(freq: float, ms: int, rate: int = SAMPLE_RATE, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(rate * ms / 1000)) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# --------------------------------------------------------------------------- pcm
def test_wav_round_trip_is_exact(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    samples = rng.integers(-32768, 32767, size=5000, dtype=np.int16)
    path = tmp_path / "sub" / "clip.wav"
    pcm.write_wav(path, AudioClip(samples, SAMPLE_RATE))
    back = pcm.read_wav(path)
    assert back.sample_rate == SAMPLE_RATE
    assert back.samples.dtype == np.int16
    np.testing.assert_array_equal(back.samples, samples)
    assert not list(tmp_path.glob("**/*.tmp"))


def test_read_stereo_averages_to_mono(tmp_path: Path) -> None:
    left = np.full(100, 1000, dtype=np.int16)
    right = np.full(100, 3000, dtype=np.int16)
    interleaved = np.stack([left, right], axis=1).reshape(-1)
    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(interleaved.astype("<i2").tobytes())
    clip = pcm.read_wav(path)
    assert clip.sample_rate == 44100
    assert len(clip.samples) == 100
    assert np.all(clip.samples == 2000)


def test_read_24_bit_and_8_bit(tmp_path: Path) -> None:
    values = np.array([0, 1 << 20, -(1 << 20), (1 << 23) - 1, -(1 << 23)], dtype=np.int32)
    raw = bytearray()
    for v in values.tolist():
        raw += int(v & 0xFFFFFF).to_bytes(3, "little")
    path = tmp_path / "deep.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(3)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(bytes(raw))
    clip = pcm.read_wav(path)
    np.testing.assert_array_equal(clip.samples, (values >> 8).astype(np.int16))

    path8 = tmp_path / "eight.wav"
    with wave.open(str(path8), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(1)
        wf.setframerate(8000)
        wf.writeframes(bytes([128, 255, 0]))
    clip8 = pcm.read_wav(path8)
    np.testing.assert_array_equal(clip8.samples, np.array([0, 127 << 8, -128 << 8], dtype=np.int16))


def test_float_int16_conversions() -> None:
    x = np.array([-32768, -1, 0, 1, 32767], dtype=np.int16)
    f = pcm.to_float(x)
    assert f.dtype == np.float32
    assert f[0] == -1.0 and f[2] == 0.0
    back = pcm.to_int16(np.array([-2.0, 0.0, 0.5, 2.0], dtype=np.float32))
    assert back.tolist() == [-32767, 0, 16384, 32767]


@pytest.mark.parametrize("src", [44100, 32000, 24000, 16000])
def test_resample_keeps_tone_peak_and_length(src: int) -> None:
    x = tone(440.0, 500, rate=src)
    y = pcm.resample(x, src, SAMPLE_RATE)
    assert y.dtype == np.float32
    expected = round(len(x) * SAMPLE_RATE / src)
    assert abs(len(y) - expected) <= 1
    spectrum = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    peak_hz = np.argmax(spectrum) * SAMPLE_RATE / len(y)
    assert abs(peak_hz - 440.0) <= SAMPLE_RATE / len(y) + 1e-6
    assert abs(np.max(np.abs(y)) - 0.5) < 0.05
    np.testing.assert_array_equal(pcm.resample(x, src, src), x)       # same rate: untouched


def test_to_canonical_resamples_only_when_needed() -> None:
    clip = AudioClip(pcm.to_int16(tone(440.0, 200, rate=44100)), 44100)
    canon = pcm.to_canonical(clip)
    assert canon.sample_rate == SAMPLE_RATE
    assert abs(canon.duration_ms - 200) <= 1
    same = AudioClip(np.zeros(10, dtype=np.int16), SAMPLE_RATE)
    assert pcm.to_canonical(same) is same


# --------------------------------------------------------------------------- dsp
def test_trim_silence_keeps_40_ms() -> None:
    silence = np.zeros(int(SAMPLE_RATE * 0.5), dtype=np.float32)
    voiced = tone(300.0, 300, amp=0.3)
    x = np.concatenate([silence, voiced, silence])
    y = dsp.trim_silence(x, threshold_dbfs=-45, keep_ms=40, frame_ms=10)
    keep = int(SAMPLE_RATE * 0.04)
    frame = int(SAMPLE_RATE * 0.01)
    # 40 ms kept on each side, plus at most one 10 ms frame of alignment slack per side
    assert len(voiced) + 2 * keep <= len(y) <= len(voiced) + 2 * keep + 2 * frame
    # the kept margin is silence, the voiced part is intact
    assert np.max(np.abs(y[:keep])) == 0.0
    assert np.max(np.abs(y[len(y) - keep:])) == 0.0
    assert np.max(np.abs(y)) == np.max(np.abs(voiced))
    assert dsp.trim_silence(np.zeros(1000, dtype=np.float32)).shape == (1000,)


def test_normalize_rms_hits_target_and_respects_max_gain() -> None:
    x = tone(220.0, 400, amp=0.05)
    y = dsp.normalize_rms(x, target_dbfs=-20, max_gain_db=40)
    assert abs(dsp.rms_dbfs(y) + 20.0) < 0.5
    quiet = tone(220.0, 400, amp=0.01)                   # ~ -43 dBFS, more than 12 dB below target
    z = dsp.normalize_rms(quiet, target_dbfs=-20, max_gain_db=12)
    assert abs(dsp.rms_dbfs(z) - (dsp.rms_dbfs(quiet) + 12.0)) < 0.01
    # voiced-frame measurement: padding a clip with silence must not change the gain
    padded = np.concatenate([np.zeros(SAMPLE_RATE, dtype=np.float32), x])
    w = dsp.normalize_rms(padded, target_dbfs=-20, max_gain_db=40)
    assert abs(dsp.rms_dbfs(w[SAMPLE_RATE:]) + 20.0) < 0.5
    assert np.all(dsp.normalize_rms(np.zeros(500, dtype=np.float32)) == 0.0)


def test_crossfade_is_continuous_and_right_length() -> None:
    a = tone(110.0, 500, amp=0.5)
    b = tone(165.0, 500, amp=0.5)
    y = dsp.equal_power_crossfade(a, b, 100)
    assert len(y) == len(a) + len(b) - int(SAMPLE_RATE * 0.1)
    assert np.max(np.abs(np.diff(y))) < 0.05
    assert np.max(np.abs(y)) <= 0.5 * np.sqrt(2) + 1e-3


def test_loop_to_length_exact_length() -> None:
    x = tone(220.0, 700, amp=0.4)
    for n in (10, len(x) - 1, len(x), len(x) + 1, len(x) * 5 + 123):
        y = dsp.loop_to_length(x, n, crossfade_ms=2000)
        assert len(y) == n
        assert y.dtype == np.float32
        assert np.max(np.abs(y)) <= 0.4 * np.sqrt(2) + 1e-3
    y = dsp.loop_to_length(x, len(x) * 4)
    assert np.max(np.abs(np.diff(y))) < 0.05                  # seams are smooth
    assert dsp.rms_dbfs(y[len(x):]) > dsp.rms_dbfs(x) - 3.0     # looping keeps the level up
    assert len(dsp.loop_to_length(np.zeros(0, dtype=np.float32), 50)) == 50


def test_fade_ramps() -> None:
    x = np.ones(1000, dtype=np.float32)
    y = dsp.fade(x, 10, 10)
    assert y[0] == 0.0 and y[500] == 1.0 and y[-1] < 0.01
    assert np.all(np.diff(y[: int(SAMPLE_RATE * 0.01)]) > 0)


def test_soft_limit_never_exceeds_ceiling() -> None:
    x = tone(100.0, 200, amp=3.0)
    y = dsp.soft_limit(x, knee_dbfs=-3, ceiling_dbfs=-1)
    assert np.max(np.abs(y)) <= dsp.db_to_gain(-1.0)
    quiet = tone(100.0, 200, amp=0.5)
    np.testing.assert_allclose(dsp.soft_limit(quiet), quiet, atol=1e-6)


def test_rms_envelope_shape_and_levels() -> None:
    loud = tone(200.0, 200, amp=0.5)
    quiet = np.zeros(int(SAMPLE_RATE * 0.2), dtype=np.float32)
    env = dsp.rms_envelope(np.concatenate([loud, quiet]), frame_ms=20)
    assert env.shape == (20,)
    assert np.all(env[:9] > -12.0)
    assert np.all(env[10:] <= -100.0)
    assert dsp.rms_envelope(np.zeros(0, dtype=np.float32)).shape == (0,)
    assert abs(dsp.rms_dbfs(np.full(100, 1.0, dtype=np.float32))) < 1e-6
    assert abs(dsp.db_to_gain(-6.0206) - 0.5) < 1e-4
