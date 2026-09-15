"""bookreader.audio.pcm - PCM sample handling: WAV I/O, int16/float conversion and resampling.

The canonical format is SAMPLE_RATE (22050 Hz), mono, int16 PCM in a RIFF WAV written with the
stdlib ``wave`` module. In memory the engine works in float32 in [-1, 1]; int16 only appears at
file boundaries and inside :class:`bookreader.types.AudioClip`.
"""
from __future__ import annotations

import logging
import os
import wave
from pathlib import Path

import numpy as np

from bookreader.types import SAMPLE_RATE, AudioClip, InputError

log = logging.getLogger(__name__)

RESAMPLE_TAPS = 101          # FIR length of the anti-aliasing low-pass used when downsampling
RESAMPLE_CUTOFF = 0.45       # cutoff as a fraction of the destination rate
RESAMPLE_KAISER_BETA = 8.6   # Kaiser window shape (~ Blackman-like stop-band attenuation)


# --------------------------------------------------------------------------- conversions
def to_float(x: np.ndarray) -> np.ndarray:
    """int16 samples -> float32 in [-1, 1). Float input is passed through as float32."""
    arr = np.asarray(x)
    if np.issubdtype(arr.dtype, np.floating):
        return np.ascontiguousarray(arr, dtype=np.float32)
    return (arr.astype(np.float32) / 32768.0).astype(np.float32)


def to_int16(x: np.ndarray, clip: bool = True) -> np.ndarray:
    """float samples in [-1, 1] -> int16 with rounding. With ``clip`` values outside the range are clamped."""
    arr = np.asarray(x, dtype=np.float64)
    if clip:
        arr = np.clip(arr, -1.0, 1.0)
    scaled = np.rint(arr * 32767.0)
    if clip:
        scaled = np.clip(scaled, -32768, 32767)
    return scaled.astype(np.int16)


# --------------------------------------------------------------------------- resampling
def _lowpass_kernel(src: int, dst: int) -> np.ndarray:
    """Kaiser-windowed sinc low-pass with cutoff ``RESAMPLE_CUTOFF * dst`` expressed at rate ``src``."""
    fc = RESAMPLE_CUTOFF * dst / src                    # cycles per input sample
    m = (RESAMPLE_TAPS - 1) / 2.0
    n = np.arange(RESAMPLE_TAPS, dtype=np.float64) - m
    kernel = 2.0 * fc * np.sinc(2.0 * fc * n) * np.kaiser(RESAMPLE_TAPS, RESAMPLE_KAISER_BETA)
    return kernel / kernel.sum()


def resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    """Resample *x* from *src* Hz to *dst* Hz; returns float32 of length ``round(n * dst / src)``.

    Downsampling first low-passes with a 101-tap Kaiser-windowed sinc FIR (``np.convolve`` mode
    ``'same'``), then linearly interpolates onto the new grid; upsampling interpolates only.
    """
    if src <= 0 or dst <= 0:
        raise ValueError(f"sample rates must be positive, got src={src} dst={dst}")
    y = to_float(x)
    if src == dst:
        return y
    n = len(y)
    n_out = int(round(n * dst / src))
    if n == 0 or n_out == 0:
        return np.zeros(n_out, dtype=np.float32)
    if dst < src:
        y = np.convolve(y.astype(np.float64), _lowpass_kernel(src, dst), mode="same")
    t_new = np.arange(n_out, dtype=np.float64) * (src / dst)
    out = np.interp(t_new, np.arange(n, dtype=np.float64), y)
    return out.astype(np.float32)


def to_canonical(clip: AudioClip) -> AudioClip:
    """Return *clip* at SAMPLE_RATE (mono int16). Clips already canonical are returned unchanged."""
    if clip.sample_rate == SAMPLE_RATE:
        return clip
    log.debug("resampling clip %d Hz -> %d Hz (%d samples)", clip.sample_rate, SAMPLE_RATE, len(clip.samples))
    return AudioClip(to_int16(resample(clip.samples, clip.sample_rate, SAMPLE_RATE)), SAMPLE_RATE)


# --------------------------------------------------------------------------- WAV I/O
def _decode_frames(data: bytes, sampwidth: int) -> np.ndarray:
    """Raw PCM frames -> int16 samples (channels still interleaved)."""
    if sampwidth == 1:                                   # unsigned 8-bit
        return ((np.frombuffer(data, dtype=np.uint8).astype(np.int16) - 128) << 8).astype(np.int16)
    if sampwidth == 2:
        return np.frombuffer(data, dtype="<i2").copy()
    if sampwidth == 3:
        raw = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        value = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
        value = np.where(value >= 1 << 23, value - (1 << 24), value)   # sign-extend 24 bits
        return (value >> 8).astype(np.int16)
    if sampwidth == 4:
        return (np.frombuffer(data, dtype="<i4") >> 16).astype(np.int16)
    raise InputError(f"unsupported WAV sample width: {sampwidth} bytes")


def read_wav(path: Path | str) -> AudioClip:
    """Read a RIFF WAV (8/16/24/32-bit PCM, any channel count) into a mono int16 AudioClip."""
    path = Path(path)
    try:
        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            rate = wf.getframerate()
            data = wf.readframes(wf.getnframes())
    except (wave.Error, EOFError) as exc:
        raise InputError(f"cannot read WAV {path}: {exc}") from exc
    samples = _decode_frames(data, sampwidth)
    if channels > 1:
        frames = samples[: len(samples) - len(samples) % channels].reshape(-1, channels)
        samples = np.rint(frames.astype(np.float64).mean(axis=1)).astype(np.int16)
    return AudioClip(samples, rate)


def write_wav(path: Path | str, clip: AudioClip) -> Path:
    """Write *clip* as mono int16 WAV atomically (temp file in the same directory, then ``os.replace``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    samples = np.ascontiguousarray(clip.samples, dtype="<i2")
    try:
        with wave.open(str(tmp), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(clip.sample_rate))
            wf.writeframes(samples.tobytes())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path
