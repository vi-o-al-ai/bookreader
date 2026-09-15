"""bookreader.audio.dsp - pure-numpy DSP primitives used by the render and mix stages.

Every function takes and returns float32 sample arrays in [-1, 1] (int16 input is converted) and
measures time in milliseconds at ``sample_rate`` (default SAMPLE_RATE). Nothing here mutates its input.
"""
from __future__ import annotations

import numpy as np

from bookreader.types import SAMPLE_RATE

from bookreader.audio.pcm import to_float

SILENCE_FLOOR_DBFS = -120.0   # reported level of an all-zero frame


# --------------------------------------------------------------------------- levels
def db_to_gain(db: float) -> float:
    """Decibels -> linear amplitude factor."""
    return float(10.0 ** (db / 20.0))


def _dbfs(rms: np.ndarray | float) -> np.ndarray | float:
    return 20.0 * np.log10(np.maximum(np.asarray(rms, dtype=np.float64), 1e-6))


def rms_dbfs(x: np.ndarray) -> float:
    """RMS level of the whole array in dBFS (1.0 full scale); empty/silent input reports the floor."""
    y = to_float(x)
    if len(y) == 0:
        return SILENCE_FLOOR_DBFS
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    return max(SILENCE_FLOOR_DBFS, float(_dbfs(rms)))


def ms_to_samples(ms: float, sample_rate: int = SAMPLE_RATE) -> int:
    """Milliseconds -> whole samples (rounded)."""
    return int(round(ms * sample_rate / 1000.0))


def rms_envelope(x: np.ndarray, frame_ms: int = 20, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Per-frame RMS level in dBFS (float64, one value per ``frame_ms`` frame; the last frame is zero-padded)."""
    y = to_float(x).astype(np.float64)
    frame = max(1, ms_to_samples(frame_ms, sample_rate))
    if len(y) == 0:
        return np.zeros(0, dtype=np.float64)
    n_frames = -(-len(y) // frame)
    padded = np.zeros(n_frames * frame, dtype=np.float64)
    padded[: len(y)] = y
    rms = np.sqrt(np.mean(padded.reshape(n_frames, frame) ** 2, axis=1))
    return np.maximum(SILENCE_FLOOR_DBFS, _dbfs(rms))


# --------------------------------------------------------------------------- trimming and normalization
def trim_silence(
    x: np.ndarray,
    threshold_dbfs: float = -45.0,
    keep_ms: int = 40,
    frame_ms: int = 10,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Cut leading/trailing frames below *threshold_dbfs*, keeping *keep_ms* of the surrounding silence.

    A clip with no frame above the threshold is returned unchanged.
    """
    y = to_float(x)
    env = rms_envelope(y, frame_ms, sample_rate)
    loud = np.flatnonzero(env > threshold_dbfs)
    if len(loud) == 0:
        return y
    frame = max(1, ms_to_samples(frame_ms, sample_rate))
    keep = ms_to_samples(keep_ms, sample_rate)
    start = max(0, int(loud[0]) * frame - keep)
    end = min(len(y), (int(loud[-1]) + 1) * frame + keep)
    return np.ascontiguousarray(y[start:end])


def normalize_rms(
    x: np.ndarray,
    target_dbfs: float = -20.0,
    max_gain_db: float = 12.0,
    frame_ms: int = 10,
    voiced_threshold_dbfs: float = -50.0,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Scale *x* so the RMS of its voiced frames (above *voiced_threshold_dbfs*) hits *target_dbfs*.

    The applied gain never exceeds *max_gain_db*; attenuation is unbounded. A clip with no voiced
    frame is returned unchanged.
    """
    y = to_float(x)
    frame = max(1, ms_to_samples(frame_ms, sample_rate))
    env = rms_envelope(y, frame_ms, sample_rate)
    voiced = env > voiced_threshold_dbfs
    if not np.any(voiced):
        return y
    mask = np.repeat(voiced, frame)[: len(y)]
    measured = rms_dbfs(y[mask])
    gain_db = min(max_gain_db, target_dbfs - measured)
    return (y * db_to_gain(gain_db)).astype(np.float32)


# --------------------------------------------------------------------------- fades and seams
def fade(x: np.ndarray, in_ms: int, out_ms: int, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Apply linear fade-in over *in_ms* and fade-out over *out_ms* (each capped at the clip length)."""
    y = to_float(x).copy()
    n = len(y)
    n_in = min(n, ms_to_samples(in_ms, sample_rate))
    n_out = min(n, ms_to_samples(out_ms, sample_rate))
    if n_in > 0:
        y[:n_in] *= np.linspace(0.0, 1.0, n_in, endpoint=False, dtype=np.float32)
    if n_out > 0:
        y[n - n_out:] *= np.linspace(0.0, 1.0, n_out, endpoint=False, dtype=np.float32)[::-1]
    return y


def _crossfade_ramps(n: int) -> tuple[np.ndarray, np.ndarray]:
    """(fade-out, fade-in) equal-power ramps: cos/sin over a quarter turn, summing to unit power."""
    theta = (np.arange(n, dtype=np.float64) + 0.5) / max(n, 1) * (np.pi / 2.0)   # mid-sample, symmetric
    return np.cos(theta).astype(np.float32), np.sin(theta).astype(np.float32)


def _crossfade_samples(a: np.ndarray, b: np.ndarray, overlap: int) -> np.ndarray:
    overlap = max(0, min(overlap, len(a), len(b)))
    if overlap == 0:
        return np.concatenate([a, b]).astype(np.float32)
    out_ramp, in_ramp = _crossfade_ramps(overlap)
    seam = a[len(a) - overlap:] * out_ramp + b[:overlap] * in_ramp
    return np.concatenate([a[: len(a) - overlap], seam, b[overlap:]]).astype(np.float32)


def equal_power_crossfade(a: np.ndarray, b: np.ndarray, overlap_ms: int, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Join *a* and *b* with a cos/sin (equal-power) crossfade of *overlap_ms*; length = len(a)+len(b)-overlap."""
    return _crossfade_samples(to_float(a), to_float(b), ms_to_samples(overlap_ms, sample_rate))


def loop_to_length(x: np.ndarray, n_samples: int, crossfade_ms: int = 2000, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Repeat *x* with equal-power seams until exactly *n_samples* long (or trim it when already longer).

    A clip shorter than twice the crossfade uses a crossfade of a quarter of its length.
    """
    y = to_float(x)
    n_samples = max(0, int(n_samples))
    if n_samples == 0 or len(y) == 0:
        return np.zeros(n_samples, dtype=np.float32)
    if len(y) >= n_samples:
        return np.ascontiguousarray(y[:n_samples])
    cf = ms_to_samples(crossfade_ms, sample_rate)
    if len(y) < 2 * cf:
        cf = len(y) // 4
    step = len(y) - cf
    out = np.zeros(n_samples + len(y), dtype=np.float32)
    out_ramp, in_ramp = _crossfade_ramps(cf) if cf > 0 else (np.zeros(0, np.float32), np.zeros(0, np.float32))
    pos = 0
    first = True
    while pos < n_samples:
        seg = y.copy()
        if cf > 0:
            if not first:
                seg[:cf] *= in_ramp
            seg[len(seg) - cf:] *= out_ramp
        out[pos: pos + len(seg)] += seg
        pos += step
        first = False
    return np.ascontiguousarray(out[:n_samples])


# --------------------------------------------------------------------------- limiting
def soft_limit(x: np.ndarray, knee_dbfs: float = -3.0, ceiling_dbfs: float = -1.0) -> np.ndarray:
    """Soft-knee limiter: samples above the knee are tanh-compressed so the output never exceeds the ceiling.

    Below the knee the signal is untouched; the result is finally hard-clipped at +-0.999.
    """
    y = to_float(x).astype(np.float64)
    knee = db_to_gain(knee_dbfs)
    ceiling = db_to_gain(ceiling_dbfs)
    if ceiling <= knee:
        raise ValueError("ceiling must be above the knee")
    headroom = ceiling - knee
    mag = np.abs(y)
    over = mag > knee
    limited = np.where(over, knee + headroom * np.tanh((mag - knee) / headroom), mag)
    out = np.sign(y) * limited
    return np.clip(out, -0.999, 0.999).astype(np.float32)
