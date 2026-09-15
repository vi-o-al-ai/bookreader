"""bookreader.providers.mock.synth - procedural audio recipes for the mock family.

Everything here is a pure function of its arguments: all randomness flows through
``np.random.default_rng(seed)`` (never Python's salted ``hash()``), so two processes given the
same seed produce byte-identical audio. All output is mono int16 at ``SAMPLE_RATE``; internally
signals are float32 in [-1, 1]. Synthesis is vectorized numpy - no per-sample Python loops -
and every FIR is applied with FFT convolution so long segments stay fast.

Public surface:

* helpers ``adsr``, ``noise``, ``bandpass``, ``harmonic_stack``
* ``voice_burst_sequence`` - the mock voice (syllable bursts of a harmonic stack)
* ``music_bed`` - one mood bed of an exact length
* ``sfx_recipe`` - keyword-matched sound effects
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Callable

import numpy as np

from bookreader.types import SAMPLE_RATE

log = logging.getLogger(__name__)

_TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------- small utilities
def _samples(ms: float, sr: int = SAMPLE_RATE) -> int:
    """Number of samples in *ms* milliseconds (rounded)."""
    return int(round(ms * sr / 1000.0))


def _time(n: int, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Sample times in seconds as float64 (phase accumulation needs the precision)."""
    return np.arange(n, dtype=np.float64) / sr


def _to_int16(x: np.ndarray) -> np.ndarray:
    """Clip a float signal to [-1, 1] and convert to int16."""
    return (np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0) * 32767.0).astype(np.int16)


def _peak_normalize(x: np.ndarray, peak: float) -> np.ndarray:
    """Scale *x* so its absolute peak equals *peak* (silence is returned unchanged)."""
    m = float(np.max(np.abs(x))) if len(x) else 0.0
    if m <= 1e-9:
        return x.astype(np.float32)
    return (x * (peak / m)).astype(np.float32)


def _fade(x: np.ndarray, in_ms: float, out_ms: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Linear fade in/out (in place on a float copy); fades shrink when the clip is shorter."""
    y = np.array(x, dtype=np.float32, copy=True)
    n = len(y)
    a = min(_samples(in_ms, sr), n // 2)
    r = min(_samples(out_ms, sr), n // 2)
    if a > 0:
        y[:a] *= np.linspace(0.0, 1.0, a, endpoint=False, dtype=np.float32)
    if r > 0:
        y[n - r:] *= np.linspace(1.0, 0.0, r, dtype=np.float32)   # ends exactly on 0
    return y


_DIRECT_CONVOLVE_MAX_TAPS = 256
_DIRECT_CONVOLVE_MAX_SAMPLES = 1 << 18


def _fft_size(target: int) -> int:
    """Smallest 5-smooth integer >= *target* (pocketfft is fastest on 2/3/5-factor lengths)."""
    best = 1 << max(1, target).bit_length()
    p2 = 1
    while p2 < best:
        p3 = p2
        while p3 < best:
            p5 = p3
            while p5 < best:
                if p5 >= target:
                    best = p5
                    break
                p5 *= 5
            p3 *= 3
        p2 *= 2
    return best


def _fft_convolve(x: np.ndarray, h: np.ndarray, offset: int, n_out: int) -> np.ndarray:
    """Full linear convolution of *x* and *h* via FFT, returning ``n_out`` samples from *offset*."""
    n, m = len(x), len(h)
    if n == 0 or m == 0 or n_out <= 0:
        return np.zeros(max(n_out, 0), dtype=np.float32)
    size = _fft_size(n + m - 1)
    y = np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(h, size), size)
    out = np.zeros(n_out, dtype=np.float64)
    avail = min(n_out, n + m - 1 - offset)
    if avail > 0:
        out[:avail] = y[offset:offset + avail]
    return out.astype(np.float32)


def _convolve_same(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Zero-phase 'same' convolution with a symmetric FIR: direct for short kernels (faster
    than an FFT below a few hundred taps), FFT-based otherwise."""
    if len(h) <= len(x) <= _DIRECT_CONVOLVE_MAX_SAMPLES and len(h) <= _DIRECT_CONVOLVE_MAX_TAPS:
        return np.convolve(np.asarray(x, dtype=np.float64), h, mode="same").astype(np.float32)
    return _fft_convolve(x, h, len(h) // 2, len(x))


def _convolve_causal(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Causal convolution truncated to ``len(x)`` (an impulse train hit with a kernel)."""
    return _fft_convolve(x, h, 0, len(x))


def _windowed_sinc_lowpass(cutoff_hz: float, taps: int, sr: int) -> np.ndarray:
    """Hamming-windowed sinc low-pass kernel with unity DC gain."""
    k = np.arange(taps, dtype=np.float64) - (taps - 1) / 2.0
    fc = cutoff_hz / sr
    h = 2.0 * fc * np.sinc(2.0 * fc * k) * np.hamming(taps)
    return h / h.sum()


# --------------------------------------------------------------------------- public helpers
def adsr(
    n: int,
    attack_ms: float = 10.0,
    decay_ms: float = 0.0,
    sustain: float = 1.0,
    release_ms: float = 20.0,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Linear attack/decay/sustain/release envelope of exactly *n* samples (float32, 0..1).

    Attack ramps 0 -> 1, decay ramps 1 -> *sustain*, the release multiplies the tail down to 0.
    When the three segments do not fit in *n* they are shrunk proportionally.
    """
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    a, d, r = _samples(attack_ms, sr), _samples(decay_ms, sr), _samples(release_ms, sr)
    total = a + d + r
    if total > n:
        a = int(a * n / total)
        d = int(d * n / total)
        r = n - a - d
    env = np.full(n, float(sustain), dtype=np.float32)
    if a > 0:
        env[:a] = np.linspace(0.0, 1.0, a, endpoint=False, dtype=np.float32)
    if d > 0:
        env[a:a + d] = np.linspace(1.0, float(sustain), d, endpoint=False, dtype=np.float32)
    if r > 0:
        env[n - r:] *= np.linspace(1.0, 0.0, r, endpoint=False, dtype=np.float32)
    return env


def noise(rng: np.random.Generator, n: int, color: str = "white", sr: int = SAMPLE_RATE) -> np.ndarray:
    """*n* samples of white, pink (1/f power) or brown (1/f^2 power) noise, peak-normalized to 1."""
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    white = rng.standard_normal(n)
    if color == "white" or n < 8:
        return _peak_normalize(white, 1.0)
    if color not in ("pink", "brown"):
        raise ValueError(f"unknown noise color {color!r}; expected white, pink or brown")
    alpha = 0.5 if color == "pink" else 1.0            # magnitude exponent: power falls as 1/f^(2*alpha)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    weights = 1.0 / np.maximum(freqs, 20.0) ** alpha
    weights[0] = 0.0                                    # no DC
    shaped = np.fft.irfft(np.fft.rfft(white) * weights, n)
    return _peak_normalize(shaped, 1.0)


def bandpass_kernel(low_hz: float, high_hz: float, sr: int = SAMPLE_RATE, taps: int = 101) -> np.ndarray:
    """Symmetric band-pass FIR kernel built from windowed-sinc low-pass kernels (``np.sinc``).

    ``low_hz <= 0`` makes it a pure low-pass; ``high_hz >= sr/2`` makes it a pure high-pass.
    """
    taps = max(3, int(taps) | 1)                        # odd length keeps the kernel symmetric
    nyquist = sr / 2.0
    low = max(0.0, float(low_hz))
    high = min(float(high_hz), nyquist * 0.999)
    if high <= low:
        raise ValueError(f"bandpass needs low_hz < high_hz < sr/2, got {low_hz}, {high_hz}")
    if high < nyquist * 0.999:
        kernel = _windowed_sinc_lowpass(high, taps, sr)
    else:
        kernel = np.zeros(taps, dtype=np.float64)
        kernel[taps // 2] = 1.0
    if low > 0.0:
        kernel = kernel - _windowed_sinc_lowpass(low, taps, sr)
    return kernel


def bandpass(x: np.ndarray, low_hz: float, high_hz: float, sr: int = SAMPLE_RATE, taps: int = 101) -> np.ndarray:
    """Zero-phase band-pass of *x* with ``bandpass_kernel(low_hz, high_hz)`` (mode 'same')."""
    return _convolve_same(x, bandpass_kernel(low_hz, high_hz, sr, taps))


def harmonic_stack(
    f0_hz: np.ndarray | float,
    n: int,
    partials: int = 6,
    rolloff: float = 1.0,
    brightness: float = 1.0,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Sum of *partials* sines at k*f0 with amplitudes ``brightness**(k-1) / k**rolloff``.

    *f0_hz* may be a scalar or a per-sample contour (phase is accumulated, so glides are
    continuous). Partials above Nyquist are dropped; the result is normalized to peak <= 1.
    """
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    f0 = np.broadcast_to(np.asarray(f0_hz, dtype=np.float64), (n,))
    # accumulate in float64, then wrap to [0, 2pi) so float32 sines stay accurate for any length
    phase = np.mod(_TWO_PI * np.cumsum(f0) / sr, _TWO_PI).astype(np.float32)
    f_max = float(np.max(f0))
    out = np.zeros(n, dtype=np.float32)
    total = 0.0
    for k in range(1, max(1, int(partials)) + 1):
        if k * f_max >= sr / 2.0:
            break
        amp = (brightness ** (k - 1)) / (k ** rolloff)
        if amp <= 1e-6:
            break
        out += np.float32(amp) * np.sin(np.float32(k) * phase)
        total += amp
    if total > 0.0:
        out /= np.float32(total)
    return out


# --------------------------------------------------------------------------- voices
_TOKEN_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*|\d+|[.!?;:,—–…-]+")
_VOWEL_RE = re.compile(r"[aeiouyà-ÿ]+", re.IGNORECASE)
_FRICATIVE_RE = re.compile(r"sh|th|ch|[fsvzhx]", re.IGNORECASE)
_LONG_PAUSE_CHARS = frozenset(".!?…")

_EMOTION_PITCH_VARIANCE: dict[str, float] = {"urgent": 1.4, "angry": 1.4, "sad": 0.7, "weary": 0.7}
_EMOTION_RATE: dict[str, float] = {"urgent": 1.15, "weary": 0.85}
_DELIVERY_GAIN_DB: dict[str, float] = {"normal": 0.0, "whisper": -12.0, "shout": 6.0, "quiet": -6.0, "strained": 0.0}

_KIND_GAP, _KIND_VOICED, _KIND_FRICATIVE = 0, 1, 2
_MIN_UTTERANCE_MS = 300.0
_BASE_PEAK = 0.45                                       # normal delivery peak; shout (+6 dB) reaches 0.9


def _voice_events(text: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Turn text into a schedule of (kind, nominal_ms) events: voiced syllable bursts of
    90-160 ms, 10 ms unvoiced bursts before fricative-heavy words, 40 ms word gaps and
    120-300 ms pauses at punctuation."""
    kinds: list[int] = []
    nominal: list[float] = []
    for tok in _TOKEN_RE.findall(text):
        if tok[0].isalnum():
            if kinds and kinds[-1] != _KIND_GAP:
                kinds.append(_KIND_GAP)
                nominal.append(40.0)
            if tok[0].isdigit():
                syllables = 1
                fricative_heavy = False
            else:
                syllables = max(1, len(_VOWEL_RE.findall(tok)))
                fricative_heavy = len(_FRICATIVE_RE.findall(tok)) / len(tok) >= 0.3
            if fricative_heavy:
                kinds.append(_KIND_FRICATIVE)
                nominal.append(10.0)
            for _ in range(syllables):
                kinds.append(_KIND_VOICED)
                nominal.append(float(rng.uniform(90.0, 160.0)))
        else:
            long_pause = any(c in _LONG_PAUSE_CHARS for c in tok)
            kinds.append(_KIND_GAP)
            nominal.append(float(rng.uniform(200.0, 300.0) if long_pause else rng.uniform(120.0, 200.0)))
    if _KIND_VOICED not in kinds:                       # empty or symbol-only text: one soft burst
        kinds.append(_KIND_VOICED)
        nominal.append(120.0)
    return np.asarray(kinds, dtype=np.int64), np.asarray(nominal, dtype=np.float64)


def _fit_lengths(nominal: np.ndarray, n: int) -> np.ndarray:
    """Scale nominal event durations so their sample lengths sum to exactly *n*."""
    lengths = np.floor(nominal / nominal.sum() * n).astype(np.int64)
    lengths[-1] += n - int(lengths.sum())
    return lengths


def _pitch_walk(rng: np.random.Generator, count: int, spread: float) -> np.ndarray:
    """Mean-reverting per-burst pitch deviation (fraction of f0), clipped to +-8 %."""
    steps = rng.normal(0.0, spread, count)
    walk = np.empty(count, dtype=np.float64)
    prev = 0.0
    for i in range(count):                              # count is the number of bursts, not samples
        prev = min(0.08, max(-0.08, 0.6 * prev + steps[i]))
        walk[i] = prev
    return walk


def voice_burst_sequence(
    text: str,
    f0: float,
    brightness: float,
    tremolo: float,
    ms_per_char: int,
    speed: float,
    emotion: str,
    delivery: str,
    seed: int,
) -> np.ndarray:
    """Synthesize *text* as a sequence of voiced syllable bursts; returns int16 at SAMPLE_RATE.

    Duration is ``max(300 ms, len(text) * ms_per_char) / (speed * emotion rate)``. Each syllable
    is a 6-partial harmonic stack at *f0* with a random-walk pitch contour, 5 Hz vibrato and a
    two-band formant emphasis; fricative-heavy words open with a 10 ms noise burst; punctuation
    inserts gaps. *tremolo* (0..1) adds 6 Hz amplitude modulation (elderly voices). Delivery:
    whisper swaps the harmonics for band-passed noise at -12 dB, shout is +15 % f0 and +6 dB,
    quiet is -6 dB, strained adds 3 % jitter. Emotion widens or narrows pitch variance and rate.
    """
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    sr = SAMPLE_RATE
    rate = max(0.25, float(speed)) * _EMOTION_RATE.get(emotion, 1.0)
    total_ms = max(_MIN_UTTERANCE_MS, len(text) * float(ms_per_char)) / rate
    n = max(1, _samples(total_ms, sr))

    kinds, nominal = _voice_events(text, rng)
    lengths = _fit_lengths(nominal, n)
    starts = np.cumsum(lengths) - lengths
    seg = np.repeat(np.arange(len(kinds)), lengths)     # per-sample event index
    t_in = np.arange(n, dtype=np.float64) - starts[seg]
    seg_len = np.maximum(lengths[seg], 1).astype(np.float64)
    seg_kind = kinds[seg]
    t = _time(n, sr)

    # amplitude envelopes ---------------------------------------------------------------
    attack = np.minimum(_samples(15.0, sr), seg_len / 3.0)
    release = np.minimum(_samples(30.0, sr), seg_len / 3.0)
    ramp = np.minimum(np.minimum(t_in / np.maximum(attack, 1.0), (seg_len - t_in) / np.maximum(release, 1.0)), 1.0)
    ramp = np.clip(ramp, 0.0, 1.0)
    burst_gain = rng.uniform(0.75, 1.0, len(kinds))
    env_voiced = ramp * (seg_kind == _KIND_VOICED) * burst_gain[seg]
    if tremolo > 0.0:
        env_voiced *= 1.0 - float(tremolo) * 0.5 * (1.0 - np.cos(_TWO_PI * 6.0 * t))
    env_fricative = np.sin(math.pi * np.clip(t_in / seg_len, 0.0, 1.0)) * (seg_kind == _KIND_FRICATIVE)

    # pitch contour ----------------------------------------------------------------------
    base_f0 = float(f0) * (1.15 if delivery == "shout" else 1.0)
    walk = _pitch_walk(rng, len(kinds), 0.025 * _EMOTION_PITCH_VARIANCE.get(emotion, 1.0))
    contour = base_f0 * (1.0 + walk[seg]) * (1.0 + 0.02 * (0.5 - t_in / seg_len))
    contour *= 1.0 + 0.006 * np.sin(_TWO_PI * 5.0 * t)
    if delivery == "strained":
        jitter = np.cumsum(rng.standard_normal(n))
        jitter -= np.convolve(jitter, np.full(256, 1.0 / 256.0), mode="same")
        jitter /= max(float(np.std(jitter)), 1e-9)
        contour *= 1.0 + 0.03 * jitter
    contour = np.clip(contour, 40.0, sr / 14.0)

    # sources ----------------------------------------------------------------------------
    partial_brightness = 0.35 + 0.65 * float(np.clip(brightness, 0.0, 1.0))
    if delivery == "whisper":
        voiced = bandpass(noise(rng, n, "white", sr), 800.0, 4000.0, sr)
    else:
        voiced = harmonic_stack(contour, n, partials=6, rolloff=1.0, brightness=partial_brightness, sr=sr)
        # formant-like emphasis: identity + F1 band + brightness-scaled F2 band as one FIR
        f2_low = 1500.0 + 1200.0 * float(np.clip(brightness, 0.0, 1.0))
        formant = 0.6 * bandpass_kernel(450.0, 950.0, sr) + 0.8 * float(brightness) * bandpass_kernel(f2_low, f2_low + 900.0, sr)
        formant[len(formant) // 2] += 1.0
        voiced = _convolve_same(voiced, formant)
    fricative = bandpass(noise(rng, n, "white", sr), 2500.0, 7000.0, sr) * env_fricative * 0.5
    signal = voiced * env_voiced + fricative

    signal = _peak_normalize(signal, _BASE_PEAK) * (10.0 ** (_DELIVERY_GAIN_DB.get(delivery, 0.0) / 20.0))
    return _to_int16(_fade(signal, 20.0, 20.0, sr))


# --------------------------------------------------------------------------- music
@dataclass(frozen=True)
class _MoodRecipe:
    root: float                      # Hz of the tonic
    intervals: tuple[int, ...]       # chord tones in semitones above the root
    tempo: float                     # bpm at energy 0.5
    mode: str                        # "pad" | "arp" | "pulse"
    partials: int                    # harmonic richness of each tone
    lfo_hz: float                    # slow amplitude LFO for pads
    level: float                     # relative loudness (0..1)
    progression: tuple[int, ...]     # chord root offsets in semitones, one per two bars
    extras: tuple[str, ...] = ()     # "soft_pulse" | "tritone_tremolo" | "noise_swell" | "filter_sweep" | "detune"


_MOOD_RECIPES: dict[str, _MoodRecipe] = {
    "calm": _MoodRecipe(110.00, (0, 4, 7, 11), 70, "pad", 1, 0.10, 0.55, (0, 5, 7, 0)),
    "warm": _MoodRecipe(130.81, (0, 4, 7), 84, "pad", 3, 0.15, 0.70, (0, 5, 0, 7), ("soft_pulse",)),
    "tense": _MoodRecipe(82.41, (0, 3, 7), 60, "pad", 6, 0.00, 1.00, (0,), ("tritone_tremolo", "noise_swell")),
    "ominous": _MoodRecipe(65.41, (0, 1, 7), 46, "pad", 6, 0.05, 0.90, (0,), ("filter_sweep",)),
    "melancholy": _MoodRecipe(98.00, (0, 3, 7, 10), 58, "arp", 2, 0.20, 0.60, (0, 8, 5, 7)),
    "sad": _MoodRecipe(87.31, (0, 3, 7), 50, "pad", 1, 0.07, 0.50, (0, 8, 3, 7)),
    "hopeful": _MoodRecipe(146.83, (0, 4, 7, 12), 100, "arp", 4, 0.25, 0.75, (0, 7, 9, 5)),
    "adventurous": _MoodRecipe(130.81, (0, 4, 7), 128, "pulse", 5, 0.00, 0.90, (0, 5, 7, 10)),
    "joyful": _MoodRecipe(196.00, (0, 4, 7, 9), 140, "pulse", 4, 0.00, 0.85, (0, 5, 9, 7)),
    "mysterious": _MoodRecipe(98.00, (0, 7, 12), 56, "pad", 2, 0.08, 0.60, (0, 6, 0, 1), ("detune",)),
    "romantic": _MoodRecipe(123.47, (0, 4, 7, 11, 14), 66, "pad", 3, 0.12, 0.65, (0, 5, 9, 7)),
}
_MUSIC_EDGE_FADE_MS = 200.0
_CHORD_SEAM_MS = 40.0   # pad chords overlap-add with an equal-power crossfade of this length


def _tone(freq: float, n: int, recipe: _MoodRecipe, energy: float) -> np.ndarray:
    """One sustained chord tone; energy brightens the partial roll-off. Triangle-like odd
    partials for 3-partial recipes, full stacks otherwise."""
    if recipe.partials <= 1:
        return harmonic_stack(freq, n, partials=1)
    rolloff = 1.6 - 0.6 * energy
    return harmonic_stack(freq, n, partials=recipe.partials, rolloff=rolloff, brightness=0.9)


def _render_pad(freqs: list[float], n: int, recipe: _MoodRecipe, energy: float) -> np.ndarray:
    out = np.zeros(n, dtype=np.float32)
    for f in freqs:
        if "detune" in recipe.extras:
            out += 0.5 * (_tone(f * 0.996, n, recipe, energy) + _tone(f * 1.004, n, recipe, energy))
        else:
            out += _tone(f, n, recipe, energy)
    return out / max(1, len(freqs))


def _seam_ramps(n: int) -> tuple[np.ndarray, np.ndarray]:
    """(fade-out, fade-in) equal-power cos/sin ramps of *n* samples for the pad chord seams."""
    theta = (np.arange(n, dtype=np.float64) + 0.5) / max(n, 1) * (math.pi / 2.0)
    return np.cos(theta).astype(np.float32), np.sin(theta).astype(np.float32)


def _render_arp(freqs: list[float], n: int, step: int, recipe: _MoodRecipe, energy: float) -> np.ndarray:
    idx = np.arange(n) // max(1, step)
    note = np.asarray(freqs, dtype=np.float64)[idx % len(freqs)]
    t_in = (np.arange(n) - idx * step) / SAMPLE_RATE
    env = np.minimum(t_in / 0.005, 1.0) * np.exp(-t_in / (0.6 * step / SAMPLE_RATE))
    rolloff = 1.6 - 0.6 * energy
    return harmonic_stack(note, n, partials=recipe.partials, rolloff=rolloff, brightness=0.9) * env.astype(np.float32)


def _render_pulse(freqs: list[float], n: int, step: int, recipe: _MoodRecipe, energy: float) -> np.ndarray:
    t_in = (np.arange(n) % max(1, step)) / SAMPLE_RATE
    env = (np.minimum(t_in / 0.004, 1.0) * np.exp(-t_in / (0.25 * step / SAMPLE_RATE))).astype(np.float32)
    out = np.zeros(n, dtype=np.float32)
    for f in freqs:
        out += _tone(f, n, recipe, energy)
    return out / max(1, len(freqs)) * env


def music_bed(mood: str, energy: float, duration_ms: int, seed: int) -> np.ndarray:
    """Compose a mood bed of exactly *duration_ms* (int16 at SAMPLE_RATE).

    Each mood maps to a root, chord intervals, tempo, timbre and LFO (see ``_MOOD_RECIPES``);
    chords change every two bars, energy scales amplitude, tempo and brightness, and 200 ms
    edge fades keep loops seamless. The seed picks the LFO phase, a sub-cent detune per tone
    and the noise content, so different books get different-but-consistent beds.
    ``"none"`` yields silence; an unknown mood logs a warning and uses the calm recipe.
    """
    n = max(0, _samples(duration_ms))
    if mood == "none" or n == 0:
        return np.zeros(n, dtype=np.int16)
    recipe = _MOOD_RECIPES.get(mood)
    if recipe is None:
        log.warning("music_bed: unknown mood %r, using calm", mood)
        recipe = _MOOD_RECIPES["calm"]
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    energy = float(np.clip(energy, 0.0, 1.0))
    sr = SAMPLE_RATE
    tempo = recipe.tempo * (0.85 + 0.3 * energy)
    beat = 60.0 / tempo
    step = max(1, _samples(beat * 500.0, sr))           # eighth note
    chord_len = max(1, _samples(8 * beat * 1000.0, sr))  # two 4/4 bars
    t = _time(n, sr)
    lfo_phase = float(rng.uniform(0.0, _TWO_PI))

    out = np.zeros(n, dtype=np.float32)
    seam = max(1, min(_samples(_CHORD_SEAM_MS, sr), chord_len))
    fade_out, fade_in = _seam_ramps(seam)
    for c in range(math.ceil(n / chord_len)):
        s0 = c * chord_len
        s1 = min(n, s0 + chord_len)
        shift = recipe.progression[c % len(recipe.progression)]
        detune = 1.0 + rng.normal(0.0, 0.001, len(recipe.intervals))
        freqs = [recipe.root * 2.0 ** ((shift + iv) / 12.0) * float(d) for iv, d in zip(recipe.intervals, detune)]
        if recipe.mode == "arp":
            piece = _render_arp(freqs, s1 - s0, step, recipe, energy)
        elif recipe.mode == "pulse":
            piece = _render_pulse(freqs, s1 - s0, step, recipe, energy)
        else:
            # Pads are sustained: each chord runs `seam` samples into the next and the two are
            # overlap-added with an equal-power crossfade, so the bed never dips at a chord change.
            s1 = min(n, s0 + chord_len + seam)
            piece = _render_pad(freqs, s1 - s0, recipe, energy)
            if c > 0:
                piece[:seam] *= fade_in[: len(piece)]
            if s1 - s0 > chord_len:
                piece[chord_len:] *= fade_out[: s1 - s0 - chord_len]
        out[s0:s1] += piece

    if recipe.lfo_hz > 0.0:
        out *= (0.8 + 0.2 * np.sin(_TWO_PI * recipe.lfo_hz * t + lfo_phase)).astype(np.float32)
    if "soft_pulse" in recipe.extras:
        out *= (0.75 + 0.25 * np.maximum(0.0, np.cos(_TWO_PI * t / beat))).astype(np.float32)
    if "tritone_tremolo" in recipe.extras:
        pedal = harmonic_stack(recipe.root * 2.0 ** (18.0 / 12.0), n, partials=3)
        out += 0.35 * pedal * (0.5 + 0.5 * np.sin(_TWO_PI * 6.0 * t)).astype(np.float32)
    if "noise_swell" in recipe.extras:
        swell = (0.5 - 0.5 * np.cos(_TWO_PI * 0.12 * t)) ** 2
        out += 0.3 * bandpass(noise(rng, n, "pink", sr), 200.0, 2500.0, sr) * swell.astype(np.float32)
    if "filter_sweep" in recipe.extras:
        low = bandpass(out, 0.0, 350.0, sr)
        mix = (0.5 + 0.5 * np.sin(_TWO_PI * 0.06 * t)).astype(np.float32)
        out = low * (1.0 - mix) + out * mix

    out = _peak_normalize(out, 0.9) * recipe.level * (0.4 + 0.6 * energy)
    return _to_int16(_fade(out, _MUSIC_EDGE_FADE_MS, _MUSIC_EDGE_FADE_MS, sr))


# --------------------------------------------------------------------------- sfx
_SfxFn = Callable[[np.random.Generator, int, str], np.ndarray]


def _wants_repeat(desc: str) -> bool:
    return any(w in desc for w in ("repeat", "again", "over and over", "continuous"))


def _place(buf: np.ndarray, start: int, piece: np.ndarray, gain: float = 1.0) -> None:
    """Add *piece* into *buf* at *start*, truncating at the buffer end."""
    if start >= len(buf) or start < 0:
        return
    m = min(len(piece), len(buf) - start)
    if m > 0:
        buf[start:start + m] += piece[:m] * gain


def _impulse_train(n: int, positions: np.ndarray, amps: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Impulses at *positions* (samples) with *amps*, each convolved with *kernel* (causal)."""
    train = np.zeros(n, dtype=np.float64)
    keep = (positions >= 0) & (positions < n)
    np.add.at(train, positions[keep].astype(np.int64), amps[keep])
    return _convolve_causal(train, np.asarray(kernel, dtype=np.float64))


def _decay(n: int, seconds: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """exp(-t / seconds) over *n* samples."""
    return np.exp(-_time(n, sr) / max(seconds, 1e-4)).astype(np.float32)


def _sfx_thunder(rng: np.random.Generator, n: int, desc: str) -> np.ndarray:
    t = _time(n)
    decay_s = float(rng.uniform(2.0, 4.0))
    main = bandpass(noise(rng, n, "brown"), 40.0, 400.0, taps=255) * np.exp(-t * 6.9 / decay_s) * np.minimum(t / 0.015, 1.0)
    delay = float(rng.uniform(0.6, 1.2))
    td = np.maximum(t - delay, 0.0)
    rumble_env = np.where(t >= delay, (1.0 - np.exp(-td / 0.3)) * np.exp(-td * 4.0 / decay_s), 0.0)
    rumble = bandpass(noise(rng, n, "brown"), 40.0, 160.0, taps=255) * rumble_env
    return (main + 0.6 * rumble).astype(np.float32)


def _sfx_bell(rng: np.random.Generator, n: int, desc: str) -> np.ndarray:
    base = float(rng.uniform(220.0, 330.0))
    ratios = (1.0, 2.0, 2.4, 3.0, 4.5)
    amps = (1.0, 0.6, 0.4, 0.3, 0.2)
    tau = 3.0 / 6.9                                     # -60 dB after 3 s
    length = min(n, _samples(3200.0))
    t = _time(length)
    strike = np.zeros(length, dtype=np.float64)
    for ratio, amp in zip(ratios, amps):
        strike += amp * np.sin(_TWO_PI * base * ratio * t) * np.exp(-t / (tau / math.sqrt(ratio)))
    transient = bandpass(noise(rng, length, "white"), 1500.0, 6000.0) * np.exp(-t / 0.005)
    strike = (strike / sum(amps) + 0.4 * transient) * np.minimum(t / 0.002, 1.0)
    out = np.zeros(n, dtype=np.float32)
    _place(out, 0, strike.astype(np.float32))
    if any(w in desc for w in ("two", "twice", "double")) or _wants_repeat(desc):
        _place(out, min(_samples(1500.0), n // 2), strike.astype(np.float32), 0.9)
    return out


def _hit_kernel(rng: np.random.Generator, thump_hz: float) -> np.ndarray:
    length = _samples(250.0)
    t = _time(length)
    crack = bandpass(noise(rng, length, "white"), 100.0, 2500.0) * np.exp(-t / 0.02)
    freq = thump_hz * (1.0 + 0.5 * np.exp(-t / 0.03))
    thump = np.sin(_TWO_PI * np.cumsum(freq) / SAMPLE_RATE) * np.exp(-t / 0.12)
    return (0.6 * crack + thump).astype(np.float32)


def _sfx_slam(rng: np.random.Generator, n: int, desc: str) -> np.ndarray:
    kernel = _hit_kernel(rng, 80.0)
    if _wants_repeat(desc):
        times = np.arange(0.0, n / SAMPLE_RATE, 0.7)
    elif "knock" in desc:
        times = np.array([0.0, 0.35, 0.7])
    else:
        times = np.array([0.0])
    positions = (times * SAMPLE_RATE).astype(np.int64)
    amps = rng.uniform(0.8, 1.0, len(positions))
    return _impulse_train(n, positions, amps, kernel)


def _sfx_snap(rng: np.random.Generator, n: int, desc: str) -> np.ndarray:
    length = min(n, _samples(60.0))
    t = _time(length)
    click = bandpass(noise(rng, length, "white"), 2000.0, 6000.0) * np.exp(-t / 0.003)
    body = np.sin(_TWO_PI * 400.0 * t) * np.exp(-t / 0.02)
    out = np.zeros(n, dtype=np.float32)
    _place(out, 0, (click + 0.3 * body).astype(np.float32))
    return out


def _sfx_wind(rng: np.random.Generator, n: int, desc: str) -> np.ndarray:
    t = _time(n)
    base = noise(rng, n, "white")
    low = bandpass(base, 150.0, 700.0)
    high = bandpass(base, 600.0, 1800.0)
    lfo = 0.5 + 0.5 * np.sin(_TWO_PI * 0.2 * t + rng.uniform(0.0, _TWO_PI))
    return (low * (0.4 + 0.6 * lfo) + 0.5 * high * lfo ** 2).astype(np.float32)


def _sfx_gulls(rng: np.random.Generator, n: int, desc: str) -> np.ndarray:
    out = np.zeros(n, dtype=np.float32)
    pos = 0
    while pos < n:
        length = _samples(rng.uniform(180.0, 320.0))
        t = _time(length)
        u = t / max(t[-1], 1e-6) if length > 1 else np.zeros(1)
        centre = float(rng.uniform(1200.0, 2200.0))
        freq = centre * (1.0 + 0.25 * np.sin(math.pi * u)) + 40.0 * np.sin(_TWO_PI * 30.0 * t)
        chirp = np.sin(_TWO_PI * np.cumsum(freq) / SAMPLE_RATE) * np.sin(math.pi * u) ** 0.7
        _place(out, pos, chirp.astype(np.float32), float(rng.uniform(0.5, 1.0)))
        pos += length + _samples(rng.uniform(250.0, 900.0))
    return out


def _sfx_fire(rng: np.random.Generator, n: int, desc: str) -> np.ndarray:
    t = _time(n)
    bed = bandpass(noise(rng, n, "pink"), 100.0, 3000.0) * (0.8 + 0.2 * np.sin(_TWO_PI * 0.3 * t))
    mask = rng.random(n) < 8.0 / SAMPLE_RATE
    positions = np.nonzero(mask)[0]
    amps = rng.uniform(0.3, 1.0, len(positions))
    kl = _samples(4.0)
    kernel = bandpass(noise(rng, kl, "white"), 2000.0, 7000.0) * _decay(kl, 0.001)
    return (0.5 * bed + _impulse_train(n, positions, amps, kernel)).astype(np.float32)


def _sfx_kettle(rng: np.random.Generator, n: int, desc: str) -> np.ndarray:
    t = _time(n)
    u = t / max(t[-1], 1e-6) if n > 1 else np.zeros(n)
    freq = (1200.0 + 1400.0 * u) * (1.0 + 0.01 * np.sin(_TWO_PI * 6.0 * t))
    tone = np.sin(_TWO_PI * np.cumsum(freq) / SAMPLE_RATE) * (0.15 + 0.85 * u)
    hiss = bandpass(noise(rng, n, "white"), 3000.0, 7000.0) * 0.15 * u
    return (tone + hiss).astype(np.float32)


def _sfx_hooves(rng: np.random.Generator, n: int, desc: str) -> np.ndarray:
    cycle = 1.0 / 2.5
    offsets = np.array([0.0, 0.12, 0.25, 0.37])
    accents = np.array([1.0, 0.7, 0.85, 0.6])
    cycles = np.arange(0.0, n / SAMPLE_RATE + cycle, cycle)
    times = (cycles[:, None] + offsets[None, :]).ravel()
    amps = np.tile(accents, len(cycles)) * rng.uniform(0.85, 1.0, len(times))
    times = times + rng.uniform(-0.005, 0.005, len(times))
    kl = _samples(80.0)
    t = _time(kl)
    kernel = bandpass(noise(rng, kl, "white"), 700.0, 2500.0) * np.exp(-t / 0.006) + 0.8 * np.sin(_TWO_PI * 70.0 * t) * np.exp(-t / 0.03)
    return _impulse_train(n, (times * SAMPLE_RATE).astype(np.int64), amps, kernel)


def _sfx_footsteps(rng: np.random.Generator, n: int, desc: str) -> np.ndarray:
    crunch = "crunch" in desc or "gravel" in desc
    times = np.arange(0.0, n / SAMPLE_RATE, 0.5)          # 2 Hz
    times = np.maximum(times + rng.uniform(-0.02, 0.02, len(times)), 0.0)
    kl = _samples(80.0)
    band = (800.0, 5000.0) if crunch else (300.0, 3000.0)
    kernel = bandpass(noise(rng, kl, "white"), *band) * _decay(kl, 0.025)
    amps = rng.uniform(0.7, 1.0, len(times))
    return _impulse_train(n, (times * SAMPLE_RATE).astype(np.int64), amps, kernel)


def _sfx_clicks(rng: np.random.Generator, n: int, desc: str) -> np.ndarray:
    mask = rng.random(n) < 6.0 / SAMPLE_RATE
    positions = np.nonzero(mask)[0]
    amps = rng.uniform(0.5, 1.0, len(positions))
    kl = _samples(2.0)
    kernel = bandpass(noise(rng, kl, "white"), 3000.0, 9000.0) * _decay(kl, 0.0006)
    return _impulse_train(n, positions, amps, kernel)


def _sfx_rain(rng: np.random.Generator, n: int, desc: str) -> np.ndarray:
    t = _time(n)
    bed = bandpass(noise(rng, n, "white"), 1000.0, 8000.0) * (0.7 + 0.3 * np.sin(_TWO_PI * 0.15 * t + rng.uniform(0.0, _TWO_PI)))
    mask = rng.random(n) < 20.0 / SAMPLE_RATE
    positions = np.nonzero(mask)[0]
    amps = rng.uniform(0.2, 0.6, len(positions))
    kl = _samples(3.0)
    kernel = bandpass(noise(rng, kl, "white"), 2000.0, 6000.0) * _decay(kl, 0.001)
    return (bed + _impulse_train(n, positions, amps, kernel)).astype(np.float32)


def _sfx_sea(rng: np.random.Generator, n: int, desc: str) -> np.ndarray:
    t = _time(n)
    wave = (0.5 - 0.5 * np.cos(_TWO_PI * 0.1 * t + rng.uniform(0.0, _TWO_PI))) ** 1.5
    return (bandpass(noise(rng, n, "brown"), 80.0, 1500.0) * (0.35 + 0.65 * wave)).astype(np.float32)


def _sfx_cough(rng: np.random.Generator, n: int, desc: str) -> np.ndarray:
    out = np.zeros(n, dtype=np.float32)
    puff_len = _samples(160.0)
    t = _time(puff_len)
    for i in range(int(rng.integers(2, 4))):
        puff = bandpass(noise(rng, puff_len, "white"), 300.0, 2500.0) * np.exp(-t / 0.05) * np.minimum(t / 0.005, 1.0)
        _place(out, i * _samples(260.0), puff.astype(np.float32), float(rng.uniform(0.7, 1.0)))
    return out


def _sfx_whistle(rng: np.random.Generator, n: int, desc: str) -> np.ndarray:
    base = float(rng.uniform(1300.0, 1900.0))
    scale = np.array([1.0, 9 / 8, 5 / 4, 3 / 2, 5 / 3, 2.0])
    notes = base * rng.choice(scale, 4)
    note_len = min(_samples(300.0), max(1, n // 4))
    t = _time(note_len)
    out = np.zeros(n, dtype=np.float32)
    env = adsr(note_len, 15.0, 0.0, 1.0, 40.0)
    for i, f in enumerate(notes):
        tone = np.sin(_TWO_PI * f * t * (1.0 + 0.008 * np.sin(_TWO_PI * 5.0 * t)))
        _place(out, i * note_len, (tone * env).astype(np.float32))
    return out


def _sfx_default(rng: np.random.Generator, n: int, desc: str) -> np.ndarray:
    t = _time(n)
    seconds = max(n / SAMPLE_RATE, 1e-3)
    return (bandpass(noise(rng, n, "white"), 300.0, 3000.0) * np.exp(-t * 4.0 / seconds) * np.minimum(t / 0.01, 1.0)).astype(np.float32)


_SFX_RECIPES: tuple[tuple[str, tuple[str, ...], _SfxFn], ...] = (
    ("thunder", ("thunder", "rumble"), _sfx_thunder),
    ("bell", ("bell", "chime", "toll", "gong"), _sfx_bell),
    ("kettle", ("kettle",), _sfx_kettle),
    ("slam", ("slam", "knock", "bang", "thud", "pound", "fist"), _sfx_slam),
    ("snap", ("snap", "twig"), _sfx_snap),
    ("wind", ("wind", "gale", "gust", "breeze", "howl"), _sfx_wind),
    ("gulls", ("gull", "seagull", "bird"), _sfx_gulls),
    ("fire", ("fire", "flame", "crackl", "hearth", "ember"), _sfx_fire),
    ("hooves", ("hoof", "hooves", "horse", "gallop", "clop"), _sfx_hooves),
    ("footsteps", ("footstep", "footfall", "crunch", "gravel", "boots", "walking", "steps"), _sfx_footsteps),
    ("clicks", ("needle", "click", "knitting", "tick", "clock"), _sfx_clicks),
    ("rain", ("rain", "drizzle", "downpour"), _sfx_rain),
    ("sea", ("sea", "wave", "surf", "ocean", "tide", "shore"), _sfx_sea),
    ("cough", ("cough",), _sfx_cough),
    ("whistle", ("whistl",), _sfx_whistle),
)
_SFX_MATCHERS: tuple[tuple[str, re.Pattern[str], _SfxFn], ...] = tuple(
    (name, re.compile(r"\b(?:" + "|".join(re.escape(k) for k in keys) + r")"), fn) for name, keys, fn in _SFX_RECIPES
)


def sfx_recipe_name(description: str) -> str:
    """The recipe a description resolves to (first keyword match in table order, else 'default')."""
    desc = description.lower()
    for name, pattern, _fn in _SFX_MATCHERS:
        if pattern.search(desc):
            return name
    return "default"


def _end_on_zero_crossing(x: np.ndarray, window: int = 200) -> np.ndarray:
    """Silence the tail after the last sign change inside the final *window* samples so a
    looped clip seams without a click; falls back to a short fade when no crossing exists."""
    y = np.array(x, dtype=np.float32, copy=True)
    n = len(y)
    if n < 2:
        return y
    lo = max(1, n - window)
    tail = y[lo - 1:]
    crossings = np.nonzero(tail[:-1] * tail[1:] <= 0.0)[0]
    if len(crossings):
        cut = lo + int(crossings[-1])
        y[cut:] = 0.0
    else:
        y[lo:] *= np.linspace(1.0, 0.0, n - lo, dtype=np.float32)
        y[-1] = 0.0
    return y


def sfx_recipe(description: str, duration_ms: int, intensity: float, loop: bool, seed: int) -> np.ndarray:
    """Render a sound effect of *duration_ms* for *description* (int16 at SAMPLE_RATE).

    The description is matched against keyword recipes (thunder, bell, slam/knock, snap, wind,
    gulls, fire, kettle, hooves, footsteps, clicks, rain, sea, cough, whistle); anything else
    gets a decaying band-passed noise burst. Intensity (0..1) scales gain; ``loop=True`` ends
    the clip on a zero crossing so it repeats without a click.
    """
    n = max(0, _samples(duration_ms))
    if n == 0:
        return np.zeros(0, dtype=np.int16)
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    desc = description.lower()
    name = sfx_recipe_name(desc)
    fn: _SfxFn = _sfx_default
    for recipe_name, _pattern, recipe_fn in _SFX_MATCHERS:
        if recipe_name == name:
            fn = recipe_fn
            break
    log.debug("sfx_recipe %r -> %s (%d ms, loop=%s)", description, name, duration_ms, loop)
    raw = fn(rng, n, desc)
    gain = 0.3 + 0.7 * float(np.clip(intensity, 0.0, 1.0))
    out = _peak_normalize(raw, 0.9) * gain
    out = _fade(out, 5.0, 0.0 if loop else 10.0)
    if loop:
        out = _end_on_zero_crossing(out)
    return _to_int16(out)
