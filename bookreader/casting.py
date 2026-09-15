"""bookreader.casting - one deterministic casting pass: bible characters -> provider voices.

The narrator is cast first (voices tagged for narration, adult, with the least "character" in
them), then every speaking character in order of line count. Each candidate voice is scored
(gender, age, timbre keywords, penalties for reuse, for clashing with a conversation partner
and for being the narrator's voice) and ties break on ``stable_seed`` so the same book always
casts the same way. When the best voice is already taken it is reused with a settings
variation so two characters never sound identical. Overrides pin voice ids by character name.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Collection, Mapping

from bookreader.types import (
    AGES,
    NARRATOR,
    Cast,
    CastBible,
    CharacterEntry,
    VoiceAssignment,
    VoiceInfo,
    VoiceSettings,
    normalize_name,
    stable_seed,
)

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- scoring table
GENDER_MATCH = 100
GENDER_UNKNOWN = 40
AGE_MATCH = 30
AGE_ADJACENT = 15                 # also used when either age is unknown
KEYWORD_MATCH = 5
ALREADY_ASSIGNED_PENALTY = 60
CO_SPEECH_PENALTY = 20            # same (gender, age) bucket as a conversation partner's voice
NARRATOR_VOICE_PENALTY = 10

AGE_ORDER: tuple[str, ...] = tuple(age for age in AGES if age != "unknown")   # child .. elderly
UNKNOWN = "unknown"

KEYWORDS: tuple[str, ...] = (
    "warm", "dry", "rasp", "gruff", "bright", "deep", "soft", "stern", "young", "old",
    "clear", "gentle", "rough", "thin", "firm", "smooth", "husky", "breathy", "light", "dark",
    "low", "high", "crisp", "calm", "elderly", "child", "boy", "girl", "teen", "nasal", "sharp",
)
NARRATOR_TAG_WEIGHTS: dict[str, int] = {"narration": 30, "audiobook": 30, "neutral": 15, "calm": 10}
NARRATOR_ADULT_BONUS = 20
NARRATOR_UNKNOWN_AGE_BONUS = 10
CHARACTERFUL_TAGS: frozenset[str] = frozenset({
    "rasp", "raspy", "gruff", "rough", "deep", "child", "boy", "girl", "teen", "elderly", "old",
    "stern", "bright", "thin", "husky", "breathy", "shrill", "nasal", "whisper",
})
CHARACTERFUL_PENALTY = 10

# Base settings derived from the bible's voice notes / age.
STYLE_RASPY = 0.35
STABILITY_STERN = 0.7
SPEED_CHILD = 1.05

# Reuse variation (applied when the best voice is already assigned).
VARIATION_SPEEDS: tuple[float, float] = (1.08, 0.92)
VARIATION_PITCH_SEMITONES = 2.0
VARIATION_STABILITY = 0.15
VARIATION_STYLE = 0.15

# settings_for deltas: (stability, style, speed)
EMOTION_DELTAS: dict[str, tuple[float, float, float]] = {
    "urgent": (-0.15, 0.20, 0.05),
    "angry": (-0.15, 0.20, 0.05),
    "afraid": (-0.10, 0.0, 0.05),
    "sad": (0.0, 0.05, -0.05),
    "tender": (0.0, 0.05, -0.05),
    "melancholy": (0.0, 0.05, -0.05),
    "calm": (0.10, 0.0, 0.0),
}
WHISPER_SPEED_FACTOR = 0.95
WHISPER_STYLE = 0.30
SHOUT_STYLE_DELTA = 0.25
SHOUT_SPEED_DELTA = 0.05
QUIET_SPEED_DELTA = -0.03
SPEED_MIN, SPEED_MAX = 0.85, 1.15

_WORD_RE = re.compile(r"[a-z]+")


# --------------------------------------------------------------------------- helpers
def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _keywords(text: str) -> set[str]:
    """Timbre keywords found in *text* (prefix match, so ``rasping`` matches ``rasp``)."""
    words = set(_WORD_RE.findall(text.lower()))
    return {kw for kw in KEYWORDS if any(word.startswith(kw) for word in words)}


def _voice_text(voice: VoiceInfo) -> str:
    return " ".join([*voice.tags, voice.description, voice.name])


def _character_text(entry: CharacterEntry) -> str:
    return f"{entry.voice_notes} {entry.description}"


def _age_score(a: str, b: str) -> int:
    if a == UNKNOWN or b == UNKNOWN:
        return AGE_ADJACENT
    if a == b:
        return AGE_MATCH
    if a in AGE_ORDER and b in AGE_ORDER and abs(AGE_ORDER.index(a) - AGE_ORDER.index(b)) == 1:
        return AGE_ADJACENT
    return 0


def _gender_score(a: str, b: str) -> int:
    if a == UNKNOWN or b == UNKNOWN:
        return GENDER_UNKNOWN
    return GENDER_MATCH if a == b else 0


def bucket(voice: VoiceInfo) -> tuple[str, str]:
    """The (gender, age) bucket a voice belongs to."""
    return (voice.gender, voice.age)


def narrator_score(voice: VoiceInfo) -> int:
    """Suitability of *voice* for narration: narration tags, adult age, few characterful tags."""
    tags = {t.lower() for t in voice.tags}
    score = sum(weight for tag, weight in NARRATOR_TAG_WEIGHTS.items() if tag in tags)
    if voice.age == "adult":
        score += NARRATOR_ADULT_BONUS
    elif voice.age == UNKNOWN:
        score += NARRATOR_UNKNOWN_AGE_BONUS
    score -= CHARACTERFUL_PENALTY * len(tags & CHARACTERFUL_TAGS)
    return score


def character_score(
    entry: CharacterEntry,
    voice: VoiceInfo,
    assigned: Mapping[str, int],
    partner_buckets: Collection[tuple[str, str]],
    narrator_voice_id: str,
) -> tuple[int, list[str]]:
    """Score one voice for one character; returns the score and the reasons that built it."""
    reasons: list[str] = []
    score = _gender_score(entry.gender, voice.gender)
    reasons.append("gender match" if score == GENDER_MATCH else "gender unknown" if score == GENDER_UNKNOWN else "gender mismatch")
    age = _age_score(entry.age, voice.age)
    score += age
    reasons.append("age match" if age == AGE_MATCH else "age adjacent" if age == AGE_ADJACENT else "age mismatch")
    overlap = sorted(_keywords(_character_text(entry)) & _keywords(_voice_text(voice)))
    if overlap:
        score += KEYWORD_MATCH * len(overlap)
        reasons.append("keywords " + "/".join(overlap))
    if assigned.get(voice.id, 0):
        score -= ALREADY_ASSIGNED_PENALTY
        reasons.append("already assigned")
    if bucket(voice) in partner_buckets:
        score -= CO_SPEECH_PENALTY
        reasons.append("same bucket as a conversation partner")
    if voice.id == narrator_voice_id:
        score -= NARRATOR_VOICE_PENALTY
        reasons.append("narrator's voice")
    return score, reasons


def base_settings(entry: CharacterEntry) -> VoiceSettings:
    """Default synthesis settings implied by the bible entry (timbre notes, age)."""
    settings = VoiceSettings()
    notes = _keywords(entry.voice_notes)
    if "rasp" in notes or "dry" in notes:
        settings.style = STYLE_RASPY
    if "stern" in notes:
        settings.stability = STABILITY_STERN
    if entry.age == "child":
        settings.speed = SPEED_CHILD
    return settings


def variation(settings: VoiceSettings, family: str, ordinal: int) -> tuple[VoiceSettings, str]:
    """The *ordinal*-th reuse of a voice (1-based): alternate speed up/down with a pitch shift
    (mock/local) or stability/style deltas (elevenlabs). Returns the new settings and a note."""
    up = ordinal % 2 == 1
    out = settings.model_copy()
    out.speed = _clamp(VARIATION_SPEEDS[0] if up else VARIATION_SPEEDS[1], SPEED_MIN, SPEED_MAX)
    if family == "elevenlabs":
        delta = VARIATION_STABILITY if up else -VARIATION_STABILITY
        out.stability = _clamp(out.stability + delta, 0.0, 1.0)
        out.style = _clamp(out.style + (VARIATION_STYLE if up else -VARIATION_STYLE), 0.0, 1.0)
        note = f"stability {out.stability:.2f} style {out.style:.2f}"
    else:
        out.pitch_shift = VARIATION_PITCH_SEMITONES if up else -VARIATION_PITCH_SEMITONES
        note = f"pitch {out.pitch_shift:+.0f} st"
    return out, f"reuse #{ordinal} with variation: speed {out.speed:.2f}, {note}"


def _resolve_overrides(overrides: Mapping[str, str], voices_by_id: Mapping[str, VoiceInfo]) -> dict[str, str]:
    """Normalize override names and reject unknown voice ids (a 400-style ValueError)."""
    resolved: dict[str, str] = {}
    for name, voice_id in overrides.items():
        if voice_id not in voices_by_id:
            raise ValueError(f"unknown voice id {voice_id!r} in cast override for {name!r}")
        resolved[normalize_name(name)] = voice_id
    return resolved


def _override_for(entry: CharacterEntry, overrides: Mapping[str, str]) -> str | None:
    for key, voice_id in overrides.items():
        if entry.matches(key):
            return voice_id
    return None


# --------------------------------------------------------------------------- public API
def cast_voices(
    bible: CastBible,
    voices: list[VoiceInfo],
    family: str,
    overrides: Mapping[str, str],
    seed_material: str,
    co_speech: Mapping[str, Collection[str]] | None = None,
) -> Cast:
    """Assign a voice (and base settings) to the narrator and every speaking character.

    *overrides* maps character names (case-insensitive; ``"narrator"`` pins the narrator) to
    voice ids; an unknown voice id raises ``ValueError``. *co_speech* maps a character to the
    characters that speak in the same paragraphs; a voice in the same (gender, age) bucket as a
    partner's voice is penalized so conversations stay distinguishable. Bible entries without
    lines are not cast.
    """
    if not voices:
        raise ValueError("cannot cast: the voice catalog is empty")
    voices_by_id = {voice.id: voice for voice in voices}
    pinned = _resolve_overrides(overrides, voices_by_id)
    known = {normalize_name(entry.name) for entry in bible.characters} | {normalize_name(NARRATOR)}
    for key in pinned:
        if key not in known and not any(entry.matches(key) for entry in bible.characters):
            log.warning("cast override for unknown character %r ignored", key)

    assigned: Counter[str] = Counter()
    partners = co_speech or {}

    narrator_pin = pinned.get(normalize_name(NARRATOR))
    if narrator_pin is not None:
        narrator_voice = voices_by_id[narrator_pin]
        narrator = VoiceAssignment(
            character=NARRATOR, voice=narrator_voice, seed=stable_seed(seed_material, NARRATOR),
            source="override", reason="pinned by override",
        )
    else:
        ranked = sorted(voices, key=lambda v: (-narrator_score(v), stable_seed(seed_material, NARRATOR, v.id)))
        narrator_voice = ranked[0]
        narrator = VoiceAssignment(
            character=NARRATOR, voice=narrator_voice, seed=stable_seed(seed_material, NARRATOR),
            reason=f"auto: narration score {narrator_score(narrator_voice)}",
        )
    assigned[narrator_voice.id] += 1

    speaking = sorted((c for c in bible.characters if c.line_count > 0), key=lambda c: (-c.line_count, c.name))
    assignments: list[VoiceAssignment] = []
    voice_of: dict[str, VoiceInfo] = {}
    for entry in speaking:
        settings = base_settings(entry)
        pin = _override_for(entry, pinned)
        if pin is not None:
            voice = voices_by_id[pin]
            source, reason = "override", "pinned by override"
        else:
            partner_buckets = {bucket(voice_of[p]) for p in partners.get(entry.name, ()) if p in voice_of}
            scored = []
            for candidate in voices:
                score, reasons = character_score(entry, candidate, assigned, partner_buckets, narrator_voice.id)
                scored.append((-score, stable_seed(seed_material, entry.name, candidate.id), candidate, score, reasons))
            scored.sort(key=lambda item: (item[0], item[1]))
            _, _, voice, score, reasons = scored[0]
            source, reason = "auto", f"auto: score {score} ({', '.join(reasons)})"
            if assigned[voice.id]:
                settings, note = variation(settings, family, assigned[voice.id])
                reason = f"{reason}; {note}"
        assigned[voice.id] += 1
        voice_of[entry.name] = voice
        assignments.append(
            VoiceAssignment(
                character=entry.name, voice=voice, settings=settings,
                seed=stable_seed(seed_material, entry.name), source=source, reason=reason,
            )
        )
        log.debug("cast %s -> %s (%s)", entry.name, voice.id, reason)
    return Cast(family=family, narrator=narrator, characters=assignments)


def settings_for(assignment: VoiceAssignment, emotion: str, delivery: str) -> VoiceSettings:
    """Per-segment settings: the assignment's base settings nudged by emotion and delivery,
    clamped to 0..1 (stability, style, similarity) and 0.85..1.15 (speed)."""
    base = assignment.settings
    stability, style, speed = base.stability, base.style, base.speed
    d_stability, d_style, d_speed = EMOTION_DELTAS.get(emotion, (0.0, 0.0, 0.0))
    stability += d_stability
    style += d_style
    speed += d_speed
    if delivery == "whisper":
        speed *= WHISPER_SPEED_FACTOR
        style = WHISPER_STYLE
    elif delivery == "shout":
        style += SHOUT_STYLE_DELTA
        speed += SHOUT_SPEED_DELTA
    elif delivery == "quiet":
        speed += QUIET_SPEED_DELTA
    return VoiceSettings(
        stability=round(_clamp(stability, 0.0, 1.0), 4),
        similarity_boost=round(_clamp(base.similarity_boost, 0.0, 1.0), 4),
        style=round(_clamp(style, 0.0, 1.0), 4),
        speed=round(_clamp(speed, SPEED_MIN, SPEED_MAX), 4),
        pitch_shift=base.pitch_shift,
    )
