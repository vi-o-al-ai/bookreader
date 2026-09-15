"""Tests for bookreader.casting: deterministic voice assignment and per-segment settings."""
from __future__ import annotations

import pytest

from bookreader.casting import cast_voices, settings_for
from bookreader.providers.mock.tts import MockTTS
from bookreader.types import NARRATOR, CastBible, CharacterEntry, VoiceAssignment, VoiceInfo, VoiceSettings

SEED = "abc123"


def make_bible() -> CastBible:
    return CastBible(characters=[
        CharacterEntry(name="Mara Quill", aliases=["Mara"], gender="female", age="adult", line_count=13, first_chapter=1),
        CharacterEntry(name="Tobias", gender="male", age="child", line_count=4, description="Tobias was only twelve", first_chapter=1),
        CharacterEntry(name="Ansel Vey", aliases=["Ansel", "the stranger"], gender="male", age="adult", line_count=9, voice_notes="dry, rasping", first_chapter=2),
        CharacterEntry(name="Hetta", aliases=["Old Hetta"], gender="female", age="elderly", line_count=5, first_chapter=2),
        CharacterEntry(name="Silent Sam", gender="male", age="adult", line_count=0),
    ])


@pytest.fixture
def voices() -> list[VoiceInfo]:
    return MockTTS().list_voices()


@pytest.fixture
def by_id(voices: list[VoiceInfo]) -> dict[str, VoiceInfo]:
    return {v.id: v for v in voices}


def test_full_cast_distinct_and_matching(voices: list[VoiceInfo], by_id: dict[str, VoiceInfo]) -> None:
    cast = cast_voices(make_bible(), voices, "mock", {}, SEED)
    assert cast.family == "mock"
    assert cast.narrator.character == NARRATOR
    assert cast.narrator.voice.id == "mock-narrator-neutral"
    names = [a.character for a in cast.characters]
    assert names == ["Mara Quill", "Ansel Vey", "Hetta", "Tobias"]          # by line count, non-speaking entry skipped
    ids = [a.voice.id for a in cast.characters] + [cast.narrator.voice.id]
    assert len(set(ids)) == len(ids)
    entries = {c.name: c for c in make_bible().characters}
    for assignment in cast.characters:
        entry = entries[assignment.character]
        assert assignment.voice.gender == entry.gender
        assert assignment.source == "auto"
        assert assignment.seed != 0
        assert "gender match" in assignment.reason
    tobias = cast.assignment_for("Tobias")
    assert tobias.voice.age == "child" and tobias.voice.id == "mock-c-boy"
    assert tobias.settings.speed == pytest.approx(1.05)                     # children speak a touch faster
    ansel = cast.assignment_for("Ansel Vey")
    assert ansel.settings.style == pytest.approx(0.35)                      # dry / rasping notes
    assert cast.assignment_for("Nobody").character == NARRATOR              # unknown speakers fall back to the narrator


def test_casting_is_deterministic(voices: list[VoiceInfo]) -> None:
    first = cast_voices(make_bible(), voices, "mock", {}, SEED)
    second = cast_voices(make_bible(), list(reversed(voices)), "mock", {}, SEED)
    assert first.model_dump() == second.model_dump()
    assert first.narrator.seed == second.narrator.seed


def test_override_pins_voice_case_insensitively(voices: list[VoiceInfo]) -> None:
    cast = cast_voices(make_bible(), voices, "mock", {"tobias": "mock-m-teen", "NARRATOR": "mock-m-adult-deep"}, SEED)
    tobias = cast.assignment_for("Tobias")
    assert tobias.voice.id == "mock-m-teen" and tobias.source == "override"
    assert cast.narrator.voice.id == "mock-m-adult-deep" and cast.narrator.source == "override"
    others = [a for a in cast.characters if a.character != "Tobias"]
    assert all(a.voice.id not in ("mock-m-teen", "mock-m-adult-deep") for a in others)
    with pytest.raises(ValueError, match="unknown voice id"):
        cast_voices(make_bible(), voices, "mock", {"Tobias": "no-such-voice"}, SEED)
    with pytest.raises(ValueError, match="empty"):
        cast_voices(make_bible(), [], "mock", {}, SEED)


def test_small_inventory_reuses_voices_with_variation(by_id: dict[str, VoiceInfo]) -> None:
    two = [by_id["mock-m-adult-deep"], by_id["mock-f-adult-warm"]]
    cast = cast_voices(make_bible(), two, "mock", {}, SEED)
    assert len(cast.characters) == 4
    combos = {(a.voice.id, a.settings.speed, a.settings.pitch_shift) for a in cast.characters}
    combos.add((cast.narrator.voice.id, cast.narrator.settings.speed, cast.narrator.settings.pitch_shift))
    assert len(combos) == 5, "every assignment must sound different even with two voices"
    reused = [a for a in cast.characters if "variation" in a.reason]
    assert reused, "reuse must be recorded in the reason"
    for assignment in reused:
        assert assignment.settings.speed in (pytest.approx(1.08), pytest.approx(0.92))
        assert abs(assignment.settings.pitch_shift) == pytest.approx(2.0)
    eleven = cast_voices(make_bible(), two, "elevenlabs", {}, SEED)
    for assignment in (a for a in eleven.characters if "variation" in a.reason):
        assert assignment.settings.pitch_shift == 0.0                       # elevenlabs varies stability/style instead
        assert assignment.settings.stability != VoiceSettings().stability


def test_co_speech_contrast_avoids_same_bucket(by_id: dict[str, VoiceInfo]) -> None:
    pool = [by_id[v] for v in ("mock-narrator-neutral", "mock-m-adult-deep", "mock-m-adult-gruff", "mock-m-young-clear", "mock-f-adult-warm")]
    bible = CastBible(characters=[
        CharacterEntry(name="Arn", gender="male", age="adult", line_count=10),
        CharacterEntry(name="Bram", gender="male", age="adult", line_count=8),
    ])
    plain = cast_voices(bible, pool, "mock", {}, SEED)
    arn, bram = plain.assignment_for("Arn"), plain.assignment_for("Bram")
    assert arn.voice.id != bram.voice.id
    assert (arn.voice.gender, arn.voice.age) == (bram.voice.gender, bram.voice.age) == ("male", "adult")

    contrasted = cast_voices(bible, pool, "mock", {}, SEED, co_speech={"Arn": {"Bram"}, "Bram": {"Arn"}})
    arn2, bram2 = contrasted.assignment_for("Arn"), contrasted.assignment_for("Bram")
    assert arn2.voice.id == arn.voice.id
    assert bram2.voice.id == "mock-m-young-clear"
    assert (bram2.voice.gender, bram2.voice.age) != (arn2.voice.gender, arn2.voice.age)
    assert "conversation partner" not in bram2.reason


def test_settings_for_deltas_and_clamps(by_id: dict[str, VoiceInfo]) -> None:
    base = VoiceAssignment(character="X", voice=by_id["mock-m-adult-deep"], settings=VoiceSettings(stability=0.1, style=0.9, speed=1.12))
    urgent = settings_for(base, "urgent", "shout")
    assert urgent.stability == 0.0                                          # 0.1 - 0.15 clamped
    assert urgent.style == 1.0                                              # 0.9 + 0.2 + 0.25 clamped
    assert urgent.speed == pytest.approx(1.15)                              # 1.12 + 0.05 + 0.05 clamped
    neutral = settings_for(base, "neutral", "normal")
    assert neutral.model_dump() == VoiceSettings(stability=0.1, style=0.9, speed=1.12).model_dump()
    whisper = settings_for(base, "afraid", "whisper")
    assert whisper.style == pytest.approx(0.3)
    assert whisper.speed == pytest.approx(min(1.15, (1.12 + 0.05) * 0.95))
    assert whisper.stability == pytest.approx(0.0)
    calm = settings_for(VoiceAssignment(character="Y", voice=by_id["mock-f-adult-warm"]), "calm", "quiet")
    assert calm.stability == pytest.approx(0.65)
    assert calm.speed == pytest.approx(0.97)
    slow = settings_for(VoiceAssignment(character="Z", voice=by_id["mock-f-adult-warm"], settings=VoiceSettings(speed=0.86)), "sad", "normal")
    assert slow.speed == pytest.approx(0.85)
    assert slow.style == pytest.approx(0.2)
    assert slow.pitch_shift == 0.0
