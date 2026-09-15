"""Cast bible: merge rules, speaker registration, finalize, fingerprints."""
from __future__ import annotations

from bookreader.analysis.bible import apply_updates, finalize, register_speaker, speaker_line_counts
from bookreader.types import NARRATOR, CastBible, ChapterScript, CharacterEntry, CharacterUpdate, Segment


def _bible(*entries: CharacterEntry) -> CastBible:
    return CastBible(characters=list(entries))


def _names(bible: CastBible) -> list[str]:
    return [c.name for c in bible.characters]


# --------------------------------------------------------------------------- alias merge
def test_single_token_then_full_name_upgrades_canonical() -> None:
    bible = apply_updates(CastBible(), [CharacterUpdate(name="Ansel", gender="male")], 2)
    assert _names(bible) == ["Ansel"]
    bible = apply_updates(bible, [CharacterUpdate(name="Ansel Vey", age="adult")], 2)
    assert _names(bible) == ["Ansel Vey"]
    entry = bible.characters[0]
    assert entry.aliases == ["Ansel"]
    assert entry.gender == "male" and entry.age == "adult"
    assert bible.find("Ansel") is entry and bible.find("Ansel Vey") is entry
    assert bible.version == 2


def test_full_name_then_single_token_becomes_alias() -> None:
    bible = apply_updates(CastBible(), [CharacterUpdate(name="Ansel Vey")], 2)
    bible = apply_updates(bible, [CharacterUpdate(name="Ansel", voice_notes="dry")], 2)
    assert _names(bible) == ["Ansel Vey"]
    assert bible.characters[0].aliases == ["Ansel"]
    assert bible.characters[0].voice_notes == "dry"


def test_subset_match_requires_uniqueness() -> None:
    bible = apply_updates(CastBible(), [CharacterUpdate(name="John Smith"), CharacterUpdate(name="John Brown")], 1)
    bible = apply_updates(bible, [CharacterUpdate(name="John")], 1)
    assert _names(bible) == ["John Smith", "John Brown", "John"]
    bible = apply_updates(bible, [CharacterUpdate(name="Mara Quill")], 1)
    bible = apply_updates(bible, [CharacterUpdate(name="Quill Mara Vey", aliases=["Mara Quill"])], 1)
    assert "Mara Quill" in _names(bible), "an alias match merges without renaming a multi-token canonical"
    assert "Quill Mara Vey" in bible.find("Mara Quill").aliases


def test_alias_given_by_update_matches_later_updates() -> None:
    bible = apply_updates(CastBible(), [CharacterUpdate(name="Mara Quill", aliases=["Mara"], gender="female")], 1)
    bible = apply_updates(bible, [CharacterUpdate(name="Mara", description="the keeper of the light")], 1)
    assert _names(bible) == ["Mara Quill"]
    assert bible.characters[0].description == "the keeper of the light"
    assert bible.characters[0].aliases == ["Mara"]


def test_alias_naming_a_trait_contradicting_entry_does_not_merge(caplog) -> None:
    bible = _bible(CharacterEntry(name="Wren", gender="female", age="adult", line_count=3), CharacterEntry(name="the man", gender="male", provisional=True))
    with caplog.at_level("WARNING", logger="bookreader.analysis.bible"):
        bible = apply_updates(bible, [CharacterUpdate(name="Corwin Tallow", aliases=["Corwin", "Wren"], gender="male", age="adult")], 1)
    assert _names(bible) == ["Wren", "the man", "Corwin Tallow"], "a new entry instead of folding a man into Wren"
    wren = bible.find("Wren")
    assert wren is not None and wren.name == "Wren" and wren.gender == "female" and wren.aliases == []
    assert bible.find("Corwin Tallow").name == "Corwin Tallow"
    assert any("Wren" in record.message and "not merging" in record.message for record in caplog.records)


def test_subset_match_requires_agreeing_traits() -> None:
    bible = _bible(CharacterEntry(name="Ash Carver", gender="male", age="adult"))
    bible = apply_updates(bible, [CharacterUpdate(name="Ash", gender="female", age="child")], 1)
    assert _names(bible) == ["Ash Carver", "Ash"]
    assert bible.find("Ash Carver").gender == "male" and bible.find("Ash Carver").age == "adult"
    bible = _bible(CharacterEntry(name="Ash Carver", gender="male"))
    bible = apply_updates(bible, [CharacterUpdate(name="Ash", gender="unknown", age="adult")], 1)
    assert _names(bible) == ["Ash Carver"], "unknown traits still agree"
    assert bible.find("Ash Carver").age == "adult"


# --------------------------------------------------------------------------- honorifics
def test_old_hetta_becomes_hetta_with_alias() -> None:
    bible = apply_updates(CastBible(), [CharacterUpdate(name="Old Hetta", gender="female", age="elderly")], 2)
    assert _names(bible) == ["Hetta"]
    assert bible.characters[0].aliases == ["Old Hetta"]
    assert bible.find("Old Hetta") is bible.characters[0]
    assert bible.find("hetta") is bible.characters[0]


def test_old_hetta_merges_into_existing_hetta() -> None:
    bible = _bible(CharacterEntry(name="Hetta", gender="female"))
    bible = apply_updates(bible, [CharacterUpdate(name="Old Hetta", age="elderly")], 2)
    assert _names(bible) == ["Hetta"]
    assert bible.characters[0].aliases == ["Old Hetta"]
    assert bible.characters[0].age == "elderly"


# --------------------------------------------------------------------------- merge_into
def test_merge_into_provisional_renames_it() -> None:
    bible = _bible(CharacterEntry(name="the stranger", gender="male", age="adult", provisional=True, first_chapter=2, line_count=2))
    bible = apply_updates(bible, [CharacterUpdate(name="Ansel Vey", aliases=["Ansel"], merge_into="the stranger")], 2)
    assert _names(bible) == ["Ansel Vey"]
    entry = bible.characters[0]
    assert entry.provisional is False
    assert set(entry.aliases) == {"Ansel", "the stranger"}
    assert entry.line_count == 2 and entry.first_chapter == 2
    assert bible.find("the stranger") is entry


def test_merge_into_named_entry_adds_alias_and_details() -> None:
    bible = _bible(CharacterEntry(name="Ansel Vey", gender="male", description="first mate", voice_notes="dry"))
    bible = apply_updates(bible, [CharacterUpdate(name="Vey", merge_into="Ansel Vey", voice_notes="rasping, dry", description="first mate of the Corvid")], 3)
    assert _names(bible) == ["Ansel Vey"]
    entry = bible.characters[0]
    assert entry.aliases == ["Vey"]
    assert entry.description == "first mate of the Corvid", "the longer description wins"
    assert entry.voice_notes == "dry, rasping", "voice notes concatenate unique parts"


def test_unresolvable_merge_into_falls_back_to_name() -> None:
    bible = apply_updates(CastBible(), [CharacterUpdate(name="Hetta", merge_into="nobody at all")], 2)
    assert _names(bible) == ["Hetta"]


# --------------------------------------------------------------------------- gender / age upgrades
def test_unknown_never_overwrites_known() -> None:
    bible = _bible(CharacterEntry(name="Mara Quill", gender="female", age="adult"))
    bible = apply_updates(bible, [CharacterUpdate(name="Mara Quill", gender="unknown", age="unknown")], 1)
    assert bible.characters[0].gender == "female" and bible.characters[0].age == "adult"
    bible = apply_updates(bible, [CharacterUpdate(name="Mara Quill", gender="male", age="child")], 1)
    assert bible.characters[0].gender == "female" and bible.characters[0].age == "adult", "known values are never overwritten"


def test_known_upgrades_unknown() -> None:
    bible = _bible(CharacterEntry(name="Tobias"))
    bible = apply_updates(bible, [CharacterUpdate(name="Tobias", gender="male", age="child")], 1)
    assert bible.characters[0].gender == "male" and bible.characters[0].age == "child"


# --------------------------------------------------------------------------- provisional entries
def test_provisional_merged_by_name_introduction_via_alias() -> None:
    bible, canonical = register_speaker(CastBible(), "the stranger", 2)
    assert canonical == "the stranger"
    assert bible.characters[0].provisional is True and bible.characters[0].line_count == 1
    bible = apply_updates(bible, [CharacterUpdate(name="Ansel Vey", aliases=["Ansel", "the stranger"], gender="male", age="adult")], 2)
    assert _names(bible) == ["Ansel Vey"]
    entry = bible.characters[0]
    assert entry.provisional is False
    assert set(entry.aliases) == {"Ansel", "the stranger"}
    assert entry.line_count == 1


def test_provisional_merged_by_description_mention() -> None:
    bible = _bible(CharacterEntry(name="the stranger", gender="male", provisional=True))
    bible = apply_updates(bible, [CharacterUpdate(name="Ansel Vey", description="the stranger washed up below the point", gender="male")], 2)
    assert _names(bible) == ["Ansel Vey"]
    assert "the stranger" in bible.characters[0].aliases


def test_provisional_not_merged_when_traits_contradict() -> None:
    bible = _bible(CharacterEntry(name="the stranger", gender="male", provisional=True))
    bible = apply_updates(bible, [CharacterUpdate(name="Anna", aliases=["the stranger"], gender="female")], 2)
    assert _names(bible) == ["the stranger", "Anna"]


def test_register_speaker_resolves_the_boy_to_the_only_child() -> None:
    bible = _bible(CharacterEntry(name="Mara Quill", gender="female", age="adult"), CharacterEntry(name="Tobias", gender="male", age="child"))
    bible, canonical = register_speaker(bible, "the boy", 1)
    assert canonical == "Tobias" and _names(bible) == ["Mara Quill", "Tobias"] and bible.find("Tobias").line_count == 1
    bible, canonical = register_speaker(bible, "the girl", 1)
    assert canonical == "the girl" and bible.characters[-1].provisional, "no female child: a provisional entry as before"
    two_boys = _bible(CharacterEntry(name="Tobias", gender="male", age="child"), CharacterEntry(name="Kit", gender="male", age="child"))
    two_boys, canonical = register_speaker(two_boys, "the boy", 1)
    assert canonical == "the boy", "ambiguous descriptors are not guessed"
    bible, canonical = register_speaker(_bible(CharacterEntry(name="Ansel Vey", gender="male", age="adult")), "the man", 2)
    assert canonical == "the man", "generic descriptors stay provisional"


def test_descriptor_update_creates_provisional_entry() -> None:
    bible = apply_updates(CastBible(), [CharacterUpdate(name="the rider", gender="male")], 3)
    assert bible.characters[0].provisional is True
    assert bible.characters[0].first_chapter == 3


# --------------------------------------------------------------------------- register_speaker
def test_register_speaker_counts_lines_and_returns_canonical() -> None:
    bible = _bible(CharacterEntry(name="Mara Quill", aliases=["Mara"]))
    out, canonical = register_speaker(bible, "Mara", 1)
    assert canonical == "Mara Quill"
    assert out.characters[0].line_count == 1 and out.characters[0].first_chapter == 1
    assert bible.characters[0].line_count == 0, "inputs are never mutated"
    out, canonical = register_speaker(out, NARRATOR, 1)
    assert canonical == NARRATOR and len(out.characters) == 1
    out, canonical = register_speaker(out, "Tobias", 2)
    assert canonical == "Tobias" and out.characters[1].provisional is False and out.characters[1].line_count == 1


# --------------------------------------------------------------------------- finalize
def test_finalize_subset_merge_and_provisional_drop() -> None:
    bible = _bible(
        CharacterEntry(name="Mara Quill", gender="female", line_count=3, first_chapter=1),
        CharacterEntry(name="Mara", age="adult", line_count=2, first_chapter=2, voice_notes="warm"),
        CharacterEntry(name="the rider", provisional=True, line_count=0),
        CharacterEntry(name="the stranger", provisional=True, line_count=2),
        CharacterEntry(name="Tobias", line_count=1),
    )
    final = finalize(bible)
    assert _names(final) == ["Mara Quill", "The Stranger", "Tobias"]
    mara = final.characters[0]
    assert mara.aliases == ["Mara"] and mara.line_count == 5 and mara.age == "adult" and mara.voice_notes == "warm"
    assert final.characters[1].provisional is True
    assert final.version == bible.version + 1
    assert _names(bible) == ["Mara Quill", "Mara", "the rider", "the stranger", "Tobias"], "input untouched"


def test_finalize_skips_ambiguous_subset() -> None:
    bible = _bible(CharacterEntry(name="Mara Quill"), CharacterEntry(name="Mara Vey"), CharacterEntry(name="Mara"))
    assert _names(finalize(bible)) == ["Mara Quill", "Mara Vey", "Mara"]


def test_finalize_is_idempotent_on_clean_bible() -> None:
    bible = _bible(CharacterEntry(name="Mara Quill", aliases=["Mara"], line_count=1))
    final = finalize(bible)
    assert final == bible


# --------------------------------------------------------------------------- fingerprints
def test_fingerprint_stable_across_json_round_trip_and_changes_with_version() -> None:
    bible = apply_updates(CastBible(), [CharacterUpdate(name="Mara Quill", aliases=["Mara"], gender="female")], 1)
    round_tripped = CastBible.model_validate_json(bible.model_dump_json())
    assert round_tripped.fingerprint() == bible.fingerprint()
    assert round_tripped == bible
    bumped = bible.model_copy(update={"version": bible.version + 1})
    assert bumped.fingerprint() != bible.fingerprint()
    updated = apply_updates(bible, [CharacterUpdate(name="Tobias")], 1)
    assert updated.fingerprint() != bible.fingerprint()
    assert updated.version == bible.version + 1


# --------------------------------------------------------------------------- helpers
def test_speaker_line_counts() -> None:
    def seg(id_: str, speaker: str, kind: str) -> Segment:
        return Segment(id=id_, paragraph_index=1, speaker=speaker, kind=kind, text="x")  # type: ignore[arg-type]

    scripts = [
        ChapterScript(chapter_index=1, title="a", segments=[seg("a", NARRATOR, "narration"), seg("b", "Mara Quill", "dialogue")]),
        ChapterScript(chapter_index=2, title="b", segments=[seg("c", "Mara Quill", "dialogue"), seg("d", "Tobias", "dialogue")]),
    ]
    assert speaker_line_counts(scripts) == {"Mara Quill": 2, "Tobias": 1}
