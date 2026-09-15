"""CHUNK_ANALYSIS_SCHEMA: closed objects everywhere, enums from the type constants, and a sample
that satisfies both the schema and ChunkAnalysis.model_validate."""
from __future__ import annotations

from typing import Any

from bookreader.analysis.schema import (
    CHARACTER_SCHEMA,
    CHUNK_ANALYSIS_SCHEMA,
    LABEL_SCHEMA,
    MUSIC_CUE_SCHEMA,
    PROMPT_VERSION,
    SFX_CUE_SCHEMA,
)
from bookreader.types import AGES, DELIVERIES, EMOTIONS, GENDERS, MOODS, ChunkAnalysis

SAMPLE: dict[str, Any] = {
    "labels": [
        {"span_id": "c1p2s0", "speaker": "Mara Quill", "emotion": "urgent", "delivery": "shout"},
        {"span_id": "c1p3s1", "speaker": "Tobias", "emotion": "afraid", "delivery": "normal"},
    ],
    "characters": [
        {
            "name": "Mara Quill", "aliases": ["Mara"], "gender": "female", "age": "adult",
            "description": "the lighthouse keeper", "voice_notes": "steady, low", "merge_into": None,
        },
        {
            "name": "Ansel Vey", "aliases": ["the stranger"], "gender": "male", "age": "adult",
            "description": "first mate of the Corvid", "voice_notes": "dry, rasping", "merge_into": "the stranger",
        },
    ],
    "sfx_cues": [
        {"span_id": "c1p4s0", "anchor_text": "Thunder split the sky", "description": "a single deep thunderclap",
         "kind": "impact", "duration_s": 3.0, "intensity": 0.9},
    ],
    "music_cues": [
        {"span_id": "c1p1s0", "action": "start", "mood": "tense", "energy": 0.7, "prompt": "low strings, storm"},
    ],
}


def walk_objects(node: Any) -> list[dict[str, Any]]:
    """Every schema node of type object, depth first."""
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        if node.get("type") == "object":
            found.append(node)
        for value in node.values():
            found.extend(walk_objects(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(walk_objects(item))
    return found


def test_every_object_is_closed_and_fully_required() -> None:
    objects = walk_objects(CHUNK_ANALYSIS_SCHEMA)
    assert len(objects) == 5, "root plus label, character, sfx cue and music cue"
    for obj in objects:
        assert obj["additionalProperties"] is False
        assert obj["required"] == sorted(obj["properties"])


def test_top_level_shape() -> None:
    assert CHUNK_ANALYSIS_SCHEMA["type"] == "object"
    assert sorted(CHUNK_ANALYSIS_SCHEMA["properties"]) == ["characters", "labels", "music_cues", "sfx_cues"]
    for key, item in (("labels", LABEL_SCHEMA), ("characters", CHARACTER_SCHEMA), ("sfx_cues", SFX_CUE_SCHEMA), ("music_cues", MUSIC_CUE_SCHEMA)):
        prop = CHUNK_ANALYSIS_SCHEMA["properties"][key]
        assert prop["type"] == "array"
        assert prop["items"] is item
    assert CHARACTER_SCHEMA["properties"]["merge_into"]["type"] == ["string", "null"]
    assert CHARACTER_SCHEMA["properties"]["aliases"] == {"type": "array", "items": {"type": "string"}}
    for name in ("duration_s", "intensity"):
        assert SFX_CUE_SCHEMA["properties"][name] == {"type": "number"}
    assert MUSIC_CUE_SCHEMA["properties"]["energy"] == {"type": "number"}


def test_enums_match_type_constants() -> None:
    assert LABEL_SCHEMA["properties"]["emotion"]["enum"] == list(EMOTIONS)
    assert LABEL_SCHEMA["properties"]["delivery"]["enum"] == list(DELIVERIES)
    assert CHARACTER_SCHEMA["properties"]["gender"]["enum"] == list(GENDERS)
    assert CHARACTER_SCHEMA["properties"]["age"]["enum"] == list(AGES)
    assert MUSIC_CUE_SCHEMA["properties"]["mood"]["enum"] == list(MOODS)
    assert MUSIC_CUE_SCHEMA["properties"]["action"]["enum"] == ["start", "change", "stop"]
    assert SFX_CUE_SCHEMA["properties"]["kind"]["enum"] == ["impact", "ambient"]


def test_sample_validates_as_chunk_analysis() -> None:
    analysis = ChunkAnalysis.model_validate(SAMPLE)
    assert [label.speaker for label in analysis.labels] == ["Mara Quill", "Tobias"]
    assert analysis.characters[1].merge_into == "the stranger"
    assert analysis.sfx_cues[0].kind == "impact"
    assert analysis.music_cues[0].action == "start"
    assert analysis.source == "heuristic", "the schema carries no source; the analyzer sets it"
    assert analysis.warnings == []


def test_sample_covers_every_required_key() -> None:
    """Belt and braces: the sample uses exactly the keys the schema requires at every level."""
    assert sorted(SAMPLE) == CHUNK_ANALYSIS_SCHEMA["required"]
    for key, item in (("labels", LABEL_SCHEMA), ("characters", CHARACTER_SCHEMA), ("sfx_cues", SFX_CUE_SCHEMA), ("music_cues", MUSIC_CUE_SCHEMA)):
        for entry in SAMPLE[key]:
            assert sorted(entry) == item["required"]


def test_prompt_version_is_a_stable_string() -> None:
    assert PROMPT_VERSION == "2"
