"""bookreader.analysis.schema - the JSON schema Claude's structured output must follow.

``CHUNK_ANALYSIS_SCHEMA`` mirrors :class:`bookreader.types.ChunkAnalysis` minus the bookkeeping
fields (``source``, ``warnings``) that the analyzer sets itself. Every object in it has
``additionalProperties: false`` and a ``required`` list naming all of its properties, which the
Anthropic structured-output API demands; enums are rendered from the constant tuples in
:mod:`bookreader.types` so the two can never drift apart.

``PROMPT_VERSION`` is part of the analyzer's ``cache_version`` and the last line of the system
prompt: bump it whenever the schema or the prompts change in a way that affects results.
"""
from __future__ import annotations

from typing import Any

from bookreader.types import AGES, DELIVERIES, EMOTIONS, GENDERS, MOODS

PROMPT_VERSION = "2"

SFX_KINDS: tuple[str, ...] = ("impact", "ambient")
MUSIC_ACTIONS: tuple[str, ...] = ("start", "change", "stop")


def _enum(values: tuple[str, ...]) -> dict[str, Any]:
    return {"type": "string", "enum": list(values)}


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    """A closed object: no extra keys, every property required (listed in sorted order)."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(properties),
        "properties": dict(properties),
    }


def _array_of(item: dict[str, Any]) -> dict[str, Any]:
    return {"type": "array", "items": item}


LABEL_SCHEMA: dict[str, Any] = _object(
    {
        "span_id": {"type": "string"},
        "speaker": {"type": "string"},
        "emotion": _enum(EMOTIONS),
        "delivery": _enum(DELIVERIES),
    }
)

CHARACTER_SCHEMA: dict[str, Any] = _object(
    {
        "name": {"type": "string"},
        "aliases": _array_of({"type": "string"}),
        "gender": _enum(GENDERS),
        "age": _enum(AGES),
        "description": {"type": "string"},
        "voice_notes": {"type": "string"},
        "merge_into": {"type": ["string", "null"]},
    }
)

SFX_CUE_SCHEMA: dict[str, Any] = _object(
    {
        "span_id": {"type": "string"},
        "anchor_text": {"type": "string"},
        "description": {"type": "string"},
        "kind": _enum(SFX_KINDS),
        "duration_s": {"type": "number"},
        "intensity": {"type": "number"},
    }
)

MUSIC_CUE_SCHEMA: dict[str, Any] = _object(
    {
        "span_id": {"type": "string"},
        "action": _enum(MUSIC_ACTIONS),
        "mood": _enum(MOODS),
        "energy": {"type": "number"},
        "prompt": {"type": "string"},
    }
)

CHUNK_ANALYSIS_SCHEMA: dict[str, Any] = _object(
    {
        "labels": _array_of(LABEL_SCHEMA),
        "characters": _array_of(CHARACTER_SCHEMA),
        "sfx_cues": _array_of(SFX_CUE_SCHEMA),
        "music_cues": _array_of(MUSIC_CUE_SCHEMA),
    }
)
