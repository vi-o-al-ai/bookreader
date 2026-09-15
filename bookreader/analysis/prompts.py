"""bookreader.analysis.prompts - the system prompt and user message for the Claude analyzer.

``SYSTEM_PROMPT`` is byte-stable: no timestamps, vocabularies rendered from the constant tuples
in :mod:`bookreader.types`, and it ends with the prompt version. It is sent with
``cache_control: ephemeral`` so every chunk of every job hits the same cached prefix.
:func:`build_user_message` renders one chunk (bible, mood in force, read-only context, the spans
to label) and, on the repair pass, the note describing what was wrong with the previous reply.
The first span of every paragraph listed in ``Chunk.scene_break_paragraphs`` carries a
``"scene_break": true`` key so the model knows where a new scene starts.
"""
from __future__ import annotations

import json
import re

from bookreader.analysis.schema import PROMPT_VERSION
from bookreader.types import AGES, DELIVERIES, EMOTIONS, GENDERS, MOODS, NARRATOR, CastBible, Chunk


_SPAN_PARAGRAPH_RE = re.compile(r"^c\d+p(\d+)s\d+$")   # span id -> its 1-based paragraph index


def _vocabulary(values: tuple[str, ...]) -> str:
    return ", ".join(values)


SYSTEM_PROMPT: str = f"""You are an audio-drama script editor. You label a novel, one chunk at a time, so it can be performed by multiple voices with music and sound effects.

## Input
The user message carries the cast bible so far (JSON), the music mood currently playing, a few previous paragraphs for context only (already processed; never label them), and a numbered list of spans to label. Each span has an id, a kind ("narration" or "quote") and its verbatim text. A span carrying "scene_break": true is the first span of a paragraph that opens a new scene: the conversation starts afresh there, so the participants and turn order of earlier paragraphs do not carry across it, and the music mood may change. Do NOT rewrite, correct or quote back any span text; refer to spans only by id.

## Labels
Output exactly one label per QUOTE span. Narration spans may be omitted: they are always spoken by {NARRATOR}. Attribute each quote in this order of evidence:
1. an explicit speech tag in the adjacent narration ("said Mara", "Tobias asked");
2. a noun-phrase tag resolved through the bible descriptions ("the boy" is the child, "the old woman" is the elderly woman);
3. a pronoun tag ("he whispered") resolved by gender to the most recently mentioned matching character;
4. conversational alternation between the participants of the exchange.
A name addressed inside a quote ("Hetta," Mara warned) is the addressee, not the speaker. A reply to "what is your name" is spoken by the person who is named in it. Use canonical names from the bible exactly as written; a person not yet in the bible is named by the most complete name the text gives.
Give each label one emotion and one delivery from the vocabularies below.

## Characters
Add a character entry only for people who speak. Give the full name, aliases (other names, descriptors such as "the stranger" once resolved), gender, age, a one-line description, and voice_notes (timbre words: dry, warm, gravelly, bright...). When a new mention is the same person as an existing bible entry, set merge_into to that entry's canonical name; otherwise merge_into is null.

## Sound effects
Emit sfx only for concrete audible events described in NARRATION spans, at most 3 per paragraph. anchor_text is copied verbatim from the span (the phrase where the sound happens). kind is "impact" for one-off sounds (thunder, a slam, a bell, a snap, hooves) or "ambient" for continuous ones (wind, rain, the sea, fire, gulls). duration_s is 0.5 to 12 for impacts and 10 to 20 for ambient beds; intensity is 0 to 1; description is a short sound-design prompt.

## Music
Emit music cues only at scene starts or clear mood shifts relative to the mood currently playing: action "start" when no music is playing, "change" on a shift, "stop" when music should fall silent. prompt is a short instrumental description with no vocals; energy is 0 to 1. Keep cues sparse: most chunks need none or one.

## Vocabularies
emotion: {_vocabulary(EMOTIONS)}
delivery: {_vocabulary(DELIVERIES)}
mood: {_vocabulary(MOODS)}
gender: {_vocabulary(GENDERS)}
age: {_vocabulary(AGES)}

Reply with JSON matching the provided schema and nothing else.
prompt-version: {PROMPT_VERSION}"""


def build_user_message(chunk: Chunk, bible: CastBible, repair_note: str | None = None) -> str:
    """The user turn for *chunk*: bible, mood in force, context, spans, and an optional repair note.

    The first span of each paragraph in ``chunk.scene_break_paragraphs`` gets ``"scene_break": true``;
    every other span is rendered as just ``id``, ``kind`` and ``text``.
    """
    scene_breaks = set(chunk.scene_break_paragraphs)
    marked: set[int] = set()
    spans: list[dict[str, object]] = []
    for span in chunk.spans:
        entry: dict[str, object] = {"id": span.id, "kind": span.kind}
        match = _SPAN_PARAGRAPH_RE.match(span.id)
        paragraph = int(match.group(1)) if match else None
        if paragraph is not None and paragraph in scene_breaks and paragraph not in marked:
            entry["scene_break"] = True
            marked.add(paragraph)
        entry["text"] = span.text
        spans.append(entry)
    parts = [
        "## Cast bible (JSON)\n" + bible.to_prompt_json(),
        "## Music mood currently playing\n" + chunk.prior_mood,
        "## Previous paragraphs (context only, already processed)\n" + chunk.context_before,
        f"## Spans to label (chapter {chunk.chapter_index}, chunk {chunk.chunk_index})\n"
        + json.dumps(spans, ensure_ascii=False, indent=1),
    ]
    if repair_note:
        parts.append("## Repair note\n" + repair_note)
    return "\n".join(parts)
