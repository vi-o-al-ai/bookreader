"""bookreader.analysis.bible - the cast bible: character identity threaded across chunks.

Every function returns a new :class:`CastBible`; inputs are never mutated. Names are compared
through :func:`bookreader.types.normalize_name` (case-insensitive, honorific-stripped).
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Iterable

from bookreader.types import NARRATOR, CastBible, ChapterScript, CharacterEntry, CharacterUpdate, normalize_name

log = logging.getLogger(__name__)

HONORIFIC_RE = re.compile(r"^(old|young|mr|mrs|ms|miss|dr|captain|sir|lady|lord|aunt|uncle)\.?\s+", re.I)
DESCRIPTOR_RE = re.compile(r"^(the|a|an)\s+\S", re.I)
UNKNOWN = "unknown"


# --------------------------------------------------------------------------- name helpers
def is_descriptor(name: str) -> bool:
    """True for descriptor names such as ``"the stranger"`` or ``"a rider"`` (provisional entries)."""
    return bool(DESCRIPTOR_RE.match(name.strip()))


def strip_honorific(name: str) -> tuple[str, str | None]:
    """``"Old Hetta"`` -> ``("Hetta", "Old Hetta")``; a name without honorific -> ``(name, None)``."""
    cleaned = " ".join(name.split())
    stripped = HONORIFIC_RE.sub("", cleaned)
    if stripped and stripped != cleaned:
        return stripped, cleaned
    return cleaned, None


def title_case(name: str) -> str:
    """Capitalize each word without ``str.title``'s apostrophe quirk (``"the stranger's"`` stays sane)."""
    return " ".join(word[:1].upper() + word[1:] for word in name.split())


def _same(a: str, b: str) -> bool:
    return normalize_name(a) == normalize_name(b)


def _agrees(a: str, b: str) -> bool:
    return a == b or UNKNOWN in (a, b)


def _name_tokens(name: str) -> list[str]:
    return [normalize_name(token) for token in name.split() if normalize_name(token)]


def is_token_subset(single: str, multi: str) -> bool:
    """True when *single* is one token that occurs among the tokens of the multi-token *multi*."""
    key = normalize_name(single)
    tokens = _name_tokens(multi)
    return bool(key) and " " not in key and len(tokens) > 1 and key in tokens


def alias_key(name: str) -> str:
    """Case- and whitespace-insensitive key for alias deduplication. Unlike ``normalize_name``
    it keeps honorifics, so ``"Old Hetta"`` survives as an alias of ``"Hetta"``."""
    return " ".join(name.lower().split())


def merge_aliases(canonical: str, *groups: Iterable[str]) -> list[str]:
    """Ordered union of alias lists, deduplicated by :func:`alias_key` and excluding *canonical*."""
    seen = {alias_key(canonical)}
    out: list[str] = []
    for group in groups:
        for alias in group:
            alias = " ".join(alias.split())
            key = alias_key(alias)
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(alias)
    return out


def merge_voice_notes(existing: str, incoming: str) -> str:
    """Comma-separated union of voice descriptors, keeping first-seen order."""
    parts: list[str] = []
    seen: set[str] = set()
    for text in (existing, incoming):
        for part in text.split(","):
            part = part.strip()
            key = part.lower()
            if part and key not in seen:
                seen.add(key)
                parts.append(part)
    return ", ".join(parts)


# --------------------------------------------------------------------------- merge rules
def _traits_agree(entry: CharacterEntry, update: CharacterUpdate) -> bool:
    return _agrees(entry.gender, update.gender) and _agrees(entry.age, update.age)


def _provisional_match(bible: CastBible, update: CharacterUpdate) -> CharacterEntry | None:
    """A provisional entry whose descriptor the update mentions (alias or description) and whose
    gender/age do not contradict the update."""
    mentioned = {normalize_name(a) for a in update.aliases}
    description = update.description.lower()
    for entry in bible.characters:
        if not entry.provisional or not _traits_agree(entry, update):
            continue
        key = normalize_name(entry.name)
        if key and (key in mentioned or key in description):
            return entry
    return None


def _subset_match(bible: CastBible, name: str) -> CharacterEntry | None:
    """The unique non-provisional entry whose single-token name is a token of the multi-token
    *name*, or whose multi-token name contains the single-token *name*."""
    hosts = [
        entry for entry in bible.characters
        if not entry.provisional and (is_token_subset(entry.name, name) or is_token_subset(name, entry.name))
    ]
    return hosts[0] if len(hosts) == 1 else None


def _find_target(bible: CastBible, update: CharacterUpdate) -> CharacterEntry | None:
    """Entry the update describes: by name, by one of its aliases, by unique token subset, or a
    provisional entry whose descriptor it mentions."""
    target = bible.find(update.name)
    for alias in update.aliases:
        if target is not None:
            break
        candidate = bible.find(alias)
        if candidate is not None and (not candidate.provisional or _traits_agree(candidate, update)):
            target = candidate
    return target or _subset_match(bible, update.name) or _provisional_match(bible, update)


def _merge_update(entry: CharacterEntry, update: CharacterUpdate) -> None:
    """Fold *update* into *entry* in place (entry is already a private copy)."""
    incoming, honorific_form = strip_honorific(update.name)
    extra_aliases: list[str] = [update.name] if honorific_form else []
    if incoming and not _same(incoming, entry.name):
        rename = (entry.provisional and not is_descriptor(incoming)) or is_token_subset(entry.name, incoming)
        if rename:
            extra_aliases.append(entry.name)
            entry.name = incoming
            entry.provisional = False
        else:
            extra_aliases.append(incoming)
    entry.aliases = merge_aliases(entry.name, entry.aliases, update.aliases, extra_aliases)
    if len(update.description) > len(entry.description):
        entry.description = update.description
    entry.voice_notes = merge_voice_notes(entry.voice_notes, update.voice_notes)
    if entry.gender == UNKNOWN:
        entry.gender = update.gender
    if entry.age == UNKNOWN:
        entry.age = update.age


def _new_entry(update: CharacterUpdate, chapter_index: int) -> CharacterEntry:
    name, honorific_form = strip_honorific(update.name)
    provisional = is_descriptor(name)
    return CharacterEntry(
        name=name,
        aliases=merge_aliases(name, update.aliases, [honorific_form] if honorific_form else []),
        gender=update.gender,
        age=update.age,
        description=update.description,
        voice_notes=merge_voice_notes("", update.voice_notes),
        first_chapter=chapter_index,
        provisional=provisional,
    )


def apply_updates(bible: CastBible, updates: list[CharacterUpdate], chapter_index: int) -> CastBible:
    """Return a new bible with every update merged in order.

    Target resolution: ``merge_into`` (when it resolves), else the update's name or one of its
    aliases, else the unique entry related by a single-token/multi-token name subset
    (``Ansel`` <-> ``Ansel Vey``), else a provisional entry whose descriptor the update mentions. A found entry gains the union of
    aliases, the longer description, concatenated voice notes, gender/age upgrades from
    'unknown', and a canonical-name upgrade (single token -> containing multi-token name, or a
    descriptor -> the real name). Otherwise a new entry is appended. ``version`` grows by one
    per applied update.
    """
    out = bible.model_copy(deep=True)
    for update in updates:
        if not update.name.strip():
            log.warning("ignoring character update with an empty name")
            continue
        target: CharacterEntry | None = None
        if update.merge_into:
            target = out.find(update.merge_into)
            if target is None:
                log.warning("merge_into %r for %r does not match any entry; treating as a plain update", update.merge_into, update.name)
        if target is None:
            target = _find_target(out, update)
        if target is None:
            out.characters.append(_new_entry(update, chapter_index))
        else:
            _merge_update(target, update)
        out.version += 1
    return out


def register_speaker(bible: CastBible, name: str, chapter_index: int) -> tuple[CastBible, str]:
    """Count one dialogue line for *name*, creating an entry when the analyzer named someone the
    bible does not know (provisional when the name is a descriptor). Returns the canonical name.
    """
    if not name.strip() or normalize_name(name) == NARRATOR.lower():
        return bible, NARRATOR
    out = bible.model_copy(deep=True)
    entry = out.find(name)
    if entry is None:
        canonical, honorific_form = strip_honorific(name)
        provisional = is_descriptor(canonical)
        entry = CharacterEntry(
            name=canonical,
            aliases=[honorific_form] if honorific_form else [],
            first_chapter=chapter_index,
            provisional=provisional,
        )
        out.characters.append(entry)
        out.version += 1
        log.warning("speaker %r missing from the cast bible; created a %s entry", name, "provisional" if provisional else "new")
    entry.line_count += 1
    if entry.first_chapter == 0:
        entry.first_chapter = chapter_index
    return out, entry.name


def _absorb(host: CharacterEntry, other: CharacterEntry) -> None:
    host.aliases = merge_aliases(host.name, host.aliases, [other.name], other.aliases)
    host.line_count += other.line_count
    if host.gender == UNKNOWN:
        host.gender = other.gender
    if host.age == UNKNOWN:
        host.age = other.age
    if len(other.description) > len(host.description):
        host.description = other.description
    host.voice_notes = merge_voice_notes(host.voice_notes, other.voice_notes)
    if other.first_chapter and (host.first_chapter == 0 or other.first_chapter < host.first_chapter):
        host.first_chapter = other.first_chapter


def finalize(bible: CastBible) -> CastBible:
    """End-of-book cleanup: a single-token name that is a token of exactly one multi-token name
    becomes that entry's alias (``Mara`` -> ``Mara Quill``); provisional entries that never spoke
    are dropped; surviving provisional names are title-cased.
    """
    out = bible.model_copy(deep=True)
    changed = False
    multis = [c for c in out.characters if not c.provisional and len(c.name.split()) > 1]
    for single in [c for c in out.characters if not c.provisional and len(c.name.split()) == 1]:
        hosts = [m for m in multis if is_token_subset(single.name, m.name)]
        if len(hosts) == 1:
            _absorb(hosts[0], single)
            out.characters.remove(single)
            changed = True
            log.info("bible finalize: merged %r into %r", single.name, hosts[0].name)
    survivors: list[CharacterEntry] = []
    for entry in out.characters:
        if entry.provisional and entry.line_count == 0:
            changed = True
            log.info("bible finalize: dropped silent provisional entry %r", entry.name)
            continue
        if entry.provisional:
            titled = title_case(entry.name)
            if titled != entry.name:
                entry.name = titled
                changed = True
        survivors.append(entry)
    out.characters = survivors
    if changed:
        out.version += 1
    return out


def speaker_line_counts(scripts: Iterable[ChapterScript]) -> dict[str, int]:
    """Dialogue lines per speaker across *scripts* (narration segments are not counted)."""
    counts: Counter[str] = Counter()
    for script in scripts:
        for segment in script.segments:
            if segment.kind == "dialogue":
                counts[segment.speaker] += 1
    return dict(counts)
