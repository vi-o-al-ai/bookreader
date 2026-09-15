"""bookreader.providers.mock.analysis - HeuristicAnalyzer, the rule-based offline text analyzer.

The mock family's ``analysis`` capability: no model, no network, no randomness. It labels every
quote span of a chunk with a speaker (explicit tags, pronoun tags, descriptor tags, leading
subjects, paragraph sharing, reply cues, conversational alternation; the conversation resets
at chapter start, at every scene break the chunk reports and on time transitions), infers emotion and
delivery from tag verbs and punctuation, harvests characters from narration, emits SFX cues
anchored to verbatim phrases, and music start/change actions from paragraph mood keywords.
:func:`bookreader.analysis.validate.validate_chunk_analysis` also uses it to fill labels a
model forgot (:meth:`HeuristicAnalyzer.label_missing`).
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from bookreader.analysis.assemble import MOOD_PROMPTS
from bookreader.analysis.bible import is_token_subset, alias_key, strip_honorific
from bookreader.providers.base import NullUsage, UsageSink
from bookreader.types import (
    NARRATOR,
    CastBible,
    CharacterUpdate,
    Chunk,
    ChunkAnalysis,
    Delivery,
    Emotion,
    MusicCueRaw,
    SfxCueRaw,
    Span,
    SpanLabel,
    normalize_name,
)

if TYPE_CHECKING:
    from bookreader.settings import Settings

log = logging.getLogger(__name__)

UNKNOWN = "unknown"

# --------------------------------------------------------------------------- vocabularies
SPEECH_VERBS: tuple[str, ...] = (
    "said", "asked", "shouted", "whispered", "warned", "replied", "called", "cried", "laughed",
    "muttered", "sniffed", "added", "answered", "snapped", "breathed", "coughed", "hissed",
    "growled", "sighed", "murmured", "yelled", "exclaimed", "demanded", "insisted", "repeated",
    "agreed", "gasped", "grumbled", "spoke", "says", "asks", "shouts", "whispers", "replies",
)

STOPWORDS: frozenset[str] = frozenset(
    """
    The A An This That These Those Some Any Every Each No All Both Either Neither Another Other Such
    I You He She It We They Me Him Her Us Them My Your His Its Our Their Mine Yours Hers Ours Theirs
    Myself Himself Herself Itself Ourselves Themselves Who Whom Whose What Which Whatever Whoever
    Someone Somebody Anyone Anybody Everyone Everybody Nobody Nothing Something Anything Everything One
    In On At By To For From With Without Of Off Over Under Into Onto Out Up Down Through Across Along
    Around Behind Beyond Before After Above Below Between Among Against During Until Till Since Toward
    Towards Inside Outside Near Beside Besides And But Or Nor So Yet If Then Than As Because Although
    Though While Whereas Unless Whether When Whenever Where Wherever Why How However Therefore Thus
    Hence Once Twice Again Also Too Even Still Already Just Only Almost Perhaps Maybe Please Indeed
    Instead Rather Quite Very Really Never Always Often Sometimes Soon Now Today Tonight Tomorrow
    Yesterday Later Earlier Afterwards Meanwhile Morning Evening Night Noon Midnight Dawn Dusk
    Somewhere Anywhere Nowhere Everywhere Here There Home Away Back Yes Not Oh Ah Well Hush Hello Hi
    Hey Look Wait Stop Go Come Run Right Fine Easy Good Bad Sorry Thanks Thank Help Fire Enough Ha Aye
    Nay Sir Ma'am Madam Lad Lass Boy Girl Man Woman Mother Father Mama Papa Mum Dad Chapter Part Book
    Prologue Epilogue Thunder Gull Gulls Point Harbour Harbor Board Sea Wind Storm Rain Sun Moon Sky
    Earth God Heaven Hell Christmas Easter January February March April May June July August September
    October November December Monday Tuesday Wednesday Thursday Friday Saturday Sunday Spring Summer
    Autumn Winter Fall North South East West Ring Stand Leave Words First Second Third Last Next Nine
    Ten Old Young Mr Mrs Ms Miss Dr Doctor Captain Aunt Uncle Lady Lord Dear Come
    """.split()
)

HONORIFICS = r"(?:Old|Young|Mr|Mrs|Ms|Miss|Dr|Captain|Sir|Lady|Lord|Aunt|Uncle)\.?"
NAME_TOKEN = r"[A-Z][\w']+"
NAME_RUN = rf"(?:{HONORIFICS} )?{NAME_TOKEN}(?: {NAME_TOKEN})?"
VERBS = "|".join(SPEECH_VERBS)
DESCRIPTOR_PHRASE = r"the (?:old (?:man|woman)|man|woman|stranger|boy|girl|rider)"

NAME_RUN_RE = re.compile(rf"\b{NAME_RUN}")
NAME_TOKEN_RE = re.compile(r"[A-Z][a-z][\w']*")
TAG_NAME_RE = re.compile(rf"\b(?P<name>{NAME_RUN})\s+(?P<verb>{VERBS})\b")
TAG_INVERTED_RE = re.compile(rf"\b(?P<verb>{VERBS})\s+(?P<name>{NAME_RUN})")
TAG_VERB_AFTER_RE = re.compile(rf"\s+(?:{VERBS})\b")
TAG_VERB_BEFORE_RE = re.compile(rf"\b(?:{VERBS})\s+$")
PRONOUN_TAG_RE = re.compile(rf"\b(?P<pron>he|she)\s+(?:\w+ly\s+)?(?P<verb>{VERBS})\b", re.I)
DESCRIPTOR_TAG_RE = re.compile(rf"\b(?P<desc>{DESCRIPTOR_PHRASE})\s+(?P<verb>{VERBS})\b", re.I)
DESCRIPTOR_RE = re.compile(rf"\b(?P<desc>{DESCRIPTOR_PHRASE})\b", re.I)
LEADING_PRONOUN_RE = re.compile(r"(?P<pron>he|she)\b", re.I)
GENDER_PRONOUN_RE = re.compile(r"\b(?:(?P<male>he|his)|(?P<female>she|her))\b", re.I)
DESCRIPTION_RE = re.compile(rf"{NAME_RUN}(?:, who [^,.;]+|,? (?:was|is|were) [^,.;]+)")
TIME_TRANSITION_RE = re.compile(
    r"^(?:Later|Afterwards|Meanwhile|That (?:night|evening|morning)|The next|A (?:week|day|month|year)|"
    r"On the \w+ day|Morning|By (?:morning|evening))\b"
)
REPLY_CUE_RE = re.compile(r"\b(?:repl(?:y|ied|ies)|answer(?:ed|s)?|respon(?:se|ded))\b", re.I)
NAME_QUESTION_RE = re.compile(r"\b(?:your name|who are you)\b", re.I)
SENTENCE_END_RE = re.compile(r"[.!?][\"'”’)]*\s*$")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
VOCATIVE_RE = re.compile(r"^(?P<name>[A-Z][\w']+)[!,]$")
ADDRESS_TAIL_RE = re.compile(r",\s+(?P<name>[A-Z][\w']+)[.!?]*$")
ADDRESS_HEAD_RE = re.compile(r"^(?P<name>[A-Z][\w']+),\s")
WORD_RE = re.compile(r"[A-Za-z][\w']*")
POSSESSIVE_RE = re.compile(r"['’]s$")
QUOTED_RE = re.compile(r"[\"“]([^\"”]+)[\"”]")
NAME_RUN_SPLIT_RE = re.compile(r"[.,;:!?]+")

DESCRIPTOR_TRAITS: dict[str, tuple[str, str]] = {
    "the man": ("male", "adult"),
    "the woman": ("female", "adult"),
    "the stranger": (UNKNOWN, "adult"),
    "the rider": (UNKNOWN, "adult"),
    "the boy": ("male", "child"),
    "the girl": ("female", "child"),
    "the old man": ("male", "elderly"),
    "the old woman": ("female", "elderly"),
}

AGE_WORDS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:twelve|eleven|ten|nine|eight|seven|six)\b(?!\s+(?:minutes|hours|days|weeks|months|years|miles|feet|yards|paces|men|women|of))|\b(?:boy|girl|lad|lass|child)\b", re.I), "child"),
    (re.compile(r"\b(?:thirteen|fourteen|fifteen|sixteen|seventeen|teenage[rd]?|youth)\b", re.I), "teen"),
    (re.compile(r"\b(?:sixty|seventy|eighty|ninety) winters\b|\belderly\b|\baged\b|\b(?:grey|gray|white)-haired\b|\bwrinkled\b|\b(?:was|is|looked|grown) (?:very )?old\b|\bold (?:man|woman|lady|fellow)\b", re.I), "elderly"),
)

VERB_CUES: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"\bshout"), "urgent", "shout"),
    (re.compile(r"\bwhisper"), "afraid", "whisper"),
    (re.compile(r"\bquiet(?:er|ly)\b|\bsoftly\b|\bmurmur"), "calm", "quiet"),
    (re.compile(r"\bwarned\b|\bsnapped\b"), "stern", "normal"),
    (re.compile(r"\blaugh|\bchuckl|\bgrinn"), "amused", "normal"),
    (re.compile(r"\bsniff"), "stern", "normal"),
    (re.compile(r"\bcough|\bshaking\b|\bwheez"), "weary", "strained"),
)
FEAR_RE = re.compile(r"\bfright|\bfear|\bshaking\b|\bcracked\b|\btrembl|\bterrif")
TENDER_RE = re.compile(r"\bI'll not forget\b|\bsaved my life\b", re.I)

VOICE_NOTE_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bdry rattle\b"), "dry, rasping"),
    (re.compile(r"\brattl"), "rasping"),
    (re.compile(r"\bhoarse"), "hoarse"),
    (re.compile(r"\bgruff"), "gruff"),
    (re.compile(r"\bdeep voice\b"), "deep"),
    (re.compile(r"\bsoft voice\b"), "soft"),
    (re.compile(r"\bshrill"), "shrill"),
    (re.compile(r"\bbooming"), "booming"),
    (re.compile(r"\bcroak"), "croaky"),
    (re.compile(r"\bwheez"), "wheezy"),
)


@dataclass(frozen=True)
class _SfxRule:
    name: str
    pattern: re.Pattern[str]
    anchor: re.Pattern[str] | None
    description: str
    kind: str
    duration_s: float
    intensity: float
    variants: tuple[tuple[re.Pattern[str], str, float], ...] = ()   # (condition in the span, description, duration)
    requires_absent: re.Pattern[str] | None = None                  # skip when this matches the paragraph


def _rule(
    name: str, pattern: str, anchor: str | None, description: str, kind: str, duration_s: float, intensity: float,
    variants: tuple[tuple[str, str, float], ...] = (), requires_absent: str | None = None,
) -> _SfxRule:
    return _SfxRule(
        name, re.compile(pattern, re.I), re.compile(anchor, re.I) if anchor else None, description, kind, duration_s, intensity,
        tuple((re.compile(cond, re.I), desc, dur) for cond, desc, dur in variants),
        re.compile(requires_absent, re.I) if requires_absent else None,
    )


SFX_RULES: tuple[_SfxRule, ...] = (
    _rule("thunder", r"\bthunder\w*", None, "a single deep thunderclap with a long distant rumble", "impact", 3.0, 0.9),
    _rule("slam", r"\bslam(?:med|ming|s)?\b", None, "a wooden shutter slamming hard against a stone wall", "impact", 1.5, 0.8,
          variants=((r"\bagain and again\b", "shutter banging repeatedly", 4.0),)),
    _rule("snap", r"\bsnap(?:ped|ping|s)?\b", None, "a taut rope snapping with a sharp crack", "impact", 0.8, 0.7),
    _rule("bell", r"\bbell\b[^.!?]*\brang\b|\brang\b[^.!?]*\bbell\b", r"\bbell\b",
          "a great bronze bell striking, deep resonant tone rolling out over water", "impact", 4.0, 0.8,
          variants=((r"\btwice\b", "two bell strikes, deep bronze tone with a long decay", 6.0),)),
    _rule("gulls", r"\bgulls?\b", None, "seagulls crying over rocks with a light sea breeze", "ambient", 12.0, 0.4),
    _rule("footsteps", r"\bboots crunch\w*|\bcrunching\b|\bfootsteps\b", r"\bcrunch\w*|\bfootsteps\b",
          "boots crunching on loose stones, hurried footsteps", "impact", 2.0, 0.5),
    _rule("fire", r"\bcrackled\b|\bcrackling fire\b", r"\bcrackl\w*", "a log fire crackling in a grate", "ambient", 15.0, 0.4),
    _rule("kettle", r"\bkettle\b[^.!?]*\bwhistl\w*", r"\bwhistl\w*", "a kettle coming to the boil, a rising whistle", "impact", 3.0, 0.5),
    _rule("hooves", r"\bhooves\b", None, "horse hooves clattering on a stone road, approaching", "impact", 4.0, 0.6),
    _rule("needles", r"\bneedles clicking\b", None, "knitting needles clicking softly and steadily", "ambient", 8.0, 0.3),
    _rule("wind", r"\bwind\b", None, "wind gusting and moaning around a stone building", "ambient", 15.0, 0.4),
    _rule("sea", r"\bsea\b|\bwaves\b[^.!?]*\bbreaking\b", r"\bsea\b|\bwaves\b", "waves breaking on rocks, steady sea wash", "ambient", 15.0, 0.4),
    _rule("knock", r"\bknock(?:ed|ing|s)?\b", None, "knuckles knocking on a wooden door", "impact", 1.0, 0.5),
    _rule("cough", r"\bcough(?:ed|ing|s)?\b", None, "a dry rattling cough", "impact", 1.0, 0.4),
    _rule("whistling", r"\bwhistling\b", None, "a man whistling a tune while he works", "impact", 3.0, 0.3, requires_absent=r"\bkettle\b"),
)
MAX_SFX_PER_PARAGRAPH = 3

MOOD_KEYWORDS: tuple[tuple[str, tuple[re.Pattern[str], ...]], ...] = (
    ("tense", tuple(re.compile(p, re.I) for p in (r"\bstorm", r"\bthunder", r"\bwind\b", r"\bslam", r"\bshout"))),
    ("calm", tuple(re.compile(p, re.I) for p in (r"\bgr[ae]y\b", r"\bquiet", r"\bgulls?\b", r"\bmorning\b", r"\bflat\b"))),
    ("warm", tuple(re.compile(p, re.I) for p in (r"\bfire\b", r"\btea\b", r"\bkettle\b", r"\bknitting\b", r"\bcottage\b"))),
    ("melancholy", tuple(re.compile(p, re.I) for p in (r"\bletter\b", r"\bclosing\b", r"\bleave\b", r"\bnobody answered\b", r"\bcrumpled\b"))),
    ("hopeful", tuple(re.compile(p, re.I) for p in (r"\bmended\b", r"\bwhistling\b", r"\bfollowed him everywhere\b"))),
)
MOOD_ENERGY: dict[str, float] = {"tense": 0.7}
DEFAULT_ENERGY = 0.3
MOOD_LOOKAHEAD = 3                      # paragraphs a new mood may look ahead to prove it persists
PRONOUN_WINDOW = 120                    # chars after a mention searched for a gendered pronoun
AGE_WINDOW = 80                         # chars after a mention searched for age words
MAX_DESCRIPTION = 140


# --------------------------------------------------------------------------- state
@dataclass
class _Character:
    name: str
    aliases: list[str] = field(default_factory=list)
    gender: str = UNKNOWN
    age: str = UNKNOWN
    description: str = ""
    voice_notes: list[str] = field(default_factory=list)
    provisional: bool = False
    bible_name: str | None = None          # canonical name in the bible when the chunk began
    mentions: int = 0
    attributed: int = 0

    def keys(self) -> set[str]:
        return {normalize_name(self.name), *(normalize_name(a) for a in self.aliases)} - {""}

    def add_alias(self, alias: str) -> None:
        alias = " ".join(alias.split())
        known = {alias_key(self.name), *(alias_key(a) for a in self.aliases)}
        if alias and alias_key(alias) not in known:
            self.aliases.append(alias)

    def rename(self, new_name: str) -> None:
        old = self.name
        self.name = new_name
        self.provisional = False
        self.add_alias(old)

    def add_voice_note(self, note: str) -> None:
        for part in note.split(","):
            part = part.strip()
            if part and part.lower() not in {n.lower() for n in self.voice_notes}:
                self.voice_notes.append(part)


@dataclass
class _Conversation:
    participants: list[_Character] = field(default_factory=list)   # most recent last
    speakers: list[_Character] = field(default_factory=list)       # last two distinct, most recent last
    by_gender: dict[str, _Character] = field(default_factory=dict)
    last_mention: _Character | None = None
    addressee: _Character | None = None
    last_quote: str = ""

    def reset(self) -> None:
        self.participants.clear()
        self.speakers.clear()
        self.by_gender.clear()
        self.last_mention = None
        self.addressee = None
        self.last_quote = ""

    @property
    def last_speaker(self) -> _Character | None:
        return self.speakers[-1] if self.speakers else None

    def mention(self, character: _Character) -> None:
        self.participants = [p for p in self.participants if p is not character] + [character]
        if character.gender != UNKNOWN:
            self.by_gender[character.gender] = character
        self.last_mention = character

    def spoke(self, character: _Character) -> None:
        self.mention(character)
        self.speakers = [s for s in self.speakers if s is not character] + [character]
        del self.speakers[:-2]

    def by_pronoun(self, gender: str) -> _Character | None:
        found = self.by_gender.get(gender)
        if found is not None:
            return found
        if self.last_mention is not None and self.last_mention.gender == UNKNOWN:
            self.last_mention.gender = gender
            self.by_gender[gender] = self.last_mention
            return self.last_mention
        return None

    def partner(self, speaker: _Character | None) -> tuple[_Character | None, int]:
        """The person *speaker* is talking to and how many candidates there were."""
        if self.addressee is not None and self.addressee is not speaker:
            return self.addressee, 1
        others = [p for p in reversed(self.participants) if p is not speaker]
        if not others:
            return None, 0
        return others[0], len(others)


@dataclass
class _Paragraph:
    index: int
    spans: list[Span]

    @property
    def narration_text(self) -> str:
        return " ".join(s.text for s in self.spans if s.kind == "narration")

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.spans)


def _first_sentence(text: str) -> str:
    return SENTENCE_SPLIT_RE.split(text.strip(), maxsplit=1)[0]


def _last_sentence(text: str) -> str:
    return SENTENCE_SPLIT_RE.split(text.strip())[-1]


def _agrees(a: str, b: str) -> bool:
    return a == b or UNKNOWN in (a, b)


def _emotion_delivery(quote: str, tag_text: str, speaker: _Character | None) -> tuple[Emotion, Delivery]:
    lowered = tag_text.lower()
    for pattern, emotion, delivery in VERB_CUES:
        if pattern.search(lowered):
            return emotion, delivery  # type: ignore[return-value]
    if FEAR_RE.search(lowered):
        return "afraid", ("shout" if "!" in quote else "normal")
    if TENDER_RE.search(quote):
        return "tender", "normal"
    if "!" in quote:
        return "urgent", "shout"
    if "..." in quote or "…" in quote:
        return "hesitant", "normal"
    if "?" in quote:
        return ("hesitant" if speaker is not None and speaker.age == "child" else "neutral"), "normal"
    return "neutral", "normal"


def _paragraphs_of(chunk: Chunk) -> list[_Paragraph]:
    groups: dict[str, _Paragraph] = {}
    for span in chunk.spans:
        match = re.match(r"c\d+p(\d+)s\d+$", span.id)
        key = match.group(1) if match else span.id
        if key not in groups:
            groups[key] = _Paragraph(index=int(key) if match else len(groups) + 1, spans=[])
        groups[key].spans.append(span)
    return list(groups.values())


# --------------------------------------------------------------------------- one chunk
class _ChunkRun:
    """All the state of one ``analyze_chunk`` call."""

    def __init__(self, chunk: Chunk, bible: CastBible) -> None:
        self.chunk = chunk
        self.paragraphs = _paragraphs_of(chunk)
        self.characters: list[_Character] = [
            _Character(
                name=entry.name,
                aliases=list(entry.aliases),
                gender=entry.gender,
                age=entry.age,
                description=entry.description,
                voice_notes=[n.strip() for n in entry.voice_notes.split(",") if n.strip()],
                provisional=entry.provisional,
                bible_name=entry.name,
            )
            for entry in bible.characters
        ]
        self.conv = _Conversation()
        self.warnings: list[str] = []
        self.labels: list[tuple[Span, _Character | None, Emotion, Delivery]] = []
        self.sfx: list[SfxCueRaw] = []
        self.scene_breaks: frozenset[int] = frozenset(chunk.scene_break_paragraphs)
        self.reset_paragraphs: set[int] = set()      # where the conversation restarted; a mood change there needs no look-ahead
        all_text = chunk.context_before + " " + " ".join(s.text for s in chunk.spans)
        self.token_counts: Counter[str] = Counter(WORD_RE.findall(all_text))

    # ------------------------------------------------------------------ driver
    def run(self) -> ChunkAnalysis:
        if not self.paragraphs:
            return ChunkAnalysis(source="heuristic")
        if self.chunk.chunk_index == 0:
            self.reset_paragraphs.add(self.paragraphs[0].index)
        else:
            self._seed_from_context(self.chunk.context_before)
        for paragraph in self.paragraphs:
            self._process_paragraph(paragraph)
        return ChunkAnalysis(
            labels=[
                SpanLabel(span_id=span.id, speaker=character.name if character else NARRATOR, emotion=emotion, delivery=delivery)
                for span, character, emotion, delivery in self.labels
            ],
            characters=self._character_updates(),
            sfx_cues=self.sfx,
            music_cues=self._music_cues(),
            source="heuristic",
            warnings=self.warnings,
        )

    # ------------------------------------------------------------------ characters
    def _lookup(self, name: str) -> _Character | None:
        key = normalize_name(name)
        if not key:
            return None
        for character in self.characters:
            if key in character.keys():
                return character
        if " " not in key:
            hits: list[_Character] = []
            for character in self.characters:
                for full in (character.name, *character.aliases):
                    tokens = full.split()
                    if len(tokens) > 1 and key in {normalize_name(t) for t in tokens} and character not in hits:
                        hits.append(character)
            if len(hits) == 1:
                return hits[0]
        return None

    def _create(self, name: str, **traits: object) -> _Character:
        character = _Character(name=name, **traits)  # type: ignore[arg-type]
        self.characters.append(character)
        return character

    def _resolve_name(self, raw: str, *, create: bool) -> _Character | None:
        """Resolve a capitalized run ('Old Hetta', "Ansel's", 'Mara') to a character."""
        cleaned, honorific_form = strip_honorific(POSSESSIVE_RE.sub("", raw.strip()))
        tokens = cleaned.split()
        while tokens and tokens[0] in STOPWORDS:
            tokens.pop(0)
        if not tokens or (len(tokens[0]) > 1 and tokens[0].isupper()):
            return None
        if len(tokens) > 1 and tokens[1] in STOPWORDS:
            tokens = tokens[:1]
        name = " ".join(tokens)
        elderly = honorific_form is not None and honorific_form.split()[0].lower() == "old"
        character = self._lookup(name)
        if character is None:
            if not create:
                return None
            character = self._create(name)
        elif normalize_name(name) != normalize_name(character.name):
            character.add_alias(name)
        if honorific_form:
            character.add_alias(honorific_form)
            if elderly and character.age == UNKNOWN:
                character.age = "elderly"
        return character

    def _resolve_descriptor(self, phrase: str, text: str, pos: int, *, create: bool) -> _Character | None:
        key = " ".join(phrase.lower().split())
        gender, age = DESCRIPTOR_TRAITS[key]
        direct = self._lookup(key)
        if direct is not None:
            return direct
        if gender == UNKNOWN:
            found = GENDER_PRONOUN_RE.search(text, pos, pos + PRONOUN_WINDOW)
            if found:
                gender = "male" if found.group("male") else "female"
        if age in ("child", "elderly"):
            candidates = [c for c in self.characters if not c.provisional and c.age == age and _agrees(c.gender, gender)]
            if candidates:
                recent = [p for p in reversed(self.conv.participants) if p in candidates]
                return recent[0] if recent else candidates[0]
        for character in self.characters:
            if character.provisional and _agrees(character.gender, gender) and _agrees(character.age, age):
                character.add_alias(key)
                if character.gender == UNKNOWN:
                    character.gender = gender
                if character.age == UNKNOWN:
                    character.age = age
                return character
        if not create:
            return None
        return self._create(key, gender=gender, age=age, provisional=True)

    def _sentence_initial(self, text: str, start: int, first_in_paragraph: bool, prev_span: Span | None) -> bool:
        before = text[:start]
        if before.strip():
            return bool(SENTENCE_END_RE.search(before))
        if first_in_paragraph or prev_span is None or prev_span.kind != "quote":
            return True
        return prev_span.text.rstrip().endswith((".", "!", "?"))

    def _infer_traits(self, character: _Character, text: str, start: int, end: int) -> None:
        if character.gender == UNKNOWN:
            found = GENDER_PRONOUN_RE.search(text, end, end + PRONOUN_WINDOW)
            if found:
                character.gender = "male" if found.group("male") else "female"
        if character.age == UNKNOWN:
            window = DESCRIPTOR_RE.sub("", _first_sentence(text[end:end + AGE_WINDOW]))
            for pattern, age in AGE_WORDS:
                if pattern.search(window):
                    character.age = age
                    break
        if not character.description:
            found = DESCRIPTION_RE.match(text, start)
            if found:
                character.description = found.group(0).strip()[:MAX_DESCRIPTION]

    def _scan(self, text: str, *, create: bool, first_in_paragraph: bool = True, prev_span: Span | None = None) -> None:
        """Harvest names and descriptor phrases from narration text in reading order."""
        found: list[tuple[int, re.Match[str], bool]] = [(m.start(), m, True) for m in NAME_RUN_RE.finditer(text)]
        found += [(m.start(), m, False) for m in DESCRIPTOR_RE.finditer(text)]
        for _, match, is_name in sorted(found, key=lambda item: item[0]):
            if is_name:
                raw = match.group(0)
                character = self._resolve_name(raw, create=False)
                if character is None and create:
                    cleaned, honorific_form = strip_honorific(POSSESSIVE_RE.sub("", raw))
                    tokens = [t for t in cleaned.split() if t not in STOPWORDS]
                    if len(tokens) == 1 and honorific_form is None and self._sentence_initial(text, match.start(), first_in_paragraph, prev_span):
                        seen_elsewhere = self.token_counts[tokens[0]] >= 2
                        tag_subject = bool(TAG_VERB_AFTER_RE.match(text, match.end()) or TAG_VERB_BEFORE_RE.search(text[:match.start()]))
                        if not (seen_elsewhere or tag_subject):
                            continue
                    character = self._resolve_name(raw, create=True)
                if character is None:
                    continue
                if create:
                    character.mentions += 1
                    self._infer_traits(character, text, match.start(), match.end())
                self.conv.mention(character)
            else:
                character = self._resolve_descriptor(match.group("desc"), text, match.end(), create=create)
                if character is None:
                    continue
                if create:
                    character.mentions += 1
                self.conv.mention(character)

    def _seed_from_context(self, context: str) -> None:
        if not context.strip():
            return
        self._scan(context, create=False)
        quotes = QUOTED_RE.findall(context)
        self.conv.last_quote = quotes[-1] if quotes else context.split("\n\n")[-1]

    def _character_updates(self) -> list[CharacterUpdate]:
        updates: list[CharacterUpdate] = []
        for character in self.characters:
            if character.mentions == 0 and character.attributed == 0:
                continue
            if character.bible_name is None and character.attributed == 0 and character.mentions < 2:
                continue
            age = character.age
            if age == UNKNOWN and not character.provisional:
                age = "adult"
            merge_into = None
            if character.bible_name and normalize_name(character.bible_name) != normalize_name(character.name):
                merge_into = character.bible_name
            updates.append(
                CharacterUpdate(
                    name=character.name,
                    aliases=list(character.aliases),
                    gender=character.gender,  # type: ignore[arg-type]
                    age=age,  # type: ignore[arg-type]
                    description=character.description,
                    voice_notes=", ".join(character.voice_notes),
                    merge_into=merge_into,
                )
            )
        return updates

    # ------------------------------------------------------------------ attribution
    def _vocative(self, quote: str) -> _Character | None:
        text = quote.strip()
        match = VOCATIVE_RE.match(text)
        if match:
            name = match.group("name")
            if name in STOPWORDS or not NAME_TOKEN_RE.fullmatch(name):
                return None
            character = self._resolve_name(name, create=True)
            if character is not None:
                character.mentions += 1
            return character
        match = ADDRESS_TAIL_RE.search(text) or ADDRESS_HEAD_RE.match(text)
        if match and match.group("name") not in STOPWORDS:
            return self._lookup(match.group("name"))
        return None

    def _introduction(self, quote: str) -> tuple[_Character, str] | None:
        if not NAME_QUESTION_RE.search(self.conv.last_quote):
            return None
        runs = [" ".join(r.split()) for r in NAME_RUN_SPLIT_RE.split(quote) if r.strip()]
        tokens = [t for r in runs for t in r.split()]
        if not 1 <= len(tokens) <= 4 or not all(NAME_TOKEN_RE.fullmatch(t) and t not in STOPWORDS for t in tokens):
            return None
        ordered = sorted(dict.fromkeys(runs), key=lambda r: (-len(r.split()), runs.index(r)))
        canonical = ordered[0]
        target = self._lookup(canonical)
        if target is None:
            provisional = [p for p in reversed(self.conv.participants) if p.provisional]
            if provisional:
                target = provisional[0]
            else:
                candidate, _ = self.conv.partner(self.conv.last_speaker)
                if candidate is not None and (candidate.provisional or is_token_subset(candidate.name, canonical)):
                    target = candidate
            if target is None:
                target = self._create(canonical)
            elif normalize_name(target.name) != normalize_name(canonical):
                target.rename(canonical)
        for alias in ordered[1:]:
            target.add_alias(alias)
        return target, "introduction"

    def _explicit_tag(self, text: str) -> _Character | None:
        for pattern in (TAG_NAME_RE, TAG_INVERTED_RE):
            for match in pattern.finditer(text):
                character = self._resolve_name(match.group("name"), create=True)
                if character is not None:
                    character.mentions += 1
                    return character
        return None

    def _pronoun_tag(self, text: str) -> _Character | None:
        match = PRONOUN_TAG_RE.search(text)
        if match:
            found = self.conv.by_pronoun("male" if match.group("pron").lower() == "he" else "female")
            if found is not None:
                return found
        match = DESCRIPTOR_TAG_RE.search(text)
        if match:
            return self._resolve_descriptor(match.group("desc"), text, match.end(), create=True)
        return None

    def _others_in(self, text: str, character: _Character) -> bool:
        for match in NAME_RUN_RE.finditer(text):
            other = self._resolve_name(match.group(0), create=False)
            if other is not None and other is not character:
                return True
        for match in DESCRIPTOR_RE.finditer(text):
            other = self._resolve_descriptor(match.group("desc"), text, match.end(), create=False)
            if other is not None and other is not character:
                return True
        return False

    def _leading_subject(self, span: Span) -> _Character | None:
        text = span.text.strip()
        character: _Character | None = None
        match = NAME_RUN_RE.match(text)
        if match:
            character = self._resolve_name(match.group(0), create=False)
        if character is None:
            match = DESCRIPTOR_RE.match(text)
            if match:
                character = self._resolve_descriptor(match.group("desc"), text, match.end(), create=False)
        if character is None:
            match = LEADING_PRONOUN_RE.match(text)
            if match:
                character = self.conv.by_pronoun("male" if match.group("pron").lower() == "he" else "female")
        if character is not None and not self._others_in(text, character):
            return character
        return None

    def _tag_rules(self, paragraph: _Paragraph, i: int) -> tuple[_Character, str] | None:
        spans = paragraph.spans
        after = spans[i + 1] if i + 1 < len(spans) and spans[i + 1].kind == "narration" else None
        before = spans[i - 1] if i > 0 and spans[i - 1].kind == "narration" else None
        texts = [t for t in (_first_sentence(after.text) if after else None, _last_sentence(before.text) if before else None) if t]
        for text in texts:
            character = self._explicit_tag(text)
            if character is not None:
                return character, "tag"
        for text in texts:
            character = self._pronoun_tag(text)
            if character is not None:
                return character, "pronoun"
        if before is not None:
            character = self._leading_subject(before)
            if character is not None:
                return character, "subject"
        return None

    def _fallback(self, paragraph: _Paragraph, i: int) -> tuple[_Character | None, str]:
        span = paragraph.spans[i]
        before = paragraph.spans[i - 1] if i > 0 and paragraph.spans[i - 1].kind == "narration" else None
        last = self.conv.last_speaker
        partner, candidates = self.conv.partner(last)
        if partner is not None:
            if before is not None and REPLY_CUE_RE.search(before.text):
                return partner, "reply"
            if candidates > 1:
                names = ", ".join(p.name for p in reversed(self.conv.participants) if p is not last)
                self.warnings.append(f"{span.id}: ambiguous attribution among {candidates} participants ({names}); guessed {partner.name!r}")
            return partner, "alternation"
        if last is not None:
            self.warnings.append(f"{span.id}: no conversation partner; attributed to the last speaker {last.name!r}")
            return last, "last"
        self.warnings.append(f"{span.id}: could not attribute the quote; spoken by the narrator")
        return None, "narrator"

    def _starts_scene(self, paragraph: _Paragraph) -> bool:
        """True when *paragraph* opens a new scene: flagged by ingest, or narration beginning with a time transition."""
        if paragraph.index in self.scene_breaks:
            return True
        first = paragraph.spans[0]
        return first.kind == "narration" and TIME_TRANSITION_RE.match(first.text) is not None

    def _process_paragraph(self, paragraph: _Paragraph) -> None:
        spans = paragraph.spans
        if self._starts_scene(paragraph):
            self.conv.reset()
            self.reset_paragraphs.add(paragraph.index)
        resolved: dict[int, tuple[_Character | None, str]] = {}
        prev: Span | None = None
        for i, span in enumerate(spans):
            if span.kind == "narration":
                self._scan(span.text, create=True, first_in_paragraph=i == 0, prev_span=prev)
            else:
                addressee = self._vocative(span.text)
                if addressee is not None:
                    self.conv.addressee = addressee
                    self.conv.mention(addressee)
                hit = self._introduction(span.text) or self._tag_rules(paragraph, i)
                self.conv.last_quote = span.text
                if hit is not None:
                    resolved[i] = hit
                    self.conv.spoke(hit[0])
            prev = span
        unresolved = [i for i, span in enumerate(spans) if span.kind == "quote" and i not in resolved]
        if unresolved:
            if resolved:
                for i in unresolved:
                    nearest = min(resolved, key=lambda j: (abs(j - i), -j))
                    resolved[i] = (resolved[nearest][0], "paragraph")
            else:
                shared = self._fallback(paragraph, unresolved[0])
                for i in unresolved:
                    resolved[i] = shared
            for i in unresolved:
                if resolved[i][0] is not None:
                    self.conv.spoke(resolved[i][0])  # type: ignore[arg-type]
        tag_text = paragraph.narration_text
        for i, span in enumerate(spans):
            if span.kind != "quote":
                continue
            character, rule = resolved[i]
            emotion, delivery = _emotion_delivery(span.text, tag_text, character)
            self.labels.append((span, character, emotion, delivery))
            if character is not None:
                character.attributed += 1
                for pattern, note in VOICE_NOTE_RULES:
                    if pattern.search(tag_text.lower()):
                        character.add_voice_note(note)
            log.debug("%s -> %s (%s, %s/%s)", span.id, character.name if character else NARRATOR, rule, emotion, delivery)
        self._sfx_for(paragraph)

    # ------------------------------------------------------------------ cues
    def _sfx_for(self, paragraph: _Paragraph) -> None:
        paragraph_text = paragraph.text
        found: list[tuple[int, int, int, SfxCueRaw]] = []
        for span_index, span in enumerate(paragraph.spans):
            if span.kind != "narration":
                continue
            for rule_index, rule in enumerate(SFX_RULES):
                if rule.requires_absent is not None and rule.requires_absent.search(paragraph_text):
                    continue
                match = rule.pattern.search(span.text)
                if match is None:
                    continue
                anchor = rule.anchor.search(span.text, match.start(), match.end()) if rule.anchor else match
                if anchor is None:
                    anchor = match
                description, duration = rule.description, rule.duration_s
                for condition, variant_description, variant_duration in rule.variants:
                    if condition.search(span.text):
                        description, duration = variant_description, variant_duration
                        break
                cue = SfxCueRaw(
                    span_id=span.id,
                    anchor_text=span.text[anchor.start():anchor.end()],
                    description=description,
                    kind=rule.kind,  # type: ignore[arg-type]
                    duration_s=duration,
                    intensity=rule.intensity,
                )
                found.append((rule_index, span_index, anchor.start(), cue))
        chosen = sorted(found, key=lambda item: item[0])[:MAX_SFX_PER_PARAGRAPH]
        for rule_index, _, _, cue in sorted(chosen, key=lambda item: (item[1], item[2])):
            log.debug("sfx %s at %s: %r", SFX_RULES[rule_index].name, cue.span_id, cue.anchor_text)
            self.sfx.append(cue)

    def _mood_of(self, text: str, current: str) -> str | None:
        scores: dict[str, int] = {}
        for mood, patterns in MOOD_KEYWORDS:
            score = sum(len(p.findall(text)) * (2 if " " in p.pattern else 1) for p in patterns)
            if score:
                scores[mood] = score
        if not scores:
            return None
        best = max(scores.values())
        tied = [mood for mood, _ in MOOD_KEYWORDS if scores.get(mood) == best]
        return current if current in tied else tied[0]

    def _music_cues(self) -> list[MusicCueRaw]:
        cues: list[MusicCueRaw] = []
        current: str = self.chunk.prior_mood
        moods: list[str | None] = []
        for paragraph in self.paragraphs:
            moods.append(self._mood_of(paragraph.text, current))
        started_at: int | None = None
        if current == "none":
            first = next((m for m in moods if m), "calm")
            cues.append(self._music_cue(self.paragraphs[0], "start", first))
            current, started_at = first, 0
        for i, (paragraph, mood) in enumerate(zip(self.paragraphs, moods)):
            if i == started_at or mood is None or mood == current:
                continue
            upcoming = next((m for m in moods[i + 1:i + 1 + MOOD_LOOKAHEAD] if m is not None), None)
            if upcoming == mood or paragraph.index in self.reset_paragraphs:
                cues.append(self._music_cue(paragraph, "change", mood))
                current = mood
        return cues

    @staticmethod
    def _music_cue(paragraph: _Paragraph, action: str, mood: str) -> MusicCueRaw:
        return MusicCueRaw(
            span_id=paragraph.spans[0].id,
            action=action,  # type: ignore[arg-type]
            mood=mood,  # type: ignore[arg-type]
            energy=MOOD_ENERGY.get(mood, DEFAULT_ENERGY),
            prompt=MOOD_PROMPTS.get(mood, ""),
        )


# --------------------------------------------------------------------------- provider
class HeuristicAnalyzer:
    """Deterministic rule-based analyzer (family ``mock``). Implements ``TextAnalyzer``."""

    family: ClassVar[str] = "mock"
    cache_version: str = "1"
    model_id: str = "heuristic-1"

    def __init__(self, usage: UsageSink | None = None) -> None:
        self.usage: UsageSink = usage or NullUsage()

    @classmethod
    def check(cls, settings: "Settings") -> list[str]:
        """Nothing to validate: no SDK, key or model files."""
        return []

    @classmethod
    def from_settings(cls, settings: "Settings", usage: UsageSink | None = None) -> "HeuristicAnalyzer":
        return cls(usage=usage)

    def warmup(self) -> None:
        """No-op: there is nothing to load."""
        return None

    def analyze_chunk(self, chunk: Chunk, bible: CastBible) -> ChunkAnalysis:
        """Label every quote span of *chunk* and emit character updates, SFX and music cues."""
        analysis = _ChunkRun(chunk, bible).run()
        self.usage.record("analysis", self.family, "calls", 1.0, meta={"chapter": chunk.chapter_index, "chunk": chunk.chunk_index})
        log.debug(
            "heuristic analysis ch%02d chunk %d: %d labels, %d characters, %d sfx, %d music, %d warnings",
            chunk.chapter_index, chunk.chunk_index, len(analysis.labels), len(analysis.characters),
            len(analysis.sfx_cues), len(analysis.music_cues), len(analysis.warnings),
        )
        return analysis

    def label_missing(self, chunk: Chunk, bible: CastBible, unlabelled_span_ids: list[str]) -> list[SpanLabel]:
        """Labels for just *unlabelled_span_ids*, computed from a full pass over the chunk."""
        wanted = set(unlabelled_span_ids)
        return [label for label in _ChunkRun(chunk, bible).run().labels if label.span_id in wanted]
