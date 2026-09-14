"""bookreader.types - the shared domain contract every module imports.

Rules:
* Anything serialized to JSON (disk or HTTP) is a pydantic v2 model with extra="forbid".
* In-memory audio is a plain dataclass holding a numpy int16 mono array.
* No module in the package may define a competing version of these types.
* Nothing here imports anything heavier than numpy/pydantic.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------- constants
SAMPLE_RATE = 22050            # canonical rate: mono, int16 PCM, stdlib wave
NARRATOR = "NARRATOR"          # reserved speaker name for narration spans
SCHEMA_VERSION = 1             # bump when any on-disk JSON shape changes (part of cache keys)

EMOTIONS: tuple[str, ...] = (
    "neutral", "calm", "happy", "amused", "tender", "sad", "melancholy", "afraid",
    "tense", "urgent", "angry", "stern", "surprised", "hesitant", "weary", "hopeful",
)
DELIVERIES: tuple[str, ...] = ("normal", "whisper", "shout", "quiet", "strained")
MOODS: tuple[str, ...] = (
    "calm", "warm", "tense", "ominous", "melancholy", "sad", "hopeful",
    "adventurous", "joyful", "mysterious", "romantic", "none",
)
GENDERS: tuple[str, ...] = ("male", "female", "nonbinary", "unknown")
AGES: tuple[str, ...] = ("child", "teen", "young_adult", "adult", "elderly", "unknown")

Emotion = Literal[
    "neutral", "calm", "happy", "amused", "tender", "sad", "melancholy", "afraid",
    "tense", "urgent", "angry", "stern", "surprised", "hesitant", "weary", "hopeful",
]
Delivery = Literal["normal", "whisper", "shout", "quiet", "strained"]
Mood = Literal[
    "calm", "warm", "tense", "ominous", "melancholy", "sad", "hopeful",
    "adventurous", "joyful", "mysterious", "romantic", "none",
]
Gender = Literal["male", "female", "nonbinary", "unknown"]
Age = Literal["child", "teen", "young_adult", "adult", "elderly", "unknown"]
AnalysisSource = Literal["llm", "heuristic"]
QuoteStyle = Literal["double", "single", "guillemet", "dash"]

_HONORIFIC_RE = re.compile(r"^(old|young|mr|mrs|ms|miss|dr|captain|sir|lady|lord|aunt|uncle)\.?\s+", re.I)


# --------------------------------------------------------------------------- errors
class BookreaderError(Exception):
    """Base for every error raised on purpose by bookreader.

    ``unit`` names the work item that failed (e.g. "ch02:c2p7s1", "ch01:chunk0", "ch03:m001")
    so the job error can point at it; stages set it before re-raising.
    """

    def __init__(self, message: str = "", *, unit: str | None = None) -> None:
        super().__init__(message)
        self.unit = unit


class InputError(BookreaderError):
    """The uploaded book cannot be read (unsupported type, empty, corrupt). Not retryable."""


class ProviderConfigError(BookreaderError):
    """Raised at startup: missing extra, missing key, missing model files. Never at job time."""


class ProviderTransientError(BookreaderError):
    """Rate limit / 5xx / network. Retried with backoff; the job fails only after retries.

    ``retry_after`` (seconds) is honored by bookreader.retry.with_retry when set.
    """

    def __init__(self, message: str = "", *, retry_after: float | None = None, unit: str | None = None) -> None:
        super().__init__(message, unit=unit)
        self.retry_after = retry_after


class ProviderPermanentError(BookreaderError):
    """4xx, invalid output that could not be repaired, unsupported request. Fails fast."""


class JobCancelled(BookreaderError):
    """Raised by JobContext.check_cancelled(); reason is 'user' or 'shutdown'."""

    def __init__(self, reason: Literal["user", "shutdown"] = "user") -> None:
        super().__init__(f"job cancelled ({reason})")
        self.reason = reason


# --------------------------------------------------------------------------- helpers
def canonical_json(obj: Any) -> str:
    """Deterministic JSON used for every cache key and fingerprint."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_key(*parts: Any) -> str:
    """sha256 hex of the canonical JSON of *parts*. All cache keys go through here."""
    return hashlib.sha256(canonical_json(list(parts)).encode("utf-8")).hexdigest()


def stable_seed(*parts: Any, bits: int = 32) -> int:
    """Deterministic integer seed (never Python hash(), which is salted per process)."""
    digest = hashlib.sha256(canonical_json(list(parts)).encode("utf-8")).hexdigest()
    return int(digest[: bits // 4], 16)


def normalize_name(name: str) -> str:
    """Lower-case, honorific-stripped, whitespace-collapsed key for name matching."""
    n = _HONORIFIC_RE.sub("", name.strip())
    return " ".join(n.lower().split())


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=False)


# --------------------------------------------------------------------------- book
class Span(StrictModel):
    """Smallest spoken unit. Deterministically split from a paragraph; text is never rewritten."""
    id: str                                  # "c1p7s2" (chapter, paragraph, span; all 1-based except s which is 0-based)
    kind: Literal["narration", "quote"]
    text: str                                # quote spans exclude the quote marks
    start_char: int                          # offsets into the paragraph text
    end_char: int


class Paragraph(StrictModel):
    index: int                               # 1-based within chapter
    text: str
    scene_break_before: bool = False
    spans: list[Span] = Field(default_factory=list)


class Chapter(StrictModel):
    index: int                               # 1-based within book
    title: str
    paragraphs: list[Paragraph]

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.paragraphs)

    @property
    def spans(self) -> list[Span]:
        return [s for p in self.paragraphs for s in p.spans]


class Book(StrictModel):
    title: str
    source_filename: str
    source_sha256: str
    quote_style: QuoteStyle = "double"
    chapters: list[Chapter]
    word_count: int = 0


class Chunk(StrictModel):
    """Whole paragraphs of one chapter, sized for one analyzer call."""
    chapter_index: int
    chunk_index: int                          # 0-based within chapter
    paragraph_start: int                      # 1-based, inclusive
    paragraph_end: int                        # inclusive
    spans: list[Span]
    context_before: str = ""                  # previous 2 paragraphs, read-only context
    prior_mood: Mood = "none"                 # music mood in force when this chunk begins

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.spans)


class Estimate(StrictModel):
    """Pre-flight numbers written by ingest before any paid call."""
    chapters: int
    paragraphs: int
    words: int
    chars: int
    quote_spans: int
    chunks: int
    tts_chars: int
    analysis_input_tokens_est: int
    cost_usd: dict[str, float] = Field(default_factory=dict)   # capability -> estimate


# --------------------------------------------------------------------------- cast bible
class CharacterEntry(StrictModel):
    name: str                                 # canonical, e.g. "Ansel Vey"
    aliases: list[str] = Field(default_factory=list)
    gender: Gender = "unknown"
    age: Age = "unknown"
    description: str = ""
    voice_notes: str = ""
    first_chapter: int = 0
    line_count: int = 0
    provisional: bool = False                 # created from a descriptor ("the stranger"), not a name

    def matches(self, name: str) -> bool:
        key = normalize_name(name)
        return key == normalize_name(self.name) or any(key == normalize_name(a) for a in self.aliases)


class CastBible(StrictModel):
    characters: list[CharacterEntry] = Field(default_factory=list)
    version: int = 0

    def find(self, name: str) -> CharacterEntry | None:
        if not name or normalize_name(name) == NARRATOR.lower():
            return None
        for c in self.characters:
            if c.matches(name):
                return c
        return None

    def fingerprint(self) -> str:
        return content_key("bible", SCHEMA_VERSION, self.model_dump(mode="json"))

    def to_prompt_json(self) -> str:
        """Compact form embedded in the LLM user message."""
        return canonical_json([
            {"name": c.name, "aliases": c.aliases, "gender": c.gender, "age": c.age,
             "description": c.description, "voice_notes": c.voice_notes}
            for c in self.characters if not c.provisional or c.line_count > 0
        ])


class CharacterUpdate(StrictModel):
    name: str
    aliases: list[str] = Field(default_factory=list)
    gender: Gender = "unknown"
    age: Age = "unknown"
    description: str = ""
    voice_notes: str = ""
    merge_into: str | None = None             # set when this entry is the same person as an existing one


# --------------------------------------------------------------------------- analysis output
class SpanLabel(StrictModel):
    span_id: str
    speaker: str                              # NARRATOR or a bible/updates name
    emotion: Emotion = "neutral"
    delivery: Delivery = "normal"


class SfxCueRaw(StrictModel):
    span_id: str
    anchor_text: str                          # verbatim substring of the span the sound belongs to
    description: str                          # provider prompt, e.g. "a single deep thunderclap, distant rumble"
    kind: Literal["impact", "ambient"] = "impact"
    duration_s: float = 2.0
    intensity: float = 0.7                    # 0..1


class MusicCueRaw(StrictModel):
    span_id: str
    action: Literal["start", "change", "stop"] = "change"
    mood: Mood = "calm"
    energy: float = 0.3                       # 0..1
    prompt: str = ""


class ChunkAnalysis(StrictModel):
    """What an analyzer returns for one Chunk. Same shape for LLM and heuristic."""
    labels: list[SpanLabel] = Field(default_factory=list)
    characters: list[CharacterUpdate] = Field(default_factory=list)
    sfx_cues: list[SfxCueRaw] = Field(default_factory=list)
    music_cues: list[MusicCueRaw] = Field(default_factory=list)
    source: AnalysisSource = "heuristic"
    warnings: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- assembled script
class Segment(StrictModel):
    id: str                                   # == span id
    paragraph_index: int
    speaker: str                              # NARRATOR or canonical character name
    kind: Literal["narration", "dialogue"]
    text: str
    emotion: Emotion = "neutral"
    delivery: Delivery = "normal"
    scene_break_before: bool = False
    source: AnalysisSource = "heuristic"


class SfxCue(StrictModel):
    id: str                                   # "c1x003"
    span_id: str
    anchor_text: str
    anchor_offset: int                        # char offset of anchor_text inside the span text (0 if not found)
    description: str
    kind: Literal["impact", "ambient"]
    duration_ms: int
    intensity: float
    end_span_id: str | None = None            # ambient cues run until this span ends (scene end)


class MusicCue(StrictModel):
    id: str                                   # "c1m001"
    start_span_id: str
    end_span_id: str
    mood: Mood
    energy: float
    prompt: str


class ChapterScript(StrictModel):
    chapter_index: int
    title: str
    segments: list[Segment]
    music: list[MusicCue] = Field(default_factory=list)
    sfx: list[SfxCue] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- voices & casting
class VoiceInfo(StrictModel):
    id: str
    name: str
    family: str
    gender: Gender = "unknown"
    age: Age = "unknown"
    tags: list[str] = Field(default_factory=list)   # lower-cased free labels: "narration", "warm", "raspy", accent...
    description: str = ""
    sample_rate: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class VoiceSettings(StrictModel):
    stability: float = 0.55
    similarity_boost: float = 0.8
    style: float = 0.15
    speed: float = 1.0
    pitch_shift: float = 0.0                  # semitones; mock/local only


class VoiceAssignment(StrictModel):
    character: str                            # canonical name or NARRATOR
    voice: VoiceInfo
    settings: VoiceSettings = Field(default_factory=VoiceSettings)
    seed: int = 0                             # 32-bit, stable per (book, character)
    source: Literal["auto", "override"] = "auto"
    reason: str = ""


class Cast(StrictModel):
    family: str
    narrator: VoiceAssignment
    characters: list[VoiceAssignment] = Field(default_factory=list)

    def assignment_for(self, speaker: str) -> VoiceAssignment:
        if speaker == NARRATOR:
            return self.narrator
        for a in self.characters:
            if a.character == speaker:
                return a
        return self.narrator


# --------------------------------------------------------------------------- provider requests
class TTSRequest(StrictModel):
    text: str
    voice_id: str
    settings: VoiceSettings = Field(default_factory=VoiceSettings)
    emotion: Emotion = "neutral"
    delivery: Delivery = "normal"
    previous_text: str | None = None
    next_text: str | None = None
    seed: int = 0

    def cache_fields(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class MusicRequest(StrictModel):
    prompt: str
    mood: Mood = "calm"
    energy: float = 0.3
    duration_ms: int = 30000
    loopable: bool = True
    seed: int = 0

    def cache_fields(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class SfxRequest(StrictModel):
    description: str
    kind: Literal["impact", "ambient"] = "impact"
    duration_ms: int = 2000
    loop: bool = False
    intensity: float = 0.7
    seed: int = 0

    def cache_fields(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def clip_key(kind: str, family: str, cache_version: str, request: TTSRequest | MusicRequest | SfxRequest) -> str:
    """The one function that turns a request into a cache key; used by timeline and render stages."""
    return content_key(kind, family, cache_version, SCHEMA_VERSION, request.cache_fields())


# --------------------------------------------------------------------------- audio (in-memory)
@dataclass
class AudioClip:
    """Mono int16 samples at *sample_rate*. Providers may return any rate; the pipeline canonicalizes."""
    samples: np.ndarray
    sample_rate: int = SAMPLE_RATE

    def __post_init__(self) -> None:
        arr = np.asarray(self.samples)
        if arr.ndim == 2:                      # (channels, n) or (n, channels) -> average to mono
            arr = arr.mean(axis=0 if arr.shape[0] <= 8 else 1)
        if arr.dtype != np.int16:
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.clip(arr, -1.0, 1.0) * 32767.0
            arr = arr.astype(np.int16)
        self.samples = np.ascontiguousarray(arr.reshape(-1))

    @property
    def duration_ms(self) -> int:
        return int(round(len(self.samples) * 1000 / self.sample_rate))

    @classmethod
    def from_pcm16_bytes(cls, data: bytes, sample_rate: int) -> "AudioClip":
        return cls(np.frombuffer(data, dtype="<i2").copy(), sample_rate)

    @classmethod
    def silence(cls, ms: int, sample_rate: int = SAMPLE_RATE) -> "AudioClip":
        return cls(np.zeros(int(ms * sample_rate / 1000), dtype=np.int16), sample_rate)


# --------------------------------------------------------------------------- render plans (per chapter, on disk)
class TtsJob(StrictModel):
    segment_id: str
    piece: int                                # 0-based piece index when a long segment is split
    request: TTSRequest
    clip_key: str


class ChapterTts(StrictModel):
    chapter_index: int
    jobs: list[TtsJob]
    durations_ms: dict[str, int] = Field(default_factory=dict)   # clip_key -> measured duration after trim


class MusicJob(StrictModel):
    cue_id: str
    request: MusicRequest
    clip_key: str


class SfxJob(StrictModel):
    cue_id: str
    request: SfxRequest
    clip_key: str


class Placement(StrictModel):
    track: Literal["voice", "music", "sfx"]
    clip_key: str
    ref_id: str                               # segment id, cue id
    start_ms: int
    end_ms: int                               # slot end; clip is trimmed or looped to fit
    gain_db: float = 0.0
    fade_in_ms: int = 10
    fade_out_ms: int = 10
    loop: bool = False
    duck: Literal["none", "music", "ambient"] = "none"


class SegmentTiming(StrictModel):
    id: str
    start_ms: int
    end_ms: int


class ChapterTimeline(StrictModel):
    chapter_index: int
    duration_ms: int
    segments: list[SegmentTiming]
    placements: list[Placement]
    music_jobs: list[MusicJob] = Field(default_factory=list)
    sfx_jobs: list[SfxJob] = Field(default_factory=list)


# --------------------------------------------------------------------------- jobs
class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"
    cancelled = "cancelled"


class Stage(str, Enum):
    ingest = "ingest"
    analyze = "analyze"
    cast = "cast"
    render = "render"        # per chapter: tts -> timeline -> music -> sfx -> mix
    finalize = "finalize"


STAGE_ORDER: tuple[Stage, ...] = (Stage.ingest, Stage.analyze, Stage.cast, Stage.render, Stage.finalize)
STAGE_WEIGHTS: dict[Stage, int] = {Stage.ingest: 2, Stage.analyze: 28, Stage.cast: 2, Stage.render: 66, Stage.finalize: 2}


class StageState(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"


class JobOptions(StrictModel):
    chapters: list[int] | None = None         # 1-based subset, None = all
    music: bool = True
    sfx: bool = True
    cast_overrides: dict[str, str] = Field(default_factory=dict)   # character name -> voice id


class JobError(StrictModel):
    stage: str
    error_type: Literal["input", "config", "provider_transient", "provider_permanent", "cancelled", "internal"]
    message: str
    retryable: bool
    unit: str | None = None                   # e.g. "ch02:c2p7s1", "ch01:chunk0", "ch03:m001"


class StageRecord(StrictModel):
    stage: Stage
    state: StageState = StageState.pending
    done: int = 0
    total: int = 0
    message: str = ""
    attempts: int = 0
    started_at: str | None = None
    finished_at: str | None = None


class Job(StrictModel):
    id: str
    title: str
    filename: str
    source_sha256: str = ""
    status: JobStatus = JobStatus.queued
    stage: Stage | None = None
    options: JobOptions = Field(default_factory=JobOptions)
    error: JobError | None = None
    cancel_requested: bool = False
    created_at: str = ""
    updated_at: str = ""
    started_at: str | None = None
    finished_at: str | None = None


class JobEvent(StrictModel):
    id: int
    job_id: str
    ts: str
    level: Literal["info", "warn", "error"]
    stage: str | None
    message: str


class UsageEvent(StrictModel):
    capability: str
    family: str
    unit_type: str                            # input_tokens|output_tokens|cache_read_input_tokens|cache_creation_input_tokens|characters|audio_seconds|calls
    units: float
    cache_hit: bool = False
    cost_usd: float = 0.0
    duration_ms: int = 0
    meta: dict[str, Any] = Field(default_factory=dict)


class UsageSummary(StrictModel):
    calls: int = 0
    cache_hits: int = 0
    cost_usd: float = 0.0
    by_capability: dict[str, dict[str, float]] = Field(default_factory=dict)   # capability -> unit_type -> units


# --------------------------------------------------------------------------- manifest (the public timing contract)
class ManifestSegment(StrictModel):
    id: str
    speaker: str
    voice_id: str
    kind: Literal["narration", "dialogue"]
    text: str
    emotion: Emotion
    delivery: Delivery
    paragraph: int
    start_ms: int
    end_ms: int
    source: AnalysisSource


class ManifestCue(StrictModel):
    id: str
    kind: Literal["music", "sfx"]
    start_ms: int
    end_ms: int
    gain_db: float
    clip: str                                 # cache key of the clip used
    mood: Mood | None = None                  # music
    prompt: str | None = None                 # music
    description: str | None = None            # sfx
    sfx_kind: Literal["impact", "ambient"] | None = None
    anchor_segment: str | None = None
    anchor_text: str | None = None


class ChapterManifest(StrictModel):
    index: int
    title: str
    duration_ms: int
    sample_rate: int = SAMPLE_RATE
    files: dict[str, str | None]              # mix, voice, music, sfx, mp3 (paths relative to the job dir)
    segments: list[ManifestSegment]
    cues: list[ManifestCue]


class JobManifest(StrictModel):
    manifest_version: int = SCHEMA_VERSION
    job_id: str
    title: str
    created_at: str
    providers: dict[str, dict[str, str]]      # capability -> {family, cache_version}
    sample_rate: int = SAMPLE_RATE
    total_duration_ms: int
    chapters: list[ChapterManifest]
    cast_file: str = "cast.json"
    usage_file: str = "usage.json"
    warnings: list[str] = Field(default_factory=list)


ProgressFn = Callable[[str, int, int, str], None]   # (stage_or_substage, done, total, message)
