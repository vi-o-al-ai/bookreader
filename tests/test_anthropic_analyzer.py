"""ClaudeAnalyzer against a stub ``anthropic`` module (installed in sys.modules) and a fake client
injected through __init__. No real SDK client is ever constructed; no network."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import bookreader.retry as retry_module
from bookreader.analysis.chunker import make_chunks
from bookreader.analysis.prompts import SYSTEM_PROMPT, build_user_message
from bookreader.analysis.schema import CHUNK_ANALYSIS_SCHEMA, PROMPT_VERSION
from bookreader.ingest import load_book
from bookreader.providers.anthropic.analysis import ClaudeAnalyzer, map_error, split_chunk
from bookreader.providers.base import TextAnalyzer
from bookreader.providers.mock.analysis import HeuristicAnalyzer
from bookreader.settings import Settings
from bookreader.types import (
    Book,
    CastBible,
    Chunk,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderTransientError,
    Span,
)

# --------------------------------------------------------------------------- stub SDK


class FakeAPIError(Exception):
    """Root of the stub hierarchy, like anthropic.APIError."""


class FakeAPIConnectionError(FakeAPIError):
    pass


class FakeAPITimeoutError(FakeAPIConnectionError):
    pass


class FakeAPIStatusError(FakeAPIError):
    status_code: int = 0

    def __init__(self, message: str = "", *, status_code: int | None = None, headers: dict[str, str] | None = None) -> None:
        super().__init__(message)
        if status_code is not None:
            self.status_code = status_code
        self.response = SimpleNamespace(headers=headers or {})


class FakeBadRequestError(FakeAPIStatusError):
    status_code = 400


class FakeAuthenticationError(FakeAPIStatusError):
    status_code = 401


class FakePermissionDeniedError(FakeAPIStatusError):
    status_code = 403


class FakeNotFoundError(FakeAPIStatusError):
    status_code = 404


class FakeRateLimitError(FakeAPIStatusError):
    status_code = 429


class FakeInternalServerError(FakeAPIStatusError):
    status_code = 500


class FakeAnthropic:
    """Stand-in for anthropic.Anthropic: records constructor kwargs, never talks to anything."""

    instances: list["FakeAnthropic"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.messages = SimpleNamespace(stream=lambda **_: pytest.fail("the real client must never be called"))
        FakeAnthropic.instances.append(self)


@pytest.fixture
def anthropic_stub(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    mod = types.ModuleType("anthropic")
    for cls in (
        FakeAPIError, FakeAPIConnectionError, FakeAPITimeoutError, FakeAPIStatusError, FakeBadRequestError,
        FakeAuthenticationError, FakePermissionDeniedError, FakeNotFoundError, FakeRateLimitError, FakeInternalServerError,
    ):
        setattr(mod, cls.__name__.removeprefix("Fake"), cls)
    mod.Anthropic = FakeAnthropic  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    FakeAnthropic.instances.clear()
    return mod


# --------------------------------------------------------------------------- fake client


def message(
    text: str | None,
    *,
    stop_reason: str = "end_turn",
    stop_details: Any = None,
    thinking: bool = True,
    usage: dict[str, int | None] | None = None,
) -> SimpleNamespace:
    """A final message the way the SDK shapes it: a thinking block, then the text block."""
    content: list[SimpleNamespace] = [SimpleNamespace(type="thinking", thinking="")] if thinking else []
    if text is not None:
        content.append(SimpleNamespace(type="text", text=text))
    tokens = {"input_tokens": 1200, "output_tokens": 340, "cache_read_input_tokens": 1000, "cache_creation_input_tokens": 0}
    tokens.update(usage or {})
    return SimpleNamespace(stop_reason=stop_reason, stop_details=stop_details, usage=SimpleNamespace(**tokens), content=content)


class FakeStream:
    def __init__(self, result: Any) -> None:
        self.result = result

    def __enter__(self) -> "FakeStream":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def get_final_message(self) -> Any:
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class FakeClient:
    """``messages.stream(**kwargs)`` records kwargs and returns the next scripted result
    (a message or an exception); the last result repeats."""

    def __init__(self, *results: Any) -> None:
        self.results = list(results)
        self.calls: list[dict[str, Any]] = []
        self.messages = SimpleNamespace(stream=self.stream)

    def stream(self, **kwargs: Any) -> FakeStream:
        self.calls.append(kwargs)
        result = self.results.pop(0) if len(self.results) > 1 else self.results[0]
        return FakeStream(result)

    def user_text(self, index: int) -> str:
        return str(self.calls[index]["messages"][0]["content"])


class RecordingUsage:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def record(self, capability: str, family: str, unit_type: str, units: float, **kw: Any) -> None:
        self.rows.append({"capability": capability, "family": family, "unit_type": unit_type, "units": units, **kw})


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def book(sample_book_path: Path) -> Book:
    return load_book(sample_book_path)


@pytest.fixture(scope="module")
def chunk(book: Book) -> Chunk:
    """Chapter 1 as one chunk (nine paragraphs, seven quote spans)."""
    chunks = make_chunks(book.chapters[0], 6000)
    assert len(chunks) == 1
    return chunks[0]


@pytest.fixture(scope="module")
def bible() -> CastBible:
    return CastBible()


@pytest.fixture(scope="module")
def good_json(chunk: Chunk, bible: CastBible) -> str:
    """A well-formed reply: the heuristic result rendered to the schema's shape."""
    analysis = HeuristicAnalyzer().analyze_chunk(chunk, bible)
    return json.dumps(analysis.model_dump(mode="json", exclude={"source", "warnings"}))


def make_analyzer(client: Any, usage: Any = None, **overrides: Any) -> ClaudeAnalyzer:
    kwargs: dict[str, Any] = dict(model_id="claude-opus-5", effort="medium", max_tokens=32000, fallback=HeuristicAnalyzer(), usage=usage)
    kwargs.update(overrides)
    return ClaudeAnalyzer(client, **kwargs)


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    delays: list[float] = []
    monkeypatch.setattr(retry_module.time, "sleep", lambda s: delays.append(s))
    return delays


def quote_ids(chunk: Chunk) -> list[str]:
    return [span.id for span in chunk.spans if span.kind == "quote"]


# --------------------------------------------------------------------------- prompts


def test_system_prompt_is_stable_and_versioned() -> None:
    assert SYSTEM_PROMPT.endswith(f"prompt-version: {PROMPT_VERSION}")
    assert SYSTEM_PROMPT.endswith("prompt-version: 2")
    for word in ("urgent", "whisper", "ominous", "nonbinary", "young_adult", "NARRATOR", "merge_into", "anchor_text"):
        assert word in SYSTEM_PROMPT


def test_user_message_sections(chunk: Chunk, bible: CastBible) -> None:
    text = build_user_message(chunk.model_copy(update={"prior_mood": "tense", "context_before": "Earlier that day."}), bible)
    assert text.startswith("## Cast bible (JSON)\n[]")
    assert "## Music mood currently playing\ntense" in text
    assert "## Previous paragraphs (context only, already processed)\nEarlier that day." in text
    assert "## Spans to label (chapter 1, chunk 0)" in text
    assert "## Repair note" not in text
    for span in chunk.spans:
        assert span.id in text
        assert span.text in text
    plain = build_user_message(chunk, bible)
    with_note = build_user_message(chunk, bible, repair_note="span c1p2s0 has no label")
    assert with_note.endswith("## Repair note\nspan c1p2s0 has no label")
    assert with_note.startswith(plain)
    assert build_user_message(chunk, bible) == plain, "deterministic"


def test_user_message_marks_scene_breaks(chunk: Chunk, bible: CastBible) -> None:
    marked = chunk.model_copy(update={"scene_break_paragraphs": [4, 8]})
    text = build_user_message(marked, bible)
    rendered = json.loads(text.split("## Spans to label (chapter 1, chunk 0)\n", 1)[1])
    assert [entry["id"] for entry in rendered if entry.get("scene_break") is True] == ["c1p4s0", "c1p8s0"]
    assert all(set(entry) == {"id", "kind", "text"} for entry in rendered if "scene_break" not in entry)
    assert [(e["id"], e["kind"], e["text"]) for e in rendered] == [(s.id, s.kind, s.text) for s in chunk.spans]
    assert "scene_break" not in build_user_message(chunk, bible)
    assert build_user_message(marked, bible) == text, "deterministic"
    assert '"scene_break": true' in SYSTEM_PROMPT


# --------------------------------------------------------------------------- request shape


def test_request_kwargs_shape(chunk: Chunk, bible: CastBible, good_json: str) -> None:
    client = FakeClient(message(good_json))
    analyzer = make_analyzer(client, effort="high", max_tokens=12345)
    assert isinstance(analyzer, TextAnalyzer)
    assert analyzer.family == "anthropic"
    assert analyzer.cache_version == "claude-opus-5:2"
    assert analyzer.model_id == "claude-opus-5"

    result = analyzer.analyze_chunk(chunk, bible)

    assert len(client.calls) == 1
    kwargs = client.calls[0]
    assert kwargs["model"] == "claude-opus-5"
    assert kwargs["max_tokens"] == 12345
    assert kwargs["system"] == [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]
    assert kwargs["messages"] == [{"role": "user", "content": build_user_message(chunk, bible)}]
    assert kwargs["output_config"]["format"]["type"] == "json_schema"
    assert kwargs["output_config"]["format"]["schema"] is CHUNK_ANALYSIS_SCHEMA
    assert kwargs["output_config"]["effort"] == "high"
    assert "thinking" not in kwargs
    assert "temperature" not in kwargs
    assert set(kwargs) == {"model", "max_tokens", "system", "messages", "output_config"}

    assert result.source == "llm"
    assert sorted(label.span_id for label in result.labels) == sorted(quote_ids(chunk))
    assert result.warnings == []


def test_text_block_found_after_thinking_block(chunk: Chunk, bible: CastBible, good_json: str) -> None:
    msg = message(good_json, thinking=True)
    assert msg.content[0].type == "thinking"
    result = make_analyzer(FakeClient(msg)).analyze_chunk(chunk, bible)
    assert result.source == "llm"
    assert len(result.labels) == len(quote_ids(chunk))


def test_llm_labels_are_canonicalized_through_the_bible(chunk: Chunk) -> None:
    """validate_chunk_analysis runs on the parsed reply: speakers are coerced to bible names."""
    bible = CastBible.model_validate({"characters": [{"name": "Mara Quill", "aliases": ["Mara"], "gender": "female", "age": "adult"}]})
    labels = [{"span_id": sid, "speaker": "Mara", "emotion": "neutral", "delivery": "normal"} for sid in quote_ids(chunk)]
    reply = json.dumps({"labels": labels, "characters": [], "sfx_cues": [], "music_cues": []})
    result = make_analyzer(FakeClient(message(reply))).analyze_chunk(chunk, bible)
    assert result.source == "llm"
    assert {label.speaker for label in result.labels} == {"Mara Quill"}
    assert any("coerced" in warning for warning in result.warnings)


# --------------------------------------------------------------------------- degraded paths


def test_refusal_falls_back_to_heuristic_with_warning(chunk: Chunk, bible: CastBible, caplog: pytest.LogCaptureFixture) -> None:
    refusal = message(None, stop_reason="refusal", stop_details=SimpleNamespace(category="policy", explanation="no"), thinking=False)
    client = FakeClient(refusal)
    with caplog.at_level("WARNING", logger="bookreader.providers.anthropic.analysis"):
        result = make_analyzer(client).analyze_chunk(chunk, bible)
    assert len(client.calls) == 1, "a refusal is never retried or repaired"
    assert result.source == "heuristic"
    assert "refusal:policy" in result.warnings
    assert "policy" in caplog.text and "no" in caplog.text
    expected = HeuristicAnalyzer().analyze_chunk(chunk, bible)
    assert [(l.span_id, l.speaker) for l in result.labels] == [(l.span_id, l.speaker) for l in expected.labels]


def test_refusal_without_details(chunk: Chunk, bible: CastBible) -> None:
    result = make_analyzer(FakeClient(message(None, stop_reason="refusal", stop_details=None))).analyze_chunk(chunk, bible)
    assert result.source == "heuristic"
    assert "refusal:unknown" in result.warnings


def test_max_tokens_splits_chunk_into_two_calls(chunk: Chunk, bible: CastBible) -> None:
    head, tail = split_chunk(chunk)
    heuristic = HeuristicAnalyzer()

    def reply_for(part: Chunk) -> str:
        return json.dumps(heuristic.analyze_chunk(part, bible).model_dump(mode="json", exclude={"source", "warnings"}))

    client = FakeClient(message(None, stop_reason="max_tokens", thinking=False), message(reply_for(head)), message(reply_for(tail)))
    result = make_analyzer(client).analyze_chunk(chunk, bible)

    assert len(client.calls) == 3
    first_ids = {span.id for span in head.spans}
    second_ids = {span.id for span in tail.spans}
    assert first_ids and second_ids and first_ids.isdisjoint(second_ids)
    assert first_ids | second_ids == {span.id for span in chunk.spans}
    assert all(sid in client.user_text(1) for sid in first_ids)
    assert not any(sid in client.user_text(1).split("## Spans to label")[1] for sid in second_ids)
    assert all(sid in client.user_text(2) for sid in second_ids)
    assert "## Repair note" not in client.user_text(1) and "## Repair note" not in client.user_text(2)
    assert result.source == "llm"
    assert sorted(label.span_id for label in result.labels) == sorted(quote_ids(chunk))


def test_split_chunk_shape(chunk: Chunk) -> None:
    head, tail = split_chunk(chunk)
    assert head.paragraph_start == chunk.paragraph_start == 1
    assert head.paragraph_end == 4
    assert tail.paragraph_start == 5
    assert tail.paragraph_end == chunk.paragraph_end == 9
    assert head.context_before == chunk.context_before
    assert tail.context_before.startswith("From the lamp room")
    assert "Thunder split the sky" in tail.context_before
    assert head.prior_mood == tail.prior_mood == chunk.prior_mood
    assert head.chapter_index == tail.chapter_index == 1 and head.chunk_index == tail.chunk_index == 0


def test_max_tokens_on_single_paragraph_is_permanent(bible: CastBible) -> None:
    spans = [
        Span(id="c1p1s0", kind="narration", text="He turned. ", start_char=0, end_char=11),
        Span(id="c1p1s1", kind="quote", text="Hello.", start_char=12, end_char=18),
    ]
    single = Chunk(chapter_index=1, chunk_index=0, paragraph_start=1, paragraph_end=1, spans=spans)
    client = FakeClient(message(None, stop_reason="max_tokens", thinking=False))
    with pytest.raises(ProviderPermanentError, match="truncated for single paragraph") as info:
        make_analyzer(client).analyze_chunk(single, bible)
    assert info.value.unit == "ch01:chunk0"
    assert len(client.calls) == 1


def test_invalid_json_triggers_one_repair_then_heuristic(chunk: Chunk, bible: CastBible) -> None:
    client = FakeClient(message("{not json"), message("still {not json"))
    result = make_analyzer(client).analyze_chunk(chunk, bible)
    assert len(client.calls) == 2
    assert "## Repair note" not in client.user_text(0)
    assert "## Repair note" in client.user_text(1)
    assert "not valid JSON" in client.user_text(1)
    assert client.calls[1]["system"] == client.calls[0]["system"], "the cached system block is byte-identical"
    assert client.calls[1]["output_config"] == client.calls[0]["output_config"]
    assert result.source == "heuristic"
    assert "llm_invalid" in result.warnings
    assert len(result.labels) == len(quote_ids(chunk))


def test_missing_text_block_is_repaired(chunk: Chunk, bible: CastBible, good_json: str) -> None:
    client = FakeClient(message(None, thinking=True), message(good_json))
    result = make_analyzer(client).analyze_chunk(chunk, bible)
    assert len(client.calls) == 2
    assert "no text block" in client.user_text(1)
    assert result.source == "llm"


def test_validation_error_repair_succeeds(chunk: Chunk, bible: CastBible, good_json: str) -> None:
    bad = json.loads(good_json)
    bad["labels"][0]["emotion"] = "ecstatic"
    client = FakeClient(message(json.dumps(bad)), message(good_json))
    result = make_analyzer(client).analyze_chunk(chunk, bible)
    assert len(client.calls) == 2
    note = client.user_text(1).split("## Repair note\n")[1]
    assert "did not match the schema" in note
    assert "emotion" in note
    assert result.source == "llm"
    assert result.warnings == []


def test_analysis_invalid_repair_note_names_offending_spans(chunk: Chunk, bible: CastBible, good_json: str) -> None:
    """Cues that reference spans outside the chunk cannot be repaired: the note names them."""
    bad = json.loads(good_json)
    bad["sfx_cues"].append({"span_id": "c2p9s4", "anchor_text": "x", "description": "x", "kind": "impact", "duration_s": 1, "intensity": 0.5})
    client = FakeClient(message(json.dumps(bad)), message(json.dumps(bad)))
    result = make_analyzer(client).analyze_chunk(chunk, bible)
    assert len(client.calls) == 2
    note = client.user_text(1).split("## Repair note\n")[1]
    assert "c2p9s4" in note
    assert result.source == "heuristic"
    assert "llm_invalid" in result.warnings


def test_too_many_unlabelled_spans_is_repaired(chunk: Chunk, bible: CastBible, good_json: str) -> None:
    empty = json.dumps({"labels": [], "characters": [], "sfx_cues": [], "music_cues": []})
    client = FakeClient(message(empty), message(good_json))
    result = make_analyzer(client).analyze_chunk(chunk, bible)
    assert len(client.calls) == 2
    note = client.user_text(1).split("## Repair note\n")[1]
    assert "have no label" in note
    assert quote_ids(chunk)[0] in note
    assert result.source == "llm"


# --------------------------------------------------------------------------- usage


def test_usage_rows_recorded_per_call(chunk: Chunk, bible: CastBible, good_json: str) -> None:
    sink = RecordingUsage()
    client = FakeClient(message(good_json, usage={"input_tokens": 1500, "output_tokens": 400, "cache_read_input_tokens": 1200, "cache_creation_input_tokens": None}))
    make_analyzer(client, usage=sink).analyze_chunk(chunk, bible)
    assert [row["unit_type"] for row in sink.rows] == ["input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"]
    assert [row["units"] for row in sink.rows] == [1500.0, 400.0, 1200.0, 0.0]
    for row in sink.rows:
        assert row["capability"] == "analysis"
        assert row["family"] == "anthropic"
        assert row["meta"]["chunk"] == "ch01:chunk0"
        assert row["duration_ms"] >= 0


def test_usage_recorded_for_every_call_including_repair(chunk: Chunk, bible: CastBible) -> None:
    sink = RecordingUsage()
    client = FakeClient(message("{bad"), message("{bad"))
    make_analyzer(client, usage=sink).analyze_chunk(chunk, bible)
    assert len(sink.rows) == 8
    assert all(row["family"] == "anthropic" for row in sink.rows)


# --------------------------------------------------------------------------- errors


def test_rate_limit_becomes_transient_after_retries(anthropic_stub: types.ModuleType, chunk: Chunk, bible: CastBible, no_sleep: list[float]) -> None:
    client = FakeClient(anthropic_stub.RateLimitError("slow down", headers={"retry-after": "7"}))
    with pytest.raises(ProviderTransientError) as info:
        make_analyzer(client).analyze_chunk(chunk, bible)
    assert len(client.calls) == 4, "with_retry's default four attempts"
    assert len(no_sleep) == 3
    assert all(delay >= 7.0 for delay in no_sleep), "retry-after is honored"
    assert info.value.retry_after == 7.0
    assert info.value.unit == "ch01:chunk0"


def test_transient_error_then_success(anthropic_stub: types.ModuleType, chunk: Chunk, bible: CastBible, good_json: str, no_sleep: list[float]) -> None:
    client = FakeClient(anthropic_stub.InternalServerError("boom"), anthropic_stub.APIConnectionError("reset"), message(good_json))
    result = make_analyzer(client).analyze_chunk(chunk, bible)
    assert result.source == "llm"
    assert len(client.calls) == 3
    assert len(no_sleep) == 2


def test_bad_request_is_permanent_without_retry(anthropic_stub: types.ModuleType, chunk: Chunk, bible: CastBible, no_sleep: list[float]) -> None:
    client = FakeClient(anthropic_stub.BadRequestError("bad schema"))
    with pytest.raises(ProviderPermanentError, match="bad schema") as info:
        make_analyzer(client).analyze_chunk(chunk, bible)
    assert len(client.calls) == 1
    assert no_sleep == []
    assert info.value.unit == "ch01:chunk0"


@pytest.mark.parametrize(
    "factory, expected",
    [
        (lambda m: m.RateLimitError("r"), ProviderTransientError),
        (lambda m: m.InternalServerError("i"), ProviderTransientError),
        (lambda m: m.APITimeoutError("t"), ProviderTransientError),
        (lambda m: m.APIConnectionError("c"), ProviderTransientError),
        (lambda m: m.APIStatusError("s", status_code=503), ProviderTransientError),
        (lambda m: m.APIStatusError("s", status_code=409), ProviderTransientError),
        (lambda m: m.APIStatusError("s", status_code=408), ProviderTransientError),
        (lambda m: m.BadRequestError("b"), ProviderPermanentError),
        (lambda m: m.AuthenticationError("a"), ProviderPermanentError),
        (lambda m: m.PermissionDeniedError("p"), ProviderPermanentError),
        (lambda m: m.NotFoundError("n"), ProviderPermanentError),
        (lambda m: m.APIStatusError("s", status_code=418), ProviderPermanentError),
        (lambda m: RuntimeError("unrelated"), ProviderPermanentError),
    ],
)
def test_map_error_table(anthropic_stub: types.ModuleType, factory: Any, expected: type) -> None:
    mapped = map_error(factory(anthropic_stub))
    assert type(mapped) is expected


def test_map_error_passes_bookreader_errors_through() -> None:
    original = ProviderTransientError("mine", retry_after=2.0)
    assert map_error(original) is original


def test_map_error_by_status_when_sdk_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", None)
    assert isinstance(map_error(FakeAPIStatusError("x", status_code=502)), ProviderTransientError)
    assert isinstance(map_error(FakeAPIStatusError("x", status_code=422)), ProviderPermanentError)


# --------------------------------------------------------------------------- lifecycle


def test_check_names_extra_when_sdk_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", None)
    settings = Settings.from_env({"ANTHROPIC_API_KEY": "k"})
    with pytest.raises(ProviderConfigError, match=r"bookreader\[anthropic\]") as info:
        ClaudeAnalyzer.check(settings)
    assert "analysis provider 'anthropic' needs the anthropic SDK" in str(info.value)
    with pytest.raises(ProviderConfigError, match=r"bookreader\[anthropic\]"):
        ClaudeAnalyzer.from_settings(settings)


def test_check_passes_with_stub(anthropic_stub: types.ModuleType) -> None:
    assert ClaudeAnalyzer.check(Settings.from_env({})) == []
    assert FakeAnthropic.instances == [], "check never builds a client"


def test_from_settings_builds_client_from_secrets(anthropic_stub: types.ModuleType) -> None:
    settings = Settings.from_env({
        "ANTHROPIC_API_KEY": "sk-test",
        "BOOKREADER_ANTHROPIC_MODEL": "claude-test-1",
        "BOOKREADER_ANTHROPIC_EFFORT": "low",
        "BOOKREADER_ANTHROPIC_MAX_TOKENS": "8000",
        "BOOKREADER_CONCURRENCY": "2",
    })
    sink = RecordingUsage()
    analyzer = ClaudeAnalyzer.from_settings(settings, sink)
    assert len(FakeAnthropic.instances) == 1
    assert FakeAnthropic.instances[0].kwargs == {"api_key": "sk-test", "max_retries": 3, "timeout": 600}
    assert analyzer.client is FakeAnthropic.instances[0]
    assert analyzer.model_id == "claude-test-1"
    assert analyzer.effort == "low"
    assert analyzer.max_tokens == 8000
    assert analyzer.concurrency == 2
    assert analyzer.cache_version == "claude-test-1:2"
    assert isinstance(analyzer.fallback, HeuristicAnalyzer)
    assert analyzer.usage is sink
    analyzer.warmup()


def test_from_settings_requires_key(anthropic_stub: types.ModuleType) -> None:
    with pytest.raises(ProviderConfigError, match="ANTHROPIC_API_KEY"):
        ClaudeAnalyzer.from_settings(Settings.from_env({}))


def test_module_never_imports_sdk_at_top_level() -> None:
    import bookreader.providers.anthropic.analysis as module

    assert not hasattr(module, "anthropic")
    assert "anthropic" not in module.__dict__
