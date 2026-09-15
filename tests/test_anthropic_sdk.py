"""ClaudeAnalyzer against the *real* ``anthropic`` SDK, offline: its exception classes go through
``map_error`` as the SDK constructs them, its typed ``Message`` objects go through the parse /
refusal / usage paths, and ``analyze_chunk`` drives a real ``anthropic.Anthropic`` client over an
``httpx2.MockTransport`` that serves canned SSE bodies. ``test_anthropic_analyzer.py`` covers the
same logic with a stub SDK; this module keeps that stub honest. Skipped when the extra is absent."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

anthropic = pytest.importorskip("anthropic")
httpx = pytest.importorskip("httpx2")

import bookreader.retry as retry_module  # noqa: E402
from bookreader.analysis.chunker import make_chunks  # noqa: E402
from bookreader.analysis.schema import CHUNK_ANALYSIS_SCHEMA  # noqa: E402
from bookreader.ingest import load_book  # noqa: E402
from bookreader.providers.anthropic.analysis import ClaudeAnalyzer, map_error  # noqa: E402
from bookreader.providers.mock.analysis import HeuristicAnalyzer  # noqa: E402
from bookreader.types import CastBible, Chunk, ProviderPermanentError, ProviderTransientError  # noqa: E402

URL = "https://api.anthropic.com/v1/messages"
SSE_HEADERS = {"content-type": "text/event-stream"}


# --------------------------------------------------------------------------- fixtures / helpers
@pytest.fixture(scope="module")
def chunk(sample_book_path: Path) -> Chunk:
    chunks = make_chunks(load_book(sample_book_path).chapters[0], 6000)
    assert len(chunks) == 1
    return chunks[0]


@pytest.fixture(scope="module")
def bible() -> CastBible:
    return CastBible()


@pytest.fixture(scope="module")
def good_json(chunk: Chunk, bible: CastBible) -> str:
    analysis = HeuristicAnalyzer().analyze_chunk(chunk, bible)
    return json.dumps(analysis.model_dump(mode="json", exclude={"source", "warnings"}))


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    delays: list[float] = []
    monkeypatch.setattr(retry_module.time, "sleep", lambda s: delays.append(s))
    return delays


class RecordingUsage:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def record(self, capability: str, family: str, unit_type: str, units: float, **kw: Any) -> None:
        self.rows.append({"capability": capability, "family": family, "unit_type": unit_type, "units": units, **kw})


def _request() -> Any:
    return httpx.Request("POST", URL)


def _response(status: int, *, headers: dict[str, str] | None = None, body: Any = None) -> Any:
    return httpx.Response(status, headers=headers or {}, json=body, request=_request())


def _sse(events: list[dict[str, Any]]) -> bytes:
    return "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode()


def _message_start() -> dict[str, Any]:
    return {
        "type": "message_start",
        "message": {
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5", "content": [],
            "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 1200, "output_tokens": 1, "cache_read_input_tokens": 1000, "cache_creation_input_tokens": 0},
        },
    }


def reply_sse(text: str, *, stop_reason: str = "end_turn") -> bytes:
    """A complete stream the way the API shapes it: a thinking block, then the text block."""
    return _sse([
        _message_start(),
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "who speaks"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": text[:40]}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": text[40:]}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None}, "usage": {"output_tokens": 340}},
        {"type": "message_stop"},
    ])


def refusal_sse() -> bytes:
    return _sse([
        _message_start(),
        {"type": "message_delta", "delta": {"stop_reason": "refusal", "stop_sequence": None, "stop_details": {"type": "refusal", "category": "general_harms", "explanation": "no"}}, "usage": {"output_tokens": 1}},
        {"type": "message_stop"},
    ])


def mid_stream_error_sse(error_type: str) -> bytes:
    """The stream starts (HTTP 200) and then the API sends an ``error`` event."""
    return _sse([_message_start(), {"type": "error", "error": {"type": error_type, "message": error_type}}])


class ScriptedServer:
    """One canned ``httpx2.Response`` (or exception) per request, the last one repeating."""

    def __init__(self, *responses: Any) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: Any) -> Any:
        self.requests.append(json.loads(request.content))
        response = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(response, Exception):
            raise response
        return response

    def client(self) -> Any:
        return anthropic.Anthropic(api_key="test-key", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(self)))


def stream_ok(body: bytes) -> Any:
    return httpx.Response(200, headers=SSE_HEADERS, content=body)


def make_analyzer(client: Any, usage: Any = None) -> ClaudeAnalyzer:
    return ClaudeAnalyzer(client, model_id="claude-opus-5", effort="medium", max_tokens=32000, fallback=HeuristicAnalyzer(), usage=usage)


# --------------------------------------------------------------------------- map_error with the SDK's own classes
def _status_error(cls_name: str, status: int, error_type: str, headers: dict[str, str] | None = None) -> Callable[[], Exception]:
    def build() -> Exception:
        body = {"type": "error", "error": {"type": error_type, "message": error_type}}
        return getattr(anthropic, cls_name)(error_type, response=_response(status, headers=headers, body=body), body=body)
    return build


@pytest.mark.parametrize(
    ("factory", "expected", "retry_after"),
    [
        (_status_error("RateLimitError", 429, "rate_limit_error", {"retry-after": "3"}), ProviderTransientError, 3.0),
        (_status_error("OverloadedError", 529, "overloaded_error"), ProviderTransientError, None),
        (_status_error("InternalServerError", 500, "api_error"), ProviderTransientError, None),
        (_status_error("ConflictError", 409, "api_error"), ProviderTransientError, None),
        (_status_error("APIStatusError", 200, "overloaded_error"), ProviderTransientError, None),   # SSE error event mid-stream
        (_status_error("APIStatusError", 200, "api_error"), ProviderTransientError, None),
        (_status_error("APIStatusError", 200, "invalid_request_error"), ProviderPermanentError, None),
        (_status_error("RequestTooLargeError", 413, "request_too_large"), ProviderPermanentError, None),
        (_status_error("BadRequestError", 400, "invalid_request_error"), ProviderPermanentError, None),
        (_status_error("AuthenticationError", 401, "authentication_error"), ProviderPermanentError, None),
        (_status_error("PermissionDeniedError", 403, "permission_error"), ProviderPermanentError, None),
        (_status_error("NotFoundError", 404, "not_found_error"), ProviderPermanentError, None),
        (lambda: anthropic.APIConnectionError(request=_request()), ProviderTransientError, None),
        (lambda: anthropic.APITimeoutError(request=_request()), ProviderTransientError, None),
    ],
    ids=lambda v: v if isinstance(v, str) else getattr(v, "__name__", repr(v)),
)
def test_map_error_with_real_sdk_exceptions(factory: Callable[[], Exception], expected: type, retry_after: float | None) -> None:
    exc = factory()
    assert isinstance(exc, anthropic.APIError)
    mapped = map_error(exc)
    assert type(mapped) is expected, f"{type(exc).__name__} -> {type(mapped).__name__}"
    if expected is ProviderTransientError:
        assert mapped.retry_after == retry_after


# --------------------------------------------------------------------------- typed Message objects
def _typed_message(content: list[Any], *, stop_reason: str = "end_turn", stop_details: Any = None) -> Any:
    from anthropic.types import Message, Usage

    return Message(
        id="msg_1", type="message", role="assistant", model="claude-opus-5", content=content,
        stop_reason=stop_reason, stop_details=stop_details,
        usage=Usage(input_tokens=1200, output_tokens=340, cache_read_input_tokens=1000, cache_creation_input_tokens=0),
    )


def test_parse_and_usage_from_a_typed_message(chunk: Chunk, bible: CastBible, good_json: str) -> None:
    from anthropic.types import TextBlock, ThinkingBlock

    usage = RecordingUsage()
    analyzer = make_analyzer(client=None, usage=usage)
    msg = _typed_message([ThinkingBlock(type="thinking", thinking="hmm", signature="sig"), TextBlock(type="text", text=good_json)])
    analysis = analyzer._parse(msg, chunk, bible)
    span_ids = {span.id for span in chunk.spans}
    assert analysis.source == "llm" and analysis.labels and all(label.span_id in span_ids for label in analysis.labels)

    analyzer._record_usage(msg, "ch01:chunk0", 12)
    assert {row["unit_type"]: row["units"] for row in usage.rows} == {
        "input_tokens": 1200.0, "output_tokens": 340.0, "cache_read_input_tokens": 1000.0, "cache_creation_input_tokens": 0.0,
    }
    assert all(row["family"] == "anthropic" and row["meta"]["model_id"] == "claude-opus-5" for row in usage.rows)


def test_typed_refusal_falls_back_with_its_category(chunk: Chunk, bible: CastBible) -> None:
    from anthropic.types import RefusalStopDetails

    details = RefusalStopDetails(type="refusal", category="general_harms", explanation="nope")
    analysis = make_analyzer(client=None)._refused(_typed_message([], stop_reason="refusal", stop_details=details), chunk, bible, "ch01:chunk0")
    assert analysis.source == "heuristic" and "refusal:general_harms" in analysis.warnings


# --------------------------------------------------------------------------- analyze_chunk over a real client + mock transport
def test_analyze_chunk_through_the_real_client(chunk: Chunk, bible: CastBible, good_json: str) -> None:
    server = ScriptedServer(stream_ok(reply_sse(good_json)))
    usage = RecordingUsage()
    analysis = make_analyzer(server.client(), usage=usage).analyze_chunk(chunk, bible)
    assert analysis.source == "llm" and analysis.warnings == []
    assert len(server.requests) == 1
    sent = server.requests[0]
    assert sent["model"] == "claude-opus-5" and sent["max_tokens"] == 32000 and sent["stream"] is True
    assert sent["output_config"] == {"format": {"type": "json_schema", "schema": CHUNK_ANALYSIS_SCHEMA}, "effort": "medium"}
    assert sent["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert sent["messages"][0]["role"] == "user" and chunk.spans[0].text[:20] in sent["messages"][0]["content"]
    assert {row["unit_type"]: row["units"] for row in usage.rows}["input_tokens"] == 1200.0    # from the real stream's usage


def test_mid_stream_overloaded_error_is_retried_through_the_real_client(chunk: Chunk, bible: CastBible, good_json: str, no_sleep: list[float]) -> None:
    server = ScriptedServer(stream_ok(mid_stream_error_sse("overloaded_error")), stream_ok(reply_sse(good_json)))
    analysis = make_analyzer(server.client()).analyze_chunk(chunk, bible)
    assert analysis.source == "llm" and len(server.requests) == 2 and len(no_sleep) == 1


def test_mid_stream_invalid_request_is_permanent_through_the_real_client(chunk: Chunk, bible: CastBible, no_sleep: list[float]) -> None:
    server = ScriptedServer(stream_ok(mid_stream_error_sse("invalid_request_error")))
    with pytest.raises(ProviderPermanentError):
        make_analyzer(server.client()).analyze_chunk(chunk, bible)
    assert len(server.requests) == 1 and no_sleep == []


def test_rate_limit_from_the_real_client_is_transient_after_four_attempts(chunk: Chunk, bible: CastBible, no_sleep: list[float]) -> None:
    body = {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}}
    server = ScriptedServer(httpx.Response(429, headers={"retry-after": "2"}, json=body))
    with pytest.raises(ProviderTransientError) as info:
        make_analyzer(server.client()).analyze_chunk(chunk, bible)
    assert info.value.retry_after == 2.0 and info.value.unit == "ch01:chunk0"
    assert len(server.requests) == 4 and len(no_sleep) == 3


def test_connection_error_from_the_real_client_is_transient(chunk: Chunk, bible: CastBible, no_sleep: list[float]) -> None:
    def refuse(request: Any) -> Any:
        raise httpx.ConnectError("connection refused", request=request)

    client = anthropic.Anthropic(api_key="test-key", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(refuse)))
    with pytest.raises(ProviderTransientError):
        make_analyzer(client).analyze_chunk(chunk, bible)
    assert len(no_sleep) == 3


def test_streamed_refusal_falls_back_to_the_heuristic(chunk: Chunk, bible: CastBible) -> None:
    server = ScriptedServer(stream_ok(refusal_sse()))
    analysis = make_analyzer(server.client()).analyze_chunk(chunk, bible)
    assert analysis.source == "heuristic" and "refusal:general_harms" in analysis.warnings and len(server.requests) == 1
