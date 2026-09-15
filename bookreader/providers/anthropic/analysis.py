"""bookreader.providers.anthropic.analysis - ClaudeAnalyzer, the LLM text analyzer.

One structured-output request per chunk (system prompt cached, JSON schema enforced, adaptive
thinking left to the model). The reply is parsed into :class:`bookreader.types.ChunkAnalysis`
and repaired by :func:`bookreader.analysis.validate.validate_chunk_analysis`; an unusable reply
gets exactly one repair request carrying a note about what was wrong, and if that fails too the
heuristic fallback labels the chunk (never a job failure). Refusals fall back the same way; a
reply truncated at ``max_tokens`` splits the chunk in half and analyzes each half, and a request
that hit the model's context window (``model_context_window_exceeded``) goes straight to the
heuristic fallback (a repair request would only be larger). SDK failures are mapped onto the
bookreader error taxonomy and transient ones are retried by :func:`bookreader.retry.with_retry`.

Retry and concurrency policy: this adapter is the only layer that retries (4 attempts, backoff
and ``retry-after`` honoured; the SDK client is built with ``max_retries=0`` so one failing
request costs at most 4 HTTP attempts) and the only layer that acquires the ``anthropic``
FamilyLimiter (the pipeline never wraps ``analyze_chunk`` in it again: nesting the same semaphore
deadlocks at width 1).

The ``anthropic`` SDK is never imported at module level: ``check``/``from_settings`` import it
under ``try`` (so the registry can report the missing extra) and the exception classes are looked
up the same way when an SDK call fails.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import ValidationError

from bookreader.analysis.chunker import carry_speakers
from bookreader.analysis.prompts import SYSTEM_PROMPT, build_user_message
from bookreader.analysis.schema import CHUNK_ANALYSIS_SCHEMA, PROMPT_VERSION
from bookreader.analysis.validate import AnalysisInvalid, validate_chunk_analysis
from bookreader.providers.base import NullUsage, TextAnalyzer, UsageSink
from bookreader.providers.mock.analysis import HeuristicAnalyzer
from bookreader.retry import FamilyLimiter, with_retry
from bookreader.types import (
    BookreaderError,
    CastBible,
    Chunk,
    ChunkAnalysis,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderTransientError,
    Span,
)

if TYPE_CHECKING:
    from bookreader.settings import Settings

log = logging.getLogger(__name__)

FAMILY = "anthropic"
INSTALL_HINT = "pip install 'bookreader[anthropic]'"
MISSING_SDK = f"analysis provider 'anthropic' needs the anthropic SDK: {INSTALL_HINT}"
CLIENT_MAX_RETRIES = 0                       # bookreader.retry.with_retry is the single retry layer
CLIENT_TIMEOUT_S = 600
USAGE_UNIT_TYPES: tuple[str, ...] = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
TRANSIENT_STATUS: frozenset[int] = frozenset({408, 409, 429})
TRANSIENT_CLASSES: tuple[str, ...] = ("RateLimitError", "InternalServerError", "APITimeoutError", "APIConnectionError")
PERMANENT_CLASSES: tuple[str, ...] = ("BadRequestError", "AuthenticationError", "PermissionDeniedError", "NotFoundError")
# ``error.type`` values from the API's error body. An SSE ``error`` event received after the
# stream has started surfaces as a bare APIStatusError whose status_code is the stream's 200, so
# the body type is the only signal that the failure was transient.
TRANSIENT_ERROR_TYPES: frozenset[str] = frozenset({"overloaded_error", "api_error", "rate_limit_error"})
PERMANENT_ERROR_TYPES: frozenset[str] = frozenset(
    {"invalid_request_error", "authentication_error", "permission_error", "not_found_error", "request_too_large"}
)
STREAM_OK_STATUS = 200
REPAIR_NOTE_MAX_CHARS = 1500
CONTEXT_PARAGRAPHS = 2
SPAN_ID_RE = re.compile(r"^c\d+p(\d+)s\d+$")


def _sdk_module() -> Any:
    """The ``anthropic`` module, or ``None`` when it is not installed (or stubbed out)."""
    try:
        import anthropic
    except ImportError:
        return None
    return anthropic


def _isinstance_named(exc: BaseException, module: Any, names: tuple[str, ...]) -> bool:
    return any(isinstance(exc, cls) for cls in (getattr(module, name, None) for name in names) if isinstance(cls, type))


def _retry_after(exc: BaseException) -> float | None:
    """Seconds from the response's ``retry-after`` header when it is numeric."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if not headers:
        return None
    try:
        items = list(dict(headers).items())
    except (TypeError, ValueError):
        return None
    for key, value in items:
        if str(key).lower() == "retry-after":
            try:
                return max(0.0, float(value))
            except (TypeError, ValueError):
                return None
    return None


def _body_error_type(exc: BaseException) -> str | None:
    """``error.type`` from the SDK exception's parsed body, when it has one."""
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return None
    error = body.get("error")
    if not isinstance(error, dict):
        return None
    kind = error.get("type")
    return str(kind) if isinstance(kind, str) else None


def map_error(exc: BaseException) -> BookreaderError:
    """Translate an SDK exception into the bookreader taxonomy.

    RateLimitError, InternalServerError, APITimeoutError, APIConnectionError and any
    APIStatusError with status 408/409/429 or 5xx become ProviderTransientError (``retry_after``
    from the response headers when present); BadRequestError, AuthenticationError,
    PermissionDeniedError, NotFoundError and every other status error become
    ProviderPermanentError. Bookreader errors pass through unchanged.

    An ``error`` SSE event after the stream has started (``overloaded_error``, ``api_error``) is
    raised by the SDK as a bare APIStatusError carrying the stream's own 200 status, so the body's
    ``error.type`` is consulted first: a transient type is always transient, and a 200-status error
    is transient unless its type is a known permanent one.
    """
    if isinstance(exc, BookreaderError):
        return exc
    sdk = _sdk_module()
    message = f"anthropic: {exc}" if str(exc) else f"anthropic: {type(exc).__name__}"
    status = getattr(exc, "status_code", None)
    error_type = _body_error_type(exc)
    if _isinstance_named(exc, sdk, TRANSIENT_CLASSES):
        return ProviderTransientError(message, retry_after=_retry_after(exc))
    if error_type in TRANSIENT_ERROR_TYPES:
        return ProviderTransientError(message, retry_after=_retry_after(exc))
    if _isinstance_named(exc, sdk, PERMANENT_CLASSES):
        return ProviderPermanentError(message)
    if isinstance(status, int) and (status in TRANSIENT_STATUS or status >= 500):
        return ProviderTransientError(message, retry_after=_retry_after(exc))
    if status == STREAM_OK_STATUS and error_type not in PERMANENT_ERROR_TYPES:
        return ProviderTransientError(message, retry_after=_retry_after(exc))
    return ProviderPermanentError(message)


def _paragraph_groups(spans: list[Span]) -> list[list[Span]]:
    """Spans grouped by paragraph, in order, using the ``c<ch>p<para>s<n>`` id convention."""
    groups: list[list[Span]] = []
    current_key: str | None = None
    for span in spans:
        match = SPAN_ID_RE.match(span.id)
        key = match.group(1) if match else span.id
        if key != current_key:
            groups.append([])
            current_key = key
        groups[-1].append(span)
    return groups


def _paragraph_number(span: Span, default: int) -> int:
    match = SPAN_ID_RE.match(span.id)
    return int(match.group(1)) if match else default


def split_chunk(chunk: Chunk) -> tuple[Chunk, Chunk]:
    """Halve *chunk* at its paragraph midpoint. The second half's ``context_before`` is the text
    of the first half's last paragraphs (as the chunker would give it); both keep ``prior_mood``.
    Raises ``ValueError`` for a single-paragraph chunk."""
    groups = _paragraph_groups(chunk.spans)
    if len(groups) < 2:
        raise ValueError("cannot split a single-paragraph chunk")
    middle = len(groups) // 2
    first, second = groups[:middle], groups[middle:]
    first_spans = [span for group in first for span in group]
    second_spans = [span for group in second for span in group]
    context = "\n\n".join(" ".join(span.text for span in group) for group in first[-CONTEXT_PARAGRAPHS:])
    head = chunk.model_copy(
        update={
            "spans": first_spans,
            "paragraph_end": _paragraph_number(first_spans[-1], chunk.paragraph_end),
        }
    )
    tail = chunk.model_copy(
        update={
            "spans": second_spans,
            "paragraph_start": _paragraph_number(second_spans[0], chunk.paragraph_start),
            "context_before": context,
        }
    )
    return head, tail


def _describe_failure(exc: BaseException) -> str:
    """The repair note for a reply that could not be used."""
    if isinstance(exc, AnalysisInvalid):
        note = f"Your previous reply could not be used: {exc}. Label every quote span listed below and reference only their ids."
    elif isinstance(exc, ValidationError):
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err.get('loc', ()))}: {err.get('msg', '')}" for err in exc.errors()[:8]
        )
        note = f"Your previous reply did not match the schema: {problems}. Use only the vocabularies given."
    elif isinstance(exc, json.JSONDecodeError):
        note = f"Your previous reply was not valid JSON ({exc.msg} at position {exc.pos}). Reply with one JSON object only."
    else:
        note = f"Your previous reply could not be used: {exc}"
    return note[:REPAIR_NOTE_MAX_CHARS]


def _first_text(msg: Any) -> str | None:
    """The first text block of a message (thinking blocks precede it and are skipped)."""
    return next((block.text for block in msg.content if getattr(block, "type", None) == "text"), None)


def _log_retry(attempt: int, exc: BaseException, delay: float) -> None:
    log.warning("anthropic call failed (attempt %d): %s; retrying in %.1fs", attempt, exc, delay)


class ClaudeAnalyzer:
    """Claude structured-output text analyzer (family ``anthropic``). Implements ``TextAnalyzer``."""

    family: ClassVar[str] = FAMILY

    def __init__(
        self,
        client: Any,
        model_id: str,
        effort: str,
        max_tokens: int,
        fallback: TextAnalyzer,
        usage: UsageSink | None = None,
        concurrency: int = 4,
    ) -> None:
        self.client = client
        self.model_id = model_id
        self.effort = effort
        self.max_tokens = int(max_tokens)
        self.fallback = fallback
        self.usage: UsageSink = usage or NullUsage()
        self.concurrency = max(1, int(concurrency))
        self.cache_version = f"{model_id}:{PROMPT_VERSION}"

    # ------------------------------------------------------------------ lifecycle
    @classmethod
    def check(cls, settings: "Settings") -> list[str]:
        """Only verify that the SDK imports; the key is checked by the registry via the manifest."""
        if _sdk_module() is None:
            raise ProviderConfigError(MISSING_SDK)
        return []

    @classmethod
    def from_settings(cls, settings: "Settings", usage: UsageSink | None = None) -> "ClaudeAnalyzer":
        """Build the real ``anthropic.Anthropic`` client from ``ANTHROPIC_API_KEY``."""
        sdk = _sdk_module()
        if sdk is None:
            raise ProviderConfigError(MISSING_SDK)
        api_key = settings.secrets.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ProviderConfigError("analysis provider 'anthropic' needs environment variable ANTHROPIC_API_KEY")
        client = sdk.Anthropic(api_key=api_key, max_retries=CLIENT_MAX_RETRIES, timeout=CLIENT_TIMEOUT_S)
        return cls(
            client,
            model_id=settings.anthropic_model,
            effort=settings.anthropic_effort,
            max_tokens=settings.anthropic_max_tokens,
            fallback=HeuristicAnalyzer(),
            usage=usage,
            concurrency=settings.concurrency,
        )

    def warmup(self) -> None:
        """No-op: nothing to load and no paid call should happen at startup."""
        return None

    # ------------------------------------------------------------------ analysis
    def analyze_chunk(self, chunk: Chunk, bible: CastBible) -> ChunkAnalysis:
        """Label *chunk* with Claude; falls back to the heuristic analyzer on refusal or an
        unrepairable reply, splits on truncation, and raises Provider*Error on SDK failures."""
        return self._analyze(chunk, bible, repair_note=None)

    @staticmethod
    def _unit(chunk: Chunk) -> str:
        return f"ch{chunk.chapter_index:02d}:chunk{chunk.chunk_index}"

    def _analyze(self, chunk: Chunk, bible: CastBible, repair_note: str | None) -> ChunkAnalysis:
        unit = self._unit(chunk)
        msg = self._request(chunk, bible, unit, repair_note)
        stop_reason = getattr(msg, "stop_reason", None)
        if stop_reason == "refusal":
            return self._refused(msg, chunk, bible, unit)
        if stop_reason == "max_tokens":
            return self._split_and_retry(chunk, bible, unit)
        if stop_reason == "model_context_window_exceeded":
            # The reply is cut off and a repair request (same prompt plus a note) is strictly larger,
            # so it would hit the same stop; halving the span list would not shrink the bible either.
            log.warning("%s: request hit the model context window; using the heuristic analyzer", unit)
            return self._fallback(chunk, bible, "context_window_exceeded")
        try:
            analysis = self._parse(msg, chunk, bible)
        except ValueError as exc:  # AnalysisInvalid, pydantic ValidationError and JSONDecodeError all subclass it
            if repair_note is None:
                note = _describe_failure(exc)
                log.warning("%s: unusable reply (%s); requesting one repair", unit, type(exc).__name__)
                return self._analyze(chunk, bible, repair_note=note)
            log.warning("%s: repair reply still unusable (%s); using the heuristic analyzer", unit, exc)
            return self._fallback(chunk, bible, "llm_invalid")
        log.debug(
            "%s: %d labels, %d characters, %d sfx, %d music, %d warnings%s",
            unit, len(analysis.labels), len(analysis.characters), len(analysis.sfx_cues), len(analysis.music_cues),
            len(analysis.warnings), " (after repair)" if repair_note else "",
        )
        return analysis

    def _parse(self, msg: Any, chunk: Chunk, bible: CastBible) -> ChunkAnalysis:
        text = _first_text(msg)
        if text is None:
            raise ValueError("the reply contained no text block")
        data = json.loads(text)
        if isinstance(data, dict):
            for character in data.get("characters") or []:
                if isinstance(character, dict) and character.get("merge_into") == "":
                    character["merge_into"] = None          # the schema's "no merge" spelling
        analysis = ChunkAnalysis.model_validate(data).model_copy(update={"source": "llm"})
        repaired, _ = validate_chunk_analysis(analysis, chunk, bible)
        return repaired

    def _request(self, chunk: Chunk, bible: CastBible, unit: str, repair_note: str | None) -> Any:
        """One structured-output call under the family semaphore, retried on transient errors.

        The semaphore is held only while a request is in flight (``with_retry`` sleeps outside
        it) and nowhere else in the process, so width 1 never deadlocks."""
        kwargs = self._request_kwargs(chunk, bible, repair_note)

        def attempt() -> Any:
            with FamilyLimiter.acquire(FAMILY, self.concurrency):
                try:
                    with self.client.messages.stream(**kwargs) as stream:
                        return stream.get_final_message()
                except BookreaderError:
                    raise
                except Exception as exc:  # noqa: BLE001 - every SDK failure is classified by map_error
                    raise map_error(exc) from exc

        started = time.monotonic()
        try:
            msg = with_retry(attempt, on_retry=_log_retry)
        except BookreaderError as exc:
            exc.unit = exc.unit or unit
            raise
        self._record_usage(msg, unit, int((time.monotonic() - started) * 1000))
        return msg

    def _request_kwargs(self, chunk: Chunk, bible: CastBible, repair_note: str | None) -> dict[str, Any]:
        return {
            "model": self.model_id,
            "max_tokens": self.max_tokens,
            "system": [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": build_user_message(chunk, bible, repair_note)}],
            "output_config": {
                "format": {"type": "json_schema", "schema": CHUNK_ANALYSIS_SCHEMA},
                "effort": self.effort,
            },
        }

    def _record_usage(self, msg: Any, unit: str, duration_ms: int) -> None:
        usage = getattr(msg, "usage", None)
        for unit_type in USAGE_UNIT_TYPES:
            self.usage.record(
                "analysis", self.family, unit_type, float(getattr(usage, unit_type, None) or 0),
                duration_ms=duration_ms, meta={"chunk": unit, "model_id": self.model_id},
            )

    # ------------------------------------------------------------------ degraded paths
    def _refused(self, msg: Any, chunk: Chunk, bible: CastBible, unit: str) -> ChunkAnalysis:
        details = getattr(msg, "stop_details", None)
        category = str(getattr(details, "category", None) or "unknown") if details is not None else "unknown"
        if details is not None:
            log.warning("%s: refusal (%s): %s", unit, category, getattr(details, "explanation", ""))
        else:
            log.warning("%s: refusal without details", unit)
        return self._fallback(chunk, bible, f"refusal:{category}")

    def _fallback(self, chunk: Chunk, bible: CastBible, warning: str) -> ChunkAnalysis:
        analysis = self.fallback.analyze_chunk(chunk, bible)
        return analysis.model_copy(update={"source": "heuristic", "warnings": [*analysis.warnings, warning]})

    def _split_and_retry(self, chunk: Chunk, bible: CastBible, unit: str) -> ChunkAnalysis:
        try:
            head, tail = split_chunk(chunk)
        except ValueError as exc:
            raise ProviderPermanentError("analysis output truncated for single paragraph", unit=unit) from exc
        log.warning(
            "%s: output truncated at %d tokens; splitting paragraphs %d-%d / %d-%d",
            unit, self.max_tokens, head.paragraph_start, head.paragraph_end, tail.paragraph_start, tail.paragraph_end,
        )
        first = self._analyze(head, bible, repair_note=None)
        tail = tail.model_copy(update={"prior_speakers": carry_speakers(chunk.prior_speakers, head, first)})
        parts = [first, self._analyze(tail, bible, repair_note=None)]
        return ChunkAnalysis(
            labels=[label for part in parts for label in part.labels],
            characters=[character for part in parts for character in part.characters],
            sfx_cues=[cue for part in parts for cue in part.sfx_cues],
            music_cues=[cue for part in parts for cue in part.music_cues],
            source="llm" if all(part.source == "llm" for part in parts) else "heuristic",
            warnings=[warning for part in parts for warning in part.warnings],
        )
