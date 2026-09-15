"""bookreader.providers.elevenlabs.client - SDK construction, PCM decoding and error mapping.

Everything the three ElevenLabs adapters share lives here. The SDK is imported only inside
:func:`make_client` (``try: import`` so tests can stub or null out ``sys.modules`` entries);
the adapters themselves only ever see an injected client object.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, TypeVar

from bookreader.retry import FamilyLimiter, with_retry
from bookreader.types import (
    SAMPLE_RATE,
    AudioClip,
    BookreaderError,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderTransientError,
)

log = logging.getLogger(__name__)

FAMILY = "elevenlabs"
INSTALL_HINT = "pip install 'bookreader[elevenlabs]'"
OUTPUT_FORMAT = "pcm_22050"          # raw 16-bit little-endian mono PCM at the canonical rate
TRANSIENT_STATUS: frozenset[int] = frozenset({408, 409, 429})

T = TypeVar("T")


def make_client(api_key: str) -> Any:
    """Build the real ``ElevenLabs`` client. Raises ProviderConfigError when the SDK is missing."""
    try:
        from elevenlabs.client import ElevenLabs
    except ImportError as exc:
        raise ProviderConfigError(f"the elevenlabs SDK is not installed; {INSTALL_HINT}") from exc
    return ElevenLabs(api_key=api_key)


def pcm_to_clip(chunks: Iterable[bytes] | bytes) -> AudioClip:
    """Join the SDK's byte iterator (``pcm_22050``) into a canonical-rate AudioClip."""
    data = chunks if isinstance(chunks, (bytes, bytearray)) else b"".join(chunks)
    return AudioClip.from_pcm16_bytes(bytes(data), SAMPLE_RATE)


def _retry_after(headers: Any) -> float | None:
    """Seconds from a ``Retry-After`` header when it is numeric (HTTP-date values are ignored)."""
    if not headers:
        return None
    try:
        items = dict(headers).items()
    except (TypeError, ValueError):
        return None
    for key, value in items:
        if str(key).lower() == "retry-after":
            try:
                return max(0.0, float(value))
            except (TypeError, ValueError):
                return None
    return None


def _is_transport_error(exc: BaseException) -> bool:
    """True for httpx transport failures (connect, read, timeout...)."""
    try:
        import httpx
    except ImportError:
        return False
    return isinstance(exc, httpx.TransportError)


def map_error(exc: BaseException) -> BookreaderError:
    """Translate an SDK/transport exception into the bookreader error taxonomy.

    * ``status_code`` 408/409/429 or any 5xx, and httpx transport errors -> ProviderTransientError
      (``retry_after`` taken from the response headers when present);
    * any other 4xx, and anything without a status code -> ProviderPermanentError.
    Bookreader errors are returned unchanged.
    """
    if isinstance(exc, BookreaderError):
        return exc
    status = getattr(exc, "status_code", None)
    message = f"elevenlabs: {exc}" if str(exc) else f"elevenlabs: {type(exc).__name__}"
    if isinstance(status, int) and (status in TRANSIENT_STATUS or status >= 500):
        return ProviderTransientError(message, retry_after=_retry_after(getattr(exc, "headers", None)))
    if _is_transport_error(exc):
        return ProviderTransientError(message)
    return ProviderPermanentError(message)


def guarded_call(fn: Callable[[], T], concurrency: int) -> T:
    """Run one API call under the family semaphore with backoff on transient errors.

    The semaphore is held only while the request is in flight (not during backoff sleeps), and
    every non-bookreader exception is passed through :func:`map_error` first.
    """

    def attempt() -> T:
        with FamilyLimiter.acquire(FAMILY, concurrency):
            try:
                return fn()
            except BookreaderError:
                raise
            except Exception as exc:  # noqa: BLE001 - every SDK failure is classified by map_error
                raise map_error(exc) from exc

    return with_retry(attempt, on_retry=_log_retry)


def _log_retry(attempt: int, exc: BaseException, delay: float) -> None:
    log.warning("elevenlabs call failed (attempt %d): %s; retrying in %.1fs", attempt, exc, delay)


def check_sdk() -> list[str]:
    """Shared ``check()``: only verify the SDK imports. Never touches the network."""
    try:
        import elevenlabs  # noqa: F401 - presence check only
    except ImportError as exc:
        raise ProviderConfigError(f"the elevenlabs SDK is not installed; {INSTALL_HINT}") from exc
    return []


def api_key_from(settings: Any) -> str:
    """The ELEVENLABS_API_KEY secret, or ProviderConfigError naming the variable."""
    key = settings.secrets.get("ELEVENLABS_API_KEY")
    if not key:
        raise ProviderConfigError("elevenlabs providers need environment variable ELEVENLABS_API_KEY")
    return str(key)
