"""bookreader.retry - exponential backoff for transient provider errors and per-family concurrency limits."""
from __future__ import annotations

import random
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator, TypeVar

from bookreader.types import ProviderTransientError

T = TypeVar("T")


def with_retry(
    fn: Callable[[], T],
    *,
    attempts: int = 4,
    base: float = 1.0,
    cap: float = 30.0,
    retry_on: tuple[type[BaseException], ...] = (ProviderTransientError,),
    sleep: Callable[[float], Any] | None = None,
    on_retry: Callable[[int, BaseException, float], Any] | None = None,
) -> T:
    """Call *fn* until it succeeds or *attempts* is exhausted.

    Delay before attempt n (1-based, n >= 2) is ``min(cap, base * 2**(n-2))`` with +-50% jitter,
    raised to the exception's ``retry_after`` attribute when present. ``sleep`` defaults to
    ``time.sleep`` looked up at call time so tests can monkeypatch ``bookreader.retry.time.sleep``.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    last: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except retry_on as exc:  # type: ignore[misc]
            last = exc
            if attempt == attempts:
                raise
            delay = min(cap, base * (2 ** (attempt - 1)))
            delay = delay * (0.5 + random.random() * 0.5)
            retry_after = getattr(exc, "retry_after", None)
            if retry_after is not None:
                try:
                    delay = max(delay, float(retry_after))
                except (TypeError, ValueError):
                    pass
            if on_retry is not None:
                on_retry(attempt, exc, delay)
            (sleep or time.sleep)(delay)
    assert last is not None  # pragma: no cover - loop always returns or raises
    raise last


class FamilyLimiter:
    """Process-wide bounded semaphore per provider family (shared across worker threads and jobs)."""

    _lock = threading.Lock()
    _semaphores: dict[str, threading.BoundedSemaphore] = {}
    _widths: dict[str, int] = {}

    @classmethod
    def get(cls, family: str, width: int) -> threading.BoundedSemaphore:
        width = max(1, int(width))
        with cls._lock:
            sem = cls._semaphores.get(family)
            if sem is None:
                sem = threading.BoundedSemaphore(width)
                cls._semaphores[family] = sem
                cls._widths[family] = width
            return sem

    @classmethod
    @contextmanager
    def acquire(cls, family: str, width: int) -> Iterator[None]:
        sem = cls.get(family, width)
        sem.acquire()
        try:
            yield
        finally:
            sem.release()

    @classmethod
    def width(cls, family: str) -> int | None:
        return cls._widths.get(family)

    @classmethod
    def reset(cls) -> None:
        """Forget every semaphore (tests only)."""
        with cls._lock:
            cls._semaphores.clear()
            cls._widths.clear()
