"""with_retry backoff and FamilyLimiter."""
import threading

import pytest

from bookreader.retry import FamilyLimiter, with_retry
from bookreader.types import ProviderPermanentError, ProviderTransientError


def test_retries_transient_then_succeeds():
    calls = []
    sleeps = []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise ProviderTransientError("busy")
        return "ok"

    assert with_retry(fn, attempts=4, base=1.0, sleep=sleeps.append) == "ok"
    assert len(calls) == 3
    assert len(sleeps) == 2
    assert 0.5 <= sleeps[0] <= 1.0
    assert 1.0 <= sleeps[1] <= 2.0


def test_gives_up_after_attempts_and_honors_retry_after():
    sleeps = []

    def fn():
        raise ProviderTransientError("rate limited", retry_after=7)

    with pytest.raises(ProviderTransientError):
        with_retry(fn, attempts=3, sleep=sleeps.append)
    assert sleeps == [7.0, 7.0]


def test_permanent_errors_are_not_retried():
    calls = []

    def fn():
        calls.append(1)
        raise ProviderPermanentError("bad request")

    with pytest.raises(ProviderPermanentError):
        with_retry(fn, attempts=5, sleep=lambda s: None)
    assert calls == [1]


def test_on_retry_hook_and_monkeypatched_time_sleep(monkeypatch):
    import bookreader.retry as r

    slept = []
    monkeypatch.setattr(r.time, "sleep", slept.append)
    seen = []
    state = {"n": 0}

    def fn():
        state["n"] += 1
        if state["n"] == 1:
            raise ProviderTransientError("once")
        return state["n"]

    assert with_retry(fn, on_retry=lambda a, e, d: seen.append((a, type(e).__name__))) == 2
    assert seen == [(1, "ProviderTransientError")]
    assert len(slept) == 1


def test_family_limiter_bounds_concurrency():
    FamilyLimiter.reset()
    sem = FamilyLimiter.get("x", 2)
    assert FamilyLimiter.get("x", 99) is sem  # width fixed on first use
    assert FamilyLimiter.width("x") == 2
    active = []
    peak = [0]
    lock = threading.Lock()
    go = threading.Event()

    def worker():
        with FamilyLimiter.acquire("x", 2):
            with lock:
                active.append(1)
                peak[0] = max(peak[0], len(active))
            go.wait(0.05)
            with lock:
                active.pop()

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak[0] == 2
    FamilyLimiter.reset()
