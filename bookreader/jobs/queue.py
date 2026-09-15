"""bookreader.jobs.queue - the three-method worker seam: submit / start / stop (+ depth).

``InProcessQueue`` runs jobs on daemon threads inside the API process; ``InlineQueue`` runs them
synchronously inside ``submit`` (tests, CLI). Both hand the runner a ``threading.Event`` that is
set on ``stop()`` so the running job's next ``check_cancelled()`` raises ``JobCancelled('shutdown')``
and the job goes back to ``queued``. An external queue only needs to implement the same three
methods and call :func:`bookreader.pipeline.run.run_job` in its consumer.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import TYPE_CHECKING, Callable, Protocol

if TYPE_CHECKING:
    from bookreader.settings import Settings

log = logging.getLogger(__name__)

JobRunner = Callable[[str, threading.Event], None]
POLL_INTERVAL_S = 0.1


class JobQueue(Protocol):
    """What the API needs from a queue backend."""

    def submit(self, job_id: str) -> None: ...

    def start(self) -> None: ...

    def stop(self, timeout: float = 30) -> None: ...

    def depth(self) -> int: ...


class InProcessQueue:
    """A ``queue.Queue`` drained by *workers* daemon threads."""

    backend = "thread"

    def __init__(self, run: JobRunner, workers: int = 1) -> None:
        self.run = run
        self.workers = max(1, int(workers))
        self.stop_event = threading.Event()
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._running: dict[str, str] = {}
        self._lock = threading.Lock()

    def submit(self, job_id: str) -> None:
        self._queue.put(job_id)

    def start(self) -> None:
        """Spawn the worker threads (idempotent)."""
        if self._threads:
            return
        self.stop_event.clear()
        for index in range(self.workers):
            thread = threading.Thread(target=self._worker, name=f"bookreader-worker-{index}", daemon=True)
            thread.start()
            self._threads.append(thread)
        log.info("job queue started with %d worker(s)", self.workers)

    def stop(self, timeout: float = 30) -> None:
        """Signal every worker to stop and wait up to *timeout* seconds for them to finish."""
        self.stop_event.set()
        for _ in self._threads:
            self._queue.put(None)
        deadline = time.monotonic() + timeout
        for thread in self._threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        stragglers = [t.name for t in self._threads if t.is_alive()]
        if stragglers:
            log.warning("job queue stopped with worker(s) still running: %s", ", ".join(stragglers))
        self._threads = []

    def depth(self) -> int:
        return self._queue.qsize()

    def running(self) -> list[str]:
        """Ids of the jobs currently being run."""
        with self._lock:
            return list(self._running.values())

    def _worker(self) -> None:
        name = threading.current_thread().name
        while not self.stop_event.is_set():
            try:
                job_id = self._queue.get(timeout=POLL_INTERVAL_S)
            except queue.Empty:
                continue
            if job_id is None:
                break
            with self._lock:
                self._running[name] = job_id
            try:
                self.run(job_id, self.stop_event)
            except Exception:
                log.exception("job %s crashed the worker loop; continuing", job_id)
            finally:
                with self._lock:
                    self._running.pop(name, None)
                self._queue.task_done()


class InlineQueue:
    """Runs each job synchronously inside ``submit`` on the calling thread."""

    backend = "inline"

    def __init__(self, run: JobRunner) -> None:
        self.run = run
        self.stop_event = threading.Event()

    def submit(self, job_id: str) -> None:
        self.run(job_id, self.stop_event)

    def start(self) -> None:
        self.stop_event.clear()

    def stop(self, timeout: float = 30) -> None:
        del timeout  # nothing runs in the background
        self.stop_event.set()

    def depth(self) -> int:
        return 0

    def running(self) -> list[str]:
        return []


def build_queue(settings: "Settings", run: JobRunner) -> InProcessQueue | InlineQueue:
    """The queue backend selected by ``settings.worker_mode`` (``thread`` or ``inline``)."""
    if settings.worker_mode == "inline":
        return InlineQueue(run)
    return InProcessQueue(run, workers=settings.workers)
