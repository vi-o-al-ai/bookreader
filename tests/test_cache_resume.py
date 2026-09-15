"""Cache reuse, failure/retry, cast overrides, cancellation and pruning through run_job."""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ClassVar

import pytest

from bookreader.jobs.db import JobStore, now_iso
from bookreader.jobs.paths import JobPaths, read_json
from bookreader.pipeline.cache import ClipCache
from bookreader.pipeline.run import create_job, run_job
from bookreader.providers.base import Providers
from bookreader.providers.mock.analysis import HeuristicAnalyzer
from bookreader.providers.mock.music import MockMusic
from bookreader.providers.mock.sfx import ProceduralSfx
from bookreader.providers.mock.tts import MockTTS
from bookreader.settings import Settings
from bookreader.types import (
    SAMPLE_RATE,
    AudioClip,
    Cast,
    ChapterScript,
    ChapterTts,
    Job,
    JobStatus,
    ProviderPermanentError,
    ProviderTransientError,
    Stage,
    StageState,
    TTSRequest,
    VoiceInfo,
)
from bookreader.usage import UsageRouter

MS_PER_CHAR = 4


# --------------------------------------------------------------------------- counting providers
class _Counting:
    """Delegating wrapper that counts calls to one method, optionally failing or hooking them."""

    family: ClassVar[str] = "mock"
    method: ClassVar[str] = ""

    def __init__(self, inner: object, fail_from: int | None = None, hook: Callable[[int], None] | None = None) -> None:
        self.inner = inner
        self.calls = 0
        self.fail_from = fail_from
        self.hook = hook
        self._lock = threading.Lock()

    def __getattr__(self, name: str) -> object:
        return getattr(self.inner, name)

    def _call(self, *args: object) -> object:
        with self._lock:
            self.calls += 1
            n = self.calls
        if self.hook is not None:
            self.hook(n)
        if self.fail_from is not None and n >= self.fail_from:
            raise ProviderTransientError(f"simulated outage on call {n}")
        return getattr(self.inner, self.method)(*args)


class CountingAnalyzer(_Counting):
    method = "analyze_chunk"

    def analyze_chunk(self, chunk: object, bible: object) -> object:
        return self._call(chunk, bible)


class CountingTTS(_Counting):
    method = "synthesize"

    def synthesize(self, req: TTSRequest) -> AudioClip:
        return self._call(req)  # type: ignore[return-value]


class CountingMusic(_Counting):
    method = "compose"

    def compose(self, req: object) -> object:
        return self._call(req)


class CountingSfx(_Counting):
    method = "generate"

    def generate(self, req: object) -> object:
        return self._call(req)


@dataclass
class Spied:
    providers: Providers
    analysis: CountingAnalyzer
    tts: CountingTTS
    music: CountingMusic
    sfx: CountingSfx

    @property
    def counts(self) -> dict[str, int]:
        return {"analysis": self.analysis.calls, "tts": self.tts.calls, "music": self.music.calls, "sfx": self.sfx.calls}


def spied(tts_fail_from: int | None = None, tts_hook: Callable[[int], None] | None = None) -> Spied:
    """Counting mock providers wired to the process-wide UsageRouter, like the API builds them."""
    router = UsageRouter()
    analysis = CountingAnalyzer(HeuristicAnalyzer(usage=router))
    tts = CountingTTS(MockTTS(ms_per_char=MS_PER_CHAR, usage=router), fail_from=tts_fail_from, hook=tts_hook)
    music = CountingMusic(MockMusic(usage=router))
    sfx = CountingSfx(ProceduralSfx(usage=router))
    return Spied(Providers(analysis=analysis, tts=tts, music=music, sfx=sfx), analysis, tts, music, sfx)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- environments
@dataclass
class Env:
    settings: Settings
    store: JobStore
    source: Path

    def new_job(self) -> Job:
        return create_job(self.store, self.settings, self.source)

    def paths(self, job: Job) -> JobPaths:
        return JobPaths(self.settings.data_dir, job.id)

    def run(self, job: Job, spy: Spied, **kw: object) -> Job:
        return run_job(job.id, self.settings, self.store, providers=spy.providers, **kw)  # type: ignore[arg-type]

    def total_tts(self, job: Job) -> int:
        keys: set[str] = set()
        for path in self.paths(job).chapters_dir.glob("*/tts.json"):
            keys.update(job_.clip_key for job_ in ChapterTts.model_validate(read_json(path)).jobs)
        return len(keys)


def make_env(root: Path, sample_book_path: Path, **overrides: object) -> Env:
    settings = Settings.from_env({}).with_overrides(
        data_dir=root / "data", worker_mode="inline", mock_ms_per_char=MS_PER_CHAR, warmup=False, **overrides,
    )
    return Env(settings=settings, store=JobStore(settings.data_dir / "bookreader.db"), source=sample_book_path)


@pytest.fixture(scope="module")
def warm(tmp_path_factory: pytest.TempPathFactory, sample_book_path: Path) -> tuple[Env, Job, Spied]:
    """A data dir whose cache is fully populated by one completed job."""
    env = make_env(tmp_path_factory.mktemp("warm"), sample_book_path)
    spy = spied()
    job = env.run(env.new_job(), spy)
    assert job.status == JobStatus.done, job.error
    return env, job, spy


@pytest.fixture
def fresh(tmp_path: Path, sample_book_path: Path) -> Env:
    return make_env(tmp_path, sample_book_path)


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("bookreader.retry.time.sleep", lambda seconds: None)


# --------------------------------------------------------------------------- tests
def test_second_job_on_same_book_is_all_cache_hits(warm: tuple[Env, Job, Spied]) -> None:
    env, first, first_spy = warm
    assert first_spy.counts["tts"] == env.total_tts(first) > 0
    assert first_spy.counts["analysis"] == 3 and first_spy.counts["music"] > 0 and first_spy.counts["sfx"] > 0

    spy = spied()
    second = env.run(env.new_job(), spy)
    assert second.status == JobStatus.done, second.error
    assert spy.counts == {"analysis": 0, "tts": 0, "music": 0, "sfx": 0}
    summary = env.store.usage_summary(second.id)
    assert summary.calls == 0 and summary.cache_hits > 0
    first_summary = env.store.usage_summary(first.id)
    assert first_summary.cache_hits == 0 and first_summary.calls == sum(first_spy.counts.values())
    assert summary.by_capability["tts"]["characters"] == first_summary.by_capability["tts"]["characters"]
    assert summary.by_capability["analysis"]["calls"] == 3.0
    first_manifest = json.loads(env.paths(first).manifest.read_text(encoding="utf-8"))
    second_manifest = json.loads(env.paths(second).manifest.read_text(encoding="utf-8"))
    for key in ("job_id", "created_at"):
        first_manifest.pop(key)
        second_manifest.pop(key)
    assert first_manifest == second_manifest


def test_transient_failure_then_retry_resumes_from_cache(fresh: Env) -> None:
    failing = spied(tts_fail_from=5)
    job = fresh.run(fresh.new_job(), failing)
    assert job.status == JobStatus.failed
    assert job.error is not None
    assert job.error.error_type == "provider_transient" and job.error.retryable is True
    assert job.error.stage == "render" and job.error.unit is not None and job.error.unit.startswith("ch01:c1p")
    assert "simulated outage" in job.error.message
    records = {r.stage: r for r in fresh.store.stage_records(job.id)}
    assert records[Stage.cast].state == StageState.done and records[Stage.render].state == StageState.failed
    assert records[Stage.finalize].state == StageState.pending
    assert failing.counts["tts"] >= 5 + 3                                   # the failing call was retried
    assert any(e.level == "warn" and "retrying" in e.message for e in fresh.store.events_after(job.id, 0, 1000))
    assert not fresh.paths(job).manifest.exists()
    cache = ClipCache(fresh.paths(job).cache_root)
    assert len(list((cache.root / "tts").glob("*.wav"))) == 4               # calls 1-4 succeeded and were cached

    fresh.store.requeue(job.id)
    healthy = spied()
    retried = fresh.run(job, healthy)
    assert retried.status == JobStatus.done, retried.error
    assert healthy.counts["analysis"] == 0
    assert healthy.counts["tts"] == fresh.total_tts(retried) - 4
    records = {r.stage: r for r in fresh.store.stage_records(job.id)}
    assert records[Stage.render].attempts == 2 and records[Stage.ingest].attempts == 1
    assert fresh.store.usage_summary(job.id).cache_hits >= 4


def test_cast_override_resynthesizes_only_that_character(warm: tuple[Env, Job, Spied]) -> None:
    env, _, _ = warm
    job = env.run(env.new_job(), spied())
    assert job.status == JobStatus.done
    paths = env.paths(job)
    before = Cast.model_validate_json(paths.cast.read_text(encoding="utf-8"))
    assert before.assignment_for("Tobias").voice.id != "mock-m-teen"
    tobias_segments = 0
    for chapter_script in paths.scripts_dir.glob("ch*.json"):
        script = ChapterScript.model_validate(read_json(chapter_script))
        tobias_segments += sum(1 for s in script.segments if s.speaker == "Tobias")
    assert tobias_segments == 4

    paths.cast_overrides.write_text(json.dumps({"tobias": "mock-m-teen"}), encoding="utf-8")
    env.store.requeue(job.id, from_stage=Stage.cast)
    spy = spied()
    job = env.run(job, spy)
    assert job.status == JobStatus.done, job.error
    assert spy.counts["analysis"] == 0
    assert spy.counts["tts"] == tobias_segments
    after = Cast.model_validate_json(paths.cast.read_text(encoding="utf-8"))
    tobias = after.assignment_for("Tobias")
    assert tobias.voice.id == "mock-m-teen" and tobias.source == "override"
    assert after.narrator.voice.id == before.narrator.voice.id
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    for chapter in manifest["chapters"]:
        for segment in chapter["segments"]:
            if segment["speaker"] == "Tobias":
                assert segment["voice_id"] == "mock-m-teen"
    records = {r.stage: r for r in env.store.stage_records(job.id)}
    assert records[Stage.analyze].attempts == 1 and records[Stage.render].attempts == 2


def test_cancel_mid_run_then_retry(fresh: Env) -> None:
    job = fresh.new_job()

    def cancel_on_third_call(n: int) -> None:
        if n == 3:
            fresh.store.request_cancel(job.id)

    cancelled = fresh.run(job, spied(tts_hook=cancel_on_third_call))
    assert cancelled.status == JobStatus.cancelled
    assert cancelled.error is not None and cancelled.error.error_type == "cancelled" and cancelled.error.retryable
    assert cancelled.error.stage == "render"
    records = {r.stage: r for r in fresh.store.stage_records(job.id)}
    assert records[Stage.render].state == StageState.pending and records[Stage.cast].state == StageState.done
    assert not fresh.paths(job).manifest.exists()

    fresh.store.requeue(job.id)
    done = fresh.run(job, spied())
    assert done.status == JobStatus.done, done.error
    assert done.cancel_requested is False and fresh.paths(job).manifest.is_file()


def test_stop_event_returns_job_to_queue(fresh: Env) -> None:
    job = fresh.new_job()
    stop = threading.Event()
    stop.set()
    returned = fresh.run(job, spied(), stop_event=stop)
    assert returned.status == JobStatus.queued and returned.error is None
    states = {r.stage: r.state for r in fresh.store.stage_records(job.id)}
    assert states[Stage.ingest] == StageState.done                          # ingest has no checkpoint; analyze does
    assert all(states[stage] == StageState.pending for stage in (Stage.analyze, Stage.cast, Stage.render, Stage.finalize))
    events = fresh.store.events_after(job.id, 0, 1000)
    assert any("shutting down" in e.message for e in events)


def test_cancel_requested_before_start(fresh: Env) -> None:
    job = fresh.new_job()
    fresh.store.request_cancel(job.id)
    result = fresh.run(job, spied())
    assert result.status == JobStatus.cancelled and result.error is not None and result.error.error_type == "cancelled"


def test_prune_removes_oldest_files(tmp_path: Path) -> None:
    cache = ClipCache(tmp_path / "cache")
    clip = AudioClip.silence(100, SAMPLE_RATE)                              # 4410 bytes of PCM + header
    now = time.time()
    for index, kind in enumerate(("tts", "tts", "music", "sfx", "tts")):
        path = cache.put(kind, f"k{index}", clip)
        stamp = now - 1000 + index * 10
        os.utime(path, (stamp, stamp))
    per_file = cache.path("tts", "k0").stat().st_size
    assert cache.size() == 5 * per_file
    assert cache.prune(0) == []                                             # 0 = unbounded
    removed = cache.prune(int(per_file * 2.5))
    assert [p.name for p in removed] == ["k0.wav", "k1.wav", "k2.wav"]
    assert cache.size() == 2 * per_file
    assert cache.has("sfx", "k3") and cache.has("tts", "k4") and not cache.has("tts", "k0")
    assert cache.get("tts", "k4") is not None and cache.get("tts", "k0") is None


def test_finalize_prunes_cache_to_tiny_cap(tmp_path: Path, sample_book_path: Path) -> None:
    env = make_env(tmp_path, sample_book_path, cache_max_mb=1)
    job = env.run(env.new_job(), spied())
    assert job.status == JobStatus.done, job.error
    cache = ClipCache(env.paths(job).cache_root)
    assert 0 < cache.size() <= 1024 * 1024
    assert env.paths(job).manifest.is_file()
    assert any("pruned" in e.message for e in env.store.events_after(job.id, 0, 1000))


# --------------------------------------------------------------------------- explicit re-runs (retry from a stage)
def _events(env: Env, job: Job) -> list[str]:
    return [e.message for e in env.store.events_after(job.id, 0, 10000)]


def test_retry_from_analyze_regenerates_scripts_and_cast(fresh: Env) -> None:
    """requeue(from_stage=analyze) must re-run analyze and cast even though their outputs exist."""
    job = fresh.run(fresh.new_job(), spied())
    assert job.status == JobStatus.done, job.error
    paths = fresh.paths(job)
    bible_before = paths.bible.read_bytes()
    os.utime(paths.bible, (1_000_000, 1_000_000))
    os.utime(paths.cast, (1_000_000, 1_000_000))
    first_events = len(_events(fresh, job))

    fresh.store.requeue(job.id, from_stage=Stage.analyze)
    paths.manifest.unlink()
    spy = spied()
    spy.analysis.inner.family = "anthropic"            # a provider switch: same text, different analysis cache keys
    spy.analysis.inner.cache_version = "99"
    retried = fresh.run(job, spy)
    assert retried.status == JobStatus.done, retried.error
    assert spy.counts["analysis"] == 3                 # really re-analyzed (cache keys differ for the new family)
    assert spy.counts["tts"] == 0                      # ... while unchanged voices/lines stay cache hits
    assert paths.bible.stat().st_mtime > 1_000_000 and paths.cast.stat().st_mtime > 1_000_000
    assert paths.bible.read_bytes() == bible_before    # the heuristic result is deterministic
    new_events = _events(fresh, job)[first_events:]
    assert not any("skipped" in m and ("analyze" in m or "cast" in m) for m in new_events), new_events
    records = {r.stage: r for r in fresh.store.stage_records(job.id)}
    assert records[Stage.ingest].attempts == 1
    assert all(records[s].attempts == 2 for s in (Stage.analyze, Stage.cast, Stage.render, Stage.finalize))


def test_retry_from_analyze_with_same_provider_is_free(warm: tuple[Env, Job, Spied]) -> None:
    env, _, _ = warm
    job = env.run(env.new_job(), spied())
    paths = env.paths(job)
    os.utime(paths.script(1), (1_000_000, 1_000_000))
    env.store.requeue(job.id, from_stage=Stage.analyze)
    spy = spied()
    retried = env.run(job, spy)
    assert retried.status == JobStatus.done, retried.error
    assert spy.counts == {"analysis": 0, "tts": 0, "music": 0, "sfx": 0}
    assert paths.script(1).stat().st_mtime > 1_000_000  # rewritten from cached chunk analyses
    assert env.store.usage_summary(job.id).by_capability["analysis"]["calls"] >= 3.0


def test_retry_from_ingest_rewrites_book(fresh: Env) -> None:
    job = fresh.run(fresh.new_job(), spied())
    assert job.status == JobStatus.done, job.error
    paths = fresh.paths(job)
    os.utime(paths.book, (1_000_000, 1_000_000))
    fresh.store.requeue(job.id, from_stage=Stage.ingest)
    retried = fresh.run(job, spied())
    assert retried.status == JobStatus.done, retried.error
    assert paths.book.stat().st_mtime > 1_000_000
    assert not any("ingest: book.json exists" in m for m in _events(fresh, job))


# --------------------------------------------------------------------------- tts provider family switch
class _OtherTTS:
    """A second TTS family with its own catalog that rejects voice ids it never advertised."""

    family = "other"
    cache_version = "1"
    max_chars = 5000

    def __init__(self) -> None:
        self.inner = MockTTS(ms_per_char=MS_PER_CHAR)
        self.voice_ids: list[str] = []

    def list_voices(self) -> list[VoiceInfo]:
        voices = [v.model_copy(update={"id": "other-" + v.id, "family": "other"}) for v in self.inner.list_voices()]
        return voices

    def synthesize(self, req: TTSRequest) -> AudioClip:
        if not req.voice_id.startswith("other-"):
            raise ProviderPermanentError(f"unknown voice id {req.voice_id!r}")
        self.voice_ids.append(req.voice_id)
        return self.inner.synthesize(req.model_copy(update={"voice_id": req.voice_id[len("other-"):]}))


@pytest.mark.parametrize("from_stage", [None, Stage.cast, Stage.render])
def test_switching_tts_family_recasts_instead_of_sending_stale_voice_ids(fresh: Env, from_stage: Stage | None) -> None:
    if from_stage is None:                                  # plain retry of a job that failed mid-render
        job = fresh.run(fresh.new_job(), spied(tts_fail_from=5))
        assert job.status == JobStatus.failed
    else:
        job = fresh.run(fresh.new_job(), spied())
        assert job.status == JobStatus.done, job.error
    paths = fresh.paths(job)
    assert Cast.model_validate(read_json(paths.cast)).family == "mock"

    other = _OtherTTS()
    spy = spied()
    providers = Providers(analysis=spy.analysis, tts=other, music=spy.music, sfx=spy.sfx)  # type: ignore[arg-type]
    fresh.store.requeue(job.id, from_stage=from_stage)
    paths.manifest.unlink(missing_ok=True)
    retried = run_job(job.id, fresh.settings, fresh.store, providers=providers)
    assert retried.status == JobStatus.done, retried.error
    cast = Cast.model_validate(read_json(paths.cast))
    assert cast.family == "other" and cast.narrator.voice.id.startswith("other-")
    assert other.voice_ids and all(v.startswith("other-") for v in other.voice_ids)
    assert any("re-casting" in m and "family 'mock'" in m for m in _events(fresh, job))
    voices = read_json(paths.voices)
    assert all(v["family"] == "other" for v in voices)                  # PUT /cast validates against the new catalog


def test_unchanged_family_retry_keeps_the_cast(fresh: Env) -> None:
    job = fresh.run(fresh.new_job(), spied())
    assert job.status == JobStatus.done, job.error
    paths = fresh.paths(job)
    os.utime(paths.cast, (1_000_000, 1_000_000))
    fresh.store.requeue(job.id, from_stage=Stage.render)
    retried = fresh.run(job, spied())
    assert retried.status == JobStatus.done, retried.error
    assert paths.cast.stat().st_mtime == 1_000_000                          # not re-cast
    assert not any("re-casting" in m for m in _events(fresh, job))


# --------------------------------------------------------------------------- concurrent cache writers
def test_concurrent_writers_of_one_cache_key_never_corrupt_it(tmp_path: Path) -> None:
    cache = ClipCache(tmp_path / "cache")
    clip = AudioClip.silence(2000, SAMPLE_RATE)
    errors: list[BaseException] = []
    short_reads: list[int] = []

    def write() -> None:
        for _ in range(60):
            try:
                cache.put("tts", "samekey", clip)
                cache.put_json("analysis", "samekey", {"n": 1})
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

    def read() -> None:
        for _ in range(120):
            got = cache.get("tts", "samekey")
            if got is not None and len(got.samples) != len(clip.samples):
                short_reads.append(len(got.samples))
            doc = cache.get_json("analysis", "samekey")
            if doc is not None and doc != {"n": 1}:
                short_reads.append(-1)

    threads = [threading.Thread(target=write) for _ in range(4)] + [threading.Thread(target=read) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [] and short_reads == []
    final = cache.get("tts", "samekey")
    assert final is not None and len(final.samples) == len(clip.samples)
    assert cache.get_json("analysis", "samekey") == {"n": 1}
    assert not [p for p in (cache.root / "tts").iterdir() if p.name.startswith(".")]   # no temp files left behind


# --------------------------------------------------------------------------- prune vs a resumed job
def test_prune_spares_entries_used_since_the_floor(tmp_path: Path) -> None:
    cache = ClipCache(tmp_path / "cache")
    clip = AudioClip.silence(100, SAMPLE_RATE)
    now = time.time()
    for index in range(4):
        path = cache.put("tts", f"k{index}", clip)
        os.utime(path, (now - 1000 + index * 10, now - 1000 + index * 10))
    per_file = cache.path("tts", "k0").stat().st_size
    assert cache.has("tts", "k1")                                            # a hit counts as a use ...
    assert cache.path("tts", "k1").stat().st_mtime > now - 1                 # ... and bumps the mtime
    removed = cache.prune(per_file, keep_newer_than=now - 985)
    assert [p.name for p in removed] == ["k0.wav"]                           # k2/k3 protected by the floor, k1 by its use
    assert cache.size() == 3 * per_file


def test_evicted_clip_is_regenerated_at_mix(fresh: Env) -> None:
    """Another job's finalize prunes the tts clips of a resumed chapter between its tts and mix substages."""
    failing = spied(tts_fail_from=200)
    job = fresh.new_job()
    paths = fresh.paths(job)
    cache = ClipCache(paths.cache_root)
    evicted: list[str] = []
    lock = threading.Lock()

    class PruningSfx(CountingSfx):
        def generate(self, req: object) -> object:
            with lock:
                if not evicted:                                              # stand-in for a concurrent finalize
                    for path in list((cache.root / "tts").glob("*.wav")):
                        evicted.append(path.stem)
                        path.unlink()
            return super().generate(req)

    sfx = PruningSfx(ProceduralSfx())
    spy = Spied(Providers(analysis=failing.analysis, tts=failing.tts, music=failing.music, sfx=sfx), failing.analysis, failing.tts, failing.music, sfx)  # type: ignore[arg-type]
    done = fresh.run(job, spy)
    assert done.status == JobStatus.done, done.error
    assert evicted and all(cache.has("tts", key) for key in evicted)          # regenerated, back in the cache
    assert any("evicted from the cache; regenerating" in m for m in _events(fresh, job))
    assert spy.counts["tts"] == fresh.total_tts(done) + len(evicted)


def test_finalize_prune_spares_a_concurrently_running_job(tmp_path: Path, sample_book_path: Path) -> None:
    env = make_env(tmp_path, sample_book_path, cache_max_mb=1)
    other = env.new_job()                                                    # pretend it is mid-render on another worker
    env.store.set_status(other.id, JobStatus.running)
    env.store.update_job(other.id, started_at=now_iso())
    job = env.run(env.new_job(), spied())
    assert job.status == JobStatus.done, job.error
    cache = ClipCache(env.paths(job).cache_root)
    assert cache.size() > 1024 * 1024                                        # cap exceeded rather than evicting in-use clips
    assert not any("pruned" in e.message for e in env.store.events_after(job.id, 0, 1000))

    env.store.set_status(other.id, JobStatus.done)
    env.store.requeue(job.id, from_stage=Stage.finalize)
    assert env.run(job, spied()).status == JobStatus.done
    assert cache.size() <= 1024 * 1024
