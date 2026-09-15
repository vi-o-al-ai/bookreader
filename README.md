# bookreader

Upload a book; get an audio drama with character voices, music and sound effects.

bookreader is one FastAPI process (Python 3.11) with a SQLite job store, an in-process worker
thread and a per-job workspace on disk. A book goes through five resumable stages:

| stage | what happens |
|---|---|
| `ingest` | txt / md / epub / pdf -> chapters -> paragraphs -> spans (narration / quote runs, split deterministically; the text is never rewritten) |
| `analyze` | an analyzer labels every span with speaker / emotion / delivery and emits sound-effect and music cues; a cast bible threads character identity across chunks |
| `cast` | one deterministic pass assigns a voice to the narrator and every speaking character (overridable) |
| `render` | per chapter, in order: tts -> timeline -> music -> sfx -> mix, so chapter 1 is listenable while chapter 30 synthesizes |
| `finalize` | `manifest.json` (the public timing contract), `usage.json`, cache pruning |

Four capabilities (analysis, tts, music, sfx) are served by provider *families* (mock, anthropic,
elevenlabs, local) chosen purely by environment variables. The mock family is deterministic,
needs no keys and drives the offline test suite, the CLI demo and the web UI.

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'

# render the sample book with the mock family (no keys, a few seconds); prints the manifest path
bookreader run tests/fixtures/sample_book.txt --out out

# start the API + web UI on http://127.0.0.1:8000
bookreader serve
```

Open <http://127.0.0.1:8000>, drop a book on the upload card and watch the stage stepper; each
chapter gets a player with a synchronized transcript as soon as it is mixed. To use real
providers, install the extras and export the keys (see [Providers](#providers)):

```bash
pip install -e '.[anthropic,elevenlabs]'
export ANTHROPIC_API_KEY=... ELEVENLABS_API_KEY=...
export BOOKREADER_ANALYSIS_PROVIDER=anthropic BOOKREADER_TTS_PROVIDER=elevenlabs \
       BOOKREADER_MUSIC_PROVIDER=elevenlabs BOOKREADER_SFX_PROVIDER=elevenlabs
bookreader providers          # every check must say ok
bookreader estimate book.epub # counts + cost estimate before any paid call
bookreader serve
```

## CLI

```
bookreader run FILE [--out DIR] [--title T] [--chapters 1-2] [--no-music] [--no-sfx]
                    [--analysis F] [--tts F] [--music F] [--sfx F] [--cast overrides.json]
                    [--from-stage ingest|analyze|cast|render|finalize]
bookreader estimate FILE [--chapters 1-2] [--analysis F] [--tts F] [--music F] [--sfx F]
bookreader providers [--analysis F] [--tts F] [--music F] [--sfx F]
bookreader serve [--host H] [--port P] [--reload]
bookreader status JOB_ID [--data-dir DIR]
```

* `run` reads the environment, layers the flags on top (`--out` -> `BOOKREADER_DATA_DIR`,
  `--tts` -> `BOOKREADER_TTS_PROVIDER`, ...), validates the providers, creates the job row in
  `DIR/bookreader.db`, runs it in-process with a console progress printer and prints the path of
  `manifest.json`. Exit code 0 on success, 1 when the job failed (the typed error is printed:
  stage, type, unit, message), 2 for a missing / unsupported input or a provider
  configuration problem. `--from-stage` re-runs the most recent job created from the same file
  (matched by content hash) from that stage onward instead of creating a new one; it keeps that
  job's title, chapters and music/sfx options (passing `--title`, `--chapters`, `--no-music` or
  `--no-sfx` with it is a usage error), while `--cast` is applied when the cast stage re-runs
  (`--from-stage cast` or earlier).
* `estimate` ingests and chunks the book without touching any provider and prints chapters,
  paragraphs, words, chars, quote spans, chunks, TTS chars, estimated analysis tokens and the
  cost per capability from the price table.
* `providers` prints the startup checks (family, class, ok / error, warnings) and exits 2 when
  any selected provider is unusable, with the exact remedy (pip extra, env var, path).
* `serve` validates the configuration (exit 2 with the message on failure) and runs uvicorn on
  the app factory `bookreader.api.app:create_app`.
* `status` prints a job's status, stage table, error and usage from a data directory.

## HTTP API

Everything lives under `/api`; the web UI is served at `/`. Errors are `{"detail": ...}`.

```bash
BASE=http://127.0.0.1:8000

# upload (202): file is required; title and options are optional form fields
curl -F file=@tests/fixtures/sample_book.txt -F title="The Lighthouse at Gull Point" \
     -F 'options={"chapters":[1,2],"music":true,"sfx":true,"cast_overrides":{}}' $BASE/api/jobs
# -> {"job_id":"9e394ee61b5a","status":"queued","status_url":"/api/jobs/9e394ee61b5a"}
# 415 unsupported extension, 413 larger than BOOKREADER_MAX_UPLOAD_MB, 400 bad options JSON

JOB=9e394ee61b5a
curl $BASE/api/jobs                              # newest first: id, title, status, stage, overall_pct, timestamps
curl $BASE/api/jobs/$JOB                         # full status: stages[], error, options, estimate, usage, chapters[], providers
curl "$BASE/api/jobs/$JOB/events?after=0&limit=200"   # {events:[{id,ts,level,stage,message}], last_id}; poll with after=last_id
curl "$BASE/api/jobs/$JOB/log?lines=100"         # text/plain tail of job.log
curl $BASE/api/jobs/$JOB/cast                    # {cast, voices} once the cast stage is done (404 before)
curl -X PUT -H 'Content-Type: application/json' \
     -d '{"overrides":{"Tobias":"mock-m-teen","narrator":"mock-m-adult-deep"}}' $BASE/api/jobs/$JOB/cast
# -> 202: cast/render/finalize re-run; only the changed characters are re-synthesized
# 400 unknown character or voice id, 409 while the job is running
curl $BASE/api/jobs/$JOB/manifest                # JobManifest; partial (warnings:["partial"]) while running, 404 before any chapter
curl $BASE/api/jobs/$JOB/usage                   # calls, cache_hits, cost_usd, by_capability, events_count
curl $BASE/api/jobs/$JOB/artifacts               # {files:[{path,bytes,url}]} - everything in the job dir except the source
curl -o ch01.wav $BASE/api/jobs/$JOB/artifacts/chapters/01/mix.wav
curl -H 'Range: bytes=0-1023' -o head.bin $BASE/api/jobs/$JOB/artifacts/chapters/01/mix.wav   # 206 + Content-Range
curl -X POST $BASE/api/jobs/$JOB/retry           # failed/cancelled -> queued (409 if running or queued)
curl -X POST -H 'Content-Type: application/json' -d '{"from_stage":"render"}' $BASE/api/jobs/$JOB/retry   # any finished job
curl -X POST $BASE/api/jobs/$JOB/cancel          # queued -> cancelled now; running -> stops at the next checkpoint; 409 if terminal
curl -X DELETE $BASE/api/jobs/$JOB               # 204: rows + job directory removed (shared cache untouched); 409 if running
curl $BASE/api/health                            # {ok, version, providers[], ffmpeg, queue:{backend,depth,running}, data_dir_writable}
curl $BASE/                                      # the web UI
```

Artifact paths are resolved and must stay inside the job directory (`400` otherwise); the
media type follows the extension (`audio/wav`, `audio/mpeg`, `application/json`, `text/plain`).

### Web UI

`bookreader/web/index.html` is one file of vanilla HTML/CSS/JS with no build step, served at
`/` and using only relative URLs (it works behind a path prefix). Header with the provider
summary from `/api/health`; upload card; jobs list (refreshed every 5 s); job panel with the
stage stepper, progress bar, estimate, usage, error box, retry / cancel / delete buttons and an
incrementally appended log; a cast table with a voice dropdown per character and an
"Apply re-cast" button; a chapters card with a seekable player per rendered chapter, stem
links and a transcript that highlights the segment being played (click a segment to seek).

## Configuration

All settings come from `BOOKREADER_*` environment variables (`bookreader/settings.py`).
Booleans accept `1/0/true/false/yes/no/on/off`; a malformed value fails at startup with a
message naming the variable.

| variable | default | meaning |
|---|---|---|
| `BOOKREADER_DATA_DIR` | `./data` | SQLite db, shared cache and job workspaces |
| `BOOKREADER_ANALYSIS_PROVIDER` | `mock` | `mock` \| `anthropic` |
| `BOOKREADER_TTS_PROVIDER` | `mock` | `mock` \| `elevenlabs` \| `local` |
| `BOOKREADER_MUSIC_PROVIDER` | `mock` | `mock` \| `elevenlabs` \| `local` |
| `BOOKREADER_SFX_PROVIDER` | `mock` | `mock` \| `elevenlabs` \| `local` |
| `BOOKREADER_WORKER_MODE` | `thread` | `thread` (daemon worker threads) \| `inline` (jobs run inside the request; tests, CLI) |
| `BOOKREADER_WORKERS` | `1` | worker threads |
| `BOOKREADER_WARMUP` | `1` | load local weights / voice catalogs at startup so nothing fails at job time |
| `BOOKREADER_CONCURRENCY` | `4` | thread pool width for TTS/SFX and the per-family semaphore |
| `BOOKREADER_ANALYSIS_CHUNK_CHARS` | `6000` | analyzer chunk size (whole paragraphs, never across chapters) |
| `BOOKREADER_ANTHROPIC_MODEL` | `claude-opus-5` | analysis model |
| `BOOKREADER_ANTHROPIC_EFFORT` | `medium` | `low` \| `medium` \| `high` \| `xhigh` \| `max` |
| `BOOKREADER_ANTHROPIC_MAX_TOKENS` | `32000` | output budget per analyzer call |
| `BOOKREADER_ELEVENLABS_TTS_MODEL` | `eleven_multilingual_v2` | TTS model id (part of every TTS cache key) |
| `BOOKREADER_ELEVENLABS_SFX_PROMPT_INFLUENCE` | `0.5` | 0..1 |
| `BOOKREADER_MAX_VOICES` | `300` | voice catalog bound |
| `BOOKREADER_LOCAL_TTS_ENGINE` | `piper` | `piper` \| `kokoro` |
| `BOOKREADER_PIPER_VOICES_DIR` | `./voices` | directory of `*.onnx` + `*.onnx.json` piper voices |
| `BOOKREADER_KOKORO_LANG` | `a` | kokoro language code |
| `BOOKREADER_MUSICGEN_MODEL` | `facebook/musicgen-small` | local music model |
| `BOOKREADER_AUDIOGEN_MODEL` | `facebook/audiogen-medium` | local sfx model |
| `BOOKREADER_LOCAL_SFX_FALLBACK` | `1` | `audiocraft` is in no pip extra (`pip install audiocraft` by hand); without it local sfx falls back to the mock family's procedural synthesis (mock-quality effects, startup warning) instead of failing. `0` makes the missing package a startup error |
| `BOOKREADER_MOCK_MS_PER_CHAR` | `45` | mock speech pacing (tests use 4) |
| `BOOKREADER_MP3` | `auto` | `auto` exports mp3 when ffmpeg is on PATH; `off` never |
| `BOOKREADER_MUSIC_GAIN_DB` | `-14.0` | music bed level |
| `BOOKREADER_SFX_GAIN_DB` | `-8.0` | sound-effect level |
| `BOOKREADER_MAX_UPLOAD_MB` | `50` | upload guard (413) |
| `BOOKREADER_MAX_CHAPTER_MINUTES` | `90` | timeline guard per chapter |
| `BOOKREADER_MAX_SEGMENTS` | `20000` | ingest guard (spans per job) |
| `BOOKREADER_CACHE_MAX_MB` | `5000` | LRU cache cap pruned at finalize (`0` = unbounded) |
| `BOOKREADER_PRICES` | `{}` | JSON merged over the default price table (see [Costs](#costs)) |
| `BOOKREADER_LOG_LEVEL` | `INFO` | root log level |
| `ANTHROPIC_API_KEY`, `ELEVENLABS_API_KEY` | | secrets; every variable ending in `_API_KEY` or `_TOKEN` is collected, never logged, never stored in job snapshots |

The sample rate is fixed at 22050 Hz mono int16 WAV. Provider families, paths, concurrency and
secrets always come from the running process; pacing / gain / mp3 / guard knobs are snapshotted
per job at creation so a retry renders consistently.

## Providers

| family | capabilities | pip extra | required env | notes |
|---|---|---|---|---|
| `mock` | analysis, tts, music, sfx | (none) | (none) | deterministic sha256-seeded procedural audio and a rule-based analyzer; 16 voices |
| `anthropic` | analysis | `anthropic` | `ANTHROPIC_API_KEY` | Claude labels spans with a JSON schema, cached system prompt, repair pass and heuristic fallback |
| `elevenlabs` | tts, music, sfx | `elevenlabs` | `ELEVENLABS_API_KEY` | `text_to_speech.convert` (pcm_22050, previous/next text, seed), `music.compose` (music_v2), `text_to_sound_effects.convert` |
| `local` | tts, music, sfx | `local` (+ `local-kokoro`) | (none) | Piper (`BOOKREADER_PIPER_VOICES_DIR`) or Kokoro (needs `espeak-ng`) TTS, MusicGen music, AudioGen sfx only after a manual `pip install audiocraft` (no extra installs it; otherwise the procedural mock synth runs under the `local` name, see `BOOKREADER_LOCAL_SFX_FALLBACK`) |
| | pdf ingest | `pdf` | | `pypdf`, imported lazily |

Families mix freely, e.g. anthropic analysis + elevenlabs speech + mock music. Startup
(`bookreader serve`, `bookreader run`, the API lifespan) validates every selected provider and
aggregates all problems into one message with the exact fix; with `BOOKREADER_WARMUP=1` local
weights and voice catalogs are loaded before the first job is accepted. Every provider carries a
`cache_version` (model id, prompt version, DSP recipe) that is part of every cache key, so
changing a model never serves stale audio.

## Artifact layout

```
data/
  bookreader.db                      jobs, job_stages, job_events, usage_events (SQLite, WAL)
  cache/                             shared, content-addressed
    analysis/<key>.json              one analyzer result per chunk
    tts/<key>.wav  music/<key>.wav  sfx/<key>.wav
    voices/<family>-<cache_version>.json   voice catalogs (24 h TTL)
  jobs/<job_id>/
    source.<ext>                     the upload (not listed by /artifacts)
    book.json  estimate.json         ingest output and pre-flight numbers
    bible.json                       cast bible (characters, aliases, gender/age, line counts)
    scripts/chNN.json                assembled chapter scripts (segments + cues)
    voices.json  cast.json           catalog and voice assignments
    cast_overrides.json              written by PUT /cast or `run --from-stage cast --cast F` (a first run's --cast stays in the job options)
    chapters/NN/
      tts.json                       planned TTS requests, clip keys and measured durations
      timeline.json  render.key      placements + the content key the outputs were rendered from
      mix.wav  voice.wav  music.wav  sfx.wav  [mix.mp3]
      manifest.json                  ChapterManifest: segments with start_ms/end_ms, cues, files
    manifest.json                    JobManifest (written last; its presence == job complete)
    usage.json  job.log
```

`manifest.json` is the public timing contract: every segment carries speaker, voice id,
emotion, delivery, text and `start_ms`/`end_ms`; every cue carries its clip key and window.
Stems (`voice`, `music`, `sfx`) sum to the pre-limiter mix.

## Resume, caching and re-casting

* **Stages are idempotent.** A stage skips work whose outputs exist (`book.json`, every
  script, `cast.json`, a chapter whose `render.key` matches). Within a stage every provider
  call goes through the content-addressed cache, so a job that failed on TTS segment 217/400
  resumes with 217 cache hits (`POST /retry`, or `bookreader run` again into the same data dir).
* **Cache keys.** `clip_key(kind, family, cache_version, request)` hashes the full request:
  for TTS that is the text, voice id, per-segment voice settings, emotion, delivery, seed **and
  the neighbouring `previous_text` / `next_text`** (the same voice's adjacent lines, truncated
  to 200 chars, which providers use for prosodic continuity). Editing one line therefore
  re-synthesizes up to three clips: the line itself and its two neighbours whose context
  changed. Analysis keys include the chunk, the prompt/schema version and the pre-chunk cast
  bible fingerprint; music and sfx keys include prompt, mood, energy and the requested duration.
* **Re-casting.** `PUT /cast` writes `cast_overrides.json`, resets cast / render / finalize,
  removes the chapter mixes and manifests (keeping `tts.json`) and requeues. Because voice ids
  are part of the TTS keys, only the re-cast characters' lines are synthesized again; the
  narrator and everyone else are cache hits, visible as `cache_hits` in `/usage`.
* **Retry from a stage.** `POST /retry {"from_stage": "analyze"}` resets that stage and every
  later one and forces them to regenerate their outputs even though they exist on disk;
  identical inputs still hit the cache, so re-running analysis with the same model costs
  nothing, while a changed analysis provider/model really re-analyzes. Switching the TTS
  family between runs is detected at cast/render time and re-casts against the new catalog.
* **Cancellation** is checked between chunks, between TTS/SFX submissions, between chapters
  and before finalize. In-flight provider calls complete and still populate the cache.
* **Crash / shutdown recovery.** On SIGTERM the worker stops with a 30 s grace period and the
  running job returns to `queued`; on boot the app requeues jobs left `running` or `queued`.
* **Pruning.** Finalize evicts least-recently-used cache files until the cache fits
  `BOOKREADER_CACHE_MAX_MB`. Cache hits bump mtimes, so hot clips survive, and entries used by
  a job still running on another worker are spared (the cap may be exceeded until it finishes);
  should a clip vanish anyway, the mix regenerates it from its recorded request.

## Costs

The usage ledger records one row per provider call (tokens, characters, audio seconds) priced
from the table below, plus one zero-cost row per cache hit so the hit ratio is visible.
`GET /api/jobs/{id}` shows `calls`, `cache_hits` and `cost_usd`; `/usage` breaks units down per
capability; `bookreader estimate` prices the book before any call using the same table.

Default prices (USD per unit; keys are `<family>:<unit_type>`):

```
anthropic:input_tokens 5e-6   anthropic:output_tokens 25e-6
anthropic:cache_read_input_tokens 0.5e-6   anthropic:cache_creation_input_tokens 6.25e-6
elevenlabs:characters 0   elevenlabs:audio_seconds 0
```

Override or extend with JSON, e.g. `BOOKREADER_PRICES='{"elevenlabs:characters": 0.00003}'`.
Families without prices cost 0.

## Adding a provider family

A family is one package whose `__init__` is a constants-only manifest:

```python
# mycompany_voices/__init__.py
FAMILY = "mycompany"
PROVIDERS = {"tts": "mycompany_voices.tts:MyTTS"}       # capability -> "module:Class"
EXTRA = None                                            # pip extra that installs the SDK, or None
REQUIRED_SECRETS = ("MYCOMPANY_API_KEY",)
```

Each class implements the protocol of its capability (`bookreader.providers.base`:
`TextAnalyzer`, `VoiceSynthesizer`, `MusicGenerator`, `SfxGenerator`) plus `family`,
`cache_version`, `check(settings) -> list[str]` (raise `ProviderConfigError` with the remedy;
import the SDK inside a `try`, never download), `from_settings(settings, usage)` (the only
place real clients are built; `__init__` takes injected clients for tests) and `warmup()`.
Call `usage.record(...)` after every external call, wrap network calls in
`bookreader.retry.with_retry` and `FamilyLimiter`, and return `AudioClip` at any sample rate.
Select it with `BOOKREADER_TTS_PROVIDER=mycompany_voices` (a dotted module path is imported as
given; built-in names resolve under `bookreader.providers.`).

## Docker

```bash
docker build -t bookreader .                                                    # mock only
docker build -t bookreader --build-arg EXTRAS=anthropic,elevenlabs,pdf --build-arg WITH_FFMPEG=1 .
docker run --rm -p 8000:8000 -v bookreader-data:/data \
    -e ANTHROPIC_API_KEY -e ELEVENLABS_API_KEY \
    -e BOOKREADER_ANALYSIS_PROVIDER=anthropic -e BOOKREADER_TTS_PROVIDER=elevenlabs \
    -e BOOKREADER_MUSIC_PROVIDER=elevenlabs -e BOOKREADER_SFX_PROVIDER=elevenlabs bookreader
```

The image runs as the non-root user `app`, stores everything under the `/data` volume
(`BOOKREADER_DATA_DIR=/data`), exposes port 8000 and has a `HEALTHCHECK` against
`/api/health`. Build args: `EXTRAS` (pip extras), `WITH_FFMPEG=1` (mp3 export),
`WITH_ESPEAK=1` (Kokoro). The `local` family needs torch; use a torch-capable base image for
GPU inference and mount the piper voices directory
(`-v ./voices:/voices -e BOOKREADER_PIPER_VOICES_DIR=/voices`).
AudioGen sound effects need `audiocraft`, which no extra installs (it pins its own torch); add
`RUN pip install audiocraft` to a derived image, otherwise `EXTRAS=local` ships procedural
(mock-quality) sound effects under the `local` name. A `.dockerignore` keeps `.venv`, `.git`,
`data/` and caches out of the build context.

## Development

```bash
pip install -e '.[dev]'
pytest                      # offline, mock family only, < 60 s; per-test timeout 40 s
```

Tests never need keys: third-party SDKs are replaced by fakes injected through constructors or
`sys.modules` stubs, and the mock family is deterministic across processes.
