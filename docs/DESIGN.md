# bookreader design

bookreader turns a book into an audio drama: distinct voices for the narrator and each
character, a music bed that follows the mood of each scene, and sound effects anchored to the
events the text describes, mixed per chapter with separate stems and a timing manifest.

It is one FastAPI process with SQLite job state, an in-process worker, a per-job workspace on
local disk, and a content-addressed clip cache. Provider families are selected by configuration.

## Pipeline

Every job runs five stages in order; each stage is skipped when its outputs already exist, so a
failed or cancelled job resumes where it stopped and a retried job never re-pays for work that
is already cached.

| Stage | Module | Input | Output |
|---|---|---|---|
| ingest | `bookreader/ingest/` | txt, md, epub, pdf | `book.json`: chapters, paragraphs, spans (narration / quote runs split deterministically; text is never rewritten) and `estimate.json` |
| analyze | `bookreader/analysis/`, `providers/<family>/analysis.py` | chunks of whole paragraphs, in book order, plus the running cast bible | `scripts/chNN.json`: one segment per span with speaker, emotion and delivery; music regions; SFX cues anchored to verbatim substrings; `bible.json` |
| cast | `bookreader/casting.py` | bible, the TTS family's voice catalog, overrides | `cast.json`: one voice per speaking character, scored on gender, age and voice notes, with deterministic tie-breaks |
| render | `bookreader/pipeline/stages.py`, `bookreader/audio/` | scripts and cast | per chapter: `tts.json`, `timeline.json`, `mix.wav`, `voice.wav`, `music.wav`, `sfx.wav`, `manifest.json` (and `mix.mp3` when ffmpeg is installed) |
| finalize | `bookreader/manifest.py`, `bookreader/usage.py` | chapter manifests, usage ledger | `manifest.json`, `usage.json`, cache pruning |

Render works chapter by chapter (tts, timeline, music, sfx, mix), so chapter 1 is playable while
later chapters are still synthesizing. The timeline is a pure function: it lays segments out with
pacing pauses, places SFX at the anchor word (loud impacts that open a sentence get a short gap
before it), opens looped ambient beds until the scene ends, and turns music regions into requests
with clip keys, all before any paid audio call is made.

## Analysis

The text analyzer only labels spans; it never produces text. Chunks carry the paragraphs, two
paragraphs of read-only context, the mood in force, the last speakers before the chunk and the
scene breaks inside it. A persistent cast bible (canonical name, aliases, gender, age,
description, voice notes) threads identity across chunks and chapters; per-chunk results are
cached under a key that includes the bible fingerprint, so re-running a book is free until the
first chunk whose inputs changed.

Two analyzers implement the same contract:

- `anthropic`: Claude (`claude-opus-5` by default) via `messages.stream` with a JSON-schema
  structured output, a cached system prompt, refusal fallback to the heuristic analyzer, splitting
  on truncated output and one repair pass on invalid JSON.
- `mock`: a rule-based heuristic (speech tags, pronouns, descriptors, vocatives, alternation,
  keyword moods, an SFX regex table). It is the offline test and demo path and the refusal
  fallback, not a production analyzer.

## Provider families

Four capabilities (analysis, tts, music, sfx) times four families, chosen per capability by
`BOOKREADER_<CAPABILITY>_PROVIDER`:

| Family | analysis | tts | music | sfx | Needs |
|---|---|---|---|---|---|
| mock | heuristic | procedural voices | mood beds | procedural recipes | nothing |
| anthropic | Claude | | | | `bookreader[anthropic]`, `ANTHROPIC_API_KEY` |
| elevenlabs | | Text to Speech | Music | Sound Generation | `bookreader[elevenlabs]`, `ELEVENLABS_API_KEY` |
| local | | Piper or Kokoro | MusicGen | AudioGen or procedural | `bookreader[local]` (+ `local-kokoro`), model files |

A family is a package whose `__init__` is a constants-only manifest (`FAMILY`, `PROVIDERS`,
`EXTRA`, `REQUIRED_SECRETS`). The registry in `bookreader/providers/base.py` imports provider
modules lazily and third-party SDKs only inside `check()` and `from_settings()`, so an unused
family costs nothing and a missing one fails at startup with the exact pip extra or variable to
set. Every provider carries a `cache_version` that is part of every cache key. Network adapters
own their own retry and per-family concurrency limit; the pipeline never wraps them again.

## Audio

Canonical format is 22050 Hz mono 16-bit WAV, written with the standard library. Everything else
is numpy: windowed-sinc resampling, silence trimming, RMS normalization that preserves delivery
dynamics, equal-power loops and crossfades, lookahead ducking of the music bed under speech with a
small lift in long gaps, and a soft limiter on the sum. Stems are the post-gain, post-duck tracks,
so `voice + music + sfx == mix` whenever the limiter does not engage. MP3 export runs only when
`ffmpeg` is on the path.

## Jobs, storage and operability

`data/bookreader.db` holds `jobs`, `job_stages`, `job_events` and `usage_events`. Each job has a
directory under `data/jobs/<id>/`; shared clips live under `data/cache/<kind>/<key>.wav`. Errors
are typed (`input`, `config`, `provider_transient`, `provider_permanent`, `cancelled`,
`internal`) with the failing unit named, retries can start from any stage, cancellation is
checked between chunks, submissions and chapters, orphaned jobs are re-queued on boot, and the
usage ledger records every provider call and every cache hit with an estimated cost.

The worker is an in-process thread behind a three-method queue interface (`submit`, `start`,
`stop`); an external queue only needs to call `run_job(job_id, settings, store)` in its consumer.

## Interfaces

- REST API under `/api` (upload, status, events, log, cast and re-cast, manifest, usage,
  artifacts with HTTP Range support, retry, cancel, delete, health).
- Single-page web UI at `/`: upload, live stage progress, cast table with a voice picker,
  per-chapter player with a synchronized transcript.
- CLI: `bookreader run | estimate | providers | serve | status`.

See `README.md` for every environment variable, curl examples and Docker usage.

## Known limitations

- The heuristic analyzer is tuned on prose conventions, not on any particular book; dense
  multi-speaker scenes with unusual tag verbs can still misattribute (it warns rather than fails).
- One quote style is detected per book; mixed conventions collapse some spans into narration.
- Early chapters keep the speaker labelled at the time; a character named later is merged in the
  bible but not retroactively re-attributed.
- Mock and MusicGen beds are short and looped; repetition is audible on very long scenes.
- The ElevenLabs and local adapters are tested against fakes and stubbed SDKs; run a live smoke
  test with real keys or model files before relying on them.
- Whole chapters are mixed in memory (about 16 MB per minute of audio across the three tracks);
  `BOOKREADER_MAX_CHAPTER_MINUTES` bounds it.
