"""bookreader.audio - pure-numpy audio engine: PCM I/O, DSP primitives, timeline, mixer, export.

Modules:
* ``pcm``      - WAV read/write (stdlib wave), int16/float conversion, resampling, canonicalization.
* ``dsp``      - silence trim, RMS normalization, fades, crossfades, looping, envelopes, soft limiter.
* ``timeline`` - the pure pacing/placement function turning a script + TTS durations into a ChapterTimeline.
* ``mixer``    - renders a ChapterTimeline into voice/music/sfx stems and a mastered mix, plus the manifest.
* ``export``   - optional MP3 export through ffmpeg when it is on PATH.
"""
