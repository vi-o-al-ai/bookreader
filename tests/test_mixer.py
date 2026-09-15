"""Tests for bookreader.audio.mixer and bookreader.audio.export with synthetic clips."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bookreader.types import (
    NARRATOR,
    SAMPLE_RATE,
    AudioClip,
    Cast,
    ChapterScript,
    ChapterTimeline,
    ChapterTts,
    MusicCue,
    Placement,
    Segment,
    SegmentTiming,
    SfxCue,
    TTSRequest,
    TtsJob,
    VoiceAssignment,
    VoiceInfo,
)

from bookreader.audio import export, mixer, pcm
from bookreader.audio.dsp import db_to_gain, rms_dbfs
from bookreader.audio.timeline import TimelineParams, build_timeline


def tone_clip(ms: int, amp: float, freq: float = 220.0, rate: int = SAMPLE_RATE) -> AudioClip:
    t = np.arange(int(rate * ms / 1000)) / rate
    return AudioClip(pcm.to_int16(amp * np.sin(2 * np.pi * freq * t)), rate)


def noise_clip(ms: int, amp: float, seed: int = 7) -> AudioClip:
    rng = np.random.default_rng(seed)
    return AudioClip(pcm.to_int16(rng.uniform(-amp, amp, int(SAMPLE_RATE * ms / 1000))), SAMPLE_RATE)


def make_loader(clips: dict[str, AudioClip]):
    def load(key: str) -> AudioClip:
        return clips[key]

    return load


def basic_timeline(voice_amp: float = 0.3, music_amp: float = 0.2, sfx_amp: float = 0.2) -> tuple[ChapterTimeline, dict[str, AudioClip]]:
    clips = {"v": tone_clip(2000, voice_amp), "m": noise_clip(1000, music_amp), "x": noise_clip(500, sfx_amp, seed=3)}
    placements = [
        Placement(track="voice", clip_key="v", ref_id="s1", start_ms=3000, end_ms=5000),
        Placement(track="music", clip_key="m", ref_id="m1", start_ms=0, end_ms=8000, gain_db=0.0, fade_in_ms=0, fade_out_ms=0, loop=True, duck="music"),
        Placement(track="sfx", clip_key="x", ref_id="x1", start_ms=1000, end_ms=1500, gain_db=0.0, fade_in_ms=20, fade_out_ms=200),
    ]
    timeline = ChapterTimeline(chapter_index=1, duration_ms=8000, segments=[SegmentTiming(id="s1", start_ms=3000, end_ms=5000)], placements=placements)
    return timeline, clips


def read_all(paths: dict[str, Path]) -> dict[str, np.ndarray]:
    return {name: pcm.read_wav(path).samples for name, path in paths.items()}


def test_stems_have_identical_length_and_names(tmp_path: Path) -> None:
    timeline, clips = basic_timeline()
    paths = mixer.mix_chapter(timeline, make_loader(clips), tmp_path / "01")
    assert set(paths) == {"mix", "voice", "music", "sfx"}
    assert all(p.parent == tmp_path / "01" and p.name == f"{name}.wav" for name, p in paths.items())
    stems = read_all(paths)
    expected = int(round(8000 * SAMPLE_RATE / 1000))
    assert all(len(s) == expected for s in stems.values())
    voice = stems["voice"]
    assert np.max(np.abs(voice[: 3000 * SAMPLE_RATE // 1000 - 5])) == 0
    assert np.max(np.abs(voice[3100 * SAMPLE_RATE // 1000: 4900 * SAMPLE_RATE // 1000])) > 5000
    assert np.max(np.abs(stems["sfx"][1050 * SAMPLE_RATE // 1000: 1300 * SAMPLE_RATE // 1000])) > 100
    assert np.max(np.abs(stems["sfx"][2000 * SAMPLE_RATE // 1000:])) == 0


def test_stems_sum_to_mix_when_under_ceiling(tmp_path: Path) -> None:
    timeline, clips = basic_timeline(voice_amp=0.3, music_amp=0.2, sfx_amp=0.15)
    paths = mixer.mix_chapter(timeline, make_loader(clips), tmp_path)
    stems = read_all(paths)
    total = stems["voice"].astype(np.int32) + stems["music"].astype(np.int32) + stems["sfx"].astype(np.int32)
    assert np.max(np.abs(stems["mix"])) < 32767 * db_to_gain(-1.0)
    assert np.max(np.abs(total - stems["mix"].astype(np.int32))) <= 1


def test_ducking_lowers_music_under_voice(tmp_path: Path) -> None:
    timeline, clips = basic_timeline()
    paths = mixer.mix_chapter(timeline, make_loader(clips), tmp_path)
    music = pcm.to_float(pcm.read_wav(paths["music"]).samples)
    s = SAMPLE_RATE // 1000
    under_voice = rms_dbfs(music[3500 * s: 4800 * s])
    # spec: under speech the bed follows its placement gain exactly (0 dB here); long gaps get +5 dB
    assert abs(under_voice - rms_dbfs(clips["m"].samples)) < 0.5
    # the impact at 1000-1500 ms splits the opening gap: the 1 s before it is too short to breathe ...
    before_impact = rms_dbfs(music[200 * s: 900 * s])
    assert abs(before_impact - under_voice) < 0.5
    # ... the 1.5 s between the impact and the voice does (release-limited, so only part of the way)
    in_gap = rms_dbfs(music[2000 * s: 2900 * s])
    assert in_gap - under_voice >= 2.0
    # the bed recovers after speech ends (long gap -> boosted level, release-limited)
    after = rms_dbfs(music[7000 * s: 8000 * s])
    assert abs((after - under_voice) - mixer.MUSIC_GAP_BOOST_DB) < 0.5


def test_music_bed_is_not_ducked_below_its_placement_gain() -> None:
    assert mixer.MUSIC_DUCK_DB == 0.0
    env = np.full(300, -80.0)
    env[100:200] = -20.0
    n = 300 * SAMPLE_RATE // 50
    curve = mixer.duck_gain(env, n, mixer.MUSIC_DUCK_DB, gap_boost_db=mixer.MUSIC_GAP_BOOST_DB, min_gap_ms=mixer.MUSIC_GAP_MIN_MS)
    assert abs(curve[150 * SAMPLE_RATE // 50] - 1.0) < 1e-3


def test_duck_gain_lookahead_and_hold() -> None:
    frame = SAMPLE_RATE // 50
    env = np.full(200, -80.0)
    env[100:120] = -20.0
    n = 200 * frame
    early = mixer.duck_gain(env, n, 8.0, gap_boost_db=5.0, min_gap_ms=1200)
    late = mixer.duck_gain(env, n, 8.0, gap_boost_db=5.0, min_gap_ms=1200, lookahead_ms=0)
    at_onset = 100 * frame + frame // 2
    assert late[at_onset] > db_to_gain(0.0)                       # no look-ahead: still nearly un-ducked at the onset
    assert early[at_onset] < db_to_gain(-5.0)                     # look-ahead: already most of the way down
    assert early[97 * frame + frame // 2] < db_to_gain(5.0) - 0.05 and abs(late[97 * frame + frame // 2] - db_to_gain(5.0)) < 1e-3
    # frames flagged in `hold` are busy: a 2 s speech-free run covered by an impact earns no boost
    hold = np.zeros(200, dtype=bool)
    hold[:100] = True
    held = mixer.duck_gain(env, n, 8.0, gap_boost_db=5.0, min_gap_ms=1200, hold=hold)
    assert held[: 100 * frame].max() < db_to_gain(-7.9)
    assert held[-1] > db_to_gain(0.0)                             # the trailing gap still breathes


def test_duck_gain_curve_shape() -> None:
    env = np.full(200, -80.0)
    env[100:120] = -20.0                                 # 400 ms of speech between two 2 s gaps
    n = 200 * SAMPLE_RATE // 50
    curve = mixer.duck_gain(env, n, depth_db=8.0, gap_boost_db=5.0, min_gap_ms=1200)
    assert curve.shape == (n,)
    assert abs(curve[0] - db_to_gain(5.0)) < 1e-3
    assert curve[110 * SAMPLE_RATE // 50] < db_to_gain(-7.0)
    assert curve[-1] > curve[120 * SAMPLE_RATE // 50]    # releasing back up after speech
    short = np.full(100, -80.0)
    short[40:60] = -20.0                                 # 800 ms gaps stay ducked (no pumping)
    assert mixer.duck_gain(short, 100 * SAMPLE_RATE // 50, 8.0, gap_boost_db=5.0, min_gap_ms=1200).max() < db_to_gain(-7.9)
    assert np.max(np.abs(np.diff(curve))) < 0.01          # smooth per sample
    ducked = mixer.duck(np.ones(n, dtype=np.float32), env, 4.0)
    assert ducked.shape == (n,) and ducked.min() >= db_to_gain(-4.0) - 1e-3 and ducked.max() <= 1.0 + 1e-6
    assert mixer.duck_gain(np.zeros(0), 10, 4.0).shape == (10,)


def test_limiter_engages_on_hot_clip(tmp_path: Path) -> None:
    timeline, clips = basic_timeline(voice_amp=0.98, music_amp=0.6)
    paths = mixer.mix_chapter(timeline, make_loader(clips), tmp_path)
    stems = read_all(paths)
    raw_sum = stems["voice"].astype(np.int32) + stems["music"].astype(np.int32) + stems["sfx"].astype(np.int32)
    assert np.max(np.abs(raw_sum)) > 32767 * db_to_gain(-1.0)          # the limiter had work to do
    assert np.max(np.abs(stems["mix"])) <= int(round(32767 * db_to_gain(-1.0)))
    assert np.max(np.abs(stems["mix"])) > 20000                            # but it did not squash the audio


def test_render_track_resamples_and_skips_out_of_range() -> None:
    clip = tone_clip(500, 0.5, rate=44100)
    placements = [
        Placement(track="voice", clip_key="a", ref_id="s", start_ms=0, end_ms=500),
        Placement(track="voice", clip_key="a", ref_id="s", start_ms=2000, end_ms=2500),   # beyond the buffer
    ]
    n = SAMPLE_RATE
    out = mixer.render_track(placements, make_loader({"a": clip}), n)
    assert out.shape == (n,)
    assert abs(np.max(np.abs(out[: n // 2])) - 0.5) < 0.05
    assert np.all(out[n // 2 + 10:] == 0)


def _script_and_cast() -> tuple[ChapterScript, Cast, ChapterTts, TimelineParams]:
    text1 = "Thunder split the sky over the lighthouse."
    text2 = "Ring the bell!"
    script = ChapterScript(
        chapter_index=2, title="The Storm Bell",
        segments=[
            Segment(id="c2p1s0", paragraph_index=1, speaker=NARRATOR, kind="narration", text=text1, emotion="tense", source="llm"),
            Segment(id="c2p2s1", paragraph_index=2, speaker="Mara Quill", kind="dialogue", text=text2, delivery="shout"),
        ],
        music=[MusicCue(id="c2m001", start_span_id="c2p1s0", end_span_id="c2p2s1", mood="tense", energy=0.5, prompt="")],
        sfx=[SfxCue(id="c2x001", span_id="c2p1s0", anchor_text="Thunder", anchor_offset=0, description="thunderclap", kind="impact", duration_ms=2000, intensity=0.9)],
    )
    cast = Cast(
        family="mock",
        narrator=VoiceAssignment(character=NARRATOR, voice=VoiceInfo(id="v-narr", name="N", family="mock")),
        characters=[VoiceAssignment(character="Mara Quill", voice=VoiceInfo(id="v-mara", name="M", family="mock"))],
    )
    tts = ChapterTts(
        chapter_index=2,
        jobs=[
            TtsJob(segment_id="c2p1s0", piece=0, request=TTSRequest(text=text1, voice_id="v-narr"), clip_key="tts-1"),
            TtsJob(segment_id="c2p2s1", piece=0, request=TTSRequest(text=text2, voice_id="v-mara"), clip_key="tts-2"),
        ],
        durations_ms={"tts-1": 2500, "tts-2": 900},
    )
    params = TimelineParams(
        music_enabled=True, sfx_enabled=True, music_family="mock", music_cache_version="m", music_min_ms=5000,
        music_max_ms=30000, sfx_family="mock", sfx_cache_version="s", sfx_max_ms=8000, music_gain_db=-14.0,
        sfx_gain_db=-8.0, seed_material="sha",
    )
    return script, cast, tts, params


def test_manifest_carries_timeline_values() -> None:
    script, cast, tts, params = _script_and_cast()
    timeline = build_timeline(script, cast, tts, params)
    files = {"mix": "chapters/02/mix.wav", "voice": "chapters/02/voice.wav", "music": "chapters/02/music.wav", "sfx": "chapters/02/sfx.wav", "mp3": None}
    manifest = mixer.build_chapter_manifest(script, cast, timeline, files, 2, "The Storm Bell")
    assert manifest.index == 2 and manifest.title == "The Storm Bell"
    assert manifest.duration_ms == timeline.duration_ms and manifest.sample_rate == SAMPLE_RATE
    assert manifest.files == files
    assert [s.id for s in manifest.segments] == ["c2p1s0", "c2p2s1"]
    timing = {t.id: t for t in timeline.segments}
    for seg, ms in zip(script.segments, manifest.segments):
        assert (ms.start_ms, ms.end_ms) == (timing[seg.id].start_ms, timing[seg.id].end_ms)
        assert ms.speaker == seg.speaker and ms.text == seg.text and ms.kind == seg.kind
        assert ms.emotion == seg.emotion and ms.delivery == seg.delivery and ms.paragraph == seg.paragraph_index
        assert ms.source == seg.source
    assert manifest.segments[0].voice_id == "v-narr" and manifest.segments[1].voice_id == "v-mara"

    cues = {c.id: c for c in manifest.cues}
    assert set(cues) == {"c2m001", "c2x001"}
    placements = {p.ref_id: p for p in timeline.placements if p.track != "voice"}
    music = cues["c2m001"]
    assert music.kind == "music" and music.mood == "tense" and music.prompt == timeline.music_jobs[0].request.prompt
    assert (music.start_ms, music.end_ms, music.gain_db, music.clip) == (
        placements["c2m001"].start_ms, placements["c2m001"].end_ms, -14.0, placements["c2m001"].clip_key)
    sfx = cues["c2x001"]
    assert sfx.kind == "sfx" and sfx.sfx_kind == "impact" and sfx.description == "thunderclap"
    assert sfx.anchor_segment == "c2p1s0" and sfx.anchor_text == "Thunder"
    assert (sfx.start_ms, sfx.end_ms, sfx.gain_db, sfx.clip) == (
        placements["c2x001"].start_ms, placements["c2x001"].end_ms, placements["c2x001"].gain_db, timeline.sfx_jobs[0].clip_key)
    assert manifest.model_dump(mode="json")["cues"][0]["id"] in cues


def test_mix_end_to_end_from_timeline(tmp_path: Path) -> None:
    script, cast, tts, params = _script_and_cast()
    timeline = build_timeline(script, cast, tts, params)
    clips = {
        "tts-1": tone_clip(2500, 0.3), "tts-2": tone_clip(900, 0.3, freq=330.0),
        timeline.music_jobs[0].clip_key: noise_clip(3000, 0.3), timeline.sfx_jobs[0].clip_key: noise_clip(2000, 0.5, seed=5),
    }
    paths = mixer.mix_chapter(timeline, make_loader(clips), tmp_path / "02")
    stems = read_all(paths)
    n = int(round(timeline.duration_ms * SAMPLE_RATE / 1000))
    assert all(len(s) == n for s in stems.values())
    assert np.max(np.abs(stems["music"])) > 0 and np.max(np.abs(stems["sfx"])) > 0
    # level design: under speech the music stem sits at bed level + music_gain_db (no extra duck)
    s = SAMPLE_RATE // 1000
    seg1 = {t.id: t for t in timeline.segments}["c2p1s0"]
    under_speech = rms_dbfs(pcm.to_float(stems["music"])[(seg1.start_ms + 1000) * s: (seg1.end_ms - 200) * s])
    bed = rms_dbfs(clips[timeline.music_jobs[0].clip_key].samples)
    assert abs(under_speech - (bed + params.music_gain_db)) < 1.0
    # the blocking gap before the first word holds the thunder, so the music must not swell into it
    gap = rms_dbfs(pcm.to_float(stems["music"])[(seg1.start_ms - 600) * s: (seg1.start_ms - 100) * s])
    assert gap < bed + params.music_gain_db + 1.0


def test_export_mp3_returns_false_without_ffmpeg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(export.shutil, "which", lambda name: None)
    assert export.ffmpeg_path() is None
    wav = tmp_path / "mix.wav"
    pcm.write_wav(wav, AudioClip.silence(100))
    assert export.export_mp3(wav, tmp_path / "mix.mp3") is False
    assert not (tmp_path / "mix.mp3").exists()


def test_export_mp3_handles_failing_binary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = tmp_path / "ffmpeg"
    fake.write_text("#!/bin/sh\nexit 1\n")
    fake.chmod(0o755)
    monkeypatch.setattr(export.shutil, "which", lambda name: str(fake))
    wav = tmp_path / "mix.wav"
    pcm.write_wav(wav, AudioClip.silence(100))
    assert export.export_mp3(wav, tmp_path / "mix.mp3") is False
