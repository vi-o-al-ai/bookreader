"""Tests for bookreader.audio.timeline.build_timeline on a hand-built chapter script."""
from __future__ import annotations

from bookreader.types import (
    NARRATOR,
    Cast,
    ChapterScript,
    ChapterTts,
    MusicCue,
    Segment,
    SfxCue,
    TTSRequest,
    TtsJob,
    VoiceAssignment,
    VoiceInfo,
    clip_key,
)

from bookreader.audio import timeline as tl
from bookreader.audio.timeline import DELIVERY_GAIN_DB, TimelineParams, build_timeline

SEG1_TEXT = "Thunder split the sky. Somewhere below, a shutter tore loose and slammed against the wall."
SEG2_TEXT = "Ring the bell! Ring it now!"
SEG3_TEXT = "Morning came grey and quiet. Gulls cried over the rocks."
DURATIONS = {"k-c1p1s0-0": 2000, "k-c1p1s0-1": 1800, "k-c1p2s1-0": 1500, "k-c1p3s0-0": 3000}


def make_script() -> ChapterScript:
    segments = [
        Segment(id="c1p1s0", paragraph_index=1, speaker=NARRATOR, kind="narration", text=SEG1_TEXT),
        Segment(id="c1p2s1", paragraph_index=2, speaker="Mara Quill", kind="dialogue", text=SEG2_TEXT, emotion="urgent"),
        Segment(id="c1p3s0", paragraph_index=3, speaker=NARRATOR, kind="narration", text=SEG3_TEXT, scene_break_before=True),
    ]
    sfx = [
        SfxCue(id="c1x001", span_id="c1p1s0", anchor_text="Thunder", anchor_offset=0, description="a single deep thunderclap",
               kind="impact", duration_ms=3000, intensity=0.9),
        SfxCue(id="c1x002", span_id="c1p2s1", anchor_text="bell", anchor_offset=SEG2_TEXT.index("bell"),
               description="a bell rings once", kind="impact", duration_ms=2000, intensity=0.5),
        SfxCue(id="c1x003", span_id="c1p3s0", anchor_text="Gulls", anchor_offset=SEG3_TEXT.index("Gulls"),
               description="seagulls crying over rocks", kind="ambient", duration_ms=8000, intensity=0.6, end_span_id="c1p3s0"),
    ]
    music = [
        MusicCue(id="c1m001", start_span_id="c1p1s0", end_span_id="c1p2s1", mood="tense", energy=0.6, prompt=""),
        MusicCue(id="c1m002", start_span_id="c1p3s0", end_span_id="c1p3s0", mood="calm", energy=0.3, prompt="soft morning pad"),
    ]
    return ChapterScript(chapter_index=1, title="The Storm Bell", segments=segments, music=music, sfx=sfx)


def make_cast() -> Cast:
    narrator = VoiceAssignment(character=NARRATOR, voice=VoiceInfo(id="v-narr", name="Narrator", family="mock"))
    mara = VoiceAssignment(character="Mara Quill", voice=VoiceInfo(id="v-mara", name="Mara", family="mock"))
    return Cast(family="mock", narrator=narrator, characters=[mara])


def make_tts() -> ChapterTts:
    def job(seg: str, piece: int, text: str, voice: str) -> TtsJob:
        return TtsJob(segment_id=seg, piece=piece, request=TTSRequest(text=text, voice_id=voice), clip_key=f"k-{seg}-{piece}")

    jobs = [
        job("c1p1s0", 0, SEG1_TEXT[:22], "v-narr"),
        job("c1p1s0", 1, SEG1_TEXT[23:], "v-narr"),
        job("c1p2s1", 0, SEG2_TEXT, "v-mara"),
        job("c1p3s0", 0, SEG3_TEXT, "v-narr"),
    ]
    return ChapterTts(chapter_index=1, jobs=jobs, durations_ms=dict(DURATIONS))


def make_params(**overrides: object) -> TimelineParams:
    base = dict(
        music_enabled=True, sfx_enabled=True, music_family="mock", music_cache_version="m1",
        music_min_ms=10000, music_max_ms=60000, sfx_family="mock", sfx_cache_version="s1", sfx_max_ms=10000,
        music_gain_db=-14.0, sfx_gain_db=-8.0, seed_material="book-sha",
    )
    base.update(overrides)
    return TimelineParams(**base)  # type: ignore[arg-type]


def test_voice_placements_are_sequential_and_paced() -> None:
    timeline = build_timeline(make_script(), make_cast(), make_tts(), make_params())
    voice = [p for p in timeline.placements if p.track == "voice"]
    segments = {seg.id: seg for seg in make_script().segments}
    assert [p.clip_key for p in voice] == ["k-c1p1s0-0", "k-c1p1s0-1", "k-c1p2s1-0", "k-c1p3s0-0"]
    for a, b in zip(voice, voice[1:]):
        assert a.end_ms <= b.start_ms
    for p in voice:
        assert p.end_ms - p.start_ms == DURATIONS[p.clip_key]
        assert p.gain_db == DELIVERY_GAIN_DB[segments[p.ref_id].delivery] and p.fade_in_ms == 10 and p.fade_out_ms == 10
        assert not p.loop and p.duck == "none"
    seg = {t.id: t for t in timeline.segments}
    # blocking gap: music intro 1500 + gap min(0.6*3000, 1500) = 1500 -> first word at 3000
    assert seg["c1p1s0"].start_ms == 3000
    assert voice[1].start_ms - voice[0].end_ms == 120                       # piece gap
    assert seg["c1p1s0"].end_ms == 3000 + 2000 + 120 + 1800
    assert seg["c1p2s1"].start_ms - seg["c1p1s0"].end_ms == 390             # new paragraph 650 x urgent 0.6
    assert seg["c1p3s0"].start_ms - seg["c1p2s1"].end_ms == 1500            # scene break
    assert [t.id for t in timeline.segments] == ["c1p1s0", "c1p2s1", "c1p3s0"]
    assert timeline.chapter_index == 1


def test_pause_table() -> None:
    a = Segment(id="a", paragraph_index=1, speaker=NARRATOR, kind="narration", text="x")
    same = Segment(id="b", paragraph_index=1, speaker=NARRATOR, kind="narration", text="y")
    other = Segment(id="c", paragraph_index=1, speaker="Mara", kind="dialogue", text="y")
    para = Segment(id="d", paragraph_index=2, speaker="Mara", kind="dialogue", text="y")
    weary = Segment(id="e", paragraph_index=2, speaker="Mara", kind="dialogue", text="y", emotion="weary")
    scene = Segment(id="f", paragraph_index=3, speaker=NARRATOR, kind="narration", text="y", scene_break_before=True)
    assert tl.pause_before_ms(a, same) == 250
    assert tl.pause_before_ms(a, other) == 400
    assert tl.pause_before_ms(a, para) == 650
    assert tl.pause_before_ms(a, weary) == 845
    assert tl.pause_before_ms(a, scene) == 1500
    hesitant_scene = scene.model_copy(update={"emotion": "hesitant"})
    assert tl.pause_before_ms(a, hesitant_scene) == 1950
    urgent_same = same.model_copy(update={"emotion": "urgent"})
    assert tl.pause_before_ms(a, urgent_same) == 150                        # clamped at the floor


def test_sfx_blocking_proportional_and_ambient() -> None:
    script = make_script()
    params = make_params()
    timeline = build_timeline(script, make_cast(), make_tts(), params)
    seg = {t.id: t for t in timeline.segments}
    sfx = {p.ref_id: p for p in timeline.placements if p.track == "sfx"}
    jobs = {j.cue_id: j for j in timeline.sfx_jobs}
    assert set(sfx) == {"c1x001", "c1x002", "c1x003"} == set(jobs)

    thunder = sfx["c1x001"]
    assert thunder.start_ms == 1500 and thunder.end_ms == 1500 + 3000
    assert thunder.gain_db == -8.0 + (0.9 - 0.7) * 10
    assert thunder.fade_in_ms == 20 and thunder.fade_out_ms == 200 and thunder.duck == "none" and not thunder.loop

    bell = sfx["c1x002"]
    t2 = seg["c1p2s1"]
    expected = round(t2.start_ms + SEG2_TEXT.index("bell") / len(SEG2_TEXT) * (t2.end_ms - t2.start_ms)) - 120
    assert bell.start_ms == expected
    assert t2.start_ms < bell.start_ms < t2.end_ms
    assert bell.end_ms == bell.start_ms + 2000
    assert bell.gain_db == -8.0 + (0.5 - 0.7) * 10

    gulls = sfx["c1x003"]
    t3 = seg["c1p3s0"]
    anchor = round(t3.start_ms + SEG3_TEXT.index("Gulls") / len(SEG3_TEXT) * (t3.end_ms - t3.start_ms))
    assert gulls.start_ms == anchor - 300
    assert gulls.end_ms == t3.end_ms
    assert gulls.loop and gulls.duck == "ambient" and gulls.gain_db == -18.0
    assert gulls.fade_in_ms == 800 and gulls.fade_out_ms == 1500

    for cue_id, job in jobs.items():
        assert job.clip_key == clip_key("sfx", "mock", "s1", job.request)
        assert sfx[cue_id].clip_key == job.clip_key
        assert 500 <= job.request.duration_ms <= params.sfx_max_ms
    assert jobs["c1x003"].request.loop and jobs["c1x003"].request.kind == "ambient"
    assert jobs["c1x003"].request.duration_ms == 10000                      # min(15000, sfx_max_ms)
    assert not jobs["c1x001"].request.loop and jobs["c1x001"].request.duration_ms == 3000
    assert jobs["c1x001"].request.seed == tl.stable_seed("book-sha", "a single deep thunderclap")


def test_music_regions_overlap_and_requests() -> None:
    script = make_script()
    params = make_params()
    timeline = build_timeline(script, make_cast(), make_tts(), params)
    seg = {t.id: t for t in timeline.segments}
    music = [p for p in timeline.placements if p.track == "music"]
    assert [p.ref_id for p in music] == ["c1m001", "c1m002"]
    first, second = music
    assert first.start_ms == seg["c1p1s0"].start_ms - 800
    assert second.start_ms == seg["c1p3s0"].start_ms - 800
    assert first.end_ms - second.start_ms == 3000
    assert second.end_ms == seg["c1p3s0"].end_ms + 1500 <= timeline.duration_ms
    assert first.fade_in_ms == 1500 and second.fade_in_ms == 2500
    assert all(p.fade_out_ms == 3000 and p.loop and p.duck == "music" and p.gain_db == -14.0 for p in music)

    assert len(timeline.music_jobs) == 2
    jobs = {j.cue_id: j for j in timeline.music_jobs}
    for cue_id, job in jobs.items():
        assert job.clip_key == clip_key("music", "mock", "m1", job.request)
        assert params.music_min_ms <= job.request.duration_ms <= params.music_max_ms
        assert job.request.loopable
    assert jobs["c1m001"].request.prompt == tl.default_music_prompt("tense")
    assert jobs["c1m002"].request.prompt == "soft morning pad"
    assert jobs["c1m001"].request.seed == tl.stable_seed("book-sha", "tense", tl.default_music_prompt("tense"))
    region1_ms = first.end_ms - first.start_ms
    assert jobs["c1m001"].request.duration_ms == max(params.music_min_ms, min(region1_ms + 4000, params.music_max_ms))
    assert {p.clip_key for p in music} == set(jobs[c].clip_key for c in jobs)


def test_music_request_clamps_to_provider_limits() -> None:
    params = make_params(music_min_ms=30000, music_max_ms=45000)
    timeline = build_timeline(make_script(), make_cast(), make_tts(), params)
    assert all(30000 <= j.request.duration_ms <= 45000 for j in timeline.music_jobs)
    assert all(j.request.duration_ms <= 120000 for j in timeline.music_jobs)


def test_duration_and_disabled_tracks() -> None:
    script = make_script()
    full = build_timeline(script, make_cast(), make_tts(), make_params())
    last_voice = max(p.end_ms for p in full.placements if p.track == "voice")
    last_sfx = max(p.end_ms for p in full.placements if p.track == "sfx")
    assert full.duration_ms == max(last_voice, last_sfx) + 2500

    silent = build_timeline(script, make_cast(), make_tts(), make_params(music_enabled=False, sfx_enabled=False))
    assert silent.music_jobs == [] and silent.sfx_jobs == []
    assert {p.track for p in silent.placements} == {"voice"}
    assert silent.segments[0].start_ms == 500                              # no music intro, no blocking gap
    assert silent.duration_ms == max(p.end_ms for p in silent.placements) + 2500

    no_music = build_timeline(script, make_cast(), make_tts(), make_params(music_enabled=False))
    assert no_music.music_jobs == [] and len(no_music.sfx_jobs) == 3
    assert no_music.segments[0].start_ms == 500 + 1500


def test_build_timeline_is_deterministic_and_pure() -> None:
    a = build_timeline(make_script(), make_cast(), make_tts(), make_params())
    b = build_timeline(make_script(), make_cast(), make_tts(), make_params())
    assert a == b
    other_book = build_timeline(make_script(), make_cast(), make_tts(), make_params(seed_material="other"))
    assert other_book.music_jobs[0].request.seed != a.music_jobs[0].request.seed


def test_default_music_prompt_covers_all_moods() -> None:
    from bookreader.types import MOODS

    for mood in MOODS:
        assert tl.default_music_prompt(mood)
    assert tl.default_music_prompt("unknown") == tl.default_music_prompt("calm")


def test_voice_gain_follows_delivery() -> None:
    """Clips are RMS-normalized in the tts substage, so the level dimension of delivery lives in the placement gain."""
    script = make_script()
    script.segments[0].delivery = "whisper"
    script.segments[1].delivery = "shout"
    script.segments[2].delivery = "quiet"
    timeline = build_timeline(script, make_cast(), make_tts(), make_params())
    gain = {p.ref_id: p.gain_db for p in timeline.placements if p.track == "voice"}
    assert gain["c1p1s0"] < gain["c1p3s0"] < 0.0 < gain["c1p2s1"]
    assert gain["c1p1s0"] == DELIVERY_GAIN_DB["whisper"] and gain["c1p2s1"] == DELIVERY_GAIN_DB["shout"]
    assert DELIVERY_GAIN_DB["normal"] == 0.0 and DELIVERY_GAIN_DB["whisper"] > -45.0 + 20.0   # whisper still ducks music


def test_repeated_ambient_in_one_scene_extends_the_running_bed() -> None:
    script = make_script()
    params = make_params()
    again = SfxCue(id="c1x004", span_id="c1p3s0", anchor_text="rocks", anchor_offset=SEG3_TEXT.index("rocks"),
                   description="seagulls crying over rocks", kind="ambient", duration_ms=8000, intensity=0.6, end_span_id="c1p3s0")
    script = script.model_copy(update={"sfx": script.sfx + [again]})
    timeline = build_timeline(script, make_cast(), make_tts(), params)
    ambients = [p for p in timeline.placements if p.track == "sfx" and p.duck == "ambient"]
    assert len(ambients) == 1 and ambients[0].ref_id == "c1x003"           # no second copy stacked on the first
    t3 = {t.id: t for t in timeline.segments}["c1p3s0"]
    assert ambients[0].end_ms == t3.end_ms
    assert len(timeline.sfx_jobs) == 3                                       # the clip is still requested once
    # a re-emission after the first bed has ended is a fresh placement
    later = again.model_copy(update={"span_id": "c1p3s0", "anchor_offset": 0})
    script2 = make_script()
    first = script2.sfx[2].model_copy(update={"span_id": "c1p1s0", "anchor_offset": 0, "end_span_id": "c1p1s0"})
    script2 = script2.model_copy(update={"sfx": [script2.sfx[0], script2.sfx[1], first, later]})
    timeline2 = build_timeline(script2, make_cast(), make_tts(), params)
    assert len([p for p in timeline2.placements if p.duck == "ambient"]) == 2


def test_unfound_anchor_never_opens_a_blocking_gap() -> None:
    seg = Segment(id="c1p1s0", paragraph_index=1, speaker=NARRATOR, kind="narration", text=SEG1_TEXT)
    found = SfxCue(id="x1", span_id="c1p1s0", anchor_text="thunder", anchor_offset=0, description="thunderclap",
                   kind="impact", duration_ms=3000, intensity=0.9)
    misquoted = found.model_copy(update={"anchor_text": "Lightning"})       # assemble stores offset 0 when not found
    empty = found.model_copy(update={"anchor_text": ""})
    assert tl.anchor_found(found, seg) and tl.is_blocking_impact(found, seg)
    assert not tl.anchor_found(misquoted, seg) and not tl.is_blocking_impact(misquoted, seg)
    assert not tl.is_blocking_impact(empty, seg)
    script = make_script().model_copy(update={"sfx": [misquoted]})
    timeline = build_timeline(script, make_cast(), make_tts(), make_params())
    assert timeline.segments[0].start_ms == 1500                             # music intro only, no 1.5 s dead gap
    impact = next(p for p in timeline.placements if p.track == "sfx")
    assert impact.start_ms == 1500 - 120                                     # proportional placement, onset lead
