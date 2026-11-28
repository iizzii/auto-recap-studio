"""Unit tests for 3-Act chapter_commentary mode, freeze roasts, Kiko 3D booth, chroma-key & QA gate."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw
import pytest

from pipeline import chapter_commentator, chapter_composer, chapter_slicer, host_animator, qa_gate
from pipeline.models import (
    ActChapterPlan,
    ChapterCommentaryScript,
    ChapterSlice,
    CommentaryCue,
    RightsRecord,
    SceneSegment,
    ThreeActSeriesPlan,
)
from scripts.generate_host_assets_ai import chroma_key_and_align


def test_plan_chapters_ci_synthetic() -> None:
    slices = chapter_slicer.plan_chapters(
        Path("dummy.mp4"), 10.0, 210.0, 2, {"chapter_commentary": {"target_chapter_sec": 98.0}}, ci_synthetic=True
    )
    assert len(slices) == 2
    assert slices[0].start_sec == 10.0
    assert slices[0].end_sec == slices[1].start_sec
    assert slices[1].end_sec == 210.0


def test_synthesize_timed_commentary_mock(tmp_path: Path) -> None:
    ch = ChapterSlice(part_number=1, total_parts=1, start_sec=0.0, end_sec=15.0, duration_sec=15.0)
    script = chapter_commentator.generate_chapter_commentary(
        tmp_path / "sheet.jpg", ch, "Test Film", {}, mock_llm=True
    )
    wav, timings_p, ass_p = chapter_commentator.synthesize_timed_commentary(
        script, 15.0, tmp_path, {}, mock_tts=True
    )
    assert wav.exists()
    assert timings_p.exists()
    assert ass_p.exists()
    exp_path = tmp_path / "03_timeline_expansion.json"
    assert exp_path.exists()
    exp_meta = json.loads(exp_path.read_text(encoding="utf-8"))
    assert exp_meta["expanded_duration_sec"] >= 15.0


def test_render_host_booth_video_mock(tmp_path: Path) -> None:
    ch = ChapterSlice(part_number=1, total_parts=1, start_sec=0.0, end_sec=15.0, duration_sec=15.0)
    script = chapter_commentator.generate_chapter_commentary(
        tmp_path / "sheet.jpg", ch, "Test Film", {}, mock_llm=True
    )
    wav, _, _ = chapter_commentator.synthesize_timed_commentary(script, 15.0, tmp_path, {}, mock_tts=True)
    timings = [{"start_sec": 0.0, "end_sec": 1.0, "emotion": "shocked", "pause_emotion": "smug"}]
    out_mp4 = tmp_path / "host.mp4"
    cfg = {"host": {"fps": 10, "sprites_dir": "assets/host/sprites", "booths_dir": "assets/host/booths"}}
    with patch("subprocess.Popen") as mock_popen:
        mock_popen.return_value.returncode = 0
        host_animator.render_host_booth_video(wav, timings, "comedy", 2.0, out_mp4, cfg)
        assert mock_popen.called


def test_compose_chapter_short_mock(tmp_path: Path) -> None:
    ch = ChapterSlice(part_number=1, total_parts=1, start_sec=0.0, end_sec=5.0, duration_sec=5.0)
    rights = RightsRecord(
        work_title="Test", creator="Test", year=2022, license="cc_by",
        source_page_url="https://ex.com", commercial_use_allowed=True,
        derivatives_allowed=True, verified_by="test", verified_on="2026-11-27",
    )
    with (
        patch("pipeline.ffutil.probe", return_value={"streams": [{"codec_type": "video", "width": 1920, "height": 1080}]}),
        patch("pipeline.ffutil.run") as mock_run,
    ):
        chapter_composer.compose_chapter_short(
            Path("src.mp4"), ch, Path("host.mp4"), Path("comm.wav"), Path("cap.ass"),
            rights, "comedy", tmp_path / "out.mp4", tmp_path, {},
            top_headline="THEY WOKE UP THE FOREST",
        )
        assert mock_run.called
        assert (tmp_path / "part_1_chrome.png").exists()


def test_generate_commentary_fails_fast_without_api_key(tmp_path: Path) -> None:
    """When mock_llm=False and GEMINI_API_KEY is missing, raise RuntimeError instead of mock fallback."""
    ch = ChapterSlice(part_number=1, total_parts=3, start_sec=0.0, end_sec=80.0, duration_sec=80.0)
    with patch.dict(os.environ, {"GEMINI_API_KEY": "", "GEMINI_API_KEY1": ""}, clear=False), pytest.raises(
        RuntimeError, match="GEMINI_API_KEY"
    ):
        chapter_commentator.generate_chapter_commentary(
            tmp_path / "sheet.jpg", ch, "Sprite Fright", {}, mock_llm=False
        )


def test_validate_commentary_rejects_placeholder_cliches_and_overlap() -> None:
    bad_script = ChapterCommentaryScript(
        genre="horror_comedy",
        voice_name="en-US-BrianNeural",
        voice_rate="+9%",
        chapter_title="Sprite Fright Part 1",
        top_headline="BIG MISTAKE IN THE WOODS",
        cues=[
            CommentaryCue(
                cue_id=1, start_sec=0.0, max_duration_sec=3.0,
                text="Watch what happens right here in scene 1!",
                emphasis_word="SCENE", emotion="shocked", pause_emotion="neutral",
            ),
            CommentaryCue(
                cue_id=2, start_sec=8.0, max_duration_sec=3.0,
                text="This camp counselor made a hilarious blunder hiking near glowing mushrooms.",
                emphasis_word="BLUNDER", emotion="laughing", pause_emotion="smug",
            ),
            CommentaryCue(
                cue_id=3, start_sec=16.0, max_duration_sec=3.0,
                text="This camp counselor made a hilarious blunder hiking near glowing mushrooms now.",
                emphasis_word="MUSHROOMS", emotion="laughing", pause_emotion="smug",
            ),
        ],
    )
    errs = chapter_commentator.validate_chapter_commentary(bad_script, min_emotions=5)
    joined = " | ".join(errs)
    assert "placeholder" in joined.lower()
    assert "banned cliche" in joined.lower()
    assert "70% token overlap" in joined.lower()
    assert "distinct emotions" in joined.lower()


def test_snap_scene_segments_cut_and_speech_safe() -> None:
    segs = [
        SceneSegment(start_sec=10.0, end_sec=28.0, why_kept="Setup"),
        SceneSegment(start_sec=45.0, end_sec=63.0, why_kept="Turn"),
    ]
    cuts = [11.2, 27.1, 44.3, 64.2]
    dialogue = [
        {"start": 26.8, "end": 27.4, "text": "Wait don't touch that!", "words": [
            {"word": "Wait", "start": 26.8, "end": 27.0},
            {"word": "touch", "start": 27.0, "end": 27.4},
        ]},
    ]
    snapped = chapter_slicer.snap_scene_segments(segs, cuts, dialogue, allowed_start=5.0, allowed_end=100.0)
    assert len(snapped) == 2
    assert snapped[0].start_sec == 11.2
    # End at 27.1 landed mid-word [27.0, 27.4] -> nudged cleanly past 27.4
    assert snapped[0].end_sec >= 27.4
    assert snapped[1].start_sec == 44.3
    assert snapped[1].end_sec == 64.2


def test_sanitize_three_act_plan_enforces_chronology() -> None:
    raw = ThreeActSeriesPlan(
        base_slug="sprite_fright",
        work_title="Sprite Fright",
        genre="horror_comedy",
        total_parts=3,
        parts=[
            ActChapterPlan(
                part_number=1, total_parts=3, top_headline="they woke up the forest",
                act_goal="Setup", segments=[SceneSegment(start_sec=12.0, end_sec=95.0, why_kept="Intro")],
                cliffhanger_summary="Sprites appear.",
            ),
            ActChapterPlan(
                part_number=2, total_parts=3, top_headline="BIG MISTAKE",
                act_goal="Escalation", segments=[SceneSegment(start_sec=80.0, end_sec=220.0, why_kept="Escalation")],
                cliffhanger_summary="Ambush begins.",
            ),
            ActChapterPlan(
                part_number=3, total_parts=3, top_headline="only one survived this night",
                act_goal="Climax", segments=[SceneSegment(start_sec=350.0, end_sec=440.0, why_kept="Climax")],
                cliffhanger_summary="",
            ),
        ],
    )
    clean = chapter_slicer.sanitize_three_act_plan(raw, 12.0, 570.0, 3, scene_cuts=[], dialogue_segments=[])
    assert len(clean.parts) == 3
    assert clean.parts[0].segments[-1].end_sec <= clean.parts[1].segments[0].start_sec
    assert clean.parts[1].segments[-1].end_sec <= clean.parts[2].segments[0].start_sec
    assert clean.parts[0].top_headline == "THEY WOKE UP THE FOREST"
    # Banned cliché headline in Part 2 replaced by safe clean uppercase default
    assert "BIG MISTAKE" not in clean.parts[1].top_headline


def test_remap_dialogue_to_chapter_busy_and_open_gaps() -> None:
    dialogue = [
        {"start": 102.0, "end": 105.0, "text": "What is that glowing thing?"},
        {"start": 210.0, "end": 213.0, "text": "Run back to the bus!"},
    ]
    mappings = [
        {"src_start_sec": 100.0, "src_end_sec": 120.0, "chapter_start_sec": 0.0, "chapter_end_sec": 20.0},
        {"src_start_sec": 200.0, "src_end_sec": 220.0, "chapter_start_sec": 20.0, "chapter_end_sec": 40.0},
    ]
    mapped, gaps, timeline_str = chapter_slicer.remap_dialogue_to_chapter(dialogue, mappings, 40.0)
    assert len(mapped) == 2
    assert mapped[0]["start_sec"] == 2.0
    assert mapped[1]["start_sec"] == 30.0
    assert len(gaps) >= 3
    assert "BUSY_DIALOGUE" in timeline_str
    assert "OPEN_GAP" in timeline_str


def test_plan_three_act_series_mock_persists_json(tmp_path: Path) -> None:
    proj = tmp_path / "projects" / "sprite_fright"
    work = tmp_path / "work"
    plan = chapter_slicer.plan_three_act_series(
        Path("dummy.mp4"), 12.0, 570.0, 3, "sprite_fright", "Sprite Fright",
        proj, work, {}, dialogue_segments=[], mock_llm=True, force=True,
    )
    assert len(plan.parts) == 3
    assert (proj / "three_act_plan.json").exists()
    # Reload without force
    reloaded = chapter_slicer.plan_three_act_series(
        Path("dummy.mp4"), 12.0, 570.0, 3, "sprite_fright", "Sprite Fright",
        proj, work, {}, dialogue_segments=[], mock_llm=True, force=False,
    )
    assert reloaded.model_dump() == plan.model_dump()


def test_freeze_frame_timeline_expansion_and_gold_captions(tmp_path: Path) -> None:
    ch = ChapterSlice(part_number=1, total_parts=3, start_sec=0.0, end_sec=30.0, duration_sec=30.0)
    script = chapter_commentator.generate_chapter_commentary(
        tmp_path / "sheet.jpg", ch, "Sprite Fright", {}, mock_llm=True
    )
    wav, timings_p, ass_p = chapter_commentator.synthesize_timed_commentary(
        script, 30.0, tmp_path, {}, mock_tts=True
    )
    assert wav.exists()
    exp_meta = json.loads((tmp_path / "03_timeline_expansion.json").read_text(encoding="utf-8"))
    timings = json.loads(timings_p.read_text(encoding="utf-8"))

    assert exp_meta["freeze_count"] >= 2
    assert exp_meta["expanded_duration_sec"] > 30.0
    assert exp_meta["sfx_wav"] and Path(exp_meta["sfx_wav"]).exists()
    # Subsequent cues after first freeze are shifted forward beyond their raw timestamp
    shifted = [t for t in timings if t["start_sec"] > t["raw_start_sec"] + 1.0]
    assert len(shifted) >= 1

    ass_text = ass_p.read_text(encoding="utf-8")
    assert r"\c&H4DD8FF&" in ass_text  # Gold #FFD84D emphasis highlight
    assert r"\fscx92\fscy92" in ass_text  # 2-frame pop-in scale animation


def test_chroma_key_protects_enclosed_emerald_eyes() -> None:
    """Border-seeded chroma-key must remove #00FF00 background while keeping Kiko's emerald eyes opaque."""
    img = Image.new("RGB", (512, 512), (0, 255, 0))
    d = ImageDraw.Draw(img)
    # Ginger cat head & torso
    d.ellipse([120, 80, 392, 350], fill=(235, 140, 55))
    d.rounded_rectangle([140, 310, 372, 511], radius=60, fill=(235, 140, 55))
    # Two bright emerald-green eyes fully enclosed inside the head
    d.ellipse([175, 170, 225, 220], fill=(25, 215, 75))
    d.ellipse([287, 170, 337, 220], fill=(25, 215, 75))

    aligned, fg_ratio = chroma_key_and_align(img, target_size=640)
    assert aligned.size == (640, 640)
    assert aligned.mode == "RGBA"
    assert 0.15 < fg_ratio < 0.85

    arr = np.array(aligned)
    # Outer top corners must be 100% transparent
    assert arr[5, 5, 3] == 0
    assert arr[5, 634, 3] == 0
    # Both enclosed emerald eyes must remain 100% opaque (zero punch-through!)
    assert int(arr[200:240, 225:265, 3].min()) == 255
    assert int(arr[200:240, 375:415, 3].min()) == 255


def test_movie_stage_filter_16_9_and_cinemascope() -> None:
    lay_16 = chapter_composer.compute_vertical_layout(1920, 1080)
    assert lay_16["divider_y"] == 1220
    assert lay_16["booth_h"] == 700
    assert lay_16["host_overlay_y"] == 1170
    assert lay_16["host_overlay_h"] == 750
    assert 200 <= lay_16["movie_y"] and lay_16["movie_y"] + lay_16["movie_h"] <= 1050

    lay_scope = chapter_composer.compute_vertical_layout(2048, 858)
    assert lay_scope["divider_y"] == 1220
    assert 580 <= lay_scope["movie_h"] <= 660
    assert 200 <= lay_scope["movie_y"] and lay_scope["movie_y"] + lay_scope["movie_h"] <= 1050

    fc = chapter_composer.build_chapter_composite_filter_complex(
        2048,
        858,
        90.0,
        "",
        lay_scope,
        has_host_alpha=True,
        feather_input_idx=4,
        chrome_input_idx=3,
        host_alpha_input_idx=5,
        ass_flt=",ass='cap.ass'",
    )
    assert "crop=1080:1220,boxblur=24:6" in fc
    assert "[mc_rgba][mc_mask]alphamerge=shortest=1[mov_feathered]" in fc
    assert "[host_rgb][host_a]alphamerge=shortest=1[host_rgba]" in fc
    assert "overlay=0:1170:shortest=1,ass='cap.ass'[out]" in fc


def test_chapter_qa_gate_passes_valid_and_rejects_bad(tmp_path: Path) -> None:
    ch = ChapterSlice(part_number=1, total_parts=3, start_sec=0.0, end_sec=80.0, duration_sec=80.0)
    script = chapter_commentator.generate_chapter_commentary(
        tmp_path / "sheet.jpg", ch, "Sprite Fright", {}, mock_llm=True
    )
    timings = [
        {"start_sec": c.start_sec, "end_sec": c.start_sec + 3.0, "emotion": c.emotion}
        for c in script.cues
    ]
    probe_ok = {
        "streams": [
            {
                "codec_type": "video", "width": 1080, "height": 1920,
                "codec_name": "h264", "pix_fmt": "yuv420p", "r_frame_rate": "30/1", "duration": "84.0",
            },
            {"codec_type": "audio", "duration": "84.0"},
        ]
    }
    exp_meta = {"active_commentary_sec": 39.0, "commentary_share": 0.464, "freeze_count": 2}
    with (
        patch("pipeline.ffutil.probe", return_value=probe_ok),
        patch("pipeline.qa_gate._measure_lufs", return_value=-14.1),
        patch("pipeline.qa_gate._black_stats", return_value=(0.0, 0.0)),
    ):
        rep = qa_gate.run_chapter_qa_gate(
            tmp_path / "final.mp4", script, timings, exp_meta, tmp_path, {}, strict=True, mock_llm=False
        )
        assert rep["passed"] is True
        assert rep["metrics"]["cta_end_padding_sec"] > 0.20
        assert (tmp_path / "qa_report.json").exists()

        # Inject placeholder text -> strict QA gate must exit code 7
        script.cues[0].text = "Watch what happens right here in scene 2!"
        with pytest.raises(SystemExit) as exc_info:
            qa_gate.run_chapter_qa_gate(
                tmp_path / "final.mp4", script, timings, exp_meta, tmp_path, {}, strict=True, mock_llm=False
            )
        assert exc_info.value.code == 7


def test_auto_bleep_profanity_preserves_length_and_bleeps_exact_span(tmp_path: Path) -> None:
    import wave

    sr = 48000
    n_frames = sr * 2  # 2.0s stereo WAV
    src_wav = tmp_path / "movie.wav"
    silence = np.zeros((n_frames, 2), dtype=np.int16)
    with wave.open(str(src_wav), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(silence.tobytes())

    words = [
        {"word": "shiitake", "start": 0.2, "end": 0.55},  # Must NOT false-positive on "shit"!
        {"word": "bitches!", "start": 1.0, "end": 1.4},
    ]
    out_wav = tmp_path / "movie_bleeped.wav"
    _, bleeped = chapter_composer.auto_bleep_profanity(src_wav, words, out_wav)

    assert len(bleeped) == 1
    assert bleeped[0]["word"] == "bitches!"
    # Surgical timing [w.start + 0.03, w.end + 0.06]
    assert bleeped[0]["start_sec"] == 1.03
    assert bleeped[0]["end_sec"] == 1.46
    with wave.open(str(out_wav), "rb") as wf:
        assert wf.getnframes() == n_frames
        assert wf.getnchannels() == 2
        assert wf.getframerate() == sr
        pcm = np.frombuffer(wf.readframes(n_frames), dtype=np.int16).astype(np.float32).reshape(-1, 2) / 32768.0

    # Non-profanity and pre-word region stay silent; surgical profanity region carries strong 1000 Hz bleep
    assert float(np.sqrt(np.mean(pcm[int(0.2 * sr) : int(0.55 * sr)] ** 2))) < 1e-5
    assert float(np.sqrt(np.mean(pcm[int(0.97 * sr) : int(1.02 * sr)] ** 2))) < 1e-5
    assert float(np.sqrt(np.mean(pcm[int(1.06 * sr) : int(1.43 * sr)] ** 2))) > 0.10


def test_shift_words_for_freezes() -> None:
    words = [
        {"word": "hey", "start": 5.0, "end": 5.3},
        {"word": "bitches", "start": 20.0, "end": 20.4},
    ]
    freezes = [{"raw_time_sec": 10.0, "freeze_dur_sec": 2.2}]
    shifted = chapter_composer.shift_words_for_freezes(words, freezes)
    assert shifted[0]["start"] == 5.0
    assert shifted[1]["start"] == 22.2
    assert shifted[1]["end"] == 22.6


def test_dialogue_collision_resolver_and_exact_end_cta() -> None:
    script = ChapterCommentaryScript(
        genre="horror_comedy",
        voice_name="en-US-ChristopherNeural",
        voice_rate="+8%",
        chapter_title="Sprite Fright Part 1",
        top_headline="THEY WOKE UP THE FOREST",
        cues=[
            CommentaryCue(
                cue_id=1, start_sec=0.0, max_duration_sec=2.5,
                text="Zero survival instincts on this camping trip already!",
                emphasis_word="SURVIVAL", emotion="smug", pause_emotion="neutral",
            ),
            CommentaryCue(
                cue_id=2, start_sec=16.0, max_duration_sec=2.8,
                text="Bro really dropped his snack wrapper right on a sacred shrine.",
                emphasis_word="SHRINE", emotion="facepalm", pause_emotion="shocked",
                freeze_frame=False,
            ),
            CommentaryCue(
                cue_id=3, start_sec=71.0, max_duration_sec=2.5,
                text="Oh they are cooked—Part 2 is live right now!",
                emphasis_word="COOKED", emotion="hyped", pause_emotion="shocked",
                freeze_frame=False,
            ),
        ],
    )
    busy = [(15.0, 22.0), (73.0, 79.4)]
    open_gaps = [
        {"start_sec": 0.0, "end_sec": 15.0, "duration_sec": 15.0},
        {"start_sec": 22.0, "end_sec": 73.0, "duration_sec": 51.0},
    ]
    resolved = chapter_commentator.resolve_dialogue_collisions_and_cta(
        script,
        chapter_duration=80.0,
        busy_intervals=busy,
        open_gaps=open_gaps,
        part_number=1,
        total_parts=3,
        last_movie_word_end=79.65,
    )
    # Cue 2 either moved cleanly outside [15, 22] or promoted to freeze_frame=True
    c2 = resolved.cues[1]
    assert c2.freeze_frame or chapter_commentator.compute_busy_overlap(c2.start_sec, c2.start_sec + 2.5, busy) <= 0.35
    # Final CTA anchored strictly AFTER final spoken movie word (max(79.88, 79.65 + 0.30) = 79.95)
    last = resolved.cues[-1]
    assert last.start_sec == 79.95
    assert last.freeze_frame is True


def test_clean_1word_pop_captions_strip_periods_and_commas(tmp_path: Path) -> None:
    from pipeline import caption_engine

    timings = [
        {
            "start_sec": 0.0,
            "end_sec": 2.4,
            "text": "We left the green woods, stop them. Look! Wait?",
            "emphasis_word": "STOP",
            "emotion": "shocked",
        }
    ]
    ass_path = tmp_path / "pop.ass"
    caption_engine.build_chapter_pop_captions(
        timings, tmp_path / "missing.wav", ass_path, {}, mock_tts=True, seam_y=None
    )
    ass_text = ass_path.read_text(encoding="utf-8")
    assert qa_gate._find_trailing_punct_caption_tokens(ass_path) == []
    assert "LOOK!" in ass_text
    assert "WAIT?" in ass_text
    assert "THEM." not in ass_text
    assert "WOODS," not in ass_text
    assert "WE LEFT" not in ass_text
    assert "THE GREEN" not in ass_text
    # Reconciled 63.5/36.5 caption band: 90pt font, Alignment=2, MarginV=755 (y=1065..1170)
    assert ",90," in ass_text
    assert ",2,40,40,755,1" in ass_text


def test_sprite_normalization_and_63_5_36_5_layout() -> None:
    # Reconciled 63.5/36.5 full-bleed layout with 50px pop-out headroom
    lay = chapter_composer.compute_vertical_layout(1920, 804)
    assert lay["divider_y"] == 1220
    assert lay["booth_h"] == 700
    assert lay["host_overlay_y"] == 1170
    assert lay["host_overlay_h"] == 750
    assert 200 <= lay["movie_y"] and lay["movie_y"] + lay["movie_h"] <= 1050

    # Chest/shoulder anchor normalization aligns torso cleanly WITHOUT squishing raised paws
    ref = Image.new("RGBA", (640, 640), (0, 0, 0, 0))
    ImageDraw.Draw(ref).rounded_rectangle([160, 120, 480, 640], radius=40, fill=(230, 140, 60, 255))

    shifted = Image.new("RGBA", (640, 640), (0, 0, 0, 0))
    d_sh = ImageDraw.Draw(shifted)
    d_sh.rounded_rectangle([210, 150, 500, 600], radius=40, fill=(230, 140, 60, 255))
    # Raised excited paw sticking up/out above head (must NOT shrink shoulder/torso scale)
    d_sh.ellipse([480, 40, 570, 160], fill=(230, 140, 60, 255))

    norm = host_animator.normalize_sprites_to_reference(
        {"neutral_closed": ref, "hyped_open": shifted}, target_size=460
    )
    bb_ref = host_animator._opaque_bbox(norm["neutral_closed"])
    bb_hy = host_animator._opaque_bbox(norm["hyped_open"])
    assert bb_ref is not None and bb_hy is not None
    assert bb_ref[3] == 460 and bb_hy[3] == 460


def test_hardened_chapter_qa_gate_catches_new_failures(tmp_path: Path) -> None:
    ch = ChapterSlice(part_number=1, total_parts=3, start_sec=0.0, end_sec=80.0, duration_sec=80.0)
    script = chapter_commentator.generate_chapter_commentary(
        tmp_path / "sheet.jpg", ch, "Sprite Fright", {}, mock_llm=True
    )
    timings = [
        {"start_sec": c.start_sec, "end_sec": min(75.0, c.start_sec + 3.0), "emotion": c.emotion, "dialogue_overlap_sec": 0.72, "freeze_frame": False}
        for c in script.cues
    ]
    bad_ass = tmp_path / "bad.ass"
    bad_ass.write_text("Dialogue: 0,0:00:00.00,0:00:00.50,Default,,0,0,0,,{\\c&HFFFFFF&}THEM.\n", encoding="utf-8")

    probe_ok = {
        "streams": [
            {
                "codec_type": "video", "width": 1080, "height": 1920,
                "codec_name": "h264", "pix_fmt": "yuv420p", "r_frame_rate": "30/1", "duration": "84.0",
            },
            {"codec_type": "audio", "duration": "84.0"},
        ]
    }
    exp_meta = {"active_commentary_sec": 39.0, "commentary_share": 0.464, "freeze_count": 1, "max_dialogue_overlap_sec": 0.72}
    with (
        patch("pipeline.ffutil.probe", return_value=probe_ok),
        patch("pipeline.qa_gate._measure_lufs", return_value=-14.1),
        patch("pipeline.qa_gate._black_stats", return_value=(0.0, 0.0)),
    ):
        rep = qa_gate.run_chapter_qa_gate(
            tmp_path / "final.mp4",
            script,
            timings,
            exp_meta,
            tmp_path,
            {},
            strict=False,
            mock_llm=False,
            part_number=1,
            total_parts=3,
            ass_path=bad_ass,
        )
        assert rep["passed"] is False
        joined = " | ".join(rep["failures"]).lower()
        assert "freeze-frame" in joined
        assert "busy_dialogue" in joined
        assert "ending cta" in joined
        assert "trailing '.' or ','" in joined


def test_qa_gate_hard_fails_on_chopped_ending_cta(tmp_path: Path) -> None:
    """QA gate MUST hard fail if last_cue.end_sec > final_video_duration - 0.20."""
    ch = ChapterSlice(part_number=1, total_parts=3, start_sec=0.0, end_sec=80.0, duration_sec=80.0)
    script = chapter_commentator.generate_chapter_commentary(
        tmp_path / "sheet.jpg", ch, "Sprite Fright", {}, mock_llm=True
    )
    timings = [
        {"start_sec": c.start_sec, "end_sec": min(83.92, c.start_sec + 4.1), "emotion": c.emotion}
        for c in script.cues
    ]
    timings[-1]["end_sec"] = 83.92  # Only 0.08s before 84.0s EOF -> CHOPPED!
    probe_ok = {
        "streams": [
            {
                "codec_type": "video", "width": 1080, "height": 1920,
                "codec_name": "h264", "pix_fmt": "yuv420p", "r_frame_rate": "30/1", "duration": "84.0",
            },
            {"codec_type": "audio", "duration": "84.0"},
        ]
    }
    exp_meta = {"active_commentary_sec": 39.0, "commentary_share": 0.464, "freeze_count": 3}
    with (
        patch("pipeline.ffutil.probe", return_value=probe_ok),
        patch("pipeline.qa_gate._measure_lufs", return_value=-14.1),
        patch("pipeline.qa_gate._black_stats", return_value=(0.0, 0.0)),
    ):
        rep = qa_gate.run_chapter_qa_gate(
            tmp_path / "final.mp4", script, timings, exp_meta, tmp_path, {}, strict=False, mock_llm=False
        )
        assert rep["passed"] is False
        assert any("chopped off" in f.lower() for f in rep["failures"])


def test_kiko_v2_ten_emotions_prosody_and_reactive_rig() -> None:
    all_10 = (
        "neutral", "laughing", "shocked", "horrified", "smug",
        "hyped", "sad", "facepalm", "disgusted", "suspicious",
    )
    for idx, em in enumerate(all_10, start=1):
        cue = CommentaryCue(
            cue_id=idx, start_sec=float(idx), max_duration_sec=2.5,
            text=f"Kiko reacts with {em} energy right here!",
            emphasis_word=em.upper(),
            emotion=em,  # type: ignore[arg-type]
            pause_emotion=em,  # type: ignore[arg-type]
        )
        assert cue.emotion == em
        rate, pitch = chapter_commentator.get_emotion_prosody(em)
        assert rate and pitch

    assert chapter_commentator.get_emotion_prosody("laughing") == ("+15%", "+12Hz")
    assert chapter_commentator.get_emotion_prosody("shocked") == ("+16%", "+10Hz")
    assert chapter_commentator.get_emotion_prosody("disgusted") == ("+6%", "-4Hz")
    assert chapter_commentator.get_emotion_prosody("suspicious") == ("+5%", "-3Hz")

    # Reactive rig curves & stuck-facepalm auto-release (>2.2s silent pause)
    assert len(host_animator.ELASTIC_SNAP_SCALE_Y) == 6
    assert max(host_animator.ELASTIC_SNAP_SCALE_Y) == pytest.approx(1.09)
    assert host_animator.get_target_camera_zoom("neutral") == 1.00
    assert host_animator.get_target_camera_zoom("suspicious") == 1.07
    assert host_animator.get_target_camera_zoom("laughing") == 1.10
    assert host_animator.get_target_camera_zoom("disgusted") == 1.14
    assert host_animator.get_target_camera_zoom("neutral", is_freeze=True) == 1.14

    cues = [
        {"start_sec": 0.0, "end_sec": 2.0, "emotion": "facepalm", "pause_emotion": "facepalm"},
        {"start_sec": 8.0, "end_sec": 10.0, "emotion": "shocked", "pause_emotion": "neutral"},
    ]
    em_early, spk_early, _ = host_animator.resolve_frame_emotion_state(3.5, cues)
    assert em_early == "facepalm" and not spk_early
    em_late, spk_late, _ = host_animator.resolve_frame_emotion_state(4.8, cues)
    assert em_late == "shocked" and not spk_late  # Auto-released after >2.2s!


def test_gemini_proxy_burned_in_mmss_timecode() -> None:
    vf = chapter_slicer.build_gemini_proxy_vf(height=480, fps=5, burn_timecode=True, fontfile="/tmp/f.ttf")
    assert "scale=-2:480,fps=5" in vf
    assert "drawtext=" in vf
    assert "gmtime" in vf and "%M\\:%S" in vf
    assert "x=w-tw-12:y=h-th-10" in vf
    assert "box=1" in vf


def test_freeze_cap_prevents_overfreezing_under_dense_dialogue(tmp_path: Path) -> None:
    """Even under heavy dialogue density, total freezes stay <=4 and max overlap stays <=0.45s."""
    cues = [
        CommentaryCue(
            cue_id=i + 1,
            start_sec=round(i * 6.0, 2),
            max_duration_sec=2.5,
            text=f"Hold up look at move {i + 1} right before the chaos hits the campers!",
            emphasis_word="CHAOS",
            emotion=["smug", "shocked", "laughing", "facepalm", "horrified", "hyped", "disgusted", "suspicious"][i % 8],  # type: ignore[arg-type]
            pause_emotion="neutral",
            freeze_frame=(i in (2, 5, 8, 11)),
            freeze_duration_sec=2.0 if i in (2, 5, 8, 11) else 0.0,
        )
        for i in range(12)
    ]
    script = ChapterCommentaryScript(
        genre="horror_comedy",
        voice_name="en-US-ChristopherNeural",
        voice_rate="+8%",
        chapter_title="Sprite Fright Part 1",
        top_headline="THEY WOKE UP THE FOREST",
        cues=cues,
    )
    busy = [(1.0, 4.5), (7.0, 10.5), (13.0, 17.5), (20.0, 25.0), (28.0, 34.0), (38.0, 44.0), (48.0, 54.0), (58.0, 64.0)]
    gaps = [{"start_sec": 4.5, "end_sec": 7.0, "duration_sec": 2.5}, {"start_sec": 25.0, "end_sec": 28.0, "duration_sec": 3.0}, {"start_sec": 34.0, "end_sec": 38.0, "duration_sec": 4.0}]
    resolved = chapter_commentator.resolve_dialogue_collisions_and_cta(
        script, chapter_duration=72.0, busy_intervals=busy, open_gaps=gaps, part_number=1, total_parts=3
    )
    assert sum(1 for c in resolved.cues if c.freeze_frame) <= 4
    chapter_commentator.synthesize_timed_commentary(
        resolved, 72.0, tmp_path, {}, mock_tts=True, busy_intervals=busy
    )
    exp = json.loads((tmp_path / "03_timeline_expansion.json").read_text(encoding="utf-8"))
    assert exp["freeze_count"] <= 6
    assert exp["max_dialogue_overlap_sec"] <= 0.45
    assert exp["cta_end_padding_sec"] >= 0.30
