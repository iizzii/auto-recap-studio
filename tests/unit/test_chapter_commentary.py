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
    with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False), pytest.raises(
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
    f16_9 = chapter_composer._build_movie_stage_filter(1920, 1080, "")
    assert "crop=1080:742" in f16_9
    assert "boxblur" not in f16_9  # 16:9 fills edge-to-edge cleanly within <=22% trim

    fscope = chapter_composer._build_movie_stage_filter(2560, 1080, "")
    assert "boxblur" in fscope  # Ultra-wide caps horizontal trim at 22% with blurred ambient fill
    assert "black" not in fscope  # Zero black bars


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
        assert (tmp_path / "qa_report.json").exists()

        # Inject placeholder text -> strict QA gate must exit code 7
        script.cues[0].text = "Watch what happens right here in scene 2!"
        with pytest.raises(SystemExit) as exc_info:
            qa_gate.run_chapter_qa_gate(
                tmp_path / "final.mp4", script, timings, exp_meta, tmp_path, {}, strict=True, mock_llm=False
            )
        assert exc_info.value.code == 7
