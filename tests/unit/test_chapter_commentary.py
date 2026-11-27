"""Unit tests for chapter_commentary mode + Kiko host animator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from pipeline import chapter_commentator, chapter_composer, chapter_slicer, host_animator
from pipeline.models import ChapterSlice, RightsRecord


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
        )
        assert mock_run.called
