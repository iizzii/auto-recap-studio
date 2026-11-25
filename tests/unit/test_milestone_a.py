"""Unit tests for Milestone A: deinterlace, end-credits, monotonic cuts, anti-template, story BGM."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from pipeline import audio_mixer, deinterlace
from pipeline.models import Beat, ProjectManifest
from pipeline.script_validator import validate_manifest


def _beat(bid: int, mood: str = "action", narration: str = "Suddenly the hero leaps across the burning bridge.") -> Beat:
    return Beat(
        beat_id=bid,
        narration=narration,
        source_window_start_sec=float(bid * 10),
        source_window_end_sec=float(bid * 10 + 8),
        visual_cue="hero leaps over bridge",
        mood=mood,  # type: ignore[arg-type]
        kind="commentary" if bid == 2 else "story",
    )


def test_build_deinterlace_overscan_filter() -> None:
    cfg = {"layout": {"overscan_trim_pct": 3.0}}
    flt = deinterlace.build_deinterlace_overscan_filter(cfg, deinterlace_applied=True)
    assert flt.startswith("bwdif=mode=0:parity=-1:deint=0,crop=iw*0.9400:ih*0.9400")

    flt_no_deint = deinterlace.build_deinterlace_overscan_filter(cfg, deinterlace_applied=False)
    assert flt_no_deint.startswith("crop=iw*0.9400")

    flt_none = deinterlace.build_deinterlace_overscan_filter({"layout": {"overscan_trim_pct": 0.0}}, False)
    assert flt_none == ""


def test_detect_combing_ffprobe_interlaced() -> None:
    fake_probe = {"streams": [{"codec_type": "video", "field_order": "tt"}]}
    with patch("pipeline.ffutil.probe", return_value=fake_probe):
        assert deinterlace.detect_combing(Path("dummy.mp4"), [1.0, 2.0]) is True


def test_banned_stock_templates_rejected() -> None:
    cfg = {
        "script": {
            "min_words": 20,
            "max_words": 150,
            "min_beats": 4,
            "max_beats": 8,
            "beat_min_words": 5,
            "beat_max_words": 25,
            "min_commentary_share": 0.25,
        }
    }
    beats = [
        _beat(1),
        _beat(2, narration="Here's the detail most people miss in this entire scene."),
        _beat(3),
        _beat(4),
    ]
    manifest = ProjectManifest(
        project_slug="test_proj",
        style="commentary",
        part_number=0,
        title="A hero leaps across a bridge #shorts",
        banner_text="THE BRIDGE LEAP",
        description="Short test.",
        hashtags=["#shorts"],
        source_title="Test",
        attribution_block="Test",
        beats=beats,
    )
    errors, _ = validate_manifest(manifest, cfg, (0.0, 60.0))
    assert any("banned film-essay cliche" in e for e in errors)


def test_infer_bgm_mood_and_numpy_fallback(tmp_path: Path) -> None:
    assert audio_mixer.infer_bgm_mood([_beat(1, "funny"), _beat(2, "funny"), _beat(3, "action")]) == "playful"
    assert audio_mixer.infer_bgm_mood([_beat(1, "emotional"), _beat(2, "calm"), _beat(3, "tense")]) == "emotional"
    assert audio_mixer.infer_bgm_mood([_beat(1, "tense"), _beat(2, "cliffhanger")]) == "suspense"

    out_wav = tmp_path / "bgm_test.wav"
    res = audio_mixer._synth_bgm_numpy(2.0, out_wav, "suspense")
    assert res.exists()
    assert res.stat().st_size > 1000


def test_should_deinterlace_vintage_year() -> None:
    from pipeline import video_composer
    cfg = {"layout": {"auto_deinterlace": True, "vintage_deinterlace_year": 1985}}
    assert video_composer.should_deinterlace(Path("dummy.mp4"), [1.0], cfg, work_year=1941) is True


def test_script_validator_bans_and_ending_reach() -> None:
    cfg = {
        "audio": {"source_audio_mode": "off"},
        "script": {
            "min_words": 20,
            "max_words": 150,
            "min_beats": 4,
            "max_beats": 8,
            "beat_min_words": 5,
            "beat_max_words": 25,
            "min_commentary_share": 0.25,
        },
    }
    beats = [
        _beat(1, narration="A girl enjoys a quiet morning in the sunny forest."),
        _beat(2, narration="The color saturation and lighting design look amazing."),
        _beat(3, narration="Loud sound effects echo across the valley."),
        _beat(4, narration="The hero walks away."),
    ]
    manifest = ProjectManifest(
        project_slug="test_proj",
        style="commentary",
        part_number=0,
        title="A hero leaps across a bridge #shorts",
        banner_text="THE BRIDGE LEAP",
        description="Short test.",
        hashtags=["#shorts"],
        source_title="Test",
        attribution_block="Test",
        beats=beats,
    )
    errors, _ = validate_manifest(manifest, cfg, (0.0, 100.0))
    assert any("slow/generic setup" in e for e in errors)
    assert any("color saturation" in e for e in errors)
    assert any("source audio" in e for e in errors)
    assert any("stops too early" in e for e in errors)
