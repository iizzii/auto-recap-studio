"""Unit tests for the bright/fast/1-word viral cartoon aesthetic overhaul."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from pipeline import audio_mixer, caption_engine, video_composer
from pipeline.models import Beat, Shot


def test_banner_off_and_on_video_composer(tmp_path: Path) -> None:
    shot = Shot(
        beat_id=1,
        shot_index=0,
        src_start_sec=0.0,
        src_end_sec=2.0,
        speed_factor=1.0,
        frames=60,
    )
    out_mp4 = tmp_path / "shot.mp4"
    cfg_off = {
        "runtime": {"fps": 30, "width": 1080, "height": 1920},
        "layout": {
            "bg_blur": "boxblur=20:5",
            "bg_brightness": -0.04,
            "bg_saturation": 1.25,
            "min_fg_height_px": 1020,
            "max_horizontal_crop_pct": 45,
            "overscan_trim_pct": 3.0,
            "banner": {"enabled": False, "y": 200},
        },
        "grade": {"enabled": True, "eq": "contrast=1.08:saturation=1.22:brightness=0.02", "unsharp": "5:5:0.7:5:5:0.0"},
        "encode": {"intermediate_preset": "ultrafast", "intermediate_crf": 16},
    }
    captured_cmds: list[list[str]] = []
    with patch("pipeline.ffutil.run", side_effect=lambda cmd: captured_cmds.append(cmd)):
        video_composer._render_single_shot(
            shot, Path("src.mp4"), None, out_mp4, cfg_off, src_dims=(640, 480), deinterlace_applied=True
        )
    cmd_str = " ".join(captured_cmds[0])
    assert "bwdif=mode=0:parity=-1:deint=0" in cmd_str
    assert "saturation=1.25" in cmd_str
    assert "banner.png" not in cmd_str

    # Also test banner_png provided (banner.enabled=True)
    captured_cmds.clear()
    with patch("pipeline.ffutil.run", side_effect=lambda cmd: captured_cmds.append(cmd)):
        video_composer._render_single_shot(
            shot, Path("src.mp4"), Path("banner.png"), out_mp4, cfg_off, src_dims=(640, 480), deinterlace_applied=False
        )
    assert "banner.png" in " ".join(captured_cmds[0])


def test_single_word_white_captions(tmp_path: Path) -> None:
    beats = [
        Beat(
            beat_id=1,
            narration="A giant rabbit smashed the tiny trap.",
            source_window_start_sec=0.0,
            source_window_end_sec=3.0,
            visual_cue="rabbit smashes trap",
            mood="action",
            kind="story",
            audio_duration_sec=2.5,
        )
    ]
    cfg = {
        "captions": {
            "mode": "single_word",
            "min_word_display_ms": 110,
            "max_chars_per_line": 16,
            "font_size": 88,
            "outline_px": 6,
            "shadow_px": 3,
            "active_color": "#FFFFFF",
            "inactive_color": "#FFFFFF",
            "alignment": 2,
            "margin_v": 500,
            "accent_rotation": ["#FFFFFF"],
        }
    }
    ass_path, ratio = caption_engine.generate_captions(
        "test_slug", beats, Path("voice.wav"), tmp_path, cfg, mock_tts=True
    )
    txt = ass_path.read_text(encoding="utf-8")
    assert ratio >= 0.95
    assert "&H00FFFFFF" in txt
    assert "\\k" not in txt
    assert "RABBIT" in txt


def test_audio_mixer_no_whoosh(tmp_path: Path) -> None:
    beats = [
        Beat(
            beat_id=1,
            narration="Hero leaps.",
            source_window_start_sec=0.0,
            source_window_end_sec=2.0,
            visual_cue="hero leaps",
            mood="action",
            kind="story",
            audio_duration_sec=1.5,
        ),
        Beat(
            beat_id=2,
            narration="Bad idea.",
            source_window_start_sec=2.0,
            source_window_end_sec=4.0,
            visual_cue="hero falls",
            mood="funny",
            kind="commentary",
            audio_duration_sec=1.5,
        ),
    ]
    cfg = {
        "audio": {
            "bgm_db_below_voice": -15,
            "sfx_on_transitions": False,
            "target_lufs": -14,
            "true_peak_db": -1.0,
            "sample_rate": 48000,
            "aac_bitrate": "192k",
        }
    }
    captured: list[list[str]] = []
    with (
        patch("pipeline.ffutil.duration", return_value=3.0),
        patch("pipeline.ffutil.run", side_effect=lambda cmd: captured.append(cmd)),
    ):
        audio_mixer.mix_audio(Path("voice.wav"), beats, tmp_path, cfg, allow_network=False)
    cmd_str = " ".join(captured[-1])
    assert "whoosh.wav" not in cmd_str
    assert "amix=inputs=2" in cmd_str
