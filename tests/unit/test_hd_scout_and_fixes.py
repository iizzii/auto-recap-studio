"""Unit tests for HD CC-BY scout providers, credit-roll detector, em-dash split, and 4:3 full-bleed."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from pipeline import caption_engine, deinterlace, story_scout, video_composer
from pipeline.models import Beat, Shot


def test_peertube_mp4_picker() -> None:
    detail = {
        "files": [{"resolution": {"id": 720}, "fileDownloadUrl": "https://ex/720.mp4"}],
        "streamingPlaylists": [
            {"files": [{"resolution": {"id": 1080}, "fileUrl": "https://ex/1080.mp4"}]}
        ],
    }
    url, h = story_scout._pick_peertube_mp4(detail)
    assert url == "https://ex/1080.mp4"
    assert h == 1080


def test_em_dash_splits_into_separate_captions(tmp_path: Path) -> None:
    beats = [
        Beat(
            beat_id=1,
            narration="Possible—walking away.",
            source_window_start_sec=0.0,
            source_window_end_sec=2.0,
            visual_cue="character walks",
            mood="calm",
            kind="story",
            audio_duration_sec=2.0,
        )
    ]
    cfg = {
        "captions": {
            "mode": "single_word",
            "min_word_display_ms": 110,
            "max_chars_per_line": 16,
            "active_color": "#FFFFFF",
            "inactive_color": "#FFFFFF",
            "accent_rotation": ["#FFFFFF"],
        }
    }
    ass_path, _ = caption_engine.generate_captions(
        "emdash_test", beats, Path("v.wav"), tmp_path, cfg, mock_tts=True
    )
    txt = ass_path.read_text(encoding="utf-8")
    assert "POSSIBLE" in txt
    assert "WALKING" in txt
    assert "POSSIBLE—WALKING" not in txt


def test_4_3_full_bleed_vs_16_9_letterbox(tmp_path: Path) -> None:
    shot = Shot(beat_id=1, shot_index=0, src_start_sec=0.0, src_end_sec=2.0, speed_factor=1.0, frames=60)
    cfg = {
        "runtime": {"fps": 30, "width": 1080, "height": 1920},
        "layout": {
            "auto_full_bleed_max_aspect": 1.45,
            "overscan_trim_pct": 0.0,
            "banner": {"enabled": False},
        },
        "grade": {"enabled": True},
        "encode": {},
    }
    cmds_43: list[list[str]] = []
    with patch("pipeline.ffutil.run", side_effect=lambda c: cmds_43.append(c)):
        video_composer._render_single_shot(
            shot, Path("s.mp4"), None, tmp_path / "out43.mp4", cfg, src_dims=(640, 480)
        )
    assert "split=2" not in " ".join(cmds_43[0])

    cmds_169: list[list[str]] = []
    with patch("pipeline.ffutil.run", side_effect=lambda c: cmds_169.append(c)):
        video_composer._render_single_shot(
            shot, Path("s.mp4"), None, tmp_path / "out169.mp4", cfg, src_dims=(1920, 1080)
        )
    assert "split=2" in " ".join(cmds_169[0])


def test_detect_end_credits_start_white_on_black() -> None:
    credit_frame = np.zeros((90, 160, 3), dtype=np.uint8)
    credit_frame[10:20, 10:150] = (230, 230, 230)  # white text on black
    color_frame = np.full((90, 160, 3), (20, 40, 180), dtype=np.uint8)  # colorful dark scene

    mock_cap = MagicMock()
    # First 2 frames are credits, next 2 are colorful
    mock_cap.read.side_effect = [
        (True, credit_frame),
        (True, credit_frame),
        (True, color_frame),
        (True, color_frame),
    ]
    with patch("cv2.VideoCapture", return_value=mock_cap):
        end_t = deinterlace.detect_end_credits_start(Path("dummy.mp4"), 100.0, max_tail_pct=30.0)
    assert end_t < 98.0
