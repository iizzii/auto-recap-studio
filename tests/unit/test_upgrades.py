"""Unit tests for Upgrades 3 & 4 (foreground crop math, min_luma_std, ending_pad_frames)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from pipeline.clip_selector import _score_proxy_segment
from pipeline.models import Beat
from pipeline.timeline import plan_shots_for_beat
from pipeline.tts_engine import _process_and_pad_beat
from pipeline.video_composer import compute_fg_dimensions


def test_compute_fg_dimensions_ultrawide_239() -> None:
    """2.35:1/2.39:1 (1920x818) scales up to 1080x760 instead of a 460px strip."""
    scaled_w, target_h = compute_fg_dimensions(
        1920, 818, width=1080, min_fg_height_px=760, max_horizontal_crop_pct=42.0
    )
    assert target_h == 760
    assert scaled_w > 1080


def test_compute_fg_dimensions_16_9() -> None:
    """16:9 (1920x1080) scales up to 1080x760 with ~20% horizontal crop."""
    scaled_w, target_h = compute_fg_dimensions(
        1920, 1080, width=1080, min_fg_height_px=760, max_horizontal_crop_pct=42.0
    )
    assert target_h == 760
    crop_pct = (scaled_w - 1080) / scaled_w
    assert 0.18 <= crop_pct <= 0.22


def test_score_proxy_segment_computes_low_luma_std_for_fog() -> None:
    """Flat grey fog frame yields low luma std (< 18.0)."""
    fog_frame = np.full((180, 320, 3), 128, dtype=np.uint8)
    cap = MagicMock()
    cap.get.return_value = 30.0
    cap.read.side_effect = [(True, fog_frame), (True, fog_frame), (False, None)]
    mean_luma, luma_std, _ = _score_proxy_segment(cap, 0.0, 1.0, sample_fps=30.0)
    assert mean_luma > 100.0
    assert luma_std < 18.0


@patch("pipeline.tts_engine.ffutil.duration", return_value=2.0)
@patch("pipeline.tts_engine.ffutil.run")
def test_ending_pad_frames_included_on_last_beat(
    mock_run: MagicMock, mock_dur: MagicMock, tmp_path
) -> None:
    """Final beat includes ending_pad_frames (15) and timeline preserves exact sum."""
    cfg = {
        "runtime": {"fps": 30},
        "tts": {"beat_gap_frames": 2, "ending_pad_frames": 15},
    }
    raw = tmp_path / "raw.wav"
    out = tmp_path / "beat.wav"
    frames, sec = _process_and_pad_beat(raw, out, cfg, is_last=True)
    # 2.0s * 30 = 60 speech frames + 15 ending_pad_frames = 75 frames
    assert frames == 75
    assert abs(sec - 2.5) < 1e-6

    beat = Beat(
        beat_id=1,
        narration="Final reveal.",
        source_window_start_sec=0.0,
        source_window_end_sec=5.0,
        visual_cue="dragon scar",
        mood="reveal",
        kind="story",
        audio_frames=frames,
        audio_duration_sec=sec,
    )
    assert sum(plan_shots_for_beat(beat, fps=30)) == frames
