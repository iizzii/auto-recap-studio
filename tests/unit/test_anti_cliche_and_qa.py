"""Unit tests for anti-cliche commentary rules, QA shot checks, and series range partitioning."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from main import compute_series_ranges
from pipeline import qa_gate
from pipeline.models import Beat, ProjectManifest, Shot
from pipeline.script_validator import validate_manifest


def _beat(bid: int, narration: str, kind: str = "story") -> Beat:
    return Beat(
        beat_id=bid,
        narration=narration,
        source_window_start_sec=float((bid - 1) * 10),
        source_window_end_sec=float((bid - 1) * 10 + 8),
        visual_cue="character moves",
        mood="action",
        kind=kind,  # type: ignore[arg-type]
    )


def test_script_validator_cliche_grammar_spacing_and_repeat() -> None:
    cfg = {
        "script": {
            "min_words": 20,
            "max_words": 200,
            "min_beats": 4,
            "max_beats": 8,
            "beat_min_words": 4,
            "beat_max_words": 20,
            "min_commentary_share": 0.12,
            "max_commentary_share": 0.25,
            "min_story_beats_between_commentary": 2,
        }
    }
    beats = [
        _beat(1, "Suddenly a angry beast officially enters the forest.", "story"),
        _beat(2, "Big mistake.", "commentary"),
        _beat(3, "Pure madness.", "commentary"),
        _beat(4, "The creature officially leaves the forest.", "story"),
    ]
    manifest = ProjectManifest(
        project_slug="test_proj",
        style="commentary",
        part_number=0,
        title="A beast enters the forest #shorts",
        banner_text="BEAST IN THE FOREST",
        description="Test.",
        hashtags=["#shorts"],
        source_title="Test",
        attribution_block="Test",
        beats=beats,
    )
    errors, _ = validate_manifest(manifest, cfg, (0.0, 45.0))
    assert any("big mistake" in e.lower() for e in errors)
    assert any("pure madness" in e.lower() for e in errors)
    assert any("grammar error" in e.lower() for e in errors)
    assert any("too close" in e.lower() for e in errors)


def test_qa_gate_rejects_overlong_and_overlapping_shots(tmp_path: Path) -> None:
    shots = [
        Shot(beat_id=1, shot_index=0, src_start_sec=10.0, src_end_sec=14.0, speed_factor=1.0, frames=120),  # 4.0s > 3.8s
        Shot(beat_id=2, shot_index=0, src_start_sec=12.0, src_end_sec=14.0, speed_factor=1.0, frames=60),   # overlaps
    ]
    cfg = {
        "runtime": {"fps": 30, "min_duration_sec": 1, "max_duration_sec": 175},
        "shots": {"hard_max_shot_sec": 3.5, "never_reuse_source_range": True},
        "layout": {"auto_deinterlace": False},
        "qa": {"av_sync_tolerance_sec": 0.5, "max_silence_gap_ms": 500},
    }
    with (
        patch("pipeline.ffutil.probe", return_value={"streams": [{"codec_type": "video", "r_frame_rate": "30/1"}]}),
        patch("pipeline.ffutil.duration", return_value=6.0),
        patch("pipeline.qa_gate._measure_lufs", return_value=-14.0),
        patch("pipeline.qa_gate._max_silence_ms", return_value=100.0),
        patch("pipeline.qa_gate._max_black_sec", return_value=0.0),
        pytest.raises(SystemExit),
    ):
        qa_gate.run_qa_gate(tmp_path / "f.mp4", shots, [], 0.99, tmp_path, cfg)


def test_compute_series_ranges_chronological() -> None:
    ranges = compute_series_ranges(20.0, 320.0, 3)
    assert len(ranges) == 3
    assert ranges[0] == (20.0, 120.0)
    assert ranges[1] == (120.0, 220.0)
    assert ranges[2] == (220.0, 320.0)
