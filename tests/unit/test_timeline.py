"""Property-based and unit tests for pipeline.timeline."""

from __future__ import annotations

from hypothesis import given, strategies as st

from pipeline.models import Beat
from pipeline.timeline import plan_shots_for_beat, plan_timeline


def _make_beat(beat_id: int, audio_frames: int) -> Beat:
    return Beat(
        beat_id=beat_id,
        narration="Sample narration text for timeline testing.",
        source_window_start_sec=0.0,
        source_window_end_sec=5.0,
        visual_cue="character runs",
        mood="action",
        kind="story",
        audio_frames=audio_frames,
        audio_duration_sec=audio_frames / 30.0,
    )


@given(audio_frames=st.integers(min_value=30, max_value=420))
def test_single_beat_exact_frame_sum(audio_frames: int) -> None:
    """Across 1.0s-14.0s beats (30-420 frames), sum(shot_frames) == beat.audio_frames."""
    beat = _make_beat(1, audio_frames)
    shots = plan_shots_for_beat(beat, fps=30)
    assert sum(shots) == audio_frames
    assert all(f >= 1 for f in shots)
    if audio_frames >= 60:
        assert all(f <= 135 for f in shots)  # hard_max_shot_sec=4.5 -> 135 frames


@given(
    frame_counts=st.lists(
        st.integers(min_value=45, max_value=300),
        min_size=1,
        max_size=26,
    )
)
def test_multi_beat_timeline_zero_drift(frame_counts: list[int]) -> None:
    """Across up to 26 beats, total shot frames equal total beat audio frames."""
    beats = [_make_beat(i + 1, f) for i, f in enumerate(frame_counts)]
    cfg = {
        "runtime": {"fps": 30},
        "shots": {
            "min_shot_sec": 2.0,
            "max_shot_sec": 4.0,
            "hard_max_shot_sec": 4.5,
            "target_shot_sec": 2.8,
        },
    }
    planned = plan_timeline(beats, cfg)
    assert len(planned) == len(beats)
    for beat, shot_frames in zip(beats, planned):
        assert sum(shot_frames) == beat.audio_frames
    assert sum(sum(s) for s in planned) == sum(frame_counts)
