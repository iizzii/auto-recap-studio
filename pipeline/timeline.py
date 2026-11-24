"""Integer-frame timeline arithmetic at 30 fps with zero cumulative drift."""

from __future__ import annotations

import math
from typing import Any

from pipeline.models import Beat, Shot


def plan_shots_for_beat(
    beat: Beat,
    fps: int = 30,
    min_shot_sec: float = 2.0,
    max_shot_sec: float = 4.0,
    hard_max_shot_sec: float = 4.5,
    target_shot_sec: float = 2.8,
) -> list[int]:
    """Split ``beat.audio_frames`` into integer frame counts for shots.

    Guarantees ``sum(frames_list) == beat.audio_frames``.
    """
    if beat.audio_frames is None or beat.audio_frames <= 0:
        raise ValueError(f"Beat {beat.beat_id} has invalid audio_frames: {beat.audio_frames}")

    total_frames = beat.audio_frames
    min_f = max(1, round(min_shot_sec * fps))
    max_f = max(min_f, round(max_shot_sec * fps))
    hard_max_f = max(max_f, round(hard_max_shot_sec * fps))
    target_f = max(1, round(target_shot_sec * fps))

    # Very short beats (< min_shot_sec): single shot
    if total_frames <= min_f:
        return [total_frames]

    # Choose shot count n near target_shot_sec while respecting [min_f, hard_max_f]
    n = max(1, round(total_frames / target_f))

    # Ensure each shot is at least min_f (reduce n if too many shots)
    while n > 1 and (total_frames // n) < min_f:
        n -= 1

    # Ensure each shot is at most hard_max_f (increase n if too few shots)
    while math.ceil(total_frames / n) > hard_max_f:
        n += 1

    # Prefer <= max_f if possible without violating min_f
    while math.ceil(total_frames / n) > max_f and (total_frames // (n + 1)) >= min_f:
        n += 1

    base = total_frames // n
    rem = total_frames % n
    counts = [base + (1 if i < rem else 0) for i in range(n)]
    assert sum(counts) == total_frames
    return counts


def plan_timeline(
    beats: list[Beat],
    config: dict[str, Any],
) -> list[list[int]]:
    """Plan integer frame counts for every beat in *beats*."""
    runtime_cfg = config.get("runtime", {})
    shots_cfg = config.get("shots", {})

    fps = int(runtime_cfg.get("fps", 30))
    min_shot_sec = float(shots_cfg.get("min_shot_sec", 2.0))
    max_shot_sec = float(shots_cfg.get("max_shot_sec", 4.0))
    hard_max_shot_sec = float(shots_cfg.get("hard_max_shot_sec", 4.5))
    target_shot_sec = float(shots_cfg.get("target_shot_sec", 2.8))

    planned: list[list[int]] = []
    for beat in beats:
        counts = plan_shots_for_beat(
            beat,
            fps=fps,
            min_shot_sec=min_shot_sec,
            max_shot_sec=max_shot_sec,
            hard_max_shot_sec=hard_max_shot_sec,
            target_shot_sec=target_shot_sec,
        )
        planned.append(counts)

    total_beat_frames = sum(b.audio_frames or 0 for b in beats)
    total_shot_frames = sum(sum(c) for c in planned)
    assert total_beat_frames == total_shot_frames
    return planned


def verify_timeline_frames(beats: list[Beat], shots: list[Shot]) -> None:
    """Assert that ``sum(shot.frames) == sum(beat.audio_frames)`` per beat and overall."""
    beat_map = {b.beat_id: b for b in beats}
    shots_by_beat: dict[int, int] = {}
    for shot in shots:
        shots_by_beat[shot.beat_id] = shots_by_beat.get(shot.beat_id, 0) + shot.frames

    for beat_id, beat in beat_map.items():
        expected = beat.audio_frames or 0
        actual = shots_by_beat.get(beat_id, 0)
        if actual != expected:
            raise AssertionError(
                f"Beat {beat_id} frame mismatch: shots={actual} != audio={expected}"
            )
