"""Scene-snapped chapter slicer and labeled contact-sheet extractor."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
from PIL import Image, ImageDraw
from scenedetect import ContentDetector, SceneManager, open_video

from pipeline.models import ChapterSlice

log = logging.getLogger(__name__)


def _detect_scene_cuts(video_path: Path, start_sec: float, end_sec: float) -> list[float]:
    """Return sorted scene-cut timestamps (seconds) inside [start_sec, end_sec]."""
    cuts: list[float] = []
    try:
        video = open_video(str(video_path))
        sm = SceneManager()
        sm.add_detector(ContentDetector(threshold=27.0))
        sm.detect_scenes(video, show_progress=False)
        for s_time, _ in sm.get_scene_list():
            t = s_time.get_seconds()
            if start_sec < t < end_sec:
                cuts.append(t)
    except Exception as exc:
        log.warning("Scene detection failed (%s); falling back to uniform splits.", exc)
    return sorted(cuts)


def plan_chapters(
    video_path: Path,
    allowed_start: float,
    allowed_end: float,
    total_parts: int,
    config: dict[str, Any],
    ci_synthetic: bool = False,
) -> list[ChapterSlice]:
    """Slice [allowed_start, allowed_end] into scene-snapped ChapterSlice objects."""
    cc_cfg = config.get("chapter_commentary", {})
    target_sec = float(cc_cfg.get("target_chapter_sec", 98.0))
    usable_dur = max(5.0, allowed_end - allowed_start)

    if total_parts <= 0:
        total_parts = 1 if usable_dur <= 135.0 else max(1, round(usable_dur / target_sec))

    if total_parts == 1:
        return [
            ChapterSlice(
                part_number=1,
                total_parts=1,
                start_sec=round(allowed_start, 2),
                end_sec=round(allowed_end, 2),
                duration_sec=round(usable_dur, 2),
            )
        ]

    cuts = [] if ci_synthetic else _detect_scene_cuts(video_path, allowed_start, allowed_end)
    boundaries: list[float] = [allowed_start]

    for k in range(1, total_parts):
        t_ideal = allowed_start + k * (usable_dur / total_parts)
        nearby = [c for c in cuts if abs(c - t_ideal) <= 12.0 and c > boundaries[-1] + 10.0]
        snapped = min(nearby, key=lambda c: abs(c - t_ideal)) if nearby else t_ideal
        boundaries.append(round(snapped, 2))
    boundaries.append(round(allowed_end, 2))

    slices: list[ChapterSlice] = []
    for k in range(total_parts):
        s_t = boundaries[k]
        e_t = boundaries[k + 1]
        slices.append(
            ChapterSlice(
                part_number=k + 1,
                total_parts=total_parts,
                start_sec=s_t,
                end_sec=e_t,
                duration_sec=round(max(1.0, e_t - s_t), 2),
            )
        )
    return slices


def extract_labeled_contact_sheet(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    interval_sec: float,
    out_path: Path,
) -> Path:
    """Extract 1 frame every interval_sec with MM:SS pills into a 4-column JPEG grid."""
    cols = 4
    tile_w, tile_h = 320, 180
    timestamps: list[float] = []
    t = start_sec
    while t < end_sec:
        timestamps.append(t)
        t += max(1.0, interval_sec)
    if not timestamps:
        timestamps = [start_sec]

    rows = max(1, (len(timestamps) + cols - 1) // cols)
    canvas = Image.new("RGB", (cols * tile_w, rows * tile_h), (12, 12, 12))
    draw = ImageDraw.Draw(canvas)

    cap = cv2.VideoCapture(str(video_path))
    try:
        for idx, ts in enumerate(timestamps):
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
            ret, frame = cap.read()
            r, c = divmod(idx, cols)
            x0, y0 = c * tile_w, r * tile_h
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tile = Image.fromarray(rgb).resize((tile_w, tile_h))
                canvas.paste(tile, (x0, y0))
            rel = max(0, round(ts - start_sec))
            lbl = f"{rel // 60:02d}:{rel % 60:02d}"
            draw.rectangle([x0 + 4, y0 + 4, x0 + 62, y0 + 24], fill=(0, 0, 0))
            draw.text((x0 + 8, y0 + 6), lbl, fill=(255, 230, 0))
    finally:
        cap.release()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="JPEG", quality=85)
    return out_path
