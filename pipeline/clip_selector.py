"""Vision-verified candidate shot selection and storyboard generation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scenedetect import ContentDetector, SceneManager, open_video

from pipeline import ffutil, timeline, vision_verifier
from pipeline.models import ProjectManifest, Shot

log = logging.getLogger(__name__)


def _extract_proxy(
    source_path: Path,
    start_sec: float,
    dur_sec: float,
    out_path: Path,
    height: int = 320,
) -> Path:
    """Extract a lightweight 320p proxy clip for scene detection and motion scoring."""
    ffutil.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start_sec:.3f}",
            "-i",
            str(source_path),
            "-t",
            f"{dur_sec:.3f}",
            "-vf",
            f"scale=-2:{height}",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "28",
            str(out_path),
        ]
    )
    return out_path


def _score_proxy_segment(
    cap: cv2.VideoCapture,
    rel_start: float,
    rel_end: float,
    sample_fps: float = 2.0,
) -> tuple[float, float]:
    """Return ``(mean_luma, motion_energy)`` for a segment in an open VideoCapture."""
    cap.set(cv2.CAP_PROP_POS_MSEC, rel_start * 1000.0)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step_frames = max(1, int(round(src_fps / max(0.5, sample_fps))))

    lumas: list[float] = []
    diffs: list[float] = []
    prev_gray: np.ndarray | None = None
    cur_t = rel_start
    frame_idx = 0

    while cur_t < rel_end:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step_frames == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            lumas.append(float(np.mean(gray)))
            if prev_gray is not None:
                diffs.append(float(np.mean(cv2.absdiff(gray, prev_gray))))
            prev_gray = gray
        frame_idx += 1
        cur_t = rel_start + (frame_idx / src_fps)

    mean_luma = float(np.mean(lumas)) if lumas else 0.0
    motion = float(np.mean(diffs)) if diffs else 0.0
    return mean_luma, motion


def _sample_three_frames(
    source_path: Path,
    start_sec: float,
    end_sec: float,
) -> list[Image.Image]:
    """Grab 3 representative RGB PIL images (start, mid, end) from *source_path*."""
    cap = cv2.VideoCapture(str(source_path))
    imgs: list[Image.Image] = []
    try:
        span = max(0.1, end_sec - start_sec)
        for frac in (0.15, 0.5, 0.85):
            t_ms = (start_sec + span * frac) * 1000.0
            cap.set(cv2.CAP_PROP_POS_MSEC, t_ms)
            ret, frame = cap.read()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                imgs.append(Image.fromarray(rgb))
    finally:
        cap.release()
    if not imgs:
        imgs = [Image.new("RGB", (320, 180), (20, 20, 20))] * 3
    while len(imgs) < 3:
        imgs.append(imgs[-1].copy())
    return imgs


def _build_storyboard(
    source_path: Path,
    shots: list[Shot],
    out_path: Path,
) -> None:
    """Render a 4-column JPEG storyboard (<350 KB, max 1200px wide) with red borders on flagged shots."""
    cols = 4
    tile_w, tile_h = 280, 175
    rows = max(1, (len(shots) + cols - 1) // cols)
    canvas = Image.new("RGB", (cols * tile_w, rows * tile_h), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)

    cap = cv2.VideoCapture(str(source_path))
    try:
        for idx, shot in enumerate(shots):
            r, c = divmod(idx, cols)
            x0, y0 = c * tile_w, r * tile_h
            mid_ms = ((shot.src_start_sec + shot.src_end_sec) * 0.5) * 1000.0
            cap.set(cv2.CAP_PROP_POS_MSEC, mid_ms)
            ret, frame = cap.read()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                thumb = Image.fromarray(rgb).resize((tile_w - 8, tile_h - 28))
                canvas.paste(thumb, (x0 + 4, y0 + 4))

            mm = int(shot.src_start_sec) // 60
            ss = int(shot.src_start_sec) % 60
            score_str = f"{shot.match_score:.0f}" if shot.match_score is not None else "-"
            label = f"B{shot.beat_id} | S{shot.shot_index + 1} | {mm:02d}:{ss:02d} | {score_str}"
            draw.text((x0 + 6, y0 + tile_h - 20), label, fill=(240, 240, 240))
            if shot.flag in ("low_match", "motion_only"):
                draw.rectangle(
                    [x0 + 1, y0 + 1, x0 + tile_w - 2, y0 + tile_h - 2],
                    outline=(255, 48, 48),
                    width=3,
                )
    finally:
        cap.release()

    if canvas.width > 1200:
        new_h = int(canvas.height * (1200 / canvas.width))
        canvas = canvas.resize((1200, new_h))

    quality = 82
    while quality >= 40:
        canvas.save(out_path, format="JPEG", quality=quality, optimize=True)
        if out_path.stat().st_size <= 350 * 1024:
            break
        quality -= 10


def select_shots(
    manifest: ProjectManifest,
    source_path: Path,
    project_dir: Path,
    work_dir: Path,
    allowed_range: tuple[float, float],
    speech_density: list[float],
    config: dict[str, Any],
    *,
    mock_vision: bool = False,
) -> list[Shot]:
    """Select vision-verified, non-overlapping shots matching exact timeline frames."""
    runtime_cfg = config.get("runtime", {})
    shots_cfg = config.get("shots", {})
    vision_cfg = config.get("vision", {})

    fps = int(runtime_cfg.get("fps", 30))
    proxy_h = int(shots_cfg.get("proxy_height", 320))
    scene_thresh = float(shots_cfg.get("scene_threshold", 27.0))
    min_luma = float(shots_cfg.get("min_luma", 15))
    sample_fps = float(shots_cfg.get("motion_sample_fps", 2))
    candidates_limit = int(shots_cfg.get("candidates_per_beat", 6))
    never_reuse = bool(shots_cfg.get("never_reuse_source_range", True))
    speed_min, speed_max = shots_cfg.get("speed_range", [0.97, 1.10])
    min_match_score = float(vision_cfg.get("min_match_score", 5))
    vision_enabled = bool(vision_cfg.get("enabled", True))

    planned_frames = timeline.plan_timeline(manifest.beats, config)
    proxy_dir = work_dir / "proxies"
    proxy_dir.mkdir(parents=True, exist_ok=True)

    beat_candidates: dict[int, list[dict[str, Any]]] = {}
    all_selected: list[Shot] = []
    used_ranges: list[tuple[float, float]] = []
    min_allowed, max_allowed = allowed_range

    for beat, _shot_frame_list in zip(manifest.beats, planned_frames):
        win_start = max(min_allowed, beat.source_window_start_sec - 5.0)
        win_end = min(max_allowed, beat.source_window_end_sec + 5.0)
        if win_end - win_start < 4.0:
            win_end = min(max_allowed, win_start + 6.0)
        proxy_dur = max(2.0, win_end - win_start)
        proxy_path = proxy_dir / f"beat_{beat.beat_id:03d}.mp4"
        _extract_proxy(source_path, win_start, proxy_dur, proxy_path, height=proxy_h)

        # Scene detection
        video = open_video(str(proxy_path))
        scene_mgr = SceneManager()
        scene_mgr.add_detector(ContentDetector(threshold=scene_thresh))
        scene_mgr.detect_scenes(video)
        scenes = scene_mgr.get_scene_list()

        raw_spans: list[tuple[float, float]] = []
        if scenes:
            for sc_start, sc_end in scenes:
                s0 = sc_start.get_seconds()
                s1 = sc_end.get_seconds()
                if s1 - s0 >= 1.5:
                    raw_spans.append((s0, s1))
        if not raw_spans:
            step = max(2.0, proxy_dur / max(1, candidates_limit))
            t = 0.0
            while t < proxy_dur - 1.0:
                raw_spans.append((t, min(proxy_dur, t + step)))
                t += step

        cap = cv2.VideoCapture(str(proxy_path))
        scored_cands: list[dict[str, Any]] = []
        try:
            for rel_s, rel_e in raw_spans:
                luma, motion = _score_proxy_segment(cap, rel_s, rel_e, sample_fps)
                if luma < min_luma:
                    continue
                abs_s = win_start + rel_s
                abs_e = win_start + rel_e
                # Speech density penalty unless emotional/reveal
                minute_idx = min(len(speech_density) - 1, max(0, int(abs_s // 60))) if speech_density else -1
                density = speech_density[minute_idx] if minute_idx >= 0 else 0.0
                penalty = 0.0 if beat.mood in ("emotional", "reveal") else density * 1.5
                heuristic = motion - penalty
                scored_cands.append(
                    {
                        "start": abs_s,
                        "end": abs_e,
                        "motion": motion,
                        "heuristic": heuristic,
                    }
                )
        finally:
            cap.release()

        if not scored_cands:
            scored_cands.append(
                {
                    "start": win_start,
                    "end": min(win_end, win_start + 4.0),
                    "motion": 1.0,
                    "heuristic": 1.0,
                }
            )

        scored_cands.sort(key=lambda c: c["heuristic"], reverse=True)
        top_cands = scored_cands[:candidates_limit]
        for cand in top_cands:
            cand["frames"] = _sample_three_frames(source_path, cand["start"], cand["end"])
        beat_candidates[beat.beat_id] = top_cands

    # Vision verification
    vision_scores: dict[int, list[dict[str, Any]]] = {}
    if vision_enabled:
        vision_scores = vision_verifier.verify_beats(
            manifest.beats,
            beat_candidates,
            config,
            mock_vision=mock_vision,
        )

    def _overlaps(s: float, e: float) -> bool:
        if not never_reuse:
            return False
        return any(min(e, ue) - max(s, us) > 0.25 for us, ue in used_ranges)

    for beat, _shot_frame_list in zip(manifest.beats, planned_frames):
        cands = beat_candidates[beat.beat_id]
        v_list = vision_scores.get(beat.beat_id, [])
        ranked: list[tuple[float, dict[str, Any], bool]] = []
        for idx, cand in enumerate(cands):
            v_score = float(v_list[idx]["score"]) if idx < len(v_list) else None
            rank_val = v_score if v_score is not None else cand["heuristic"]
            ranked.append((rank_val, cand, v_score is not None))
        ranked.sort(key=lambda x: x[0], reverse=True)

        for s_idx, frames_needed in enumerate(shot_frame_list):
            target_sec = frames_needed / fps
            chosen_cand = ranked[0][1]
            chosen_score = ranked[0][0] if ranked[0][2] else None
            has_vision = ranked[0][2]

            for r_val, cand, is_vis in ranked:
                if not _overlaps(cand["start"], cand["end"]):
                    chosen_cand = cand
                    chosen_score = r_val if is_vis else None
                    has_vision = is_vis
                    break

            c_start = chosen_cand["start"]
            c_end = chosen_cand["end"]
            c_dur = max(0.5, c_end - c_start)
            speed = 1.0
            if c_dur < target_sec:
                speed = max(float(speed_min), min(1.0, c_dur / target_sec))
            else:
                speed = min(float(speed_max), max(1.0, min(1.05, c_dur / target_sec)))

            src_need = target_sec * speed
            src_end = min(max_allowed, c_start + src_need)
            if src_end <= c_start:
                src_end = c_start + 0.5
            used_ranges.append((c_start, src_end))

            flag: str | None = None
            if not has_vision:
                flag = "motion_only"
            elif chosen_score is not None and chosen_score < min_match_score:
                flag = "low_match"

            all_selected.append(
                Shot(
                    beat_id=beat.beat_id,
                    shot_index=s_idx,
                    src_start_sec=round(c_start, 3),
                    src_end_sec=round(src_end, 3),
                    speed_factor=round(speed, 4),
                    frames=frames_needed,
                    match_score=chosen_score,
                    flag=flag,  # type: ignore[arg-type]
                )
            )

    timeline.verify_timeline_frames(manifest.beats, all_selected)
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "04_shot_list.json").write_text(
        json.dumps([s.model_dump() for s in all_selected], indent=2) + "\n",
        encoding="utf-8",
    )
    _build_storyboard(source_path, all_selected, project_dir / "04_storyboard.jpg")
    return all_selected
