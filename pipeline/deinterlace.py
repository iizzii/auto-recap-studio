"""Auto-detect interlacing/combing and overscan in vintage video sources."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pipeline import ffutil

log = logging.getLogger(__name__)


def detect_combing(source_path: Path, sample_timestamps: list[float], threshold: float = 1.35) -> bool:
    """Return True if the video appears interlaced/combed.

    Uses two signals:
    1. ffprobe ``field_order`` != progressive.
    2. Pixel-level combing ratio: median(mean(|row[i]-row[i+1]|) / mean(|row[i]-row[i+2]|)) > threshold
       measured in high-motion regions of sampled frames.
    """
    # Signal 1: ffprobe field_order
    try:
        info = ffutil.probe(source_path)
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "video":
                field_order = stream.get("field_order", "progressive")
                if field_order in {"tt", "bb", "tb", "bt"}:
                    log.info("detect_combing: field_order=%s → interlaced", field_order)
                    return True
                break
    except Exception as exc:
        log.warning("detect_combing: ffprobe failed: %s", exc)

    # Signal 2: pixel combing ratio
    cap = cv2.VideoCapture(str(source_path))
    ratios: list[float] = []
    try:
        for ts in sample_timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            # Only look at rows with high motion (top 30% of frame by std)
            row_stds = np.std(gray, axis=1)
            thresh_std = np.percentile(row_stds, 70)
            active_rows = np.where(row_stds > thresh_std)[0]
            if len(active_rows) < 4:
                continue
            # Combing: adjacent-row diff vs skip-row diff
            adj_diffs = []
            skip_diffs = []
            for r in active_rows:
                if r + 2 >= gray.shape[0]:
                    continue
                adj = float(np.mean(np.abs(gray[r] - gray[r + 1])))
                skip = float(np.mean(np.abs(gray[r] - gray[r + 2])))
                if skip > 0.5:
                    adj_diffs.append(adj)
                    skip_diffs.append(skip)
            if adj_diffs and skip_diffs:
                ratio = float(np.median(np.array(adj_diffs) / (np.array(skip_diffs) + 1e-6)))
                ratios.append(ratio)
    finally:
        cap.release()

    if ratios:
        median_ratio = float(np.median(ratios))
        log.info("detect_combing: median combing ratio=%.3f (threshold=%.2f)", median_ratio, threshold)
        return median_ratio > threshold
    return False


def detect_end_credits_start(
    source_path: Path,
    duration_sec: float,
    max_tail_pct: float = 20.0,
    fallback_tail_pct: float = 3.0,
) -> float:
    """Walk backwards from the end to find where end-credits begin.

    Returns the timestamp (seconds) where allowed content ends.
    A frame is classified as dark end-credits if >= 75% of luma pixels < 25.
    Stops at 2 consecutive non-credit frames. Falls back to
    ``duration_sec * (1 - fallback_tail_pct/100)`` if no credits detected.
    """
    max_walk_sec = duration_sec * max_tail_pct / 100.0
    step_sec = 4.0
    cap = cv2.VideoCapture(str(source_path))
    consecutive_non_credit = 0
    t = duration_sec - 2.0
    end_t = t
    try:
        while t > duration_sec - max_walk_sec:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ret, frame = cap.read()
            if not ret:
                t -= step_sec
                continue
            small = cv2.resize(frame, (160, 90))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            dark_ratio_25 = float(np.mean(gray < 25))
            dark_ratio_35 = float(np.mean(gray < 35))
            mean_sat = float(np.mean(hsv[:, :, 1]))
            is_credit_frame = (dark_ratio_25 >= 0.75) or (dark_ratio_35 >= 0.60 and mean_sat < 20.0)
            if is_credit_frame:
                consecutive_non_credit = 0
                t -= step_sec
            else:
                consecutive_non_credit += 1
                if consecutive_non_credit >= 2:
                    end_t = t + 2.0 * step_sec - 3.0
                    log.info("detect_end_credits_start: credits start ~%.1fs (%.1f%% from end)", end_t, (duration_sec - end_t) / duration_sec * 100)
                    return max(end_t, duration_sec * 0.5)
                t -= step_sec
    finally:
        cap.release()

    fallback = duration_sec * (1.0 - fallback_tail_pct / 100.0)
    log.info("detect_end_credits_start: no credits found, fallback=%.1fs", fallback)
    return fallback


def build_deinterlace_overscan_filter(
    config: dict[str, Any],
    deinterlace_applied: bool,
) -> str:
    """Build the FFmpeg vf filter prefix for deinterlacing and overscan cropping.

    Returns an empty string if neither is active.
    """
    parts: list[str] = []
    layout = config.get("layout", {})

    if deinterlace_applied:
        parts.append("bwdif=mode=0:parity=-1:deint=0")

    p = float(layout.get("overscan_trim_pct", 0.0)) / 100.0
    if p > 0.0:
        # crop=iw*(1-2*p):ih*(1-2*p):iw*p:ih*p
        # Expressed as fractions so FFmpeg evaluates at runtime with actual w/h
        parts.append(
            f"crop=iw*{1.0 - 2*p:.4f}:ih*{1.0 - 2*p:.4f}:iw*{p:.4f}:ih*{p:.4f}"
        )

    return ",".join(parts)
