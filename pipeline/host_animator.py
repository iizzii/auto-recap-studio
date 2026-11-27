"""Pure-Python RMS lip-sync animator for Kiko host booth (1080x690 @ 30fps)."""

from __future__ import annotations

import math
import random
import subprocess
import wave
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from pipeline.video_composer import select_h264_encoder


def _compute_frame_rms(wav_path: Path, n_frames: int, fps: int = 30) -> np.ndarray:
    """Compute 2-frame smoothed normalized RMS loudness per video frame."""
    if not wav_path.exists():
        return np.zeros(n_frames, dtype=np.float32)
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    spf = max(1, round(sr / fps))
    rms = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        seg = pcm[i * spf : (i + 1) * spf]
        if len(seg) > 0:
            rms[i] = float(np.sqrt(np.mean(seg * seg)))
    # 2-frame moving average
    if n_frames > 1:
        rms[1:] = 0.5 * (rms[1:] + rms[:-1])
    return rms


def render_host_booth_video(
    wav_path: Path,
    cue_timings: list[dict[str, Any]],
    genre: str,
    duration_sec: float,
    out_path: Path,
    config: dict[str, Any],
) -> Path:
    """Render 1080x690 30fps host_booth.mp4 by piping raw rgb24 frames to FFmpeg."""
    host_cfg = config.get("host", {})
    fps = int(host_cfg.get("fps", 30))
    sprites_dir = Path(str(host_cfg.get("sprites_dir", "assets/host/sprites")))
    booths_dir = Path(str(host_cfg.get("booths_dir", "assets/host/booths")))
    rms_open = float(host_cfg.get("rms_open_threshold", 0.085))
    rms_half = float(host_cfg.get("rms_half_threshold", 0.028))
    x_ratio = float(host_cfg.get("character_x_ratio", 0.38))

    booth_path = booths_dir / f"{genre}.png"
    if not booth_path.exists():
        booth_path = booths_dir / "comedy.png"
    booth_img = Image.open(booth_path).convert("RGBA").resize((1080, 690))

    sprite_cache: dict[str, Image.Image] = {}
    for p in sprites_dir.glob("*.png"):
        sprite_cache[p.stem] = Image.open(p).convert("RGBA")

    n_frames = max(1, round(duration_sec * fps))
    rms_arr = _compute_frame_rms(wav_path, n_frames, fps=fps)

    # Precompute blink windows
    rng = random.Random(42)
    blink_frames: set[int] = set()
    t_b = rng.uniform(2.8, 4.6)
    while t_b < duration_sec:
        f0 = int(t_b * fps)
        for df in range(4):
            blink_frames.add(f0 + df)
        t_b += rng.uniform(2.8, 4.6)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    enc_flags = select_h264_encoder(config, intermediate=True)
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1080x690", "-r", str(fps),
        "-i", "pipe:0",
        *enc_flags,
        "-pix_fmt", "yuv420p", "-r", str(fps),
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert proc.stdin is not None

    prev_emotion = cue_timings[0]["emotion"] if cue_timings else "neutral"
    pop_frames = 0

    for i in range(n_frames):
        t = i / fps
        rms = float(rms_arr[i])

        # Active emotion
        emotion = cue_timings[0]["emotion"] if cue_timings else "neutral"
        for _idx, c in enumerate(cue_timings):
            if c["start_sec"] <= t <= c["end_sec"]:
                emotion = c["emotion"]
                break
            if t > c["end_sec"]:
                emotion = c.get("pause_emotion", "neutral")

        if emotion != prev_emotion:
            pop_frames = 5
            prev_emotion = emotion

        mouth = "open" if rms >= rms_open else ("half" if rms >= rms_half else "closed")
        if i in blink_frames and mouth in ("closed", "half") and emotion != "laughing" and "blink" in sprite_cache:
            spr = sprite_cache["blink"]
        else:
            spr = sprite_cache.get(f"{emotion}_{mouth}") or sprite_cache.get("neutral_closed")

        frame = booth_img.copy()
        if spr is not None:
            dy_idle = round(3.5 * math.sin(2.0 * math.pi * t / 2.6))
            dy_talk = -int(min(7, rms * 45))
            dy_pop = -4 if pop_frames > 0 else 0
            if pop_frames > 0:
                pop_frames -= 1
            x0 = int(1080 * x_ratio - spr.width / 2)
            y0 = 690 - spr.height + dy_idle + dy_talk + dy_pop
            frame.alpha_composite(spr, (x0, y0))

        proc.stdin.write(frame.convert("RGB").tobytes())

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg host_booth pipe failed with code {proc.returncode}")
    return out_path
