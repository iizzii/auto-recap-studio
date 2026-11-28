"""Pure-Python RMS lip-sync animator for Kiko 3D host booth (1080x970 @ 30fps)."""

from __future__ import annotations

import logging
import math
import random
import subprocess
import wave
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from pipeline.video_composer import select_h264_encoder

log = logging.getLogger(__name__)

_EMOTION_TILT_DEG: dict[str, float] = {
    "neutral": 0.0,
    "laughing": 2.4,
    "hyped": 2.1,
    "smug": -1.9,
    "shocked": -2.2,
    "horrified": -2.5,
    "sad": -1.6,
    "facepalm": 1.5,
}

_GENRE_DESK_PALETTES: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "comedy": ((42, 26, 16), (255, 196, 78)),
    "horror_comedy": ((22, 16, 36), (48, 224, 196)),
    "action_scifi": ((14, 22, 38), (38, 224, 255)),
    "emotional_drama": ((20, 26, 44), (255, 180, 96)),
}


def _compute_frame_rms(wav_path: Path, n_frames: int, fps: int = 30) -> np.ndarray:
    """Compute 2-frame smoothed normalized RMS loudness per video frame."""
    if not wav_path.exists():
        return np.zeros(n_frames, dtype=np.float32)
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        if nch > 1:
            pcm = pcm.reshape(-1, nch).mean(axis=1)
    spf = max(1, round(sr / fps))
    rms = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        seg = pcm[i * spf : (i + 1) * spf]
        if len(seg) > 0:
            rms[i] = float(np.sqrt(np.mean(seg * seg)))
    if n_frames > 1:
        rms[1:] = 0.5 * (rms[1:] + rms[:-1])
    return rms


def _sample_movie_screen_glow(
    movie_path: Path | None,
    n_frames: int,
    fps: int,
    sample_stride: int = 3,
) -> np.ndarray | None:
    """Sample per-frame movie average RGB color (0..255) every few frames for reactive glow."""
    if movie_path is None or not movie_path.exists():
        return None
    cap = cv2.VideoCapture(str(movie_path))
    if not cap.isOpened():
        return None
    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or float(fps)
        colors = np.zeros((n_frames, 3), dtype=np.float32)
        last_rgb = np.array([120.0, 128.0, 145.0], dtype=np.float32)
        for i in range(0, n_frames, max(1, sample_stride)):
            t_sec = i / float(fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, round(t_sec * src_fps))
            ret, frame = cap.read()
            if ret and frame is not None:
                small = cv2.resize(frame, (32, 18), interpolation=cv2.INTER_AREA)
                bgr_mean = small.mean(axis=(0, 1))
                # Slight saturation boost so the 6% top rim light feels alive
                rgb = np.array([bgr_mean[2], bgr_mean[1], bgr_mean[0]], dtype=np.float32)
                mean_l = float(rgb.mean())
                rgb = np.clip(mean_l + (rgb - mean_l) * 1.35, 0.0, 255.0)
                last_rgb = rgb
            colors[i] = last_rgb

        # Linearly interpolate between sampled keyframes
        if sample_stride > 1 and n_frames > 1:
            keys = list(range(0, n_frames, sample_stride))
            if keys[-1] != n_frames - 1:
                colors[n_frames - 1] = last_rgb
                keys.append(n_frames - 1)
            for k0, k1 in zip(keys, keys[1:]):
                span = k1 - k0
                if span > 1:
                    c0, c1 = colors[k0], colors[k1]
                    for j in range(1, span):
                        a = j / float(span)
                        colors[k0 + j] = c0 * (1.0 - a) + c1 * a

        # Temporal smoothing so glow gently breathes with scene lighting
        for i in range(1, n_frames):
            colors[i] = 0.80 * colors[i - 1] + 0.20 * colors[i]
        return colors
    except Exception as exc:
        log.warning("Screen-glow sampling skipped (%s).", exc)
        return None
    finally:
        cap.release()


def _build_desk_and_vignette_strip(
    width: int,
    booth_h: int,
    genre: str,
    strip_top_y: int,
    desk_y: int,
) -> Image.Image:
    """Build foreground studio desk edge + smooth bottom dark vignette for Reels UI legibility."""
    strip_h = max(1, booth_h - strip_top_y)
    desk_rgb, accent_rgb = _GENRE_DESK_PALETTES.get(genre, _GENRE_DESK_PALETTES["comedy"])

    arr = np.zeros((strip_h, width, 4), dtype=np.float32)
    rel_desk = max(0, desk_y - strip_top_y)

    # 1. Soft dark bottom vignette starting from strip_top_y down to bottom (y=1680..1920 on 1080x1920)
    ys = np.arange(strip_h, dtype=np.float32)
    vig_t = np.clip(ys / max(1.0, float(strip_h - 1)), 0.0, 1.0)
    vig_alpha = (vig_t**1.65) * 195.0
    arr[:, :, 0] = 6.0
    arr[:, :, 1] = 7.0
    arr[:, :, 2] = 12.0
    arr[:, :, 3] = vig_alpha[:, None]

    # 2. Foreground studio desk console (seats Kiko behind the desk edge)
    if rel_desk < strip_h:
        dh = strip_h - rel_desk
        dy = np.linspace(0.0, 1.0, dh, dtype=np.float32)[:, None]
        # Subtle wood/matte desk surface darkening smoothly toward bottom
        shade = 1.0 - 0.58 * dy
        desk_r = desk_rgb[0] * shade
        desk_g = desk_rgb[1] * shade
        desk_b = desk_rgb[2] * shade
        desk_a = np.clip(225.0 + 25.0 * dy, 0.0, 250.0)

        # Blend desk layer over vignette
        arr[rel_desk:, :, 0] = desk_r
        arr[rel_desk:, :, 1] = desk_g
        arr[rel_desk:, :, 2] = desk_b
        arr[rel_desk:, :, 3] = np.maximum(arr[rel_desk:, :, 3], desk_a)

        # Soft bevel highlight strip along the top edge of the desk
        rim_h = min(6, dh)
        for ry in range(rim_h):
            w_rim = 1.0 - (ry / float(rim_h))
            idx = rel_desk + ry
            arr[idx, :, 0] = arr[idx, :, 0] * (1.0 - 0.35 * w_rim) + accent_rgb[0] * (0.35 * w_rim)
            arr[idx, :, 1] = arr[idx, :, 1] * (1.0 - 0.35 * w_rim) + accent_rgb[1] * (0.35 * w_rim)
            arr[idx, :, 2] = arr[idx, :, 2] * (1.0 - 0.35 * w_rim) + accent_rgb[2] * (0.35 * w_rim)
            arr[idx, :, 3] = np.maximum(arr[idx, :, 3], 235.0)

        # Subtle soft drop shadow right above desk lip for depth
        sh_h = min(14, rel_desk)
        for sy in range(1, sh_h + 1):
            idx = rel_desk - sy
            sh_a = (1.0 - sy / float(sh_h + 1)) * 75.0
            arr[idx, :, 3] = np.maximum(arr[idx, :, 3], sh_a)

    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGBA")
    # Crisp 2px subtle studio rim accent line across top of desk
    draw = ImageDraw.Draw(img)
    if 0 <= rel_desk < strip_h:
        draw.line([(0, rel_desk), (width, rel_desk)], fill=(*accent_rgb, 145), width=2)
    return img


def _normalize_sprite(img: Image.Image, target_size: int) -> Image.Image:
    """Resize sprite cleanly to target_size x target_size RGBA while preserving aspect ratio."""
    rgba = img.convert("RGBA")
    if rgba.size == (target_size, target_size):
        return rgba
    contained = ImageOps.contain(rgba, (target_size, target_size), method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))
    ox = (target_size - contained.width) // 2
    oy = target_size - contained.height
    canvas.alpha_composite(contained, (ox, oy))
    return canvas


def render_host_booth_video(
    wav_path: Path,
    cue_timings: list[dict[str, Any]],
    genre: str,
    duration_sec: float,
    out_path: Path,
    config: dict[str, Any],
    *,
    movie_path: Path | None = None,
) -> Path:
    """Render 1080x970 30fps Kiko host_booth.mp4 with smooth animation, desk foreground & screen-glow."""
    host_cfg = config.get("host", {})
    fps = int(host_cfg.get("fps", 30))
    booth_w = int(host_cfg.get("booth_width", 1080))
    booth_h = int(host_cfg.get("booth_height", 970))
    sprite_size = int(host_cfg.get("sprite_size", 620))
    sprites_dir = Path(str(host_cfg.get("sprites_dir", "assets/host/sprites")))
    booths_dir = Path(str(host_cfg.get("booths_dir", "assets/host/booths")))
    rms_open = float(host_cfg.get("rms_open_threshold", 0.085))
    rms_half = float(host_cfg.get("rms_half_threshold", 0.028))
    x_ratio = float(host_cfg.get("character_x_ratio", 0.39))
    glow_strength = float(host_cfg.get("screen_glow_strength", 0.06))
    xfade_total = max(1, int(host_cfg.get("emotion_crossfade_frames", 4)))
    max_tilt = float(host_cfg.get("max_head_tilt_deg", 2.5))

    booth_path = booths_dir / f"{genre}.png"
    if not booth_path.exists():
        booth_path = booths_dir / "comedy.png"
    booth_raw = Image.open(booth_path).convert("RGBA")
    booth_img = ImageOps.fit(booth_raw, (booth_w, booth_h), method=Image.Resampling.LANCZOS)

    sprite_cache: dict[str, Image.Image] = {}
    for p in sprites_dir.glob("*.png"):
        sprite_cache[p.stem] = _normalize_sprite(Image.open(p), sprite_size)

    # Pre-render foreground studio desk + smooth bottom vignette (maps to y=1680..1920 of 1080x1920)
    strip_top_y = max(0, int(booth_h * 0.74))
    desk_y = max(strip_top_y + 12, int(booth_h * 0.81))
    desk_vignette = _build_desk_and_vignette_strip(booth_w, booth_h, genre, strip_top_y, desk_y)

    n_frames = max(1, round(duration_sec * fps))
    rms_arr = _compute_frame_rms(wav_path, n_frames, fps=fps)
    glow_colors = (
        _sample_movie_screen_glow(movie_path, n_frames, fps=fps, sample_stride=3)
        if glow_strength > 0.0
        else None
    )

    # Precompute vertical top-down screen-glow falloff mask over upper 56% of booth
    glow_h = int(booth_h * 0.56)
    glow_weights: np.ndarray | None = None
    if glow_colors is not None and glow_h > 0:
        gy = np.linspace(0.0, 1.0, glow_h, dtype=np.float32)
        glow_weights = (((1.0 - gy) ** 1.8) * glow_strength)[:, None, None]

    # Precompute natural blink windows (120ms = ~4 frames every 2.8-4.6s)
    blink_range = host_cfg.get("blink_interval_sec", [2.8, 4.6])
    blink_lo = float(blink_range[0]) if len(blink_range) >= 2 else 2.8
    blink_hi = float(blink_range[1]) if len(blink_range) >= 2 else 4.6
    blink_dur_frames = max(2, round(float(host_cfg.get("blink_duration_ms", 120)) / 1000.0 * fps))

    rng = random.Random(42)
    blink_frames: set[int] = set()
    t_b = rng.uniform(blink_lo, blink_hi)
    while t_b < duration_sec:
        f0 = int(t_b * fps)
        for df in range(blink_dur_frames):
            blink_frames.add(f0 + df)
        t_b += rng.uniform(blink_lo, blink_hi)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    enc_flags = select_h264_encoder(config, intermediate=True)
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{booth_w}x{booth_h}", "-r", str(fps),
        "-i", "pipe:0",
        *enc_flags,
        "-pix_fmt", "yuv420p", "-r", str(fps),
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert proc.stdin is not None

    prev_emotion = cue_timings[0]["emotion"] if cue_timings else "neutral"
    xfade_from_emotion = prev_emotion
    xfade_frames_left = 0
    pop_frames = 0
    cur_tilt = 0.0
    rms_smooth = 0.0
    rot_cache: dict[tuple[str, int], Image.Image] = {}

    def _get_sprite(em: str, mouth: str, is_blink: bool) -> tuple[str, Image.Image | None]:
        if is_blink and mouth in ("closed", "half") and em not in ("laughing", "facepalm") and "blink" in sprite_cache:
            return ("blink", sprite_cache["blink"])
        key = f"{em}_{mouth}"
        if key in sprite_cache:
            return (key, sprite_cache[key])
        for fallback in (f"{em}_closed", f"neutral_{mouth}", "neutral_closed"):
            if fallback in sprite_cache:
                return (fallback, sprite_cache[fallback])
        return ("none", None)

    def _rotated(key: str, spr: Image.Image, deg: float) -> Image.Image:
        q = int(round(deg * 2.0))  # 0.5-degree quantization for high cache hit rate
        if q == 0:
            return spr
        ckey = (key, q)
        cached = rot_cache.get(ckey)
        if cached is not None:
            return cached
        angle = q / 2.0
        rot = spr.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False, center=(spr.width / 2, spr.height * 0.72))
        if len(rot_cache) < 320:
            rot_cache[ckey] = rot
        return rot

    # Anchor Kiko so shoulders sit comfortably behind the foreground desk lip
    base_y = desk_y - int(sprite_size * 0.91)
    base_x = int(booth_w * x_ratio - sprite_size / 2)

    for i in range(n_frames):
        t = i / float(fps)
        rms = float(rms_arr[i])
        rms_smooth = 0.60 * rms_smooth + 0.40 * rms

        # Determine active emotion
        emotion = cue_timings[0]["emotion"] if cue_timings else "neutral"
        for c in cue_timings:
            if c["start_sec"] <= t <= c["end_sec"]:
                emotion = c["emotion"]
                break
            if t > c["end_sec"]:
                emotion = c.get("pause_emotion", "neutral")

        if emotion != prev_emotion:
            xfade_from_emotion = prev_emotion
            xfade_frames_left = xfade_total
            pop_frames = 4
            prev_emotion = emotion

        mouth = "open" if rms >= rms_open else ("half" if rms >= rms_half else "closed")
        is_blink = i in blink_frames

        # Smooth micro head-tilt clamped to ±max_tilt degrees
        target_tilt = _EMOTION_TILT_DEG.get(emotion, 0.0) + 0.45 * math.sin(2.0 * math.pi * t / 4.2)
        target_tilt = max(-max_tilt, min(max_tilt, target_tilt))
        cur_tilt = 0.78 * cur_tilt + 0.22 * target_tilt

        curr_key, curr_spr = _get_sprite(emotion, mouth, is_blink)
        if curr_spr is not None:
            curr_spr = _rotated(curr_key, curr_spr, cur_tilt)

        if xfade_frames_left > 0 and curr_spr is not None:
            prev_key, prev_spr = _get_sprite(xfade_from_emotion, mouth, False)
            if prev_spr is not None and prev_key != curr_key:
                prev_rot = _rotated(prev_key, prev_spr, cur_tilt)
                alpha = 1.0 - (xfade_frames_left / float(xfade_total + 1))
                spr_to_draw = Image.blend(prev_rot, curr_spr, alpha)
            else:
                spr_to_draw = curr_spr
            xfade_frames_left -= 1
        else:
            spr_to_draw = curr_spr

        frame = booth_img.copy()
        if spr_to_draw is not None:
            dy_idle = round(3.8 * math.sin(2.0 * math.pi * t / 2.6))
            dy_talk = -int(min(8, rms_smooth * 50.0))
            dy_pop = -4 if pop_frames > 0 else 0
            if pop_frames > 0:
                pop_frames -= 1
            x0 = max(0, base_x)
            y0 = max(0, base_y + dy_idle + dy_talk + dy_pop)
            frame.alpha_composite(spr_to_draw, (x0, y0))

        # Overlay foreground studio desk edge + smooth bottom Reels vignette
        frame.alpha_composite(desk_vignette, (0, strip_top_y))

        rgb_img = frame.convert("RGB")
        if glow_colors is not None and glow_weights is not None:
            arr = np.array(rgb_img, dtype=np.uint8)
            top = arr[:glow_h].astype(np.float32)
            col = glow_colors[i].reshape(1, 1, 3)
            arr[:glow_h] = np.clip(top * (1.0 - glow_weights) + col * glow_weights, 0, 255).astype(np.uint8)
            proc.stdin.write(arr.tobytes())
        else:
            proc.stdin.write(rgb_img.tobytes())

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg host_booth pipe failed with code {proc.returncode}")
    return out_path
