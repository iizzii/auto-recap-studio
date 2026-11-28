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


def _apply_booth_vignette(booth_img: Image.Image, center_x_ratio: float = 0.39) -> Image.Image:
    """Apply a rich radial cinema vignette to the studio booth so 3D Kiko pops cleanly off the wall."""
    arr = np.array(booth_img.convert("RGBA"), dtype=np.float32)
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx = w * center_x_ratio
    cy = h * 0.40
    dx = (xx - cx) / max(1.0, w * 0.62)
    dy = (yy - cy) / max(1.0, h * 0.62)
    dist = np.sqrt(dx * dx + dy * dy)
    # Center stays vibrant (1.03x), outer corners smoothly darken to 0.54x
    vig = np.clip(1.03 - 0.48 * np.clip(dist - 0.25, 0.0, 1.25) ** 1.35, 0.50, 1.03)
    arr[:, :, 0] *= vig
    arr[:, :, 1] *= vig
    arr[:, :, 2] *= vig
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGBA")


def _build_desk_and_vignette_strip(
    width: int,
    booth_h: int,
    genre: str,
    strip_top_y: int,
    desk_y: int,
    *,
    char_x_ratio: float = 0.39,
) -> Image.Image:
    """Build 3D-shaded foreground podcast desk (bevel rim, contact shadow, mic base) + bottom vignette."""
    strip_h = max(1, booth_h - strip_top_y)
    desk_rgb, accent_rgb = _GENRE_DESK_PALETTES.get(genre, _GENRE_DESK_PALETTES["comedy"])

    arr = np.zeros((strip_h, width, 4), dtype=np.float32)
    rel_desk = max(0, desk_y - strip_top_y)

    # 1. Soft dark bottom vignette starting from strip_top_y down to bottom for Reels UI legibility
    ys = np.arange(strip_h, dtype=np.float32)
    vig_t = np.clip(ys / max(1.0, float(strip_h - 1)), 0.0, 1.0)
    vig_alpha = (vig_t**1.65) * 190.0
    arr[:, :, 0] = 6.0
    arr[:, :, 1] = 7.0
    arr[:, :, 2] = 12.0
    arr[:, :, 3] = vig_alpha[:, None]

    # 2. 3D-Shaded Foreground Podcast Desk overlapping Kiko's lower torso
    if rel_desk < strip_h:
        dh = strip_h - rel_desk
        top_surface_h = min(48, max(20, dh // 5))
        bevel_h = min(8, max(4, dh - top_surface_h))
        front_start = min(strip_h, rel_desk + top_surface_h + bevel_h)

        # 2a. Perspective tabletop surface with subtle horizontal walnut/slate grain
        ty = np.linspace(0.0, 1.0, top_surface_h, dtype=np.float32)[:, None]
        tx = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
        # Subtle warm studio overhead specular sheen + subtle wood-grain ripples
        center_sheen = 1.0 + 0.24 * np.exp(-((tx - char_x_ratio) ** 2) / 0.14)
        grain = 1.0 + 0.04 * np.sin(ty * 42.0 + 1.8 * np.sin(tx * 14.0))
        top_shade = (0.72 + 0.42 * ty) * center_sheen * grain
        arr[rel_desk : rel_desk + top_surface_h, :, 0] = np.clip(desk_rgb[0] * top_shade * 1.15, 0, 255)
        arr[rel_desk : rel_desk + top_surface_h, :, 1] = np.clip(desk_rgb[1] * top_shade * 1.12, 0, 255)
        arr[rel_desk : rel_desk + top_surface_h, :, 2] = np.clip(desk_rgb[2] * top_shade * 1.12, 0, 255)
        arr[rel_desk : rel_desk + top_surface_h, :, 3] = 252.0

        # 2b. Soft elliptical cast contact shadow on tabletop directly under Kiko's chest
        yy_s, xx_s = np.mgrid[0:top_surface_h, 0:width].astype(np.float32)
        cx = width * char_x_ratio
        rx = max(60.0, width * 0.22)
        ry = max(8.0, top_surface_h * 0.52)
        sd = ((xx_s - cx) / rx) ** 2 + ((yy_s - top_surface_h * 0.22) / ry) ** 2
        shadow_mul = 1.0 - 0.56 * np.exp(-sd * 1.65)
        arr[rel_desk : rel_desk + top_surface_h, :, 0] *= shadow_mul
        arr[rel_desk : rel_desk + top_surface_h, :, 1] *= shadow_mul
        arr[rel_desk : rel_desk + top_surface_h, :, 2] *= shadow_mul

        # 2c. Soft rear-edge bevel highlight tinted with booth accent color
        rim_h = min(5, top_surface_h)
        for ry_i in range(rim_h):
            w_rim = 1.0 - (ry_i / float(rim_h))
            idx = rel_desk + ry_i
            arr[idx, :, 0] = arr[idx, :, 0] * (1.0 - 0.34 * w_rim) + accent_rgb[0] * (0.34 * w_rim)
            arr[idx, :, 1] = arr[idx, :, 1] * (1.0 - 0.34 * w_rim) + accent_rgb[1] * (0.34 * w_rim)
            arr[idx, :, 2] = arr[idx, :, 2] * (1.0 - 0.34 * w_rim) + accent_rgb[2] * (0.34 * w_rim)

        # 2d. Front chamfer bevel strip separating tabletop from vertical studio console front
        bevel_y0 = rel_desk + top_surface_h
        for bi in range(bevel_h):
            idx = bevel_y0 + bi
            if idx >= strip_h:
                break
            bw = 1.0 - abs((bi / max(1.0, bevel_h - 1.0)) - 0.35)
            arr[idx, :, 0] = desk_rgb[0] * (0.55 + 0.35 * bw) + accent_rgb[0] * (0.14 * bw)
            arr[idx, :, 1] = desk_rgb[1] * (0.55 + 0.35 * bw) + accent_rgb[1] * (0.14 * bw)
            arr[idx, :, 2] = desk_rgb[2] * (0.55 + 0.35 * bw) + accent_rgb[2] * (0.14 * bw)
            arr[idx, :, 3] = 255.0

        # 2e. Dark matte studio console front fascia fading into bottom Reels vignette
        if front_start < strip_h:
            fh = strip_h - front_start
            fy = np.linspace(0.0, 1.0, fh, dtype=np.float32)[:, None]
            f_shade = 0.44 * (1.0 - 0.72 * fy)
            arr[front_start:, :, 0] = desk_rgb[0] * f_shade + 5.0 * fy
            arr[front_start:, :, 1] = desk_rgb[1] * f_shade + 6.0 * fy
            arr[front_start:, :, 2] = desk_rgb[2] * f_shade + 10.0 * fy
            arr[front_start:, :, 3] = 255.0

        # 2f. Subtle ambient occlusion shadow just above rear desk lip for 3D depth
        sh_h = min(16, rel_desk)
        for sy in range(1, sh_h + 1):
            idx = rel_desk - sy
            sh_a = (1.0 - sy / float(sh_h + 1)) * 85.0
            arr[idx, :, 3] = np.maximum(arr[idx, :, 3], sh_a)

    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGBA")
    draw = ImageDraw.Draw(img)
    if 0 <= rel_desk < strip_h:
        # Crisp 2px studio rim highlight along rear edge of desk
        draw.line([(0, rel_desk), (width, rel_desk)], fill=(*accent_rgb, 155), width=2)
        # Subtle specular highlight along front chamfer edge
        front_edge_y = min(strip_h - 1, rel_desk + min(48, max(20, (strip_h - rel_desk) // 5)))
        draw.line([(0, front_edge_y), (width, front_edge_y)], fill=(*accent_rgb, 95), width=2)

        # 3. Grounded dark broadcast desk-mic stand base & arm silhouette on left foreground
        mic_cx = int(width * 0.125)
        mic_by = min(strip_h - 6, rel_desk + 26)
        # Contact shadow under mic base
        draw.ellipse(
            [mic_cx - 64, mic_by - 12, mic_cx + 64, mic_by + 16],
            fill=(4, 5, 10, 185),
        )
        # Weighted circular matte-charcoal studio base
        draw.ellipse(
            [mic_cx - 54, mic_by - 16, mic_cx + 54, mic_by + 12],
            fill=(20, 23, 32, 248),
            outline=(*accent_rgb, 110),
            width=2,
        )
        # Matte vertical stem + subtle boom joint angled gently inward (never occluding Kiko's face)
        stem_top_y = max(2, rel_desk - 42)
        draw.rounded_rectangle(
            [mic_cx - 8, stem_top_y, mic_cx + 8, mic_by - 2],
            radius=4,
            fill=(24, 28, 38, 250),
            outline=(70, 78, 96, 170),
            width=1,
        )
        # Small broadcast arm knuckle + muted LED power dot
        draw.ellipse(
            [mic_cx - 11, stem_top_y - 9, mic_cx + 11, stem_top_y + 11],
            fill=(30, 34, 46, 252),
            outline=(*accent_rgb, 140),
            width=1,
        )
        draw.ellipse(
            [mic_cx - 3, stem_top_y - 2, mic_cx + 3, stem_top_y + 4],
            fill=(*accent_rgb, 235),
        )

    return img


def clean_sprite_alpha_halo(img: Image.Image) -> Image.Image:
    """Clean faint 1px light edge fringes around matte-charcoal headphones and fur silhouette."""
    arr = np.array(img.convert("RGBA"), dtype=np.float32)
    alpha = arr[:, :, 3]
    arr[alpha < 18.0, 3] = 0.0
    edge = (alpha >= 18.0) & (alpha < 235.0)
    if np.any(edge):
        a_norm = alpha[edge] / 255.0
        luma = 0.299 * arr[edge, 0] + 0.587 * arr[edge, 1] + 0.114 * arr[edge, 2]
        darken = np.where(luma > 130.0, 0.70 + 0.30 * a_norm, 0.88 + 0.12 * a_norm)
        arr[edge, 0] *= darken
        arr[edge, 1] *= darken
        arr[edge, 2] *= darken
        arr[edge, 3] = np.clip((a_norm**1.18) * 255.0, 0.0, 255.0)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGBA")


def _opaque_bbox(img: Image.Image, alpha_thresh: int = 24) -> tuple[int, int, int, int] | None:
    alpha = np.array(img.getchannel("A"))
    ys, xs = np.where(alpha > alpha_thresh)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _normalize_sprite(img: Image.Image, target_size: int) -> Image.Image:
    """Resize a single sprite cleanly to target_size x target_size RGBA while preserving aspect ratio."""
    rgba = clean_sprite_alpha_halo(img)
    if rgba.size == (target_size, target_size):
        return rgba
    contained = ImageOps.contain(rgba, (target_size, target_size), method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))
    ox = (target_size - contained.width) // 2
    oy = target_size - contained.height
    canvas.alpha_composite(contained, (ox, oy))
    return canvas


def normalize_sprites_to_reference(
    raw_sprites: dict[str, Image.Image],
    target_size: int,
    *,
    reference_key: str = "neutral_closed",
) -> dict[str, Image.Image]:
    """Align every sprite to neutral_closed's exact opaque width, horizontal center, and bottom baseline."""
    if not raw_sprites:
        return {}
    cleaned = {k: clean_sprite_alpha_halo(v) for k, v in raw_sprites.items()}
    ref = cleaned.get(reference_key) or next(iter(cleaned.values()))
    ref_bb = _opaque_bbox(ref)
    if ref_bb is None:
        return {k: _normalize_sprite(v, target_size) for k, v in cleaned.items()}

    ref_w = max(1, ref_bb[2] - ref_bb[0])
    ref_h = max(1, ref_bb[3] - ref_bb[1])
    base_scale = min(
        (target_size * 0.94) / float(ref_w),
        (target_size * 0.96) / float(ref_h),
    )
    target_ref_w = ref_w * base_scale

    normalized: dict[str, Image.Image] = {}
    for name, spr in cleaned.items():
        bb = _opaque_bbox(spr)
        if bb is None:
            normalized[name] = _normalize_sprite(spr, target_size)
            continue
        x0, y0, x1, y1 = bb
        w = max(1, x1 - x0)
        h = max(1, y1 - y0)
        cropped = spr.crop((x0, y0, x1, y1))

        # Match reference opaque width cleanly (clamped to [0.88, 1.12] relative to base_scale)
        desired_scale = target_ref_w / float(w)
        rel = max(0.88, min(1.12, desired_scale / max(1e-6, base_scale)))
        eff_scale = base_scale * rel
        eff_scale = min(eff_scale, target_size / float(w), target_size / float(h))

        nw = max(1, min(target_size, int(round(w * eff_scale))))
        nh = max(1, min(target_size, int(round(h * eff_scale))))
        resized = cropped.resize((nw, nh), resample=Image.Resampling.LANCZOS)

        canvas = Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))
        ox = (target_size - nw) // 2
        oy = target_size - nh
        canvas.alpha_composite(resized, (ox, oy))
        normalized[name] = canvas

    return normalized


def render_host_booth_video(
    wav_path: Path,
    cue_timings: list[dict[str, Any]],
    genre: str,
    duration_sec: float,
    out_path: Path,
    config: dict[str, Any],
    *,
    movie_path: Path | None = None,
    booth_height: int | None = None,
) -> Path:
    """Render 1080xH 30fps Kiko host_booth.mp4 with normalized sprites, 3D desk & screen-glow."""
    host_cfg = config.get("host", {})
    fps = int(host_cfg.get("fps", 30))
    booth_w = int(host_cfg.get("booth_width", 1080))
    booth_h = int(booth_height if booth_height is not None else host_cfg.get("booth_height", 970))
    booth_h -= booth_h % 2
    base_sprite_size = int(host_cfg.get("sprite_size", 620))
    sprite_size = min(740, max(base_sprite_size, int(booth_h * 0.64)))
    sprites_dir = Path(str(host_cfg.get("sprites_dir", "assets/host/sprites")))
    booths_dir = Path(str(host_cfg.get("booths_dir", "assets/host/booths")))
    rms_open = float(host_cfg.get("rms_open_threshold", 0.085))
    rms_half = float(host_cfg.get("rms_half_threshold", 0.028))
    x_ratio = float(host_cfg.get("character_x_ratio", 0.39))
    desk_y_ratio = float(host_cfg.get("desk_y_ratio", 0.69))
    glow_strength = float(host_cfg.get("screen_glow_strength", 0.06))
    xfade_total = max(1, int(host_cfg.get("emotion_crossfade_frames", 4)))
    max_tilt = float(host_cfg.get("max_head_tilt_deg", 2.5))

    booth_path = booths_dir / f"{genre}.png"
    if not booth_path.exists():
        booth_path = booths_dir / "comedy.png"
    booth_raw = Image.open(booth_path).convert("RGBA")
    booth_fitted = ImageOps.fit(booth_raw, (booth_w, booth_h), method=Image.Resampling.LANCZOS)
    booth_img = _apply_booth_vignette(booth_fitted, center_x_ratio=x_ratio)

    raw_sprites: dict[str, Image.Image] = {}
    for p in sprites_dir.glob("*.png"):
        raw_sprites[p.stem] = Image.open(p).convert("RGBA")
    sprite_cache = normalize_sprites_to_reference(raw_sprites, sprite_size)

    # Foreground 3D podcast desk anchored at desk_y_ratio (default 0.69 -> y=670..970 on 970px booth)
    desk_y = max(int(booth_h * 0.58), min(int(booth_h * 0.80), int(booth_h * desk_y_ratio)))
    strip_top_y = max(0, desk_y - 68)
    desk_vignette = _build_desk_and_vignette_strip(
        booth_w, booth_h, genre, strip_top_y, desk_y, char_x_ratio=x_ratio
    )

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

    # Ground Kiko behind the foreground podcast desk so lower ~22% of torso sits tucked under desk edge
    base_y = max(48, desk_y - int(sprite_size * 0.78))
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

        # Overlay 3D foreground studio podcast desk ON TOP of Kiko's lower torso + bottom Reels vignette
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
