"""Reactive 3D Kiko 2.0 host animator with 3D pop-out RGBA dual-pipe (1080x750 @ 30fps)."""

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
from PIL import Image, ImageDraw, ImageFont, ImageOps

from pipeline.video_composer import select_h264_encoder

log = logging.getLogger(__name__)

# 6-frame elastic expression snap-in curve (volume-preserving scale_y; scale_x = 1/scale_y)
ELASTIC_SNAP_SCALE_Y: tuple[float, ...] = (1.00, 1.09, 1.04, 0.97, 1.01, 1.00)

# Dynamic booth + character camera punch-in scale per emotion (smoothly interpolated)
EMOTION_ZOOM_TARGET: dict[str, float] = {
    "neutral": 1.00,
    "sad": 1.00,
    "smug": 1.07,
    "suspicious": 1.07,
    "hyped": 1.10,
    "laughing": 1.10,
    "shocked": 1.14,
    "horrified": 1.14,
    "disgusted": 1.14,
    "facepalm": 1.08,
}

_EMOTION_TILT_DEG: dict[str, float] = {
    "neutral": 0.0,
    "laughing": 2.6,
    "hyped": 2.2,
    "smug": -2.2,
    "shocked": -2.4,
    "horrified": -2.8,
    "sad": -1.8,
    "facepalm": 1.8,
    "disgusted": -3.0,
    "suspicious": 2.4,
}

_GENRE_DESK_PALETTES: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "comedy": ((38, 24, 16), (255, 210, 78)),
    "horror_comedy": ((18, 15, 30), (46, 230, 197)),
    "action_scifi": ((14, 20, 34), (44, 224, 255)),
    "emotional_drama": ((18, 22, 36), (255, 184, 92)),
}

_ACTIVE_WATCH_FALLBACKS: tuple[str, ...] = ("suspicious", "shocked", "horrified")


def host_alpha_path_for(rgb_mp4: Path) -> Path:
    """Return the paired grayscale alpha matte MP4 path for a rendered host overlay."""
    return rgb_mp4.with_name(f"{rgb_mp4.stem}_alpha{rgb_mp4.suffix}")


def get_target_camera_zoom(emotion: str, is_freeze: bool = False) -> float:
    """Return target booth/character zoom scale for the active emotion or freeze-frame beat."""
    if is_freeze:
        return 1.14
    return EMOTION_ZOOM_TARGET.get(emotion, 1.00)


def resolve_frame_emotion_state(
    t: float,
    cue_timings: list[dict[str, Any]],
    freeze_events: list[dict[str, Any]] | None = None,
    *,
    facepalm_release_sec: float = 2.2,
) -> tuple[str, bool, bool]:
    """Resolve ``(emotion, is_speaking, is_freeze)`` at time *t* with stuck-facepalm release."""
    is_freeze = False
    for ev in freeze_events or ():
        es = float(ev.get("expanded_start_sec", -1.0))
        ee = float(ev.get("expanded_end_sec", -1.0))
        if es <= t <= ee:
            is_freeze = True
            break

    if not cue_timings:
        return ("neutral", False, is_freeze)

    emotion = str(cue_timings[0].get("emotion", "neutral"))
    is_speaking = False
    pause_since: float | None = None
    next_emotion: str | None = None

    for idx, c in enumerate(cue_timings):
        cs = float(c.get("start_sec", 0.0))
        ce = float(c.get("end_sec", 0.0))
        if cs <= t <= ce:
            emotion = str(c.get("emotion", "neutral"))
            is_speaking = True
            if bool(c.get("freeze_frame", False)):
                is_freeze = True
            pause_since = None
            break
        if t > ce:
            emotion = str(c.get("pause_emotion", "neutral"))
            pause_since = ce
            if idx + 1 < len(cue_timings):
                next_emotion = str(cue_timings[idx + 1].get("emotion", "suspicious"))

    # Never leave Kiko stuck in facepalm during silent pauses > 2.2s
    if not is_speaking and emotion == "facepalm" and pause_since is not None and (t - pause_since) > facepalm_release_sec:
        emotion = next_emotion if next_emotion in _ACTIVE_WATCH_FALLBACKS else "suspicious"

    return (emotion, is_speaking, is_freeze)


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
        last_rgb = np.array([110.0, 120.0, 140.0], dtype=np.float32)
        for i in range(0, n_frames, max(1, sample_stride)):
            t_sec = i / float(fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, round(t_sec * src_fps))
            ret, frame = cap.read()
            if ret and frame is not None:
                small = cv2.resize(frame, (32, 18), interpolation=cv2.INTER_AREA)
                bgr_mean = small.mean(axis=(0, 1))
                rgb = np.array([bgr_mean[2], bgr_mean[1], bgr_mean[0]], dtype=np.float32)
                mean_l = float(rgb.mean())
                rgb = np.clip(mean_l + (rgb - mean_l) * 1.35, 0.0, 255.0)
                last_rgb = rgb
            colors[i] = last_rgb

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

        for i in range(1, n_frames):
            colors[i] = 0.80 * colors[i - 1] + 0.20 * colors[i]
        return colors
    except Exception as exc:
        log.warning("Screen-glow sampling skipped (%s).", exc)
        return None
    finally:
        cap.release()


def _apply_booth_vignette(booth_img: Image.Image, center_x_ratio: float = 0.36) -> Image.Image:
    """Apply rich dark radial cinema vignette so Kiko 2.0 pops dramatically off the booth wall."""
    arr = np.array(booth_img.convert("RGBA"), dtype=np.float32)
    h, w = arr.shape[:2]
    # Keep booth dramatically darker than the bright movie world above
    luma = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    mean_l = float(luma.mean())
    if mean_l > 62.0:
        gain = 48.0 / max(1.0, mean_l)
        arr[:, :, :3] *= gain

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx = w * center_x_ratio
    cy = h * 0.38
    dx = (xx - cx) / max(1.0, w * 0.62)
    dy = (yy - cy) / max(1.0, h * 0.62)
    dist = np.sqrt(dx * dx + dy * dy)
    vig = np.clip(1.02 - 0.52 * np.clip(dist - 0.22, 0.0, 1.30) ** 1.35, 0.44, 1.02)
    arr[:, :, 0] *= vig
    arr[:, :, 1] *= vig
    arr[:, :, 2] *= vig
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGBA")


def _build_neon_divider_strip(width: int, genre: str, half_glow: int = 10) -> Image.Image:
    """Build glowing 4px genre-accent neon divider strip rendered behind Kiko's 3D pop-out."""
    _, accent = _GENRE_DESK_PALETTES.get(genre, _GENRE_DESK_PALETTES["comedy"])
    h = half_glow * 2 + 1
    arr = np.zeros((h, width, 4), dtype=np.uint8)
    for y in range(h):
        dist = abs(y - half_glow)
        if dist <= 1:
            r = min(255, int(accent[0] * 0.72 + 255 * 0.28))
            g = min(255, int(accent[1] * 0.72 + 255 * 0.28))
            b = min(255, int(accent[2] * 0.72 + 255 * 0.28))
            a = 250
        elif dist == 2:
            r, g, b, a = accent[0], accent[1], accent[2], 238
        else:
            falloff = max(0.0, 1.0 - (dist - 2) / float(max(1, half_glow - 1)))
            r, g, b = accent
            a = int(148 * (falloff**1.8))
        arr[y, :, 0] = r
        arr[y, :, 1] = g
        arr[y, :, 2] = b
        arr[y, :, 3] = a
    return Image.fromarray(arr, mode="RGBA")


def _build_desk_and_vignette_strip(
    width: int,
    overlay_h: int,
    genre: str,
    strip_top_y: int,
    desk_top_y: int,
    desk_bottom_y: int,
    *,
    char_x_ratio: float = 0.36,
) -> Image.Image:
    """Build raised 3D studio desk (rel_y=385..495 -> canvas y=1555..1665) + bottom UI dark fade."""
    strip_h = max(1, overlay_h - strip_top_y)
    desk_rgb, accent_rgb = _GENRE_DESK_PALETTES.get(genre, _GENRE_DESK_PALETTES["comedy"])

    arr = np.zeros((strip_h, width, 4), dtype=np.float32)
    rel_desk = max(0, desk_top_y - strip_top_y)
    rel_desk_bot = min(strip_h, max(rel_desk + 48, desk_bottom_y - strip_top_y))

    # 1. Raised 3D Foreground Studio Desk overlapping Kiko's lower torso (ABOVE Shorts/Reels UI!)
    if rel_desk < strip_h:
        desk_span = max(36, rel_desk_bot - rel_desk)
        top_surface_h = min(46, max(24, int(desk_span * 0.42)))
        bevel_h = min(8, max(5, int(desk_span * 0.08)))
        front_start = min(strip_h, rel_desk + top_surface_h + bevel_h)

        # 1a. Perspective studio tabletop with soft warm center sheen + subtle grain
        ty = np.linspace(0.0, 1.0, top_surface_h, dtype=np.float32)[:, None]
        tx = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
        center_sheen = 1.0 + 0.28 * np.exp(-((tx - char_x_ratio) ** 2) / 0.14)
        grain = 1.0 + 0.035 * np.sin(ty * 40.0 + 1.8 * np.sin(tx * 14.0))
        top_shade = (0.76 + 0.40 * ty) * center_sheen * grain
        arr[rel_desk : rel_desk + top_surface_h, :, 0] = np.clip(desk_rgb[0] * top_shade * 1.25, 0, 255)
        arr[rel_desk : rel_desk + top_surface_h, :, 1] = np.clip(desk_rgb[1] * top_shade * 1.22, 0, 255)
        arr[rel_desk : rel_desk + top_surface_h, :, 2] = np.clip(desk_rgb[2] * top_shade * 1.22, 0, 255)
        arr[rel_desk : rel_desk + top_surface_h, :, 3] = 254.0

        # 1b. Elliptical cast contact shadow directly under Kiko's hoodie chest
        yy_s, xx_s = np.mgrid[0:top_surface_h, 0:width].astype(np.float32)
        cx = width * char_x_ratio
        rx = max(60.0, width * 0.20)
        ry = max(8.0, top_surface_h * 0.54)
        sd = ((xx_s - cx) / rx) ** 2 + ((yy_s - top_surface_h * 0.20) / ry) ** 2
        shadow_mul = 1.0 - 0.58 * np.exp(-sd * 1.65)
        arr[rel_desk : rel_desk + top_surface_h, :, :3] *= shadow_mul[:, :, None]

        # 1c. Subtle accent rim light along rear lip of desk
        rim_h = min(5, top_surface_h)
        for ry_i in range(rim_h):
            w_rim = 1.0 - (ry_i / float(rim_h))
            idx = rel_desk + ry_i
            arr[idx, :, 0] = arr[idx, :, 0] * (1.0 - 0.36 * w_rim) + accent_rgb[0] * (0.36 * w_rim)
            arr[idx, :, 1] = arr[idx, :, 1] * (1.0 - 0.36 * w_rim) + accent_rgb[1] * (0.36 * w_rim)
            arr[idx, :, 2] = arr[idx, :, 2] * (1.0 - 0.36 * w_rim) + accent_rgb[2] * (0.36 * w_rim)

        # 1d. Front chamfer bevel strip separating tabletop from studio console fascia
        bevel_y0 = rel_desk + top_surface_h
        for bi in range(bevel_h):
            idx = bevel_y0 + bi
            if idx >= strip_h:
                break
            bw = 1.0 - abs((bi / max(1.0, bevel_h - 1.0)) - 0.35)
            arr[idx, :, 0] = desk_rgb[0] * (0.62 + 0.36 * bw) + accent_rgb[0] * (0.16 * bw)
            arr[idx, :, 1] = desk_rgb[1] * (0.62 + 0.36 * bw) + accent_rgb[1] * (0.16 * bw)
            arr[idx, :, 2] = desk_rgb[2] * (0.62 + 0.36 * bw) + accent_rgb[2] * (0.16 * bw)
            arr[idx, :, 3] = 255.0

        # 1e. Visible vertical studio console fascia (front_start..rel_desk_bot)
        if front_start < rel_desk_bot:
            fh = rel_desk_bot - front_start
            fy = np.linspace(0.0, 1.0, fh, dtype=np.float32)[:, None]
            f_shade = 0.56 * (1.0 - 0.45 * fy)
            arr[front_start:rel_desk_bot, :, 0] = desk_rgb[0] * f_shade + 6.0 * fy
            arr[front_start:rel_desk_bot, :, 1] = desk_rgb[1] * f_shade + 7.0 * fy
            arr[front_start:rel_desk_bot, :, 2] = desk_rgb[2] * f_shade + 11.0 * fy
            arr[front_start:rel_desk_bot, :, 3] = 255.0

        # 1f. Bottom Shorts/Reels UI dark gradient fade (rel_desk_bot..strip_h -> canvas y=1665..1920)
        if rel_desk_bot < strip_h:
            bh = strip_h - rel_desk_bot
            by = np.linspace(0.0, 1.0, bh, dtype=np.float32)[:, None]
            arr[rel_desk_bot:, :, 0] = desk_rgb[0] * 0.24 * (1.0 - 0.78 * by) + 4.0 * by
            arr[rel_desk_bot:, :, 1] = desk_rgb[1] * 0.24 * (1.0 - 0.78 * by) + 5.0 * by
            arr[rel_desk_bot:, :, 2] = desk_rgb[2] * 0.24 * (1.0 - 0.78 * by) + 9.0 * by
            arr[rel_desk_bot:, :, 3] = 255.0

        # 1g. Ambient occlusion shadow just above rear desk edge
        sh_h = min(14, rel_desk)
        for sy in range(1, sh_h + 1):
            idx = rel_desk - sy
            sh_a = (1.0 - sy / float(sh_h + 1)) * 88.0
            arr[idx, :, 3] = np.maximum(arr[idx, :, 3], sh_a)

    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGBA")
    draw = ImageDraw.Draw(img)
    if 0 <= rel_desk < strip_h:
        draw.line([(0, rel_desk), (width, rel_desk)], fill=(*accent_rgb, 165), width=2)
        front_edge_y = min(strip_h - 1, rel_desk + min(46, max(24, int(max(36, rel_desk_bot - rel_desk) * 0.42))))
        draw.line([(0, front_edge_y), (width, front_edge_y)], fill=(*accent_rgb, 105), width=2)

        # Grounded broadcast desk-mic stand base & arm silhouette on left foreground
        mic_cx = int(width * 0.115)
        mic_by = min(strip_h - 6, rel_desk + 22)
        draw.ellipse([mic_cx - 58, mic_by - 10, mic_cx + 58, mic_by + 14], fill=(4, 5, 10, 190))
        draw.ellipse(
            [mic_cx - 48, mic_by - 14, mic_cx + 48, mic_by + 10],
            fill=(18, 21, 30, 250),
            outline=(*accent_rgb, 120),
            width=2,
        )
        stem_top_y = max(2, rel_desk - 38)
        draw.rounded_rectangle(
            [mic_cx - 7, stem_top_y, mic_cx + 7, mic_by - 2],
            radius=3,
            fill=(24, 28, 38, 252),
            outline=(72, 80, 98, 175),
            width=1,
        )
        draw.ellipse(
            [mic_cx - 10, stem_top_y - 8, mic_cx + 10, stem_top_y + 10],
            fill=(30, 34, 46, 252),
            outline=(*accent_rgb, 150),
            width=1,
        )
        draw.ellipse([mic_cx - 3, stem_top_y - 2, mic_cx + 3, stem_top_y + 4], fill=(*accent_rgb, 240))

    return img


def _load_fx_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    fonts_dir = Path(__file__).resolve().parent.parent / "assets" / "fonts"
    for candidate in (
        fonts_dir / "Montserrat-ExtraBold.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    ):
        if candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), size)
            except OSError:
                continue
    return ImageFont.load_default()


def _build_comic_fx_stamps() -> dict[str, Image.Image]:
    """Pre-render lightweight Pillow RGBA comic FX stamps (sweat drop, !?, laugh glints)."""
    stamps: dict[str, Image.Image] = {}

    # 1. Horrified cyan sweat drop (56x76 RGBA)
    sweat = Image.new("RGBA", (56, 76), (0, 0, 0, 0))
    sd = ImageDraw.Draw(sweat)
    # Teardrop silhouette
    sd.polygon([(28, 4), (48, 44), (8, 44)], fill=(54, 218, 255, 240))
    sd.ellipse([8, 26, 48, 68], fill=(54, 218, 255, 240), outline=(18, 86, 130, 235), width=3)
    sd.polygon([(28, 6), (45, 42), (11, 42)], fill=(54, 218, 255, 245))
    # Specular white highlight inside teardrop
    sd.ellipse([17, 34, 27, 52], fill=(235, 252, 255, 235))
    sd.ellipse([22, 54, 27, 60], fill=(235, 252, 255, 220))
    stamps["sweat"] = sweat

    # 2. Shocked "!?" comic exclamation pop (128x110 RGBA — pops cleanly across y=1220!)
    shock = Image.new("RGBA", (128, 110), (0, 0, 0, 0))
    shd = ImageDraw.Draw(shock)
    fnt = _load_fx_font(76)
    shd.text((16, 14), "!?", font=fnt, fill=(14, 16, 24, 220), stroke_width=8, stroke_fill=(14, 16, 24, 220))
    shd.text((12, 10), "!", font=fnt, fill=(255, 216, 77, 255), stroke_width=6, stroke_fill=(18, 20, 28, 255))
    shd.text((46, 10), "?", font=fnt, fill=(255, 92, 92, 255), stroke_width=6, stroke_fill=(18, 20, 28, 255))
    stamps["shock_mark"] = shock

    # 3. Laughing golden 4-point eye/cheek sparkle glint (48x48 RGBA)
    glint = Image.new("RGBA", (48, 48), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glint)
    pts = [
        (24, 2), (29, 19), (46, 24), (29, 29),
        (24, 46), (19, 29), (2, 24), (19, 19),
    ]
    gd.polygon(pts, fill=(255, 226, 96, 248), outline=(255, 255, 255, 220))
    gd.ellipse([19, 19, 29, 29], fill=(255, 255, 255, 255))
    stamps["glint"] = glint

    return stamps


def _paste_rgba_clipped(dst: Image.Image, fg: Image.Image, x: int, y: int) -> None:
    """Alpha-composite *fg* onto *dst* safely even if (x, y) extends outside canvas edges."""
    dw, dh = dst.size
    fw, fh = fg.size
    src_x0 = max(0, -x)
    src_y0 = max(0, -y)
    src_x1 = min(fw, dw - x)
    src_y1 = min(fh, dh - y)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return
    dst_x = max(0, x)
    dst_y = max(0, y)
    if src_x0 == 0 and src_y0 == 0 and src_x1 == fw and src_y1 == fh:
        dst.alpha_composite(fg, (dst_x, dst_y))
    else:
        dst.alpha_composite(fg.crop((src_x0, src_y0, src_x1, src_y1)), (dst_x, dst_y))


def _scale_stamp_with_alpha(stamp: Image.Image, scale: float, alpha_mul: float) -> Image.Image:
    nw = max(4, round(stamp.width * scale))
    nh = max(4, round(stamp.height * scale))
    out = stamp.resize((nw, nh), Image.Resampling.BILINEAR) if (nw != stamp.width or nh != stamp.height) else stamp.copy()
    if alpha_mul < 0.99:
        r, g, b, a = out.split()
        a = a.point(lambda v: int(v * alpha_mul))
        out = Image.merge("RGBA", (r, g, b, a))
    return out


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


def _torso_band_span(img: Image.Image, bb: tuple[int, int, int, int], alpha_thresh: int = 24) -> tuple[float, float]:
    """Measure horizontal shoulder/chest span (width, center_x_rel) in bottom 28% torso band."""
    x0, y0, x1, y1 = bb
    h = max(1, y1 - y0)
    w = max(1, x1 - x0)
    band_top = y0 + int(h * 0.72)
    alpha_band = np.array(img.getchannel("A"))[band_top:y1, x0:x1]
    _by, bx = np.where(alpha_band > alpha_thresh)
    if len(bx) > 8:
        return float(bx.max() - bx.min() + 1), float((bx.min() + bx.max()) * 0.5)
    return float(w), float(w) * 0.5


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
    """Align sprites by chest/shoulder anchor so raised paws/ears NEVER squish Kiko's torso/head."""
    if not raw_sprites:
        return {}
    cleaned = {k: clean_sprite_alpha_halo(v) for k, v in raw_sprites.items()}
    ref = cleaned.get(reference_key) or next(iter(cleaned.values()))
    ref_bb = _opaque_bbox(ref)
    if ref_bb is None:
        return {k: _normalize_sprite(v, target_size) for k, v in cleaned.items()}

    ref_w = max(1, ref_bb[2] - ref_bb[0])
    ref_h = max(1, ref_bb[3] - ref_bb[1])
    ref_sh_w, _ref_sh_cx = _torso_band_span(ref, ref_bb)
    base_scale = min(
        (target_size * 0.90) / float(ref_w),
        (target_size * 0.92) / float(ref_h),
    )
    target_sh_w = ref_sh_w * base_scale

    normalized: dict[str, Image.Image] = {}
    for name, spr in cleaned.items():
        bb = _opaque_bbox(spr)
        if bb is None:
            normalized[name] = _normalize_sprite(spr, target_size)
            continue
        x0, y0, x1, y1 = bb
        w = max(1, x1 - x0)
        h = max(1, y1 - y0)
        sh_w, sh_cx = _torso_band_span(spr, bb)
        cropped = spr.crop((x0, y0, x1, y1))

        # Anchor by chest/shoulder width so wide/tall raised-paw poses stay full head/torso scale!
        desired_scale = target_sh_w / max(1.0, sh_w)
        rel = max(0.88, min(1.12, desired_scale / max(1e-6, base_scale)))
        eff_scale = base_scale * rel
        eff_scale = min(eff_scale, target_size / float(w), target_size / float(h))

        nw = max(1, min(target_size, int(round(w * eff_scale))))
        nh = max(1, min(target_size, int(round(h * eff_scale))))
        resized = cropped.resize((nw, nh), resample=Image.Resampling.LANCZOS)

        canvas = Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))
        scaled_cx = round(sh_cx * eff_scale)
        ox = max(0, min(target_size - nw, (target_size // 2) - scaled_cx))
        oy = target_size - nh
        canvas.alpha_composite(resized, (ox, oy))
        normalized[name] = canvas

    return normalized


def _zoom_booth_frame(
    booth_base: Image.Image,
    zoom: float,
    booth_w: int,
    booth_h: int,
    center_x_ratio: float,
    cache: dict[int, Image.Image],
) -> Image.Image:
    """Smoothly zoom the 1080x700 dark cinema booth around Kiko's horizontal axis (quantized cache)."""
    q = int(round(zoom * 200.0))  # 0.005 zoom precision (~28 cached levels)
    if q <= 200:
        return booth_base
    cached = cache.get(q)
    if cached is not None:
        return cached
    z = q / 200.0
    crop_w = max(16, round(booth_w / z))
    crop_h = max(16, round(booth_h / z))
    cx = booth_w * center_x_ratio
    cy = booth_h * 0.44
    x0 = max(0, min(booth_w - crop_w, int(round(cx - crop_w / 2.0))))
    y0 = max(0, min(booth_h - crop_h, int(round(cy - crop_h / 2.0))))
    zoomed = booth_base.crop((x0, y0, x0 + crop_w, y0 + crop_h)).resize((booth_w, booth_h), Image.Resampling.BILINEAR)
    cache[q] = zoomed
    return zoomed


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
    freeze_events: list[dict[str, Any]] | None = None,
) -> Path:
    """Render 1080x750 30fps Kiko 2.0 RGB + grayscale Alpha dual-pipe overlay for true 3D pop-out.

    Geometry mapped to canvas y=1170..1920:
    - rel_y = 0..50   : transparent 3D pop-out headroom (Kiko ears/paws/FX cross divider y=1220!)
    - rel_y = 50      : glowing 4px genre-accent neon divider (drawn BEHIND Kiko!)
    - rel_y = 50..750 : opaque 1080x700 dark cinema booth + smooth 1.00x..1.14x punch-in
    - rel_y = 385..495: raised visible 3D studio desk (canvas y=1555..1665, above Shorts/Reels UI)
    - rel_y = 495..750: dark studio console fade for Shorts/Reels UI legibility
    """
    host_cfg = config.get("host", {})
    fps = int(host_cfg.get("fps", 30))
    booth_w = int(host_cfg.get("booth_width", 1080))
    booth_h = int(booth_height if booth_height is not None else host_cfg.get("booth_height", 700))
    booth_h -= booth_h % 2
    headroom = int(host_cfg.get("popout_headroom_px", 50))
    headroom -= headroom % 2
    overlay_h = booth_h + headroom

    base_sprite_size = int(host_cfg.get("sprite_size", 460))
    sprite_size = min(560, max(360, base_sprite_size))
    sprites_dir = Path(str(host_cfg.get("sprites_dir", "assets/host/sprites")))
    booths_dir = Path(str(host_cfg.get("booths_dir", "assets/host/booths")))
    rms_open = float(host_cfg.get("rms_open_threshold", 0.085))
    rms_half = float(host_cfg.get("rms_half_threshold", 0.028))
    x_ratio = float(host_cfg.get("character_x_ratio", 0.36))
    desk_top_rel_y = int(host_cfg.get("desk_top_rel_y", headroom + 335))
    desk_bottom_rel_y = int(host_cfg.get("desk_bottom_rel_y", headroom + 445))
    glow_strength = float(host_cfg.get("screen_glow_strength", 0.06))
    xfade_total = max(1, int(host_cfg.get("emotion_crossfade_frames", 3)))
    max_tilt = float(host_cfg.get("max_head_tilt_deg", 4.0))

    booth_path = booths_dir / f"{genre}.png"
    if not booth_path.exists():
        booth_path = booths_dir / "comedy.png"
    booth_raw = Image.open(booth_path).convert("RGBA")
    booth_fitted = ImageOps.fit(booth_raw, (booth_w, booth_h), method=Image.Resampling.LANCZOS)
    booth_img = _apply_booth_vignette(booth_fitted, center_x_ratio=x_ratio)
    booth_zoom_cache: dict[int, Image.Image] = {}

    divider_half_glow = 10
    divider_strip = _build_neon_divider_strip(booth_w, genre, half_glow=divider_half_glow)
    fx_stamps = _build_comic_fx_stamps()

    raw_sprites: dict[str, Image.Image] = {}
    for p in sprites_dir.glob("*.png"):
        raw_sprites[p.stem] = Image.open(p).convert("RGBA")
    sprite_cache = normalize_sprites_to_reference(raw_sprites, sprite_size)

    # Raised 3D foreground studio desk at rel_y=385..495 (canvas y=1555..1665) + bottom UI dark fade
    desk_top_rel_y = max(headroom + 260, min(overlay_h - 140, desk_top_rel_y))
    desk_bottom_rel_y = max(desk_top_rel_y + 70, min(overlay_h - 40, desk_bottom_rel_y))
    strip_top_y = max(headroom, desk_top_rel_y - 54)
    desk_vignette = _build_desk_and_vignette_strip(
        booth_w,
        overlay_h,
        genre,
        strip_top_y,
        desk_top_rel_y,
        desk_bottom_rel_y,
        char_x_ratio=x_ratio,
    )

    n_frames = max(1, round(duration_sec * fps))
    rms_arr = _compute_frame_rms(wav_path, n_frames, fps=fps)
    glow_colors = (
        _sample_movie_screen_glow(movie_path, n_frames, fps=fps, sample_stride=3)
        if glow_strength > 0.0
        else None
    )

    # Reactive 6% top-down movie screen-glow applied across upper 55% of the booth (rel_y=headroom..headroom+glow_h)
    glow_h = int(booth_h * 0.55)
    glow_weights: np.ndarray | None = None
    if glow_colors is not None and glow_h > 0:
        gy = np.linspace(0.0, 1.0, glow_h, dtype=np.float32)
        glow_weights = (((1.0 - gy) ** 1.8) * glow_strength)[:, None, None]

    # Precompute natural blink windows (120ms every 2.8-4.6s)
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
    alpha_path = host_alpha_path_for(out_path)
    enc_flags = select_h264_encoder(config, intermediate=True)

    rgb_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{booth_w}x{overlay_h}", "-r", str(fps),
        "-i", "pipe:0",
        *enc_flags,
        "-pix_fmt", "yuv420p", "-r", str(fps),
        str(out_path),
    ]
    alpha_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{booth_w}x{overlay_h}", "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "14",
        "-pix_fmt", "yuv420p", "-r", str(fps),
        str(alpha_path),
    ]
    proc_rgb = subprocess.Popen(rgb_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    proc_alpha = subprocess.Popen(alpha_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert proc_rgb.stdin is not None and proc_alpha.stdin is not None

    prev_emotion = str(cue_timings[0].get("emotion", "neutral")) if cue_timings else "neutral"
    emotion_start_t = 0.0
    xfade_from_emotion = prev_emotion
    xfade_frames_left = 0
    snap_frame_idx = len(ELASTIC_SNAP_SCALE_Y)
    cur_tilt = 0.0
    cur_zoom = 1.00
    rms_smooth = 0.0
    rot_cache: dict[tuple[str, int], Image.Image] = {}

    def _get_sprite(em: str, mouth: str, is_blink: bool) -> tuple[str, Image.Image | None]:
        if is_blink and mouth in ("closed", "half") and em not in ("laughing", "facepalm"):
            for bk in ("neutral_blink", "blink"):
                if bk in sprite_cache:
                    return (bk, sprite_cache[bk])
        key = f"{em}_{mouth}"
        if key in sprite_cache:
            return (key, sprite_cache[key])
        for fallback in (f"{em}_closed", f"neutral_{mouth}", "neutral_closed"):
            if fallback in sprite_cache:
                return (fallback, sprite_cache[fallback])
        return ("none", None)

    def _rotated(key: str, spr: Image.Image, deg: float) -> Image.Image:
        q = int(round(deg * 2.0))  # 0.5-degree quantization
        if q == 0:
            return spr
        ckey = (key, q)
        cached = rot_cache.get(ckey)
        if cached is not None:
            return cached
        angle = q / 2.0
        rot = spr.rotate(
            angle,
            resample=Image.Resampling.BILINEAR,
            expand=False,
            center=(spr.width / 2, spr.height * 0.74),
        )
        if len(rot_cache) < 360:
            rot_cache[ckey] = rot
        return rot

    # Ground Kiko behind desk_top_rel_y so lower torso tucks cleanly behind desk while ears/paws pop across y=1220!
    anchor_bottom_y = desk_top_rel_y + int(sprite_size * 0.145)
    anchor_center_x = int(booth_w * x_ratio)

    for i in range(n_frames):
        t = i / float(fps)
        rms = float(rms_arr[i])
        rms_smooth = 0.58 * rms_smooth + 0.42 * rms

        emotion, _is_speaking, is_freeze = resolve_frame_emotion_state(t, cue_timings, freeze_events)

        if emotion != prev_emotion:
            xfade_from_emotion = prev_emotion
            xfade_frames_left = xfade_total
            snap_frame_idx = 0
            emotion_start_t = t
            prev_emotion = emotion

        t_in_emotion = max(0.0, t - emotion_start_t)
        mouth = "open" if rms >= rms_open else ("half" if rms >= rms_half else "closed")
        is_blink = i in blink_frames

        # 1. Smooth dynamic booth camera punch-in (1.00x .. 1.14x over ~10 frames)
        target_zoom = get_target_camera_zoom(emotion, is_freeze=is_freeze)
        cur_zoom = 0.78 * cur_zoom + 0.22 * target_zoom

        # 2. Emotion micro-physics (tilt, shake, recoil, rock, sway, RMS bounce & scale pulse)
        dx_phys = 0.0
        dy_phys = 0.0
        rms_scale_bonus = 0.0

        if emotion == "laughing":
            target_tilt = 4.0 * math.sin(2.0 * math.pi * t * 3.8)
            dy_phys = -min(10.0, rms_smooth * 68.0) - abs(math.sin(2.0 * math.pi * t * 3.8)) * 4.0
        elif emotion == "hyped":
            target_tilt = 2.5 * math.sin(2.0 * math.pi * t * 3.2)
            dy_phys = -min(9.0, rms_smooth * 58.0)
            rms_scale_bonus = min(0.04, rms_smooth * 0.26)
        elif emotion in ("horrified", "shocked"):
            target_tilt = _EMOTION_TILT_DEG.get(emotion, 0.0)
            if t_in_emotion <= 0.45:
                decay = 1.0 - (t_in_emotion / 0.45)
                dx_phys = math.sin(i * 2.7) * 3.5 * decay
                dy_phys = math.cos(i * 3.3) * 2.5 * decay
            dy_phys += -min(7.0, rms_smooth * 46.0)
        elif emotion == "disgusted":
            target_tilt = -3.0
            dx_phys = -10.0
            dy_phys = -min(6.0, rms_smooth * 40.0)
        elif emotion == "suspicious":
            target_tilt = 2.4 + 0.4 * math.sin(2.0 * math.pi * t / 3.2)
            dx_phys = 4.0
            rms_scale_bonus = 0.015
            dy_phys = -min(6.0, rms_smooth * 42.0)
        else:
            target_tilt = _EMOTION_TILT_DEG.get(emotion, 0.0) + 0.55 * math.sin(2.0 * math.pi * t / 3.8)
            dx_phys = 2.0 * math.sin(2.0 * math.pi * t / 3.0)
            dy_phys = 2.2 * math.sin(2.0 * math.pi * t / 3.0) - min(8.0, rms_smooth * 50.0)

        target_tilt = max(-max_tilt, min(max_tilt, target_tilt))
        cur_tilt = 0.72 * cur_tilt + 0.28 * target_tilt

        # 3. 6-frame elastic expression snap-in (volume-preserving squash & stretch)
        if snap_frame_idx < len(ELASTIC_SNAP_SCALE_Y):
            snap_sy = ELASTIC_SNAP_SCALE_Y[snap_frame_idx]
            snap_sx = 1.0 / snap_sy
            snap_frame_idx += 1
        else:
            snap_sy = 1.0
            snap_sx = 1.0

        curr_key, curr_spr = _get_sprite(emotion, mouth, is_blink)
        if curr_spr is not None:
            curr_spr = _rotated(curr_key, curr_spr, cur_tilt)

        if xfade_frames_left > 0 and curr_spr is not None:
            prev_key, prev_spr = _get_sprite(xfade_from_emotion, mouth, False)
            if prev_spr is not None and prev_key != curr_key:
                prev_rot = _rotated(prev_key, prev_spr, cur_tilt)
                alpha_mix = 1.0 - (xfade_frames_left / float(xfade_total + 1))
                spr_to_draw = Image.blend(prev_rot, curr_spr, alpha_mix)
            else:
                spr_to_draw = curr_spr
            xfade_frames_left -= 1
        else:
            spr_to_draw = curr_spr

        # 4. Assemble 1080x750 RGBA frame:
        #    (a) transparent headroom rel_y=0..headroom
        #    (b) opaque zoomed 1080x700 dark cinema booth at rel_y=headroom..overlay_h
        #    (c) glowing neon divider at rel_y=headroom (BEHIND Kiko so ears/paws/FX pop in FRONT!)
        frame = Image.new("RGBA", (booth_w, overlay_h), (0, 0, 0, 0))
        booth_cur = _zoom_booth_frame(booth_img, cur_zoom, booth_w, booth_h, x_ratio, booth_zoom_cache)
        frame.paste(booth_cur, (0, headroom))
        _paste_rgba_clipped(frame, divider_strip, 0, headroom - divider_half_glow)

        # (d) Kiko 2.0 sprite scaled by booth punch-in + elastic snap-in + RMS pulse (pops into headroom!)
        spr_top_y = headroom
        spr_right_x = anchor_center_x + sprite_size // 2
        if spr_to_draw is not None:
            eff_sx = cur_zoom * snap_sx * (1.0 + rms_scale_bonus)
            eff_sy = cur_zoom * snap_sy * (1.0 + rms_scale_bonus)
            draw_w = max(32, round(spr_to_draw.width * eff_sx))
            draw_h = max(32, round(spr_to_draw.height * eff_sy))
            spr_scaled = (
                spr_to_draw.resize((draw_w, draw_h), Image.Resampling.BILINEAR)
                if (draw_w != spr_to_draw.width or draw_h != spr_to_draw.height)
                else spr_to_draw
            )
            x0 = int(round(anchor_center_x - draw_w / 2.0 + dx_phys))
            # Keep lower torso grounded at desk while extra zoom/elastic stretch rises upward!
            y0 = int(round(anchor_bottom_y - draw_h + dy_phys))
            spr_top_y = y0
            spr_right_x = x0 + draw_w
            _paste_rgba_clipped(frame, spr_scaled, x0, y0)

        # (e) Lightweight Pillow RGBA comic FX stamps popping cleanly near head / across divider
        if emotion == "horrified" and t_in_emotion <= 0.80:
            prog = t_in_emotion / 0.80
            a_fx = min(1.0, prog / 0.15) * (1.0 - max(0.0, (prog - 0.65) / 0.35))
            sw_img = _scale_stamp_with_alpha(fx_stamps["sweat"], 0.85, a_fx)
            sw_x = int(anchor_center_x + sprite_size * 0.22 * cur_zoom)
            sw_y = int(max(12, spr_top_y + 72 + prog * 18.0))
            _paste_rgba_clipped(frame, sw_img, sw_x, sw_y)
        elif emotion == "shocked" and t_in_emotion <= 0.55:
            prog = t_in_emotion / 0.55
            pop_sc = (0.75 + 0.35 * min(1.0, prog / 0.18)) * (1.0 - 0.08 * max(0.0, prog - 0.25))
            a_fx = 1.0 - max(0.0, (prog - 0.72) / 0.28)
            sh_img = _scale_stamp_with_alpha(fx_stamps["shock_mark"], pop_sc, a_fx)
            # Place right above Kiko's right ear so "!?" pops cleanly into rel_y=6..50 above y=1220!
            sh_x = int(min(booth_w - sh_img.width - 12, spr_right_x - int(sprite_size * 0.26)))
            sh_y = int(max(4, headroom - int(sh_img.height * 0.52) - int(10 * min(1.0, prog / 0.2))))
            _paste_rgba_clipped(frame, sh_img, sh_x, sh_y)
        elif emotion == "laughing":
            pulse = 0.65 + 0.35 * abs(math.sin(2.0 * math.pi * t * 4.2))
            g_big = _scale_stamp_with_alpha(fx_stamps["glint"], 0.78 * pulse, 0.92)
            g_sm = _scale_stamp_with_alpha(fx_stamps["glint"], 0.52 * (1.3 - 0.3 * pulse), 0.85)
            _paste_rgba_clipped(frame, g_big, int(anchor_center_x - sprite_size * 0.32), max(10, spr_top_y + 58))
            _paste_rgba_clipped(frame, g_sm, int(anchor_center_x + sprite_size * 0.28), max(14, spr_top_y + 44))

        # (f) Overlay raised 3D foreground studio desk ON TOP of Kiko's lower torso + bottom UI fade
        _paste_rgba_clipped(frame, desk_vignette, 0, strip_top_y)

        # (g) Extract RGBA -> apply reactive 6% top-down movie screen-glow on opaque booth/host pixels -> dual pipe!
        rgba_arr = np.array(frame, dtype=np.uint8)
        rgb_arr = rgba_arr[:, :, :3]
        alpha_arr = rgba_arr[:, :, 3]

        if glow_colors is not None and glow_weights is not None:
            g_end = min(overlay_h, headroom + glow_h)
            if g_end > headroom:
                top_slice = rgb_arr[headroom:g_end].astype(np.float32)
                w_slice = glow_weights[: g_end - headroom]
                col = glow_colors[i].reshape(1, 1, 3)
                rgb_arr[headroom:g_end] = np.clip(
                    top_slice * (1.0 - w_slice) + col * w_slice, 0, 255
                ).astype(np.uint8)

        proc_rgb.stdin.write(rgb_arr.tobytes())
        proc_alpha.stdin.write(alpha_arr.tobytes())

    proc_rgb.stdin.close()
    proc_alpha.stdin.close()
    proc_rgb.wait()
    proc_alpha.wait()
    if proc_rgb.returncode != 0:
        raise RuntimeError(f"FFmpeg host_booth RGB pipe failed with code {proc_rgb.returncode}")
    if proc_alpha.returncode != 0:
        raise RuntimeError(f"FFmpeg host_booth Alpha pipe failed with code {proc_alpha.returncode}")
    return out_path
