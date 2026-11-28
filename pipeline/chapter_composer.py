"""Single-pass 63.5/36.5 full-bleed 9:16 composer with 3D pop-out alpha merge, freeze zoom & surgical auto-bleep."""

from __future__ import annotations

import json
import logging
import re
import statistics
import wave
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pipeline import audio_mixer, ffutil, host_animator, video_composer
from pipeline.models import ChapterSlice, RightsRecord

log = logging.getLogger(__name__)

_GENRE_ACCENTS: dict[str, tuple[int, int, int]] = {
    "comedy": (255, 216, 77),
    "horror_comedy": (46, 230, 197),
    "action_scifi": (44, 224, 255),
    "emotional_drama": (255, 184, 92),
}

PROFANITY_WORDS: frozenset[str] = frozenset(
    {
        "fuck",
        "fuckin",
        "fucking",
        "fucked",
        "fucker",
        "fuckers",
        "motherfucker",
        "motherfucking",
        "shit",
        "shitting",
        "shitty",
        "bullshit",
        "bitch",
        "bitches",
        "bitching",
        "bitchy",
        "asshole",
        "assholes",
        "bastard",
        "bastards",
        "cunt",
        "cunts",
        "dick",
        "dicks",
        "shag",
        "shagged",
        "shagging",
        "whore",
        "whores",
        "slut",
        "sluts",
    }
)

_PROFANITY_PREFIXES: tuple[str, ...] = ("fuck", "motherfuck", "bitch")


def _is_profanity_word(word: str) -> bool:
    norm = re.sub(r"[^a-z]", "", str(word or "").lower())
    if not norm:
        return False
    if norm in PROFANITY_WORDS:
        return True
    return any(norm.startswith(p) for p in _PROFANITY_PREFIXES)


def shift_words_for_freezes(
    chapter_words: list[dict[str, Any]],
    freeze_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Map raw chapter word timestamps onto the freeze-expanded movie timeline."""
    if not chapter_words:
        return []
    events = sorted(freeze_events or [], key=lambda e: float(e["raw_time_sec"]))
    shifted: list[dict[str, Any]] = []
    for w in chapter_words:
        ws = float(w.get("start", 0.0))
        we = float(w.get("end", ws))
        s_shift = sum(float(ev["freeze_dur_sec"]) for ev in events if float(ev["raw_time_sec"]) <= ws + 1e-4)
        e_shift = sum(float(ev["freeze_dur_sec"]) for ev in events if float(ev["raw_time_sec"]) <= we + 1e-4)
        shifted.append(
            {
                "word": str(w.get("word", "")),
                "start": round(ws + s_shift, 3),
                "end": round(max(ws + s_shift + 0.04, we + e_shift), 3),
            }
        )
    return shifted


def auto_bleep_profanity(
    movie_wav_path: Path,
    chapter_words: list[dict[str, Any]],
    out_wav_path: Path,
) -> tuple[Path, list[dict[str, Any]]]:
    """Replace matched profanity spans [w.start + 0.03, w.end + 0.06] with a clean 1000 Hz -14 dBFS bleep.

    Preserves exact WAV sample count, sample rate, sample width, and channel count.
    """
    with wave.open(str(movie_wav_path), "rb") as wf:
        nch = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)

    if sampwidth != 2 or n_frames == 0:
        out_wav_path.parent.mkdir(parents=True, exist_ok=True)
        out_wav_path.write_bytes(movie_wav_path.read_bytes())
        return out_wav_path, []

    pcm = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    samples = pcm.reshape(-1, max(1, nch)).copy()

    total_dur = n_frames / float(sr)
    bleeped_words: list[dict[str, Any]] = []
    bleep_amp = float(10.0 ** (-14.0 / 20.0))
    fade_samples_max = max(1, int(round(0.010 * sr)))

    for w in chapter_words or []:
        raw_word = str(w.get("word", "")).strip()
        if not _is_profanity_word(raw_word):
            continue
        # Surgical timing [w.start + 0.03, w.end + 0.06]: zero early bleep, zero tail leak
        raw_s = float(w.get("start", 0.0))
        raw_e = float(w.get("end", raw_s))
        t_start = max(0.0, min(total_dur, raw_s + 0.03))
        t_end = min(total_dur, max(t_start + 0.06, raw_e + 0.06))
        s_idx = max(0, int(round(t_start * sr)))
        e_idx = min(n_frames, int(round(t_end * sr)))
        span = e_idx - s_idx
        if span <= 16:
            continue

        t = np.arange(span, dtype=np.float32) / float(sr)
        sine = bleep_amp * np.sin(2.0 * np.pi * 1000.0 * t)
        fade_n = min(fade_samples_max, span // 4)
        env = np.ones(span, dtype=np.float32)
        if fade_n > 0:
            ramp = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
            env[:fade_n] = ramp
            env[-fade_n:] = ramp[::-1]

        bleep_seg = (sine * env)[:, None]
        keep_env = (1.0 - env)[:, None]
        samples[s_idx:e_idx] = samples[s_idx:e_idx] * keep_env + bleep_seg
        bleeped_words.append(
            {
                "word": raw_word,
                "start_sec": round(t_start, 3),
                "end_sec": round(t_end, 3),
            }
        )

    out_int16 = np.clip(samples.reshape(-1) * 32767.0, -32768, 32767).astype(np.int16)
    out_wav_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_wav_path), "wb") as wf:
        wf.setnchannels(nch)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(out_int16.tobytes())

    if bleeped_words:
        log.info("Auto-bleeped %d profanity word(s): %s", len(bleeped_words), bleeped_words)
    return out_wav_path, bleeped_words


def detect_active_crop(source_path: Path, sample_count: int = 5) -> tuple[int, int, int, int] | None:
    """Detect baked-in letterbox/pillarbox black bars via FFmpeg cropdetect on 5 timestamps.

    Returns ``(cw, ch, cx, cy)`` only if meaningful black bars (>=16px) are found; else ``None``.
    """
    try:
        info = ffutil.probe(source_path)
    except Exception:
        return None
    v_stream = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    if not v_stream:
        return None
    src_w = int(v_stream.get("width", 0))
    src_h = int(v_stream.get("height", 0))
    dur = float(v_stream.get("duration") or info.get("format", {}).get("duration") or 0.0)
    if src_w < 64 or src_h < 64 or dur <= 0.5:
        return None

    fracs = [0.12 + 0.76 * (i / max(1, sample_count - 1)) for i in range(max(1, sample_count))]
    boxes: list[tuple[int, int, int, int]] = []
    pat = re.compile(r"crop=(\d+):(\d+):(\d+):(\d+)")
    for f in fracs:
        ts = max(0.1, min(dur - 0.1, dur * f))
        res = ffutil.run(
            [
                "ffmpeg", "-y",
                "-ss", f"{ts:.3f}",
                "-i", str(source_path),
                "-frames:v", "3",
                "-vf", "cropdetect=limit=24:round=2:reset=0",
                "-f", "null", "-",
            ],
            check=False,
        )
        matches = pat.findall(res.stderr)
        if matches:
            cw, ch, cx, cy = (int(v) for v in matches[-1])
            if 0.65 * src_w <= cw <= src_w and 0.55 * src_h <= ch <= src_h:
                boxes.append((cw, ch, cx, cy))

    if not boxes:
        return None
    cw = int(round(statistics.median(b[0] for b in boxes))) & ~1
    ch = int(round(statistics.median(b[1] for b in boxes))) & ~1
    cx = max(0, min(src_w - cw, int(round(statistics.median(b[2] for b in boxes))) & ~1))
    cy = max(0, min(src_h - ch, int(round(statistics.median(b[3] for b in boxes))) & ~1))

    if (src_h - ch) < 16 and (src_w - cw) < 16:
        return None
    log.info("Detected active picture crop on %s: %dx%d+%d+%d (from %dx%d)", source_path.name, cw, ch, cx, cy, src_w, src_h)
    return (cw, ch, cx, cy)


def compute_vertical_layout(active_w: int, active_h: int) -> dict[str, int]:
    """Compute reconciled 63.5% Movie (y=0..1220) / 36.5% Compact Booth (y=1220..1920) layout.

    - Upper Movie World (y=0..1220): full-bleed ambient blur, header at y=95..185, sharp movie
      centered inside y=200..1050 (<=26% crop for 2.39:1 cinemascope, <=18% for 16:9) with 18px
      feathered top/bottom alpha edge, and 1-word pop captions at y=1065..1170 (MarginV=755).
    - Compact Dark Booth (y=1220..1920): 1080x700 booth rendered inside a 1080x750 RGBA overlay
      mapped to y=1170..1920 so Kiko's ears, raised paws & comic FX pop 15-30px above y=1220.
    """
    aspect = active_w / max(1, active_h)
    is_cinemascope = aspect >= 2.1
    max_crop = 0.26 if is_cinemascope else 0.18
    max_scaled_w = round(1080 / (1.0 - max_crop))
    raw_h = int(round(max_scaled_w / max(0.5, aspect))) & ~1
    movie_h = max(480, min(660, raw_h)) if is_cinemascope else max(540, min(840, raw_h))

    stage_top = 200
    stage_bottom = 1050
    movie_y = (stage_top + max(0, (stage_bottom - stage_top - movie_h) // 2)) & ~1
    divider_y = 1220
    booth_h = 1920 - divider_y  # 700
    popout_headroom = 50
    host_overlay_y = divider_y - popout_headroom  # 1170
    host_overlay_h = booth_h + popout_headroom  # 750

    return {
        "movie_world_h": divider_y,
        "movie_y": movie_y,
        "movie_h": movie_h,
        "divider_y": divider_y,
        "booth_h": booth_h,
        "popout_headroom": popout_headroom,
        "host_overlay_y": host_overlay_y,
        "host_overlay_h": host_overlay_h,
        "max_scaled_w": max_scaled_w,
    }


def _build_feather_mask_png(width: int, height: int, out_png: Path, feather_px: int = 18) -> Path:
    """Build a fast grayscale alpha matte PNG with *feather_px* soft top/bottom edges."""
    mask = np.full((height, width), 255, dtype=np.uint8)
    f = max(1, min(feather_px, height // 4))
    ramp = np.linspace(0.0, 255.0, f, dtype=np.float32)
    smooth = np.clip(0.5 * (1.0 - np.cos(np.pi * (ramp / 255.0))) * 255.0, 0.0, 255.0).astype(np.uint8)
    mask[:f, :] = smooth[:, None]
    mask[height - f :, :] = smooth[::-1, None]
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(out_png, format="PNG")
    return out_png


def _load_bold_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        video_composer._FONTS_DIR / "Montserrat-ExtraBold.ttf",
        Path("/usr/share/fonts/truetype/montserrat/Montserrat-ExtraBold.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    ):
        if candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), size)
            except OSError:
                continue
    return ImageFont.load_default()


def _render_header_and_divider_png(
    part_number: int,
    total_parts: int,
    top_headline: str,
    genre: str,
    out_png: Path,
    *,
    movie_y: int = 200,
    divider_y: int = 1220,
    include_divider: bool = False,
) -> Path:
    """Render 1080x1920 RGBA top chrome: soft scrim, dark-glass PART pill & headline at y=95..185.

    When ``include_divider=False`` (default 3D pop-out mode), the glowing neon divider at y=1220
    is rendered inside ``host_animator.py`` BEHIND Kiko so his ears/paws/FX pop in front of it.
    """
    w, h = 1080, 1920
    accent = _GENRE_ACCENTS.get(genre, _GENRE_ACCENTS["comedy"])

    arr = np.zeros((h, w, 4), dtype=np.uint8)

    # Top gradient scrim behind header (y=0..220) so headline always pops over blurred movie bleed
    scrim_h = min(h, max(210, min(260, movie_y + 18)))
    ys = np.linspace(0.0, 1.0, scrim_h, dtype=np.float32)
    scrim_a = (178.0 * ((1.0 - ys) ** 1.15)).astype(np.uint8)
    arr[:scrim_h, :, 0] = 8
    arr[:scrim_h, :, 1] = 10
    arr[:scrim_h, :, 2] = 16
    arr[:scrim_h, :, 3] = scrim_a[:, None]

    if include_divider:
        div_center = max(20, min(h - 20, int(divider_y)))
        for y in range(div_center - 10, div_center + 11):
            dist = abs(y - div_center)
            if dist <= 1:
                r = min(255, int(accent[0] * 0.72 + 255 * 0.28))
                g = min(255, int(accent[1] * 0.72 + 255 * 0.28))
                b = min(255, int(accent[2] * 0.72 + 255 * 0.28))
                a = 248
            elif dist == 2:
                r, g, b, a = accent[0], accent[1], accent[2], 235
            else:
                falloff = max(0.0, 1.0 - (dist - 2) / 9.0)
                r, g, b = accent
                a = int(145 * (falloff**1.8))
            arr[y, :, 0] = r
            arr[y, :, 1] = g
            arr[y, :, 2] = b
            arr[y, :, 3] = a

    img = Image.fromarray(arr, mode="RGBA")
    draw = ImageDraw.Draw(img)

    # 1. Semi-transparent dark-glass PART k/N pill (y=95..133, safely below Reels/Shorts top icons)
    pill_text = f"PART {part_number}/{total_parts}" if total_parts > 1 else "MOVIE RECAP"
    pill_font = _load_bold_font(23)
    pb = draw.textbbox((0, 0), pill_text, font=pill_font)
    pw = (pb[2] - pb[0]) + 40
    ph = 38
    px0 = (w - pw) // 2
    py0 = 95
    draw.rounded_rectangle(
        [px0, py0, px0 + pw, py0 + ph],
        radius=19,
        fill=(14, 16, 24, 215),
        outline=(*accent, 232),
        width=2,
    )
    ptx = (w - (pb[2] - pb[0])) // 2 - pb[0]
    pty = py0 + (ph - (pb[3] - pb[1])) // 2 - pb[1]
    draw.text((ptx, pty), pill_text, font=pill_font, fill=(*accent, 255))

    # 2. Crisp uppercase 4-6 word top_headline centered at y=139..188
    headline = " ".join(str(top_headline or "").strip().upper().split())
    if headline:
        max_w = 970
        chosen_font = _load_bold_font(44)
        chosen_lines = [headline]
        for fsize in (46, 42, 38, 34):
            fnt = _load_bold_font(fsize)
            hb = draw.textbbox((0, 0), headline, font=fnt)
            if (hb[2] - hb[0]) <= max_w:
                chosen_font = fnt
                chosen_lines = [headline]
                break
        else:
            words = headline.split()
            mid = max(1, len(words) // 2)
            chosen_font = _load_bold_font(32)
            chosen_lines = [" ".join(words[:mid]), " ".join(words[mid:])]

        line_heights: list[int] = []
        line_bboxes: list[tuple[int, int, int, int]] = []
        for ln in chosen_lines:
            bb = draw.textbbox((0, 0), ln, font=chosen_font)
            line_bboxes.append(bb)
            line_heights.append(bb[3] - bb[1])

        gap = 4
        total_h = sum(line_heights) + gap * (len(chosen_lines) - 1)
        cur_y = 139 + max(0, (48 - total_h) // 2)
        for ln, bb, lh in zip(chosen_lines, line_bboxes, line_heights):
            lw = bb[2] - bb[0]
            tx = (w - lw) // 2 - bb[0]
            ty = cur_y - bb[1]
            draw.text((tx + 2, ty + 3), ln, font=chosen_font, fill=(0, 0, 0, 215))
            draw.text(
                (tx, ty),
                ln,
                font=chosen_font,
                fill=(255, 255, 255, 255),
                stroke_width=4,
                stroke_fill=(10, 12, 18, 255),
            )
            cur_y += lh + gap

    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png, format="PNG")
    return out_png


def expand_movie_with_freezes(
    chapter_raw_mp4: Path,
    freeze_events: list[dict[str, Any]],
    work_dir: Path,
    config: dict[str, Any],
    *,
    crop_box: tuple[int, int, int, int] | None = None,
) -> Path:
    """Insert frame-exact ease-out 1.00x->1.12x zoom freeze holds with warm contrast/vignette pop."""
    if not freeze_events:
        return chapter_raw_mp4

    fps = int(config.get("runtime", {}).get("fps", 30))
    zoom_end = float(config.get("chapter_commentary", {}).get("freeze_zoom_end", 1.12))
    zoom_delta = max(0.03, zoom_end - 1.0)

    info = ffutil.probe(chapter_raw_mp4)
    v_stream = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), {})
    src_w = int(v_stream.get("width", 1920))
    src_h = int(v_stream.get("height", 1080))
    src_w -= src_w % 2
    src_h -= src_h % 2
    apply_crop = bool(crop_box and (crop_box[0] != src_w or crop_box[1] != src_h))
    out_w = (crop_box[0] & ~1) if (apply_crop and crop_box) else src_w
    out_h = (crop_box[1] & ~1) if (apply_crop and crop_box) else src_h
    crop_flt = f"crop={crop_box[0]}:{crop_box[1]}:{crop_box[2]}:{crop_box[3]}," if (apply_crop and crop_box) else ""
    has_audio = any(s.get("codec_type") == "audio" for s in info.get("streams", []))
    raw_dur = ffutil.duration(chapter_raw_mp4)

    enc_flags = video_composer.select_h264_encoder(config, intermediate=True)
    frz_dir = work_dir / "freeze_pieces"
    frz_dir.mkdir(parents=True, exist_ok=True)

    events = sorted(freeze_events, key=lambda e: float(e["raw_time_sec"]))
    vid_pieces: list[Path] = []
    aud_pieces: list[Path] = []
    piece_idx = 0
    prev_t = 0.0

    def _cut_normal(t_start: float, t_end: float) -> None:
        nonlocal piece_idx
        span = t_end - t_start
        if span <= 0.04:
            return
        n_frames = max(1, round(span * fps))
        exact_dur = n_frames / float(fps)
        vp = frz_dir / f"p{piece_idx:02d}_norm_v.mp4"
        ap = frz_dir / f"p{piece_idx:02d}_norm_a.wav"
        ffutil.run(
            [
                "ffmpeg", "-y",
                "-ss", f"{t_start:.3f}",
                "-i", str(chapter_raw_mp4),
                "-t", f"{exact_dur + 0.25:.3f}",
                "-vf", f"{crop_flt}fps={fps},scale={out_w}:{out_h},setsar=1,tpad=stop_mode=clone:stop_duration=0.5",
                "-frames:v", str(n_frames),
                "-an",
                *enc_flags,
                "-pix_fmt", "yuv420p", "-r", str(fps), "-video_track_timescale", "90000",
                str(vp),
            ]
        )
        if has_audio:
            fade_d = min(0.015, exact_dur / 4.0)
            fade_out_st = max(0.0, exact_dur - fade_d)
            ffutil.run(
                [
                    "ffmpeg", "-y",
                    "-ss", f"{t_start:.3f}",
                    "-i", str(chapter_raw_mp4),
                    "-t", f"{exact_dur + 0.2:.3f}",
                    "-vn",
                    "-af", (
                        f"apad=whole_dur={exact_dur:.6f},atrim=0:{exact_dur:.6f},asetpts=PTS-STARTPTS,"
                        f"afade=t=in:st=0:d={fade_d:.4f},afade=t=out:st={fade_out_st:.4f}:d={fade_d:.4f}"
                    ),
                    "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le",
                    str(ap),
                ]
            )
        else:
            ffutil.run(
                [
                    "ffmpeg", "-y",
                    "-f", "lavfi", "-i", f"anullsrc=r=48000:cl=stereo:d={exact_dur:.6f}",
                    "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le",
                    str(ap),
                ]
            )
        vid_pieces.append(vp)
        aud_pieces.append(ap)
        piece_idx += 1

    for ev in events:
        ev_t = max(prev_t, min(raw_dur, float(ev["raw_time_sec"])))
        _cut_normal(prev_t, ev_t)

        # Extract exact frame at ev_t (with EOF-safe fallback for ending CTA freezes)
        n_frz = max(int(round(1.0 * fps)), int(ev.get("freeze_frames") or round(float(ev["freeze_dur_sec"]) * fps)))
        frz_dur = n_frz / float(fps)
        still_png = frz_dir / f"p{piece_idx:02d}_still.png"
        still_vf = ["-vf", f"crop={crop_box[0]}:{crop_box[1]}:{crop_box[2]}:{crop_box[3]}"] if (apply_crop and crop_box) else []
        still_seek = max(0.0, min(raw_dur - 0.06, ev_t))
        res_still = ffutil.run(
            [
                "ffmpeg", "-y",
                "-ss", f"{still_seek:.3f}",
                "-i", str(chapter_raw_mp4),
                *still_vf,
                "-frames:v", "1",
                str(still_png),
            ],
            check=False,
        )
        if res_still.returncode != 0 or not still_png.exists() or still_png.stat().st_size < 100:
            ffutil.run(
                [
                    "ffmpeg", "-y",
                    "-sseof", "-0.12",
                    "-i", str(chapter_raw_mp4),
                    *still_vf,
                    "-frames:v", "1",
                    str(still_png),
                ]
            )

        vp = frz_dir / f"p{piece_idx:02d}_freeze_v.mp4"
        ap = frz_dir / f"p{piece_idx:02d}_freeze_a.wav"
        denom = max(1, n_frz - 1)
        # Quadratic ease-out 1.00x -> 1.12x zoom + warm contrast/saturation pop + dark vignette
        zp_vf = (
            f"scale={out_w * 2}:{out_h * 2}:flags=lanczos,"
            f"zoompan=z='1+{zoom_delta:.4f}*(2*(on/{denom})-(on/{denom})*(on/{denom}))':"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"d={n_frz}:s={out_w}x{out_h}:fps={fps},setsar=1,"
            f"eq=contrast=1.10:saturation=1.14:brightness=0.02,vignette=PI/5.5,format=yuv420p"
        )
        ffutil.run(
            [
                "ffmpeg", "-y",
                "-i", str(still_png),
                "-vf", zp_vf,
                "-frames:v", str(n_frz),
                "-an",
                *enc_flags,
                "-pix_fmt", "yuv420p", "-r", str(fps), "-video_track_timescale", "90000",
                str(vp),
            ]
        )
        ffutil.run(
            [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", f"anullsrc=r=48000:cl=stereo:d={frz_dur:.6f}",
                "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le",
                str(ap),
            ]
        )
        vid_pieces.append(vp)
        aud_pieces.append(ap)
        piece_idx += 1
        prev_t = ev_t

    _cut_normal(prev_t, raw_dur)

    if not vid_pieces:
        return chapter_raw_mp4

    v_list = frz_dir / "v_concat.txt"
    v_list.write_text("\n".join(f"file '{p.resolve()}'" for p in vid_pieces) + "\n", encoding="utf-8")
    a_list = frz_dir / "a_concat.txt"
    a_list.write_text("\n".join(f"file '{p.resolve()}'" for p in aud_pieces) + "\n", encoding="utf-8")

    concat_v = frz_dir / "expanded_v.mp4"
    concat_a = frz_dir / "expanded_a.wav"
    ffutil.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(v_list), "-c", "copy", str(concat_v)])
    ffutil.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(a_list), "-c", "copy", str(concat_a)])

    expanded_mp4 = work_dir / f"{chapter_raw_mp4.stem}_expanded.mp4"
    ffutil.run(
        [
            "ffmpeg", "-y",
            "-i", str(concat_v),
            "-i", str(concat_a),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest", "-movflags", "+faststart",
            str(expanded_mp4),
        ]
    )
    log.info("Expanded %s with %d freeze roasts -> %s", chapter_raw_mp4.name, len(events), expanded_mp4.name)
    return expanded_mp4


def _build_movie_stage_filter(src_w: int, src_h: int, pre: str, stage_w: int = 1080, stage_h: int = 612) -> str:
    """Build crisp graded movie stage filter cropped cleanly to stage_w x stage_h (<=26% cinemascope / <=18% 16:9)."""
    aspect = src_w / max(1, src_h)
    grade = "eq=contrast=1.06:saturation=1.18:brightness=0.02,unsharp=5:5:0.6:5:5:0.0"
    max_crop = 0.26 if aspect >= 2.1 else 0.18
    max_scaled_w = round(stage_w / (1.0 - max_crop))
    needed_w_for_full_h = round(stage_h * aspect)

    if needed_w_for_full_h <= max_scaled_w + 8:
        return (
            f"[mov_stage_in]{pre}scale={stage_w}:{stage_h}:force_original_aspect_ratio=increase,"
            f"crop={stage_w}:{stage_h},{grade}[mov_crisp]"
        )

    fg_h = max(2, min(stage_h, int(round(max_scaled_w / aspect)) & ~1))
    return (
        f"[mov_stage_in]{pre}scale={max_scaled_w}:{fg_h}:force_original_aspect_ratio=increase,"
        f"crop={stage_w}:{fg_h},{grade}[mov_crisp]"
    )


def build_chapter_composite_filter_complex(
    active_w: int,
    active_h: int,
    dur: float,
    pre: str,
    layout: dict[str, int],
    *,
    has_host_alpha: bool,
    feather_input_idx: int,
    chrome_input_idx: int,
    host_alpha_input_idx: int | None = None,
    ass_flt: str = "",
) -> str:
    """Construct the reconciled 63.5/36.5 full-bleed + feathered movie + 3D pop-out FFmpeg filtergraph."""
    movie_world_h = int(layout.get("movie_world_h", 1220))
    movie_y = int(layout["movie_y"])
    movie_h = int(layout["movie_h"])
    divider_y = int(layout["divider_y"])
    booth_h = int(layout["booth_h"])
    host_overlay_y = int(layout.get("host_overlay_y", divider_y - 50))
    host_overlay_h = int(layout.get("host_overlay_h", booth_h + 50))

    stage_filter = _build_movie_stage_filter(active_w, active_h, pre, stage_w=1080, stage_h=movie_h)

    if has_host_alpha and host_alpha_input_idx is not None:
        host_chain = (
            f"[1:v]fps=30,scale=1080:{host_overlay_h},format=rgba[host_rgb];"
            f"[{host_alpha_input_idx}:v]fps=30,scale=1080:{host_overlay_h},format=gray,"
            f"lut=c0='if(lt(val,12),0,if(gt(val,244),255,val))'[host_a];"
            f"[host_rgb][host_a]alphamerge=shortest=1[host_rgba];"
            f"[s3][host_rgba]overlay=0:{host_overlay_y}:shortest=1{ass_flt}[out]"
        )
    else:
        host_chain = (
            f"[1:v]fps=30,scale=1080:{booth_h}[host];"
            f"[s3][host]overlay=0:{divider_y}:shortest=1{ass_flt}[out]"
        )

    return (
        f"color=c=0x0b0b10:s=1080x1920:d={dur:.3f}[base];"
        f"[0:v]fps=30,split=2[mov_amb_in][mov_stage_in];"
        f"[mov_amb_in]{pre}scale=1080:{movie_world_h}:force_original_aspect_ratio=increase,"
        f"crop=1080:{movie_world_h},boxblur=24:6,eq=brightness=-0.10:saturation=1.25[mov_amb];"
        f"{stage_filter};"
        f"[mov_crisp]format=rgba[mc_rgba];"
        f"[{feather_input_idx}:v]scale=1080:{movie_h},format=gray[mc_mask];"
        f"[mc_rgba][mc_mask]alphamerge=shortest=1[mov_feathered];"
        f"[base][mov_amb]overlay=0:0[s1];"
        f"[s1][mov_feathered]overlay=0:{movie_y}[s2];"
        f"[s2][{chrome_input_idx}:v]overlay=0:0:eof_action=repeat[s3];"
        f"{host_chain}"
    )


def compose_chapter_short(
    source_path: Path,
    chapter_slice: ChapterSlice,
    host_booth_mp4: Path,
    commentary_wav: Path,
    ass_path: Path,
    rights: RightsRecord,
    genre: str,
    out_mp4: Path,
    work_dir: Path,
    config: dict[str, Any],
    *,
    top_headline: str = "",
    expansion_meta: dict[str, Any] | None = None,
    source_is_assembled: bool = False,
    chapter_words: list[dict[str, Any]] | None = None,
    layout: dict[str, int] | None = None,
    crop_box: tuple[int, int, int, int] | None = None,
    host_alpha_mp4: Path | None = None,
) -> Path:
    """Render 63.5/36.5 full-bleed 1080x1920 30fps Short with 3D pop-out alphamerge, feathered movie & auto-bleep."""
    cc_cfg = config.get("chapter_commentary", {})
    audio_cfg = config.get("audio", {})
    normal_db = float(cc_cfg.get("source_audio_normal_db", -6.0))
    bgm_db = float(cc_cfg.get("bgm_fallback_db", -15.0))
    target_lufs = float(audio_cfg.get("target_lufs", -14.0))
    true_peak = float(audio_cfg.get("true_peak_db", -1.0))

    dur = (
        float(expansion_meta["expanded_duration_sec"])
        if expansion_meta and expansion_meta.get("expanded_duration_sec")
        else chapter_slice.duration_sec
    )
    info = ffutil.probe(source_path)
    v_stream = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), {})
    src_w = int(v_stream.get("width", 1920))
    src_h = int(v_stream.get("height", 1080))
    has_src_audio = any(s.get("codec_type") == "audio" for s in info.get("streams", []))

    active_w = crop_box[0] if crop_box else src_w
    active_h = crop_box[1] if crop_box else src_h
    lay = layout or compute_vertical_layout(active_w, active_h)
    movie_y = int(lay["movie_y"])
    movie_h = int(lay["movie_h"])
    divider_y = int(lay["divider_y"])

    alpha_cand = host_alpha_mp4 if host_alpha_mp4 is not None else host_animator.host_alpha_path_for(host_booth_mp4)
    has_host_alpha = bool(host_alpha_mp4 is not None or (alpha_cand.exists() and alpha_cand.stat().st_size > 100))

    sfx_str = str(expansion_meta.get("sfx_wav", "")).strip() if expansion_meta else ""
    sfx_path = Path(sfx_str) if sfx_str else None
    has_sfx = bool(sfx_path and sfx_path.is_file() and sfx_path.stat().st_size > 100)

    # 1. Mix Audio: -6dB movie audio (surgically auto-bleeped) -> sidechain-ducked + -9 dBFS freeze SFX
    master_m4a = work_dir / "chapter_master_audio.m4a"
    seek_args = [] if source_is_assembled else ["-ss", f"{chapter_slice.start_sec:.3f}"]
    bleeped_words: list[dict[str, Any]] = []

    if not rights.mute_source_audio and has_src_audio:
        src_aud = work_dir / "src_chapter_audio.wav"
        ffutil.run(
            [
                "ffmpeg", "-y",
                *seek_args,
                "-t", f"{dur:.3f}",
                "-i", str(source_path),
                "-vn", "-ar", "48000", "-ac", "2", str(src_aud),
            ]
        )
        movie_aud_for_mix = src_aud
        if chapter_words and src_aud.exists():
            shifted = shift_words_for_freezes(
                chapter_words,
                list(expansion_meta.get("freeze_events", [])) if expansion_meta else [],
            )
            bleeped_aud = work_dir / "src_chapter_audio_bleeped.wav"
            movie_aud_for_mix, bleeped_words = auto_bleep_profanity(src_aud, shifted, bleeped_aud)

        aud_cmd = ["ffmpeg", "-y", "-i", str(commentary_wav), "-i", str(movie_aud_for_mix)]
        if has_sfx and sfx_path is not None:
            aud_cmd.extend(["-i", str(sfx_path)])
            mix_fc = (
                f"[0:a]asplit=2[v_sc][v_mix];"
                f"[1:a]volume={normal_db}dB[bg_lvl];"
                f"[bg_lvl][v_sc]sidechaincompress=threshold=0.018:ratio=9:attack=12:release=280:makeup=1[bg_duck];"
                f"[2:a]volume=0dB[sfx_lvl];"
                f"[v_mix][bg_duck][sfx_lvl]amix=inputs=3:duration=first:normalize=0,"
                f"loudnorm=I={target_lufs}:TP={true_peak}:LRA=7[out]"
            )
        else:
            mix_fc = (
                f"[0:a]asplit=2[v_sc][v_mix];"
                f"[1:a]volume={normal_db}dB[bg_lvl];"
                f"[bg_lvl][v_sc]sidechaincompress=threshold=0.018:ratio=9:attack=12:release=280:makeup=1[bg_duck];"
                f"[v_mix][bg_duck]amix=inputs=2:duration=first:normalize=0,"
                f"loudnorm=I={target_lufs}:TP={true_peak}:LRA=7[out]"
            )
        aud_cmd.extend(["-filter_complex", mix_fc, "-map", "[out]", "-ar", "48000", "-ac", "2", "-c:a", "aac", "-b:a", "192k", str(master_m4a)])
        ffutil.run(aud_cmd)
    else:
        bgm_wav, _ = audio_mixer._pick_or_synth_bgm(dur, work_dir, genre, allow_network=True)
        aud_cmd = ["ffmpeg", "-y", "-i", str(commentary_wav), "-i", str(bgm_wav)]
        if has_sfx and sfx_path is not None:
            aud_cmd.extend(["-i", str(sfx_path)])
            mix_fc = (
                f"[0:a]asplit=2[v_sc][v_mix];"
                f"[1:a]highpass=f=80,equalizer=f=1800:t=q:w=1.2:g=-4,volume={bgm_db}dB[bg_lvl];"
                f"[bg_lvl][v_sc]sidechaincompress=threshold=0.025:ratio=5:attack=15:release=240[bg_duck];"
                f"[2:a]volume=0dB[sfx_lvl];"
                f"[v_mix][bg_duck][sfx_lvl]amix=inputs=3:duration=first:normalize=0,"
                f"loudnorm=I={target_lufs}:TP={true_peak}:LRA=7[out]"
            )
        else:
            mix_fc = (
                f"[0:a]asplit=2[v_sc][v_mix];"
                f"[1:a]highpass=f=80,equalizer=f=1800:t=q:w=1.2:g=-4,volume={bgm_db}dB[bg_lvl];"
                f"[bg_lvl][v_sc]sidechaincompress=threshold=0.025:ratio=5:attack=15:release=240[bg_duck];"
                f"[v_mix][bg_duck]amix=inputs=2:duration=first:normalize=0,"
                f"loudnorm=I={target_lufs}:TP={true_peak}:LRA=7[out]"
            )
        aud_cmd.extend(["-filter_complex", mix_fc, "-map", "[out]", "-ar", "48000", "-ac", "2", "-c:a", "aac", "-b:a", "192k", str(master_m4a)])
        ffutil.run(aud_cmd)

    # Persist bleeped_words for run_report.json & QA audit
    (work_dir / "bleeped_words.json").write_text(json.dumps(bleeped_words, indent=2) + "\n", encoding="utf-8")
    if expansion_meta is not None:
        expansion_meta["bleeped_words"] = bleeped_words

    # 2. Render top header (PART pill + uppercase top_headline at y=95..185) & 18px movie feather mask
    headline = top_headline or chapter_slice.top_headline or ""
    chrome_png = _render_header_and_divider_png(
        chapter_slice.part_number,
        chapter_slice.total_parts,
        headline,
        genre,
        work_dir / f"part_{chapter_slice.part_number}_chrome.png",
        movie_y=movie_y,
        divider_y=divider_y,
        include_divider=not has_host_alpha,
    )
    feather_png = _build_feather_mask_png(
        1080, movie_h, work_dir / f"part_{chapter_slice.part_number}_movie_feather.png", feather_px=18
    )

    # 3. Single-pass 63.5/36.5 full-bleed composite + 3D pop-out host alphamerge + final ASS caption burn
    sample_ts = [2.0] if source_is_assembled else [chapter_slice.start_sec + 2.0]
    deint = video_composer.should_deinterlace(source_path, sample_ts, config, work_year=rights.year)
    pre_parts: list[str] = []
    if deint:
        pre_parts.append("bwdif=mode=0:parity=-1:deint=0")
    if crop_box and (crop_box[0] != src_w or crop_box[1] != src_h) and (crop_box[0] + crop_box[2] <= src_w) and (crop_box[1] + crop_box[3] <= src_h):
        cw, ch, cx, cy = crop_box
        pre_parts.append(f"crop={cw}:{ch}:{cx}:{cy}")
    pre = ",".join(pre_parts) + "," if pre_parts else ""

    ass_esc = str(ass_path).replace("\\", "/").replace(":", "\\:")
    fonts_dir = str(video_composer._FONTS_DIR).replace("\\", "/").replace(":", "\\:")
    ass_flt = f",ass='{ass_esc}':fontsdir='{fonts_dir}'" if ass_path.exists() and ass_path.stat().st_size > 10 else ""

    # Inputs: 0=movie, 1=host_rgb, 2=master_audio, 3=chrome_png, 4=feather_png, [5=host_alpha]
    cmd = [
        "ffmpeg", "-y",
        *seek_args,
        "-t", f"{dur:.3f}",
        "-i", str(source_path),
        "-i", str(host_booth_mp4),
        "-i", str(master_m4a),
        "-i", str(chrome_png),
        "-loop", "1", "-framerate", "30", "-t", f"{dur + 1.0:.3f}", "-i", str(feather_png),
    ]
    host_alpha_idx: int | None = None
    if has_host_alpha:
        cmd.extend(["-i", str(alpha_cand)])
        host_alpha_idx = 5

    fc = build_chapter_composite_filter_complex(
        active_w,
        active_h,
        dur,
        pre,
        lay,
        has_host_alpha=has_host_alpha,
        feather_input_idx=4,
        chrome_input_idx=3,
        host_alpha_input_idx=host_alpha_idx,
        ass_flt=ass_flt,
    )

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    enc_flags = video_composer.select_h264_encoder(config, intermediate=False)
    cmd.extend(
        [
            "-filter_complex", fc,
            "-map", "[out]", "-map", "2:a:0",
            *enc_flags,
            "-pix_fmt", "yuv420p", "-r", "30",
            "-c:a", "copy", "-shortest", "-movflags", "+faststart",
            str(out_mp4),
        ]
    )
    ffutil.run(cmd)
    return out_mp4
