"""Single-pass zero-dead-space 9:16 split-screen composer + freeze-frame expander."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pipeline import audio_mixer, ffutil, video_composer
from pipeline.models import ChapterSlice, RightsRecord

log = logging.getLogger(__name__)

_GENRE_ACCENTS: dict[str, tuple[int, int, int]] = {
    "comedy": (255, 216, 77),
    "horror_comedy": (46, 230, 197),
    "action_scifi": (44, 224, 255),
    "emotional_drama": (255, 184, 92),
}


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
) -> Path:
    """Render 1080x1920 RGBA chrome: top gradient scrim, PART pill, headline, and glowing y=946 divider."""
    w, h = 1080, 1920
    accent = _GENRE_ACCENTS.get(genre, _GENRE_ACCENTS["comedy"])

    # Fast NumPy creation of smooth top dark gradient scrim (y=0..208) + glowing divider (y=938..956)
    arr = np.zeros((h, w, 4), dtype=np.uint8)

    # Top scrim so blurred ambient header keeps rich color while headline pops crisply
    scrim_h = 208
    ys = np.linspace(0.0, 1.0, scrim_h, dtype=np.float32)
    scrim_a = (165.0 * (1.0 - 0.52 * ys)).astype(np.uint8)
    arr[:scrim_h, :, 0] = 8
    arr[:scrim_h, :, 1] = 10
    arr[:scrim_h, :, 2] = 16
    arr[:scrim_h, :, 3] = scrim_a[:, None]

    # Subtle glowing 4px genre-accent divider centered at y=946 (halo y=936..956)
    div_center = 946
    for y in range(div_center - 10, div_center + 11):
        dist = abs(y - div_center)
        if dist <= 1:
            # Luminous core
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

    # 1. Semi-transparent dark-glass PART k/N pill (y=56..102)
    pill_text = f"PART {part_number}/{total_parts}" if total_parts > 1 else "MOVIE RECAP"
    pill_font = _load_bold_font(25)
    pb = draw.textbbox((0, 0), pill_text, font=pill_font)
    pw = (pb[2] - pb[0]) + 42
    ph = 44
    px0 = (w - pw) // 2
    py0 = 56
    draw.rounded_rectangle(
        [px0, py0, px0 + pw, py0 + ph],
        radius=22,
        fill=(14, 16, 24, 212),
        outline=(*accent, 230),
        width=2,
    )
    ptx = (w - (pb[2] - pb[0])) // 2 - pb[0]
    pty = py0 + (ph - (pb[3] - pb[1])) // 2 - pb[1]
    draw.text((ptx, pty), pill_text, font=pill_font, fill=(*accent, 255))

    # 2. Bold uppercase 4-6 word top_headline centered at y=110..196
    headline = " ".join(str(top_headline or "").strip().upper().split())
    if headline:
        max_w = 970
        chosen_font = _load_bold_font(46)
        chosen_lines = [headline]
        for fsize in (48, 44, 40, 36):
            fnt = _load_bold_font(fsize)
            hb = draw.textbbox((0, 0), headline, font=fnt)
            if (hb[2] - hb[0]) <= max_w:
                chosen_font = fnt
                chosen_lines = [headline]
                break
        else:
            # Wrap cleanly onto 2 balanced lines if still wide
            words = headline.split()
            mid = max(1, len(words) // 2)
            chosen_font = _load_bold_font(36)
            chosen_lines = [" ".join(words[:mid]), " ".join(words[mid:])]

        line_heights: list[int] = []
        line_bboxes: list[tuple[int, int, int, int]] = []
        for ln in chosen_lines:
            bb = draw.textbbox((0, 0), ln, font=chosen_font)
            line_bboxes.append(bb)
            line_heights.append(bb[3] - bb[1])

        gap = 6
        total_h = sum(line_heights) + gap * (len(chosen_lines) - 1)
        cur_y = 112 + max(0, (82 - total_h) // 2)
        for ln, bb, lh in zip(chosen_lines, line_bboxes, line_heights):
            lw = bb[2] - bb[0]
            tx = (w - lw) // 2 - bb[0]
            ty = cur_y - bb[1]
            # Soft drop shadow + crisp dark stroke for high legibility on bright frames
            draw.text((tx + 2, ty + 3), ln, font=chosen_font, fill=(0, 0, 0, 210))
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
) -> Path:
    """Insert frame-exact 1.00x->1.07x slow-zoom freeze holds with muted movie audio at roast beats."""
    if not freeze_events:
        return chapter_raw_mp4

    fps = int(config.get("runtime", {}).get("fps", 30))
    zoom_end = float(config.get("chapter_commentary", {}).get("freeze_zoom_end", 1.07))
    zoom_delta = max(0.02, zoom_end - 1.0)

    info = ffutil.probe(chapter_raw_mp4)
    v_stream = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), {})
    src_w = int(v_stream.get("width", 1920))
    src_h = int(v_stream.get("height", 1080))
    src_w -= src_w % 2
    src_h -= src_h % 2
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
                "-vf", f"fps={fps},scale={src_w}:{src_h},setsar=1,tpad=stop_mode=clone:stop_duration=0.5",
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
        ev_t = max(prev_t, min(raw_dur - 0.1, float(ev["raw_time_sec"])))
        _cut_normal(prev_t, ev_t)

        # Extract exact frame at ev_t and render silky 1.00x -> 1.07x zoom freeze hold
        n_frz = max(int(round(1.0 * fps)), int(ev.get("freeze_frames") or round(float(ev["freeze_dur_sec"]) * fps)))
        frz_dur = n_frz / float(fps)
        still_png = frz_dir / f"p{piece_idx:02d}_still.png"
        ffutil.run(
            [
                "ffmpeg", "-y",
                "-ss", f"{ev_t:.3f}",
                "-i", str(chapter_raw_mp4),
                "-frames:v", "1",
                str(still_png),
            ]
        )
        vp = frz_dir / f"p{piece_idx:02d}_freeze_v.mp4"
        ap = frz_dir / f"p{piece_idx:02d}_freeze_a.wav"
        denom = max(1, n_frz - 1)
        zp_vf = (
            f"scale={src_w * 2}:{src_h * 2}:flags=lanczos,"
            f"zoompan=z='1+{zoom_delta:.4f}*on/{denom}':"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"d={n_frz}:s={src_w}x{src_h}:fps={fps},setsar=1,format=yuv420p"
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
        # Movie audio cleanly mutes during freeze-frame roast beat
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


def _build_movie_stage_filter(src_w: int, src_h: int, pre: str, stage_w: int = 1080, stage_h: int = 742) -> str:
    """Build zero-black-bar movie stage filter with <=22% horizontal trim and ambient blur fallback."""
    aspect = src_w / max(1, src_h)
    grade = "eq=contrast=1.06:saturation=1.18:brightness=0.02,unsharp=5:5:0.6:5:5:0.0"

    # Max 22% horizontal trim => scaled_w <= stage_w / 0.78 (~1384px)
    max_scaled_w = round(stage_w / 0.78)
    needed_w_for_full_h = round(stage_h * aspect)

    if needed_w_for_full_h <= max_scaled_w:
        # 16:9, 4:3, and moderately wide films fill 1080x742 edge-to-edge cleanly within <=22% trim
        return (
            f"[mov_stage_in]{pre}scale={stage_w}:{stage_h}:force_original_aspect_ratio=increase,"
            f"crop={stage_w}:{stage_h},{grade}[mov_stage]"
        )

    # Ultra-wide CinemaScope (>~1.87:1): cap trim at 22% and back remaining strip with vibrant blurred movie
    fg_h = max(2, int(round(max_scaled_w / aspect)) & ~1)
    fg_h = min(stage_h, fg_h)
    return (
        f"[mov_stage_in]{pre}split=2[ms_bg][ms_fg];"
        f"[ms_bg]scale={stage_w}:{stage_h}:force_original_aspect_ratio=increase,"
        f"crop={stage_w}:{stage_h},boxblur=22:5,eq=brightness=-0.10:saturation=1.22[ms_amb];"
        f"[ms_fg]scale={max_scaled_w}:{fg_h}:force_original_aspect_ratio=increase,"
        f"crop={stage_w}:{fg_h},{grade}[ms_crisp];"
        f"[ms_amb][ms_crisp]overlay=0:(H-h)/2[mov_stage]"
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
) -> Path:
    """Render viral zero-dead-space 1080x1920 30fps Short with ambient header, Kiko booth & smart ducking."""
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

    sfx_str = str(expansion_meta.get("sfx_wav", "")).strip() if expansion_meta else ""
    sfx_path = Path(sfx_str) if sfx_str else None
    has_sfx = bool(sfx_path and sfx_path.is_file() and sfx_path.stat().st_size > 100)

    # 1. Mix Audio: -6dB rich movie audio -> sidechain-ducked to ~-18dB when Kiko talks (+ optional freeze SFX)
    master_m4a = work_dir / "chapter_master_audio.m4a"
    seek_args = [] if source_is_assembled else ["-ss", f"{chapter_slice.start_sec:.3f}"]

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
        aud_cmd = ["ffmpeg", "-y", "-i", str(commentary_wav), "-i", str(src_aud)]
        if has_sfx and sfx_path is not None:
            aud_cmd.extend(["-i", str(sfx_path)])
            mix_fc = (
                f"[0:a]asplit=2[v_sc][v_mix];"
                f"[1:a]volume={normal_db}dB[bg_lvl];"
                f"[bg_lvl][v_sc]sidechaincompress=threshold=0.018:ratio=9:attack=12:release=280:makeup=1[bg_duck];"
                f"[2:a]volume=-6dB[sfx_lvl];"
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
                f"[2:a]volume=-6dB[sfx_lvl];"
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

    # 2. Render top header (PART pill + uppercase top_headline) + glowing y=946 divider overlay PNG
    headline = top_headline or chapter_slice.top_headline or ""
    chrome_png = _render_header_and_divider_png(
        chapter_slice.part_number,
        chapter_slice.total_parts,
        headline,
        genre,
        work_dir / f"part_{chapter_slice.part_number}_chrome.png",
    )

    # 3. Single-pass zero-dead-space 1080x1920 composite
    sample_ts = [2.0] if source_is_assembled else [chapter_slice.start_sec + 2.0]
    deint = video_composer.should_deinterlace(source_path, sample_ts, config, work_year=rights.year)
    pre = "bwdif=mode=0:parity=-1:deint=0," if deint else ""

    ass_esc = str(ass_path).replace("\\", "/").replace(":", "\\:")
    fonts_dir = str(video_composer._FONTS_DIR).replace("\\", "/").replace(":", "\\:")
    ass_flt = f",ass='{ass_esc}':fontsdir='{fonts_dir}'" if ass_path.exists() and ass_path.stat().st_size > 10 else ""

    stage_filter = _build_movie_stage_filter(src_w, src_h, pre, stage_w=1080, stage_h=742)
    fc = (
        f"color=c=0x0b0b10:s=1080x1920:d={dur:.3f}[base];"
        f"[0:v]fps=30,split=2[mov_top_in][mov_stage_in];"
        f"[mov_top_in]{pre}scale=1080:210:force_original_aspect_ratio=increase,"
        f"crop=1080:206,boxblur=24:6,eq=brightness=-0.08:saturation=1.28[top_amb];"
        f"{stage_filter};"
        f"[1:v]fps=30,scale=1080:974[host];"
        f"[base][top_amb]overlay=0:0[s1];"
        f"[s1][mov_stage]overlay=0:205[s2];"
        f"[s2][host]overlay=0:946[s3];"
        f"[s3][3:v]overlay=0:0:eof_action=repeat{ass_flt}[out]"
    )

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    enc_flags = video_composer.select_h264_encoder(config, intermediate=False)
    ffutil.run(
        [
            "ffmpeg", "-y",
            *seek_args,
            "-t", f"{dur:.3f}",
            "-i", str(source_path),
            "-i", str(host_booth_mp4),
            "-i", str(master_m4a),
            "-i", str(chrome_png),
            "-filter_complex", fc,
            "-map", "[out]", "-map", "2:a:0",
            *enc_flags,
            "-pix_fmt", "yuv420p", "-r", "30",
            "-c:a", "copy", "-shortest", "-movflags", "+faststart",
            str(out_mp4),
        ]
    )
    return out_mp4
