"""Single-pass split-screen composer for chapter_commentary mode."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pipeline import audio_mixer, ffutil, video_composer
from pipeline.models import ChapterSlice, RightsRecord

log = logging.getLogger(__name__)


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
) -> Path:
    """Render 1080x1920 30fps split-screen chapter Short with ducked source audio (or CC-BY BGM)."""
    cc_cfg = config.get("chapter_commentary", {})
    audio_cfg = config.get("audio", {})
    ducked_db = float(cc_cfg.get("source_audio_ducked_db", -16.0))
    target_lufs = float(audio_cfg.get("target_lufs", -14.0))
    true_peak = float(audio_cfg.get("true_peak_db", -1.0))

    dur = chapter_slice.duration_sec
    info = ffutil.probe(source_path)
    v_stream = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), {})
    src_w = int(v_stream.get("width", 1920))
    src_h = int(v_stream.get("height", 1080))
    aspect = src_w / max(1, src_h)

    # 1. Mix Audio (preserve original movie audio unless mute_source_audio=True)
    master_m4a = work_dir / "chapter_master_audio.m4a"
    has_src_audio = any(s.get("codec_type") == "audio" for s in info.get("streams", []))

    if not rights.mute_source_audio and has_src_audio:
        src_aud = work_dir / "src_chapter_audio.wav"
        ffutil.run([
            "ffmpeg", "-y",
            "-ss", f"{chapter_slice.start_sec:.3f}",
            "-t", f"{dur:.3f}",
            "-i", str(source_path),
            "-vn", "-ar", "48000", "-ac", "2", str(src_aud),
        ])
        ffutil.run([
            "ffmpeg", "-y",
            "-i", str(commentary_wav),
            "-i", str(src_aud),
            "-filter_complex",
            (
                f"[0:a]asplit=2[v_sc][v_mix];"
                f"[1:a]volume={ducked_db}dB[bg_lvl];"
                f"[bg_lvl][v_sc]sidechaincompress=threshold=0.025:ratio=6:attack=15:release=320[bg_duck];"
                f"[v_mix][bg_duck]amix=inputs=2:duration=first:normalize=0,"
                f"loudnorm=I={target_lufs}:TP={true_peak}:LRA=7[out]"
            ),
            "-map", "[out]", "-ar", "48000", "-ac", "2",
            "-c:a", "aac", "-b:a", "192k", str(master_m4a),
        ])
    else:
        bgm_wav, _ = audio_mixer._pick_or_synth_bgm(dur, work_dir, genre, allow_network=True)
        ffutil.run([
            "ffmpeg", "-y",
            "-i", str(commentary_wav),
            "-i", str(bgm_wav),
            "-filter_complex",
            (
                f"[0:a]asplit=2[v_sc][v_mix];"
                f"[1:a]highpass=f=80,equalizer=f=1800:t=q:w=1.2:g=-4,volume=-15dB[bg_lvl];"
                f"[bg_lvl][v_sc]sidechaincompress=threshold=0.03:ratio=4:attack=15:release=220[bg_duck];"
                f"[v_mix][bg_duck]amix=inputs=2:duration=first:normalize=0,"
                f"loudnorm=I={target_lufs}:TP={true_peak}:LRA=7[out]"
            ),
            "-map", "[out]", "-ar", "48000", "-ac", "2",
            "-c:a", "aac", "-b:a", "192k", str(master_m4a),
        ])

    # 2. Composite 1080x1920 Split-Screen
    deint = video_composer.should_deinterlace(source_path, [chapter_slice.start_sec + 2.0], config, work_year=rights.year)
    pre = "bwdif=mode=0:parity=-1:deint=0," if deint else ""

    ass_esc = str(ass_path).replace("\\", "/").replace(":", "\\:")
    fonts_dir = str(video_composer._FONTS_DIR).replace("\\", "/").replace(":", "\\:")
    ass_flt = f",ass='{ass_esc}':fontsdir='{fonts_dir}'" if ass_path.exists() and ass_path.stat().st_size > 10 else ""

    if aspect > 1.45:
        fc = (
            f"color=c=0x0b0b10:s=1080x1920:d={dur:.3f}[base];"
            f"[0:v]{pre}fps=30,scale=1080:608:force_original_aspect_ratio=decrease,"
            f"pad=1080:608:(ow-iw)/2:(oh-ih)/2:black[mov];"
            f"[1:v]fps=30,scale=1080:690[host];"
            f"[base][mov]overlay=0:220[with_mov];"
            f"[with_mov][host]overlay=0:945{ass_flt}[out]"
        )
    else:
        fc = (
            f"color=c=0x0b0b10:s=1080x1920:d={dur:.3f}[base];"
            f"[0:v]{pre}fps=30,scale=1014:760:force_original_aspect_ratio=decrease,"
            f"pad=1014:760:(ow-iw)/2:(oh-ih)/2:black[mov];"
            f"[1:v]fps=30,scale=1080:630[host];"
            f"[base][mov]overlay=33:150[with_mov];"
            f"[with_mov][host]overlay=0:1015{ass_flt}[out]"
        )

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    enc_flags = video_composer.select_h264_encoder(config, intermediate=False)
    ffutil.run([
        "ffmpeg", "-y",
        "-ss", f"{chapter_slice.start_sec:.3f}",
        "-t", f"{dur:.3f}",
        "-i", str(source_path),
        "-i", str(host_booth_mp4),
        "-i", str(master_m4a),
        "-filter_complex", fc,
        "-map", "[out]", "-map", "2:a:0",
        *enc_flags,
        "-pix_fmt", "yuv420p", "-r", "30",
        "-c:a", "copy", "-shortest", "-movflags", "+faststart",
        str(out_mp4),
    ])
    return out_mp4
