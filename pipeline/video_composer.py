"""Render-then-concat 9:16 video composer with blurred_letterbox and banner overlay."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from pipeline import ffutil
from pipeline.models import ProjectManifest, Shot

log = logging.getLogger(__name__)

_FONTS_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"


def _render_banner_png(banner_text: str, width: int, out_png: Path) -> Path:
    """Render a rounded dark pill banner PNG with white bold text."""
    pad_x, pad_y = 44, 22
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont = ImageFont.load_default()
    for candidate in (
        _FONTS_DIR / "Montserrat-ExtraBold.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ):
        if candidate.exists():
            font = ImageFont.truetype(str(candidate), 46)
            break

    dummy = Image.new("RGBA", (width, 160), (0, 0, 0, 0))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), banner_text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    pill_w = min(width - 80, tw + pad_x * 2)
    pill_h = th + pad_y * 2
    img = Image.new("RGBA", (width, pill_h + 10), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    x0 = (width - pill_w) // 2
    d.rounded_rectangle(
        [x0, 4, x0 + pill_w, 4 + pill_h],
        radius=24,
        fill=(15, 15, 20, 215),
        outline=(255, 230, 0, 230),
        width=3,
    )
    tx = (width - tw) // 2 - bbox[0]
    ty = 4 + (pill_h - th) // 2 - bbox[1]
    d.text((tx, ty), banner_text, font=font, fill=(255, 255, 255, 255))
    img.save(out_png, format="PNG")
    return out_png


def _render_single_shot(
    shot: Shot,
    source_path: Path,
    banner_png: Path | None,
    out_mp4: Path,
    config: dict[str, Any],
) -> Path:
    """Extract, scale/blur/zoom, grade, and frame-lock a single shot intermediate."""
    rt = config.get("runtime", {})
    layout = config.get("layout", {})
    grade = config.get("grade", {})
    enc = config.get("encode", {})

    fps = int(rt.get("fps", 30))
    width = int(rt.get("width", 1080))
    height = int(rt.get("height", 1920))
    bg_blur = str(layout.get("bg_blur", "boxblur=25:5"))
    bg_bright = float(layout.get("bg_brightness", -0.22))
    fg_zoom = float(layout.get("fg_zoom_pct", 2))
    banner_y = int(layout.get("banner", {}).get("y", 200))

    fg_w = int(round(width * (1.0 + fg_zoom / 100.0)))
    if fg_w % 2:
        fg_w += 1

    src_dur = max(0.2, shot.src_end_sec - shot.src_start_sec)
    pts_expr = f"PTS/{shot.speed_factor:.4f}"

    fg_chain = f"scale={fg_w}:-2,crop={width}:ih:(iw-{width})/2:0"
    if grade.get("enabled", True):
        eq_str = str(grade.get("eq", "contrast=1.04:saturation=1.06"))
        unsharp_str = str(grade.get("unsharp", "3:3:0.3"))
        fg_chain += f",eq={eq_str},unsharp={unsharp_str}"

    fc_parts = [
        f"[0:v]setpts={pts_expr},fps={fps},tpad=stop_mode=clone:stop_duration=2,split=2[bg_in][fg_in]",
        f"[bg_in]scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},{bg_blur},eq=brightness={bg_bright}[bg]",
        f"[fg_in]{fg_chain}[fg]",
        "[bg][fg]overlay=(W-w)/2:(H-h)/2[comp]",
    ]

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{shot.src_start_sec:.3f}",
        "-i",
        str(source_path),
        "-t",
        f"{src_dur + 0.5:.3f}",
    ]
    if banner_png is not None:
        cmd.extend(["-i", str(banner_png)])
        fc_parts.append(f"[comp][1:v]overlay=0:{banner_y}[out]")
        out_label = "[out]"
    else:
        out_label = "[comp]"

    cmd.extend(
        [
            "-filter_complex",
            ";".join(fc_parts),
            "-map",
            out_label,
            "-an",
            "-frames:v",
            str(shot.frames),
            "-c:v",
            "libx264",
            "-preset",
            str(enc.get("intermediate_preset", "ultrafast")),
            "-crf",
            str(enc.get("intermediate_crf", 16)),
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(fps),
            str(out_mp4),
        ]
    )
    ffutil.run(cmd)
    return out_mp4


def compose_video(
    manifest: ProjectManifest,
    shots: list[Shot],
    source_path: Path,
    master_audio: Path,
    ass_path: Path,
    work_dir: Path,
    config: dict[str, Any],
) -> tuple[Path, list[Path]]:
    """Render all shot intermediates, concat, burn captions, and mux final_short.mp4."""
    rt = config.get("runtime", {})
    layout = config.get("layout", {})
    enc = config.get("encode", {})

    fps = int(rt.get("fps", 30))
    width = int(rt.get("width", 1080))
    threads = int(rt.get("threads", 2))
    parallel_jobs = max(1, int(enc.get("parallel_shot_jobs", 2)))
    preset = str(enc.get("preset", "veryfast"))
    crf = str(enc.get("crf", 20))

    inter_dir = work_dir / "intermediates"
    inter_dir.mkdir(parents=True, exist_ok=True)
    out_dir = work_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    banner_png: Path | None = None
    if layout.get("banner", {}).get("enabled", True) and manifest.banner_text:
        banner_png = _render_banner_png(
            manifest.banner_text, width, work_dir / "banner.png"
        )

    tasks: list[tuple[Shot, Path]] = [
        (s, inter_dir / f"shot_{s.beat_id:03d}_{s.shot_index:02d}.mp4")
        for s in shots
    ]

    with ThreadPoolExecutor(max_workers=parallel_jobs) as pool:
        futures = [
            pool.submit(_render_single_shot, s, source_path, banner_png, p, config)
            for s, p in tasks
        ]
        intermediates = [f.result() for f in futures]

    concat_txt = inter_dir / "shots_concat.txt"
    concat_txt.write_text(
        "\n".join(f"file '{p.resolve()}'" for p in intermediates) + "\n",
        encoding="utf-8",
    )
    concat_mp4 = work_dir / "concat_video.mp4"
    ffutil.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_txt),
            "-c",
            "copy",
            str(concat_mp4),
        ]
    )

    expected_frames = sum(s.frames for s in shots)
    actual_frames = ffutil.frame_count(concat_mp4)
    if actual_frames != expected_frames:
        raise AssertionError(
            f"Concat frame count {actual_frames} != expected {expected_frames}"
        )

    final_mp4 = out_dir / "final_short.mp4"
    ass_escaped = str(ass_path).replace("\\", "/").replace(":", "\\:")
    fonts_dir = str(_FONTS_DIR).replace("\\", "/").replace(":", "\\:")
    vf = f"ass='{ass_escaped}':fontsdir='{fonts_dir}'"

    ffutil.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(concat_mp4),
            "-i",
            str(master_audio),
            "-vf",
            vf,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            crf,
            "-threads",
            str(threads),
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(fps),
            "-c:a",
            "copy",
            "-shortest",
            "-movflags",
            "+faststart",
            str(final_mp4),
        ]
    )
    return final_mp4, intermediates
