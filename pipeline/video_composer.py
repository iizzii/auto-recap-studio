"""Render-then-concat 9:16 video composer with blurred_letterbox and banner overlay."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from pipeline import deinterlace, ffutil
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


def compute_fg_dimensions(
    src_w: int,
    src_h: int,
    width: int = 1080,
    min_fg_height_px: int = 760,
    max_horizontal_crop_pct: float = 42.0,
) -> tuple[int, int]:
    """Compute ``(scaled_w, target_h)`` for the foreground layer, boosting ultra-wide 2.39:1."""
    aspect = max(0.1, src_w / max(1, src_h))
    nat_h = round(width / aspect)
    target_h = max(nat_h, min_fg_height_px)
    scaled_w = round(target_h * aspect)
    if scaled_w > width and (scaled_w - width) / scaled_w > max_horizontal_crop_pct / 100.0:
        scaled_w = int(width / (1.0 - max_horizontal_crop_pct / 100.0))
        target_h = int(scaled_w / aspect)
    target_h = target_h - (target_h % 2)
    return scaled_w, max(2, target_h)


def should_deinterlace(
    source_path: Path,
    sample_timestamps: list[float],
    config: dict[str, Any],
    work_year: int | None = None,
) -> bool:
    """Return True if deinterlacing should be applied (vintage year <= 1985 or combing detected)."""
    layout_cfg = config.get("layout", {})
    if not layout_cfg.get("auto_deinterlace", True):
        return False
    vintage_year = int(layout_cfg.get("vintage_deinterlace_year", 1985))
    if work_year is not None and work_year <= vintage_year:
        return True
    threshold = float(layout_cfg.get("combing_threshold", 1.12))
    return deinterlace.detect_combing(source_path, sample_timestamps, threshold=threshold)


def _render_single_shot(
    shot: Shot,
    source_path: Path,
    banner_png: Path | None,
    out_mp4: Path,
    config: dict[str, Any],
    src_dims: tuple[int, int] = (1920, 1080),
    deinterlace_applied: bool = False,
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
    min_fg_h = int(layout.get("min_fg_height_px", 760))
    max_crop_pct = float(layout.get("max_horizontal_crop_pct", 42.0))
    banner_y = int(layout.get("banner", {}).get("y", 200))

    _, target_h = compute_fg_dimensions(
        src_dims[0],
        src_dims[1],
        width=width,
        min_fg_height_px=min_fg_h,
        max_horizontal_crop_pct=max_crop_pct,
    )

    src_dur = max(0.2, shot.src_end_sec - shot.src_start_sec)
    pts_expr = f"PTS/{shot.speed_factor:.4f}"

    fg_chain = f"scale=-2:{target_h},crop={width}:{target_h}:(iw-{width})/2:0"
    if grade.get("enabled", True):
        eq_str = str(grade.get("eq", "contrast=1.04:saturation=1.06"))
        unsharp_str = str(grade.get("unsharp", "3:3:0.3"))
        fg_chain += f",eq={eq_str},unsharp={unsharp_str}"

    pre_filters: list[str] = []
    if deinterlace_applied:
        pre_filters.append("bwdif=mode=0:parity=-1:deint=0")
    overscan_trim_pct = float(layout.get("overscan_trim_pct", 0.0))
    if overscan_trim_pct > 0:
        p = overscan_trim_pct / 100.0
        pre_filters.append(
            f"crop=trunc(iw*(1-2*{p:.4f})/2)*2:trunc(ih*(1-2*{p:.4f})/2)*2:"
            f"trunc(iw*{p:.4f}/2)*2:trunc(ih*{p:.4f}/2)*2"
        )
    pre_prefix = ",".join(pre_filters) + "," if pre_filters else ""

    fc_parts = [
        f"[0:v]{pre_prefix}setpts={pts_expr},fps={fps},tpad=stop_mode=clone:stop_duration=2,split=2[bg_in][fg_in]",
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
    work_year: int | None = None,
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

    info = ffutil.probe(source_path)
    v_stream = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), {})
    src_dims = (int(v_stream.get("width", 1920)), int(v_stream.get("height", 1080)))

    sample_ts = [round((s.src_start_sec + s.src_end_sec) * 0.5, 2) for s in shots[:10]]
    deinterlace_applied = should_deinterlace(source_path, sample_ts, config, work_year=work_year) if shots else False

    with ThreadPoolExecutor(max_workers=parallel_jobs) as pool:
        futures = [
            pool.submit(
                _render_single_shot,
                s,
                source_path,
                banner_png,
                p,
                config,
                src_dims,
                deinterlace_applied,
            )
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
